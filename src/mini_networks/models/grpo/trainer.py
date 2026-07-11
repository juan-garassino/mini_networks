"""GRPO trainer: rlhf's pipeline with group-relative advantages, no value net."""
from __future__ import annotations

import copy
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.models.grpo.config import GRPOConfig
from mini_networks.models.rlhf.model import shakespearean_score
from mini_networks.models.rlhf.trainer import (
    RLHFTrainer,
    _log_probs_of_tokens,
    make_rlhf_dataloader,
)

log = logging.getLogger(__name__)


class GRPOTrainer(RLHFTrainer):
    def _collect_group(self, model, config: GRPOConfig, prompt_ids: list[int]) -> list[dict]:
        """G sampled responses for ONE prompt — the group is the baseline."""
        group = []
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=config.device)
        g = max(2, config.limit_steps(config.group_size, s_cap=2, m_cap=4))
        model.eval()
        for _ in range(g):
            with torch.no_grad():
                full = model.generate(
                    prompt,
                    max_new_tokens=config.limit_steps(config.rollout_max_new, s_cap=8, m_cap=16),
                    temperature=config.rollout_temperature,
                )
            response = full[0, len(prompt_ids):].tolist()
            text = self.tokenizer.decode(response) if self.tokenizer else ""
            group.append({"full_tokens": full, "reward": shakespearean_score(text)})
        return group

    def _grpo_update(self, model, ref_model, group: list[dict], config: GRPOConfig) -> float:
        """Clipped-ratio update with the GROUP as the baseline (no value net):
        A_i = (r_i - mean_group) / (std_group + eps)."""
        rewards = torch.tensor([g["reward"] for g in group], dtype=torch.float32,
                               device=config.device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        optimizer = optim.AdamW(model.parameters(), lr=config.rlhf_lr)
        total = 0.0
        for adv, rb in zip(advantages, group):
            tokens = rb["full_tokens"]
            if tokens.shape[1] < 2:
                continue
            model.train()
            log_probs_cur = _log_probs_of_tokens(model, tokens)
            with torch.no_grad():
                log_probs_ref = _log_probs_of_tokens(ref_model, tokens)
            kl = (log_probs_ref - log_probs_cur).mean()
            ratio = torch.exp(log_probs_cur - log_probs_ref.detach())
            clipped = torch.clamp(ratio, 1 - config.ppo_clip, 1 + config.ppo_clip)
            policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
            loss = policy_loss + config.kl_coef * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        return total / max(1, len(group))

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, GRPOConfig)
        ds = dataloader.dataset
        if hasattr(ds, "tokenizer"):
            self.tokenizer = ds.tokenizer
            vocab_size = ds.vocab_size
        else:
            vocab_size = config.vocab_size
        effective = config.model_copy(update={"vocab_size": vocab_size})
        logger.log_config(effective.model_dump())

        log.info("  [GRPO] Stage 1: pretraining LM")
        model = self._pretrain(effective, dataloader, logger, vocab_size)
        self.model = model
        ref = copy.deepcopy(model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self.ref_model = ref

        corpus = getattr(ds, "text", "")
        if not corpus:
            raise RuntimeError("GRPO dataset did not expose raw text for prompts.")
        all_ids = self.tokenizer.encode(corpus) if self.tokenizer else [
            ord(c) % vocab_size for c in corpus]
        prompt_len = min(16, config.seq_len // 4)
        prompts = [all_ids[i: i + prompt_len]
                   for i in range(0, max(1, len(all_ids) - prompt_len), prompt_len)]

        n_iters = config.limit_steps(config.n_ppo_iters, s_cap=1, m_cap=2)
        offset = config.tier_epochs(config.pretrain_epochs, medium_cap=2)
        log.info("  [GRPO] Stage 2: group-relative fine-tuning")
        for it in range(n_iters):
            group = self._collect_group(model, effective, prompts[it % len(prompts)])
            avg_reward = sum(g["reward"] for g in group) / max(1, len(group))
            loss = self._grpo_update(model, ref, group, effective)
            logger.log_metrics(offset + it, {"grpo_loss": loss, "avg_reward": avg_reward})
            log.info(f"    GRPO iter {it}  loss {loss:.4f}  reward {avg_reward:.4f}")

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        if self.tokenizer:
            self.tokenizer.save(str(logger.artifact_path("tokenizer.json")))


def make_grpo_dataloader(config: GRPOConfig, split: str = "train") -> DataLoader:
    return make_rlhf_dataloader(config, split=split)
