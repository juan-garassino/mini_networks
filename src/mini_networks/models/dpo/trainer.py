"""DPO trainer: pretrain (inherited from rlhf) → preference pairs → DPO loss."""
from __future__ import annotations

import copy
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.models.dpo.config import DPOConfig
from mini_networks.models.rlhf.model import shakespearean_score
from mini_networks.models.rlhf.trainer import (
    RLHFTrainer,
    _log_probs_of_tokens,
    make_rlhf_dataloader,
)

log = logging.getLogger(__name__)


class DPOTrainer(RLHFTrainer):
    def _collect_pairs(self, model, config: DPOConfig, prompts: list[list[int]]) -> list[dict]:
        """Sample TWO responses per prompt; the heuristic score labels
        chosen vs rejected (self-labelled preferences). Ties are skipped."""
        pairs = []
        model.eval()
        n = min(config.limit_steps(config.n_rollouts, s_cap=1, m_cap=8), len(prompts))
        max_new = config.limit_steps(config.rollout_max_new, s_cap=8, m_cap=16)
        for i in range(n):
            prompt_ids = prompts[i % len(prompts)]
            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=config.device)
            cand = []
            with torch.no_grad():
                for _ in range(2):
                    full = model.generate(prompt, max_new_tokens=max_new,
                                          temperature=config.rollout_temperature)
                    text = self.tokenizer.decode(full[0, len(prompt_ids):].tolist()) if self.tokenizer else ""
                    cand.append((full, shakespearean_score(text)))
            if cand[0][1] == cand[1][1]:
                continue
            cand.sort(key=lambda c: c[1], reverse=True)
            pairs.append({"chosen": cand[0][0], "rejected": cand[1][0]})
        return pairs

    def _dpo_update(self, model, ref_model, pairs: list[dict], config: DPOConfig) -> float:
        optimizer = optim.AdamW(model.parameters(), lr=config.rlhf_lr)
        total = 0.0
        model.train()
        for pb in pairs:
            # Sequence log-prob margins vs the frozen reference — the KL
            # anchor is implicit in these ref terms (no PPO machinery).
            lp_c = _log_probs_of_tokens(model, pb["chosen"]).sum()
            lp_r = _log_probs_of_tokens(model, pb["rejected"]).sum()
            with torch.no_grad():
                lp_c_ref = _log_probs_of_tokens(ref_model, pb["chosen"]).sum()
                lp_r_ref = _log_probs_of_tokens(ref_model, pb["rejected"]).sum()
            margin = (lp_c - lp_c_ref) - (lp_r - lp_r_ref)
            loss = -F.logsigmoid(config.dpo_beta * margin)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        return total / max(1, len(pairs))

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, DPOConfig)
        ds = dataloader.dataset
        if hasattr(ds, "tokenizer"):
            self.tokenizer = ds.tokenizer
            vocab_size = ds.vocab_size
        else:
            vocab_size = config.vocab_size
        effective_config = config.model_copy(update={"vocab_size": vocab_size})
        logger.log_config(effective_config.model_dump())

        log.info("  [DPO] Stage 1: pretraining LM")
        model = self._pretrain(effective_config, dataloader, logger, vocab_size)
        self.model = model
        ref = copy.deepcopy(model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self.ref_model = ref

        corpus = getattr(ds, "text", "")
        if not corpus:
            raise RuntimeError("DPO dataset did not expose raw text for prompts.")
        all_ids = self.tokenizer.encode(corpus) if self.tokenizer else [ord(c) % vocab_size for c in corpus]
        prompt_len = min(16, config.seq_len // 4)
        prompts = [all_ids[i: i + prompt_len]
                   for i in range(0, max(1, len(all_ids) - prompt_len), prompt_len)]

        n_iters = config.limit_steps(config.n_ppo_iters, s_cap=1, m_cap=3)
        offset = config.tier_epochs(config.pretrain_epochs, medium_cap=2)
        log.info("  [DPO] Stage 2: preference optimization")
        for it in range(n_iters):
            pairs = self._collect_pairs(model, effective_config, prompts)
            if not pairs:
                logger.log_metrics(offset + it, {"dpo_loss": 0.0, "n_pairs": 0})
                continue
            loss = self._dpo_update(model, ref, pairs, effective_config)
            logger.log_metrics(offset + it, {"dpo_loss": loss, "n_pairs": len(pairs)})
            log.info(f"    DPO iter {it}  loss {loss:.4f}  pairs {len(pairs)}")

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        if self.tokenizer:
            self.tokenizer.save(str(logger.artifact_path("tokenizer.json")))


def make_dpo_dataloader(config: DPOConfig, split: str = "train") -> DataLoader:
    return make_rlhf_dataloader(config, split)
