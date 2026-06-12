"""RLHF trainer — PPO fine-tuning of TransformerLM with Shakespearean reward.

Pipeline
--------
  Stage 1 — Pretrain  a TransformerLM on the text corpus (standard CE loss).
  Stage 2 — PPO loop  iteratively:
               a. Sample `n_rollouts` prompts from the corpus.
               b. Generate a response for each with the current policy.
               c. Score every response with the heuristic reward function.
               d. Compute per-token log-prob ratio vs frozen reference policy.
               e. Update the policy with PPO clip + KL penalty.

Educational notes
-----------------
  Reference policy (frozen copy of Stage 1 LM):
    KL(π || π_ref) penalises the policy from deviating too far.
    This prevents reward hacking — the model shouldn't just memorise
    archaic words without maintaining fluent text.

  Why token-level rewards?
    Each generated token gets the episode's scalar reward spread uniformly.
    This is a simplification of full RLHF (which uses token-level credit
    assignment), but it captures the essential PPO update.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.rlhf.config import RLHFConfig
from mini_networks.models.rlhf.model import RewardModel, shakespearean_score
from mini_networks.models.transformer.model import TransformerLM
from mini_networks.models.transformer.tokenizer import CharTokenizer

import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: token log-probabilities under a model
# ---------------------------------------------------------------------------

def _log_probs_of_tokens(
    model: TransformerLM,
    tokens: torch.Tensor,          # [1, T]
) -> torch.Tensor:
    """Return log P(tokens[t] | tokens[:t]) for all t > 0 — shape [T-1]."""
    logits, _ = model(tokens)                      # [1, T, V]
    # Shift: predict tokens[1:] from positions [0:-1]
    logits = logits[0, :-1, :]                     # [T-1, V]
    targets = tokens[0, 1:]                        # [T-1]
    return -F.cross_entropy(logits, targets, reduction="none")  # [T-1]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RLHFTrainer(BaseTrainer):
    def __init__(self):
        self.model: TransformerLM | None = None
        self.ref_model: TransformerLM | None = None
        self.tokenizer: CharTokenizer | None = None

    def _build_lm(self, config: RLHFConfig, vocab_size: int) -> TransformerLM:
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            seq_len=config.seq_len,
            dropout=config.dropout,
        ).to(config.device)

    # ------------------------------------------------------------------
    # Stage 1: supervised pretraining
    # ------------------------------------------------------------------

    def _pretrain(
        self,
        config: RLHFConfig,
        dataloader: DataLoader,
        logger: Logger,
        vocab_size: int,
    ) -> TransformerLM:
        model = self._build_lm(config, vocab_size)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        epochs = config.tier_epochs(config.pretrain_epochs, medium_cap=2)

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(config.device), y.to(config.device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(1, len(dataloader))
            logger.log_metrics(epoch, {"pretrain_loss": avg})
            log.info(f"  [Pretrain] epoch {epoch}  loss {avg:.4f}")

        return model

    # ------------------------------------------------------------------
    # Stage 2: PPO fine-tuning
    # ------------------------------------------------------------------

    def _collect_rollouts(
        self,
        model: TransformerLM,
        config: RLHFConfig,
        prompts: list[list[int]],
    ) -> list[dict]:
        """Generate responses and score them. Returns list of rollout dicts."""
        rollouts = []
        model.eval()
        n = min(config.limit_steps(config.n_rollouts, s_cap=1, m_cap=4), len(prompts))
        for i in range(n):
            prompt_ids = prompts[i % len(prompts)]
            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=config.device)
            with torch.no_grad():
                full = model.generate(
                    prompt,
                    max_new_tokens=config.limit_steps(config.rollout_max_new, s_cap=8, m_cap=16),
                    temperature=config.rollout_temperature,
                )
            response_ids = full[0, len(prompt_ids):].tolist()
            response_text = self.tokenizer.decode(response_ids) if self.tokenizer else ""
            reward = shakespearean_score(response_text)
            rollouts.append({
                "full_tokens": full,          # [1, T_prompt + T_resp]
                "reward": reward,
            })
        return rollouts

    def _ppo_update(
        self,
        model: TransformerLM,
        ref_model: TransformerLM,
        rollouts: list[dict],
        config: RLHFConfig,
    ) -> float:
        """One PPO iteration over the collected rollouts."""
        optimizer = optim.AdamW(model.parameters(), lr=config.rlhf_lr)
        total_loss = 0.0

        for _ in range(config.limit_steps(config.ppo_epochs, s_cap=1, m_cap=2)):
            for rb in rollouts:
                tokens = rb["full_tokens"]        # [1, T]
                reward = rb["reward"]
                T = tokens.shape[1]
                if T < 2:
                    continue

                # Current and reference log-probs
                model.train()
                log_probs_cur = _log_probs_of_tokens(model, tokens)         # [T-1]
                with torch.no_grad():
                    log_probs_ref = _log_probs_of_tokens(ref_model, tokens)  # [T-1]

                # Token-level KL divergence penalty
                kl = (log_probs_ref - log_probs_cur).mean()

                # PPO surrogate — reward broadcast to all tokens
                r_tensor = torch.tensor(reward, dtype=torch.float32, device=config.device)
                advantage = r_tensor - 0.5   # centre around median reward (0.5)

                ratio = torch.exp(log_probs_cur - log_probs_ref.detach())
                clipped = torch.clamp(ratio, 1 - config.ppo_clip, 1 + config.ppo_clip)
                policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()

                loss = policy_loss + config.kl_coef * kl

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        n = max(1, len(rollouts) * config.limit_steps(config.ppo_epochs, s_cap=1, m_cap=2))
        return total_loss / n

    # ------------------------------------------------------------------
    # BaseTrainer contract
    # ------------------------------------------------------------------

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, RLHFConfig)

        ds = dataloader.dataset
        if hasattr(ds, "tokenizer"):
            self.tokenizer = ds.tokenizer
            vocab_size = ds.vocab_size
        else:
            vocab_size = config.vocab_size

        effective_config = config.model_copy(update={"vocab_size": vocab_size})
        logger.log_config(effective_config.model_dump())

        # --- Stage 1: pretrain ---
        log.info("  [RLHF] Stage 1: pretraining LM")
        model = self._pretrain(effective_config, dataloader, logger, vocab_size)
        self.model = model

        # Freeze a reference copy
        ref = copy.deepcopy(model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self.ref_model = ref

        # Build prompt bank from dataset text
        corpus = getattr(ds, "text", "")
        if not corpus:
            raise RuntimeError("RLHF dataset did not expose raw text for prompts.")

        if self.tokenizer:
            all_ids = self.tokenizer.encode(corpus)
        else:
            all_ids = [ord(c) % vocab_size for c in corpus]

        prompt_len = min(16, config.seq_len // 4)
        prompts = [
            all_ids[i: i + prompt_len]
            for i in range(0, max(1, len(all_ids) - prompt_len), prompt_len)
        ]

        # --- Stage 2: PPO ---
        n_iters = config.limit_steps(config.n_ppo_iters, s_cap=1, m_cap=2)
        offset = config.tier_epochs(config.pretrain_epochs, medium_cap=2)
        log.info("  [RLHF] Stage 2: PPO fine-tuning")
        for it in range(n_iters):
            rollouts = self._collect_rollouts(model, effective_config, prompts)
            avg_reward = sum(r["reward"] for r in rollouts) / max(1, len(rollouts))
            loss = self._ppo_update(model, ref, rollouts, effective_config)
            logger.log_metrics(offset + it, {
                "ppo_loss": loss,
                "avg_reward": avg_reward,
            })
            log.info(f"    PPO iter {it}  loss {loss:.4f}  reward {avg_reward:.4f}")

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        if self.tokenizer:
            self.tokenizer.save(str(logger.artifact_path("tokenizer.json")))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, RLHFConfig)
        ds = dataloader.dataset
        vocab_size = ds.vocab_size if hasattr(ds, "vocab_size") else config.vocab_size
        if self.model is None:
            self.model = self._build_lm(config, vocab_size)
        model = self.model
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(config.device), y.to(config.device)
                logits, _ = model(x)
                total_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        return {"eval_loss": total_loss / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, RLHFConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        prompt_text = inputs.get("prompt", "") if isinstance(inputs, dict) else str(inputs)
        max_new = inputs.get("max_new_tokens", 32) if isinstance(inputs, dict) else 32
        temperature = inputs.get("temperature", 1.0) if isinstance(inputs, dict) else 1.0

        if self.tokenizer and prompt_text:
            ids = self.tokenizer.encode(prompt_text)
        else:
            ids = [0]
        prompt = torch.tensor([ids], dtype=torch.long, device=config.device)
        model.eval()
        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=max_new, temperature=temperature)
        generated = self.tokenizer.decode(output[0].tolist()) if self.tokenizer else ""
        return {
            "generated": generated,
            "reward": shakespearean_score(generated),
        }


    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        """Load model.pt + tokenizer.json. Infers vocab_size from state dict."""
        from pathlib import Path
        assert isinstance(config, RLHFConfig)
        path = Path(artifacts_dir)
        state = torch.load(path / "model.pt", map_location=config.device, weights_only=True)
        vocab_size = state["token_embed.weight"].shape[0]
        self.model = self._build_lm(config, vocab_size)
        self.model.load_state_dict(state)
        self.model.eval()
        tok_path = path / "tokenizer.json"
        if tok_path.exists():
            self.tokenizer = CharTokenizer.load(str(tok_path))


def make_rlhf_dataloader(config: RLHFConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
        file_path=config.text_file,
        seq_len=config.seq_len,
    )
