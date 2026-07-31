"""Mini VLM: a multimodal LM composed from two zoo atoms.

The molecule the taxonomy chapter is about: take the vision patch-encoder
atom (ViT-style patchify, `models/multimodal/encoders.py::VisionPatchEncoder`)
and the causal-attention LM atom (transformer), and wire them the way real
VLMs (LLaVA-style) do — project the image's patch tokens into the LM's
embedding space and prepend them as a PREFIX; the text then attends to the
image through ordinary causal self-attention. Nothing multimodal is invented:
it is pure composition.

Task: templated MNIST QA — "what digit is this?" / "is it even or odd?" /
"is it more than four?", answered in words, char-tokenized. Loss is CE on
the ANSWER span only (the question is context, not target).

Honesty evidence (house rule): evaluate() reports `answer_accuracy` (the
gate metric, held-out MNIST test images) AND `blind_accuracy` — the same
eval with the image zeroed. The language prior alone gets ~1/2 on even/odd
and ~1/10 on digit naming; the gap proves the answers come THROUGH the
image prefix, not from text statistics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataset
from mini_networks.core.logging.logger import Logger
from mini_networks.models.multimodal.encoders import VisionPatchEncoder
from mini_networks.models.transformer.tokenizer import CharTokenizer

import logging

log = logging.getLogger(__name__)

DIGIT_WORDS = ("zero", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine")

# (question, answer_fn) — answers end with '.' so generation can terminate
QA_TEMPLATES = (
    ("what digit is this?", lambda y: DIGIT_WORDS[y]),
    ("is it even or odd?", lambda y: "even" if y % 2 == 0 else "odd"),
    ("is it more than four?", lambda y: "yes" if y > 4 else "no"),
)


class VLMConfig(BaseConfig):
    model_name: str = "vlm"
    dataset: str = "mnist"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    patch_size: int = 4        # 28/4 -> 49 image prefix tokens (all patches, LLaVA-style)
    max_text_len: int = 40     # "Q: <q> A: <ans>." fits comfortably
    eval_samples: int = 256    # held-out images scored per evaluate()


def _corpus() -> str:
    parts = ["Q: A: ."]
    for q, a_fn in QA_TEMPLATES:
        parts.append(q)
        parts.extend(a_fn(y) for y in range(10))
    return " ".join(parts)


def build_tokenizer() -> CharTokenizer:
    return CharTokenizer.from_text(_corpus())


class _QADataset(Dataset):
    """(image, tokens, labels): labels are next-token targets, -1 outside the
    answer span so the question contributes zero loss."""

    def __init__(self, base: Dataset, tokenizer: CharTokenizer, max_len: int,
                 pad_id: int, seed: int = 5, limit: int | None = None):
        self._base = base
        self._tok = tokenizer
        self._max_len = max_len
        self._pad = pad_id
        self._seed = seed
        self._limit = min(limit, len(base)) if limit else len(base)

    def __len__(self) -> int:
        return self._limit

    def encode_qa(self, question: str, answer: str | None):
        """Returns (tokens [max_len], labels [max_len-1]) — labels -1 except
        where the NEXT token is part of ' <answer>.'."""
        prompt = f"Q: {question} A:"
        prompt_ids = self._tok.encode(prompt)
        full = prompt_ids + (self._tok.encode(f" {answer}.") if answer else [])
        full = full[: self._max_len]
        tokens = torch.full((self._max_len,), self._pad, dtype=torch.long)
        tokens[: len(full)] = torch.tensor(full)
        labels = torch.full((self._max_len - 1,), -1, dtype=torch.long)
        # target positions: predicting tokens after the prompt, up to len(full)
        for i in range(len(prompt_ids) - 1, len(full) - 1):
            labels[i] = full[i + 1]
        return tokens, labels

    def __getitem__(self, idx: int):
        image, y = self._base[idx]
        g = torch.Generator().manual_seed(self._seed * 100_003 + idx)
        q, a_fn = QA_TEMPLATES[int(torch.randint(0, len(QA_TEMPLATES), (1,), generator=g))]
        tokens, labels = self.encode_qa(q, a_fn(int(y)))
        return image, tokens, labels, int(y)


class VLMNet(nn.Module):
    """Vision patch prefix + causal text transformer, one shared stream."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, patch_size: int = 4, max_text_len: int = 40):
        super().__init__()
        self.vision = VisionPatchEncoder(patch_size=patch_size, d_model=d_model)
        self.proj = nn.Linear(d_model, d_model)  # the "connector" (LLaVA's MLP)
        self.pad_id = vocab_size
        self.tok_embed = nn.Embedding(vocab_size + 1, d_model)  # +1 pad
        self.n_prefix = self.vision.n_patches
        self.pos = nn.Embedding(self.n_prefix + max_text_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model,
            batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """-> logits over TEXT positions [B, T_text, V]."""
        prefix = self.proj(self.vision(images))               # [B, P, D]
        text = self.tok_embed(tokens)                         # [B, T, D]
        x = torch.cat([prefix, text], dim=1)
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        return self.head(self.norm(x[:, self.n_prefix:]))

    @torch.no_grad()
    def answer(self, images: torch.Tensor, tokenizer: CharTokenizer,
               question: str, max_new: int = 12) -> list[str]:
        """Greedy decode ' <answer>.' after 'Q: <q> A:' — batch of images."""
        self.eval()
        device = images.device
        prompt = tokenizer.encode(f"Q: {question} A:")
        stop = tokenizer.encode(".")[0]
        tokens = torch.tensor(prompt, device=device).unsqueeze(0).expand(images.shape[0], -1)
        tokens = tokens.clone()
        done = torch.zeros(images.shape[0], dtype=torch.bool, device=device)
        for _ in range(max_new):
            logits = self(images, tokens)
            nxt = logits[:, -1].argmax(dim=-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            done |= nxt == stop
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
            if done.all():
                break
        outs = []
        for row in tokens[:, len(prompt):].tolist():
            text = tokenizer.decode([t for t in row if t < self.pad_id])
            outs.append(text.split(".")[0].strip())
        return outs


class VLM:
    def __init__(self):
        self.model: VLMNet | None = None
        self.tokenizer: CharTokenizer | None = None

    def _loaders(self, config: VLMConfig):
        tok = build_tokenizer()
        pad = len(tok.stoi)
        limit = config.dataset_sample_limit
        train_base = get_dataset(config.dataset, config.data_root, split="train")
        test_base = get_dataset(config.dataset, config.data_root, split="test")
        train_ds = _QADataset(train_base, tok, config.max_text_len, pad,
                              seed=config.seed, limit=limit)
        test_ds = _QADataset(test_base, tok, config.max_text_len, pad,
                             seed=config.seed + 1, limit=config.eval_samples)
        return tok, (
            DataLoader(train_ds, batch_size=config.effective_batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=config.effective_batch_size, shuffle=False),
        )

    def train_all(self, config: VLMConfig, logger: Logger) -> dict:
        tok, (dl_train, dl_test) = self._loaders(config)
        self.tokenizer = tok
        model = VLMNet(
            vocab_size=len(tok.stoi), d_model=config.d_model, n_heads=config.n_heads,
            n_layers=config.n_layers, patch_size=config.patch_size,
            max_text_len=config.max_text_len,
        ).to(config.device)
        self.model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        max_batches = config.max_train_batches or len(dl_train)
        for epoch in range(config.effective_epochs):
            model.train()
            total, n = 0.0, 0
            for images, tokens, labels, _ in dl_train:
                if n >= max_batches:
                    break
                logits = model(images.to(config.device), tokens.to(config.device))
                # next-token CE on the answer span only (labels are -1 elsewhere)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    labels.to(config.device).reshape(-1),
                    ignore_index=-1,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                n += 1
            logger.log_metrics(epoch, {"loss": total / max(1, n), "epoch": epoch})
            log.info(f"  epoch {epoch}  loss {total / max(1, n):.4f}")

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        metrics = self._evaluate(config, dl_test)
        logger.log_summary({"status": "completed", **metrics})
        return {**metrics, "run_dir": str(logger.run_dir)}

    @torch.no_grad()
    def _evaluate(self, config: VLMConfig, dl_test) -> dict:
        """answer_accuracy on held-out images; blind_accuracy = image zeroed —
        the gap proves the answers flow through the image prefix."""
        model, tok = self.model, self.tokenizer
        correct = blind_correct = total = 0
        for images, _, _, ys in dl_test:
            images = images.to(config.device)
            for t_idx, (q, a_fn) in enumerate(QA_TEMPLATES):
                preds = model.answer(images, tok, q)
                blind = model.answer(torch.zeros_like(images), tok, q)
                for p, b, y in zip(preds, blind, ys.tolist()):
                    truth = a_fn(int(y))
                    correct += int(p == truth)
                    blind_correct += int(b == truth)
                    total += 1
        return {
            "answer_accuracy": correct / max(1, total),
            "blind_accuracy": blind_correct / max(1, total),
        }
