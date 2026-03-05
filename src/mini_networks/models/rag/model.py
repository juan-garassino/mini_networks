"""NanoRAG: TF-IDF retrieval + TransformerLM generation.

Architecture (inspired by legacy/014-rag/miniRAG.py)
-----------------------------------------------------
  1. Document store   — plain text chunks indexed by TF-IDF vectors
  2. Retriever        — cosine similarity between query and chunk TF-IDF
  3. Generator        — our own TransformerLM (not GPT-2)
     The retrieved context is prepended to the user prompt before encoding.

Educational notes
-----------------
  - TF-IDF is computed from scratch using only the standard library and torch.
  - Cosine similarity retrieval is O(n_docs * vocab) — fast for small corpora.
  - The generator is the same TransformerLM used in models/transformer, so
    RAG is purely additive: no new model weights, just smarter prompting.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TF-IDF Index
# ---------------------------------------------------------------------------

class TFIDFIndex:
    """Simple TF-IDF document index using pure Python + torch."""

    def __init__(self):
        self.documents: list[str] = []
        self._idf: Optional[torch.Tensor] = None   # [vocab_size]
        self._tfidf: Optional[torch.Tensor] = None  # [n_docs, vocab_size]
        self._word2idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build(self, documents: list[str]) -> None:
        self.documents = list(documents)
        n = len(documents)

        # Build vocabulary from all documents
        all_words: set[str] = set()
        doc_tokens: list[list[str]] = []
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_tokens.append(tokens)
            all_words.update(tokens)

        self._word2idx = {w: i for i, w in enumerate(sorted(all_words))}
        V = len(self._word2idx)

        # Compute TF per document
        tf = torch.zeros(n, V)
        for d_idx, tokens in enumerate(doc_tokens):
            if not tokens:
                continue
            counts = Counter(tokens)
            total = len(tokens)
            for word, cnt in counts.items():
                if word in self._word2idx:
                    tf[d_idx, self._word2idx[word]] = cnt / total

        # Compute IDF
        doc_freq = (tf > 0).float().sum(dim=0)  # [V]
        idf = torch.log((n + 1.0) / (doc_freq + 1.0)) + 1.0
        self._idf = idf

        # TF-IDF matrix
        self._tfidf = tf * idf  # [n_docs, V]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[float, str]]:
        """Return top-k (score, document) pairs ranked by cosine similarity."""
        if self._tfidf is None or len(self.documents) == 0:
            return []

        tokens = self._tokenize(query)
        V = len(self._word2idx)
        q_tf = torch.zeros(V)
        if tokens:
            counts = Counter(tokens)
            total = len(tokens)
            for word, cnt in counts.items():
                if word in self._word2idx:
                    q_tf[self._word2idx[word]] = cnt / total

        q_tfidf = q_tf * self._idf  # [V]
        q_norm = q_tfidf.norm()
        if q_norm < 1e-8:
            return []

        doc_norms = self._tfidf.norm(dim=1).clamp(min=1e-8)  # [n_docs]
        scores = (self._tfidf @ q_tfidf) / (doc_norms * q_norm)  # [n_docs]

        k = min(top_k, len(self.documents))
        top_scores, top_idx = scores.topk(k)
        return [(top_scores[i].item(), self.documents[top_idx[i]]) for i in range(k)]


# ---------------------------------------------------------------------------
# NanoRAG
# ---------------------------------------------------------------------------

class NanoRAG:
    """Retrieval-Augmented Generation using TF-IDF + TransformerLM.

    Workflow:
      1. `add_documents(texts)` — indexes text chunks into TF-IDF store
      2. `retrieve(query, k)` — returns most relevant chunks
      3. `generate(query, model, tokenizer, ...)` — prepends context to query,
         encodes it, feeds to TransformerLM, returns decoded generation
    """

    def __init__(self, top_k: int = 3, chunk_size: int = 200):
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.index = TFIDFIndex()

    def add_documents(self, texts: list[str]) -> None:
        """Chunk and index a list of documents."""
        chunks: list[str] = []
        for text in texts:
            for i in range(0, max(1, len(text)), self.chunk_size):
                chunk = text[i: i + self.chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
        self.index.build(chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[str]:
        """Return top-k most relevant document chunks."""
        k = top_k or self.top_k
        results = self.index.retrieve(query, top_k=k)
        return [doc for _, doc in results]

    def build_prompt(self, query: str, top_k: Optional[int] = None) -> str:
        """Retrieve context and prepend it to the query."""
        chunks = self.retrieve(query, top_k=top_k)
        context = " | ".join(chunks)
        if context:
            return f"[Context: {context}] {query}"
        return query

    def generate(
        self,
        query: str,
        model: torch.nn.Module,
        tokenizer,
        device: str = "cpu",
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> str:
        """Full RAG pipeline: retrieve → augment prompt → generate."""
        prompt_text = self.build_prompt(query)
        ids = tokenizer.encode(prompt_text)
        if not ids:
            ids = [0]
        prompt = torch.tensor([ids], dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            output = model.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )
        return tokenizer.decode(output[0].tolist())
