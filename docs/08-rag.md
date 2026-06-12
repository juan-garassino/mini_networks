# Chapter 08 — Retrieval-Augmented Generation (RAG)

## Theory recap

A language model can only "know" what was in its training data and what fits in
its context window. Retrieval-Augmented Generation splits the problem in two:
a **retriever** finds the most relevant text chunks for a query, and a
**generator** conditions its output on those chunks. The retriever here is
classical information retrieval — TF-IDF vectors compared by cosine
similarity — which makes the whole mechanism inspectable: term frequency (how
often a word appears in a chunk) times inverse document frequency (how rare
the word is across all chunks) gives each chunk a sparse vector, and the
chunks whose vectors point most in the same direction as the query vector win.
No embeddings model, no vector database — just counting and a dot product.

## In this repo

- `src/mini_networks/models/rag/model.py` — `TFIDFIndex` is a from-scratch
  TF-IDF index using only the standard library and torch. `build(documents)`
  tokenizes with a lowercase `[a-z0-9]+` regex, computes per-document term
  frequencies, then `idf = log((n+1)/(doc_freq+1)) + 1`, and stores the
  `[n_docs, vocab]` TF-IDF matrix. `retrieve(query, top_k)` builds the query
  TF-IDF vector and ranks documents by cosine similarity
  (`(self._tfidf @ q_tfidf) / (doc_norms * q_norm)`), returning
  `(score, document)` pairs via `topk`.
- `NanoRAG` (same file) is the pipeline wrapper:
  `add_documents(texts)` slices each text into `chunk_size`-character chunks
  and builds the index → `retrieve(query)` returns top-k chunks →
  `build_prompt(query)` joins them as `"[Context: c1 | c2 | ...] {query}"` →
  `generate(query, model, tokenizer)` encodes that prompt and calls
  `model.generate(...)`. The generator is the repo's own `TransformerLM`
  from `src/mini_networks/models/transformer/model.py` — RAG adds no new
  model weights, just smarter prompting.
- `src/mini_networks/models/rag/trainer.py` — `RAGTrainer.train()` first
  trains the `TransformerLM` on the corpus with plain cross-entropy (the same
  loop as the transformer chapter), saves `model.pt` + `tokenizer.json`, then
  builds the `NanoRAG` index from the dataset's raw `ds.text`. `infer()`
  returns `{"generated": ..., "retrieved": [...]}` so you can see exactly
  which chunks conditioned the output.
- `load_checkpoint()` reloads the LM weights (inferring `vocab_size` from
  `state["token_embed.weight"].shape[0]`) and the tokenizer, then **rebuilds
  the TF-IDF index from the dataset** rather than deserializing it — the
  index is derived data, fully reconstructable from the corpus, so it is
  never saved as an artifact.
- Config: `src/mini_networks/models/rag/config.py` — `top_k=3`,
  `chunk_size=200`, plus a small TransformerLM (`d_model=64`, 2 layers).
  Registered as `"rag"` in the API and CLI.

## Where it composes

Two compositions use RAG as a building block (see chapter 11):

- `src/mini_networks/compositions/rag_guided_generation.py` — the standalone
  retrieve-then-generate pipeline as a composition: trains an LM on a text
  file, indexes the corpus, and answers queries with retrieved context.
- `src/mini_networks/compositions/rag_conditioned_diffusion.py` — chains
  `RAGGuidedGeneration` with `CLIPGuidedDiffusion`: RAG generates a prompt
  from a seed text, and that prompt steers CLIP-guided image generation.

## Try it

```bash
uv run python main.py train --model rag --fast_demo
curl -X POST http://localhost:8000/infer/rag \
  -H "Content-Type: application/json" -d '{"query": "To be"}'
```

## Latest results

<!-- results:start items=rag -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| rag | pass | eval_loss | 3.8786 | n/a |

<!-- results:end -->
