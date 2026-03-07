# Runs and Logging

All training and evaluation outputs follow the same structure to keep experiments comparable.

**Output Layout**
- `runs/<project>/<timestamp>/metrics.jsonl`
- `runs/<project>/<timestamp>/config.yaml`
- `runs/<project>/<timestamp>/artifacts/`

**Metrics**
`metrics.jsonl` is line‑delimited JSON where each record includes:
- `step` or `epoch`
- `key`
- `value`

**Artifacts**
Artifacts are model‑specific and may include:
- `model.pt`
- `generator.pt`, `discriminator.pt`
- `tokenizer.json`
- `samples/` or plotted images

**Why It Helps**
Uniform logging makes it easy to compare models, diagnose regressions, and share Colab runs without custom scripts.
