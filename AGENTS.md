# Repository Guidelines

mini_networks is a unified educational ML lab: ~31 models + ~19 compositions in
`src/mini_networks/`, sharing one runtime contract (`core/runtime.py::BaseTrainer`),
one data registry (`core/data/registry.py`), one logger
(`runs/<name>/<ts>/{metrics.jsonl,config.yaml,artifacts/}`), and one quality gate
(`main.py sweep --check`). `legacy/` is reference-only; never add binary data to git.

See `CLAUDE.md` for the full architecture map, tier system (S/M/L budgets via
`config.effective_*`), quality-gate semantics, and conventions. Design spec:
`docs/superpowers/specs/2026-06-12-ultimate-educational-resource-design.md`.

## Commands

- `uv sync --dev` — install
- `make test` — fast pytest suite
- `make validate-s` — full S-tier check sweep (CI parity)
- `python main.py list | train | compose | sweep | serve` — CLI entry point

## Rules of thumb

- Use `config.effective_epochs/batch_size/timesteps`, never raw fields.
- New model/composition ⇒ add an `EvalSpec` entry in `core/evalspec.py`
  (the coverage unit test enforces this) and keep thresholds justified.
- stdlib logging via `log = logging.getLogger(__name__)`; `logger` names the
  metrics Logger parameter in trainers. argparse only. `torch.load(..., weights_only=True)`.
- Local machine is slow: targeted tests + single S-tier runs only; full sweeps
  run in CI or on Colab GPU.
