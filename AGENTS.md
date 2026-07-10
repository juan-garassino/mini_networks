# Repository Guidelines

mini_networks is a unified educational ML lab: ~31 models + ~19 compositions in
`src/mini_networks/`, sharing one runtime contract (`core/runtime.py::BaseTrainer`),
one data registry (`core/data/registry.py`), one logger
(`runs/<name>/<ts>/{metrics.jsonl,config.yaml,artifacts/}`), and one quality gate
(`main.py sweep --check`). `legacy/` is reference-only; never add binary data to git.

A graphical **playground** (`frontend/` SPA + `/web` read-layer) reads that same
contract live; **M/L training can run on GCP ephemeral Cloud Run Jobs**
(`infra/gcp/`) persisting to MLflow (Neon + GCS). The UI and cloud are pure
readers/writers of the contract — trainers never change. See `CLAUDE.md` for the
architecture map, tier system, quality-gate semantics, env vars, and conventions.
Design specs: `docs/superpowers/specs/2026-06-12-…` and `…/2026-06-24-playground-and-gcp-training-design.md`.

## Commands

- `uv sync --dev` (add `--extra cloud` for mlflow/gcp) — install
- `make test` — fast pytest suite
- `make validate-s` — full S-tier check sweep (CI parity)
- `make -C infra/gcp validate` — terraform fmt-check + validate (static)
- `python main.py list | train | compose | sweep | serve` — CLI; `serve` opens the playground at `/`

## Rules of thumb

- Use `config.effective_epochs/batch_size/timesteps`, never raw fields.
- New model/composition ⇒ add an `EvalSpec` entry in `core/evalspec.py`
  (the coverage unit test enforces this) and keep thresholds justified.
- stdlib logging via `log = logging.getLogger(__name__)`; `logger` names the
  metrics Logger parameter in trainers. argparse only. `torch.load(..., weights_only=True)`.
- Local machine is slow: targeted tests + single S-tier runs only; full sweeps
  run in CI or on Colab GPU.
