# Repository Guidelines

## Unification Process (One Product from Many Experiments)
This repo is a set of independent ML experiments under `legacy/`. The goal is to turn them into one educational product that runs dynamically in Colab with standardized data loading, logging, and cross-model composition (e.g., CLIP-guided diffusion, Transformer + MoE intersections). Use this step-by-step process:
1. **Inventory + entrypoints**: For each `legacy/<nnn>-<name>/`, identify the runnable entrypoint (`main.py`, `-run` script, or a single-file prototype) and its dependencies.
2. **Define a shared runtime contract**: Create a single interface all projects must implement: `train(config, data_module, logger, output_dir)` and `evaluate(config, data_module, logger, output_dir)`.
3. **Standardize config**: Centralize CLI/Colab params into a shared `Config` schema (YAML/JSON + argparse). Map each legacy project’s flags into this schema.
4. **Unify data loading**: Build one `data/` module with dataset registry + transforms. Migrate per-project loaders (e.g., MNIST/Fashion in `legacy/006-lora/src/data.py`) to this registry.
5. **Unify logging**: Write a common logger that emits to console + `runs/<project>/<timestamp>/` with `metrics.jsonl`, `config.yaml`, and `artifacts/`.
6. **Adapter layer**: For each project, add an adapter that maps legacy training code to the shared contract without rewriting model internals.
7. **Composition layer**: Define a small interface for model composition: `encode()`, `score()`, `guided_step()`, and `sample()`. Implement adapters that can plug into each other (e.g., CLIP provides `score(text, image)`; Diffusion exposes `guided_step(latent, t, guidance_fn)` and `sample(...)`).
8. **Cross-model integrations**: Build explicit compositions as separate adapters:
   - **CLIP-guided diffusion**: use CLIP similarity to guide diffusion sampling.
   - **Diffusion family bridge**: standardize a sampler API so `legacy/004-diffusion`, `legacy/005-guided-diffusion`, and `legacy/013-autoregressive-diff` can share a single driver, logging, and data pipeline.
   - **Transformer + MoE**: swap or augment FFN blocks with MoE experts and keep a shared tokenizer/embedding interface.
9. **Colab entrypoint**: Provide a single `colab_launcher.py` that selects a project or composition by name, prepares data, and runs training/eval.
10. **Validation pass**: Run a minimal smoke test for each adapter (1 epoch, small batch) to verify data/logging compatibility.

## Project Structure & Module Organization
- All content lives under `legacy/`.
- `legacy/001-data/` contains raw MNIST/FashionMNIST files.
- Most projects follow a package layout (e.g., `miniDiffusion/`, `miniTransformer/`), plus `scripts/`, `tests/`, and `requirements.txt`.
- Some are single-file prototypes: `legacy/011-transformer-moe/`, `legacy/012-transformer-gptrl/`, `legacy/013-autoregressive-diff/`, `legacy/014-rag/`.

## Build, Test, and Development Commands
Commands are project-scoped; run them inside each `legacy/<nnn>-<name>/` directory.
- `make install_requirements`: install dependencies.
- `make test`: run pytest with coverage (if tests exist).
- `make black` / `make check_code`: format/lint.

Example:
```bash
cd legacy/004-diffusion
make install_requirements
python -m miniDiffusion.main
```

## Coding Style & Naming Conventions
- Python: 4-space indent, Black formatting, `flake8` linting (where configured).
- Packages commonly use `mini<Name>` naming; keep adapters consistent.
- Store project outputs under `runs/<project>/<timestamp>/`.

## Testing Guidelines
- Most projects expect `pytest` via `make test`, but many tests are minimal or empty.
- For unification, add a smoke test per adapter to ensure data loading and logging succeed.

## Commit & Pull Request Guidelines
- This checkout has no `.git` history; use concise, imperative commit messages.
- PRs should list affected legacy projects, include run commands, and show sample outputs (plots or logs).
