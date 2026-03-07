# Documentation Index

**Start Here**
- `docs/datasets.md` Data registry, shared task modes, and dataset mapping.
- `docs/adding_datasets.md` How to add a new dataset to the shared registry.
- `docs/multimodal_blocks.md` Encoders, cross‑attention, and fusion blocks.
- `docs/models.md` Model inventory with short descriptions.
- `docs/compositions.md` Composition interfaces and pipelines.
- `docs/runs_and_logging.md` Logging format and artifact layout.

**Architecture Layers**
Data loading → preprocessing → building blocks → networks → multimodal compositions → inference/evaluation.

**Refactor Map**
- `core/data/` datasets + shared preprocessing
- `core/blocks/` reusable layers and blocks
- `core/runtime.py` shared trainer base classes (e.g., `SupervisedTrainer`, `ContrastiveTrainer`)
- `core/diffusion/` shared diffusion sampling driver
- `models/` model definitions + trainers
- `compositions/` cross‑model pipelines + composition bases

**Core Concepts**
mini_networks standardizes how experiments are executed so different models can be compared and composed without bespoke glue code.

Key contracts:
- `train(config, dataloader, logger)` for training.
- `evaluate(config, dataloader, logger)` for evaluation.
- `infer(config, inputs)` for inference.

Core bases:
- `BaseTrainer`, `SupervisedTrainer`, `ContrastiveTrainer` in `core/runtime.py`
- `CompositionBase`, `ContrastiveCompositionBase` in `compositions/base.py`
- `sample_loop` in `core/diffusion/sampling.py` for diffusion sampling

Everything depends on a shared data registry and a common logging format. This keeps Colab workflows simple and makes cross‑model experiments reproducible.
