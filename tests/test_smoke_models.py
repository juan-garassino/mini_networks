"""Registry-wide smoke tests for model train/eval contract."""
import os
import tempfile

from mini_networks.api.dependencies import get_model_registry
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


def _build_config(config_cls):
    config = config_cls(
        fast_demo=True,
        epochs=1,
        batch_size=4,
        data_root=DATA_ROOT,
        device="cpu",
    )
    overrides = {}
    fields = getattr(config, "model_fields", {})

    if "timesteps" in fields:
        overrides["timesteps"] = 50
    if "pretrain_epochs" in fields:
        overrides["pretrain_epochs"] = 1
    if "finetune_epochs" in fields:
        overrides["finetune_epochs"] = 1
    if "n_ppo_iters" in fields:
        overrides["n_ppo_iters"] = 1
    if "ppo_epochs" in fields:
        overrides["ppo_epochs"] = 1
    if "n_rollouts" in fields:
        overrides["n_rollouts"] = 4
    if "rollout_max_new" in fields:
        overrides["rollout_max_new"] = 8
    if "seq_len" in fields:
        overrides["seq_len"] = min(getattr(config, "seq_len", 64), 64)
    if "patch_size" in fields:
        overrides["patch_size"] = min(getattr(config, "patch_size", 4), 4)
    if "require_downloads" in fields:
        overrides["require_downloads"] = False

    if overrides:
        config = config.model_copy(update=overrides)
    return config


def test_registry_train_eval_smoke():
    registry = get_model_registry()
    for name, (ConfigClass, TrainerClass, dataloader_fn) in registry.items():
        config = _build_config(ConfigClass)
        trainer = TrainerClass()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name=f"smoke-{name}")
            dl = dataloader_fn(config, split="train")
            trainer.train(config, dl, logger)
            metrics = trainer.evaluate(config, dl, logger)
            assert isinstance(metrics, dict)
