import tempfile

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import SupervisedTrainer


class _TinyConfig(BaseConfig):
    epochs: int = 2
    batch_size: int = 2
    learning_rate: float = 1e-2


class _TinyTrainer(SupervisedTrainer):
    def __init__(self):
        self.model = None

    def _build(self, config):
        import torch.nn as nn

        return nn.Linear(4, 2)

    def infer(self, config, inputs):
        return self.model(inputs)


def _dataloader():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randn(4, 4)
    y = torch.randint(0, 2, (4,))
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_logger_uses_explicit_run_dir_without_double_timestamp():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=f"{tmpdir}/runs/model/20260414-120000", run_name="20260414-120000")
        assert str(logger.run_dir).endswith("runs/model/20260414-120000")


def test_training_state_is_saved_for_resume():
    config = _TinyConfig(fast_demo=False, device="cpu", resume=True)
    trainer = _TinyTrainer()

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="resume-test")
        trainer.train(config, _dataloader(), logger)
        state = logger.load_training_state()
        assert state is not None
        assert state["completed"] is True
        assert logger.artifact_path("model.pt").exists()
