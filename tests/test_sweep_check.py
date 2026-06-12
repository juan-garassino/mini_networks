"""Unit tests for the sweep quality gate: evalspec, S-checks, report writer."""
import json

from mini_networks.core.evalspec import EVAL_SPECS, EvalSpec, get_eval_spec, passes_threshold
from mini_networks.core.sweep_report import CheckResult, render_markdown, write_report
from mini_networks.colab.gate import loss_series_check


def _metrics(key, values):
    return [{"step": i, "key": key, "value": v} for i, v in enumerate(values)]


class TestEvalSpecs:
    def test_covers_every_model_and_composition(self):
        from mini_networks.colab.launcher import MODELS, COMPOSITIONS
        expected = set(MODELS) | set(COMPOSITIONS)
        assert set(EVAL_SPECS) == expected

    def test_get_eval_spec_unknown_raises(self):
        import pytest
        with pytest.raises(KeyError, match="EvalSpec"):
            get_eval_spec("not_a_model")

    def test_thresholds_only_m_and_l(self):
        for name, spec in EVAL_SPECS.items():
            assert "S" not in spec.thresholds, f"{name}: S tier must not have a metric bar"

    def test_passes_threshold_directions(self):
        assert passes_threshold(0.9, 0.8, higher_is_better=True)
        assert not passes_threshold(0.7, 0.8, higher_is_better=True)
        assert passes_threshold(1.5, 2.0, higher_is_better=False)
        assert not passes_threshold(2.5, 2.0, higher_is_better=False)


class TestLossSeriesCheck:
    SPEC = EvalSpec(metric=None, loss_keys=("loss",))

    def test_decreasing_passes(self):
        ok, info = loss_series_check(_metrics("loss", [2.0, 1.5, 1.0]), self.SPEC)
        assert ok and info["trend"] == "ok"

    def test_increasing_fails(self):
        ok, info = loss_series_check(_metrics("loss", [1.0, 1.5, 2.0]), self.SPEC)
        assert not ok and "no loss decreased" in info["trend"]

    def test_nan_fails(self):
        ok, info = loss_series_check(_metrics("loss", [1.0, float("nan")]), self.SPEC)
        assert not ok and not info["finite"]

    def test_single_point_skips_trend(self):
        ok, info = loss_series_check(_metrics("loss", [1.0]), self.SPEC)
        assert ok and info["trend"] == "skipped (single point)"

    def test_missing_series_fails(self):
        ok, info = loss_series_check(_metrics("other", [1.0, 0.5]), self.SPEC)
        assert not ok and "no loss series found" in info["trend"]

    def test_finite_mode_skips_trend(self):
        spec = EvalSpec(metric=None, loss_keys=("reward",), s_mode="finite")
        ok, info = loss_series_check(_metrics("reward", [0.1, 0.05]), spec)
        assert ok and "s_mode=finite" in info["trend"]

    def test_one_of_many_keys_decreasing_passes(self):
        spec = EvalSpec(metric=None, loss_keys=("g_loss", "d_loss"))
        metrics = _metrics("g_loss", [1.0, 2.0]) + _metrics("d_loss", [1.0, 0.5])
        ok, info = loss_series_check(metrics, spec)
        assert ok


class TestReport:
    def _results(self):
        return [
            CheckResult(item_type="model", name="classifier", status="pass", tier="S",
                        metric="accuracy", value=0.91, threshold=None, roundtrip="ok",
                        duration_s=3.2),
            CheckResult(item_type="composition", name="latent_diffusion", status="fail",
                        tier="S", s_check={"trend": "no loss decreased"}, duration_s=8.0),
            CheckResult(item_type="model", name="gan", status="error", tier="S",
                        error="Traceback ...\nRuntimeError: boom", duration_s=1.0),
        ]

    def test_write_report(self, tmp_path):
        md, js = write_report(self._results(), tmp_path / "sweep" / "x", {"tier": "S"})
        assert md.exists() and js.exists()
        payload = json.loads(js.read_text())
        assert len(payload["results"]) == 3
        assert payload["meta"]["tier"] == "S"

    def test_markdown_contains_counts_and_failures(self):
        text = render_markdown(self._results(), {"tier": "S"})
        assert "1 pass / 1 fail / 1 error" in text
        assert "### latent_diffusion (fail)" in text
        assert "RuntimeError: boom" in text
