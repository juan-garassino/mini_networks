"""Tests for scripts/render_results.py marker injection."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "render_results", Path(__file__).parent.parent / "scripts" / "render_results.py"
)
render_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_results)


REPORT = {
    "meta": {"tier": "M", "device": "cuda"},
    "results": [
        {"name": "classifier", "status": "pass", "metric": "accuracy",
         "value": 0.96, "threshold": 0.85},
        {"name": "gan", "status": "fail", "metric": "judge_score",
         "value": 0.10, "threshold": 0.15},
    ],
}

DOC = """# Chapter

Intro text.

<!-- results:start items=classifier,gan,missing_model -->
stale content
<!-- results:end -->

Outro text.
"""


def _setup(tmp_path):
    doc = tmp_path / "chapter.md"
    doc.write_text(DOC)
    sweep = tmp_path / "sweep" / "20260612-000000"
    sweep.mkdir(parents=True)
    (sweep / "report.json").write_text(json.dumps(REPORT))
    return doc, tmp_path / "sweep"


def test_injects_table(tmp_path):
    doc, sweep_root = _setup(tmp_path)
    report = render_results.latest_report(sweep_root)
    assert render_results.render_file(doc, report)
    text = doc.read_text()
    assert "| classifier | pass | accuracy | 0.9600 | 0.8500 |" in text
    assert "| gan | fail |" in text
    assert "_not in last sweep_" in text
    assert "stale content" not in text
    assert "Intro text." in text and "Outro text." in text


def test_idempotent(tmp_path):
    doc, sweep_root = _setup(tmp_path)
    report = render_results.latest_report(sweep_root)
    render_results.render_file(doc, report)
    assert not render_results.render_file(doc, report)


def test_no_marker_untouched(tmp_path):
    doc = tmp_path / "plain.md"
    doc.write_text("# No markers here\n")
    assert not render_results.render_file(doc, REPORT)
