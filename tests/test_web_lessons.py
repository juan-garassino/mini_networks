"""Lessons read-layer endpoint (curriculum chapters from docs/)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from mini_networks.api.main import create_app


def test_lessons_list_and_read():
    c = TestClient(create_app())
    lessons = c.get("/web/lessons").json()
    assert lessons and all({"id", "num", "title"} <= set(l) for l in lessons)
    first = lessons[0]["id"]
    body = c.get(f"/web/lessons/{first}").json()
    assert body["id"] == first and len(body["markdown"]) > 0


def test_lesson_unknown_404():
    c = TestClient(create_app())
    assert c.get("/web/lessons/99-nope").status_code == 404


def test_lessons_dir_override(tmp_path, monkeypatch):
    (tmp_path / "03-foo.md").write_text("# Three — Foo\n\nbody\n")
    (tmp_path / "notes.md").write_text("# not a chapter\n")  # unnumbered → excluded
    monkeypatch.setenv("MN_DOCS_DIR", str(tmp_path))
    c = TestClient(create_app())
    ids = [l["id"] for l in c.get("/web/lessons").json()]
    assert ids == ["03-foo"]
    assert c.get("/web/lessons/03-foo").json()["markdown"].startswith("# Three")
