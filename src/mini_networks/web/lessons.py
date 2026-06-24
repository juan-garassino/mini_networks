"""Curriculum reader — lists the numbered docs/NN-*.md chapters for the Quest view.

A pure reader of docs/; serves the chapter list + raw markdown (rendered client
side). Path-sandboxed to numbered chapters directly under docs/.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_CHAPTER = re.compile(r"^\d\d-.+\.md$")


def _docs_dir() -> Path:
    env = os.environ.get("MN_DOCS_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "docs"


def _title(p: Path) -> str:
    try:
        for line in p.read_text(errors="ignore").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return p.stem


def list_lessons() -> list[dict]:
    d = _docs_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("[0-9][0-9]-*.md")):
        out.append({"id": p.stem, "num": p.name[:2], "title": _title(p)})
    return out


def read_lesson(lesson_id: str) -> str:
    d = _docs_dir()
    p = (d / f"{lesson_id}.md").resolve()
    if p.parent != d.resolve() or not _CHAPTER.match(p.name) or not p.exists():
        raise FileNotFoundError(lesson_id)
    return p.read_text(errors="ignore")
