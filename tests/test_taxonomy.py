"""Sync tests: the taxonomy must cover the registry exactly and stay a DAG."""
from pathlib import Path

from mini_networks.colab.catalog import COMPOSITIONS, MODELS
from mini_networks.core.taxonomy import (
    COMPOSITION_TAXONOMY,
    MECHANISMS,
    MODEL_TAXONOMY,
    atoms,
    dependency_edges,
    mermaid,
    molecules,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_covers_every_model_exactly():
    assert set(MODEL_TAXONOMY) == set(MODELS)


def test_covers_every_composition_exactly():
    assert set(COMPOSITION_TAXONOMY) == set(COMPOSITIONS)


def test_builds_on_references_valid_models():
    for name, taxon in MODEL_TAXONOMY.items():
        for parent in taxon.builds_on:
            assert parent in MODEL_TAXONOMY, f"{name} builds_on unknown {parent!r}"
            assert parent != name


def test_composes_references_valid_models():
    for name, models in COMPOSITION_TAXONOMY.items():
        assert models, f"composition {name} composes nothing"
        for m in models:
            assert m in MODEL_TAXONOMY, f"{name} composes unknown {m!r}"


def test_elementary_iff_no_parents():
    for name, taxon in MODEL_TAXONOMY.items():
        if taxon.level == "elementary":
            assert not taxon.builds_on, f"atom {name} has parents"
            assert taxon.introduces, f"atom {name} introduces nothing"
        else:
            assert taxon.level == "derived"
            assert taxon.builds_on, f"derived {name} has no parents"


def test_atom_count_is_locked():
    """16 atoms as designed — changing this number is a curriculum decision."""
    assert len(atoms()) == 16
    assert len(atoms()) + len(molecules()) == len(MODELS)


def test_dag_acyclic():
    edges = [(c, p) for c, p in dependency_edges() if c in MODEL_TAXONOMY]
    children: dict[str, list[str]] = {}
    for child, parent in edges:
        children.setdefault(parent, []).append(child)

    state: dict[str, int] = {}

    def visit(node: str):
        if state.get(node) == 1:
            raise AssertionError(f"cycle through {node}")
        if state.get(node) == 2:
            return
        state[node] = 1
        for ch in children.get(node, []):
            visit(ch)
        state[node] = 2

    for node in MODEL_TAXONOMY:
        visit(node)


def test_mechanisms_introduced_exactly_once():
    seen: dict[str, str] = {}
    for name, taxon in MODEL_TAXONOMY.items():
        for mech in taxon.introduces:
            assert mech in MECHANISMS, f"{name} introduces unknown mechanism {mech!r}"
            assert mech not in seen, f"{mech!r} introduced by both {seen[mech]} and {name}"
            seen[mech] = name
    assert set(seen) == set(MECHANISMS), (
        f"orphan mechanisms (in MECHANISMS but introduced by nobody): "
        f"{set(MECHANISMS) - set(seen)}"
    )


def test_mechanism_homes_exist():
    for name, mech in MECHANISMS.items():
        assert (REPO_ROOT / mech.home).exists(), f"{name}: missing home {mech.home}"


def test_mermaid_mentions_every_model():
    diagram = mermaid()
    for name in MODEL_TAXONOMY:
        assert name in diagram
