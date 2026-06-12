"""Interactive rich TUI and the `python -m mini_networks.colab.launcher` CLI."""
from __future__ import annotations

import os
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from mini_networks.colab.catalog import CATEGORY, COMPOSITIONS, DESCRIPTIONS, MODELS
from mini_networks.colab.runners import run_composition, run_model

console = Console()


def install_deps() -> None:
    """Install the package in the current Python environment (pip-based)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[dev]", "-q"])



def _make_models_table(items: list[str], title: str) -> Table:
    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_lines=False, highlight=True)
    tbl.add_column("#", style="bold cyan", justify="right", width=4)
    tbl.add_column("Name", style="bold white", width=30)
    tbl.add_column("Category", style="dim", width=22)
    tbl.add_column("Description", style="white")
    for i, name in enumerate(items, 1):
        tbl.add_row(str(i), name, CATEGORY.get(name, ""), DESCRIPTIONS.get(name, ""))
    return tbl



def list_models() -> None:
    console.print(_make_models_table(MODELS, "Available Models"))


def list_compositions() -> None:
    console.print(_make_models_table(COMPOSITIONS, "Available Compositions"))


# ---------------------------------------------------------------------------


def interactive_menu() -> None:
    """Display an interactive text menu for choosing a model or composition."""
    console.print(Panel(
        "[bold cyan]mini_networks[/bold cyan] — Educational ML Playground\n"
        f"[dim]{len(MODELS)} models · {len(COMPOSITIONS)} compositions · unified logging · FastAPI[/dim]",
        border_style="bright_blue",
    ))

    console.print("\nWhat would you like to explore?")
    console.print("  [bold cyan][1][/bold cyan] Train a single model")
    console.print("  [bold magenta][2][/bold magenta] Run a multi-model composition")
    console.print("  [bold red][q][/bold red] Quit")

    choice = console.input("\n[bold]Enter choice:[/bold] ").strip().lower()
    if choice == "q":
        return
    if choice not in ("1", "2"):
        console.print("[red]Invalid choice.[/red]")
        return

    if choice == "1":
        console.print()
        console.print(_make_models_table(MODELS, "Available Models"))
        idx = console.input("\n[bold]Enter model number or name:[/bold] ").strip()
        try:
            model = MODELS[int(idx) - 1] if idx.isdigit() else idx
        except IndexError:
            console.print("[red]Invalid selection.[/red]")
            return
        if model not in MODELS:
            console.print(f"[red]Unknown model: {model!r}[/red]")
            return
    else:
        console.print()
        console.print(_make_models_table(COMPOSITIONS, "Available Compositions"))
        idx = console.input("\n[bold]Enter composition number or name:[/bold] ").strip()
        try:
            comp = COMPOSITIONS[int(idx) - 1] if idx.isdigit() else idx
        except IndexError:
            console.print("[red]Invalid selection.[/red]")
            return
        if comp not in COMPOSITIONS:
            console.print(f"[red]Unknown composition: {comp!r}[/red]")
            return

    tier = (console.input("[bold]Training tier[/bold] [S/M/L] (default: M): ").strip().upper() or "M")
    if tier not in {"S", "M", "L"}:
        console.print("[red]Invalid tier.[/red]")
        return
    fast_demo = tier == "S"
    device = console.input("[bold]Device[/bold] [cpu/cuda/mps] (default: cpu): ").strip() or "cpu"

    if choice == "1":
        run_model(model, fast_demo=fast_demo, training_tier=tier, device=device)
    else:
        run_composition(comp, fast_demo=fast_demo, training_tier=tier, device=device)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="mini_networks training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models:       " + ", ".join(MODELS) + "\n"
            "Compositions: " + ", ".join(COMPOSITIONS)
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model",       choices=MODELS,       help="Single model to train")
    group.add_argument("--composition", choices=COMPOSITIONS, help="Multi-model composition to run")
    parser.add_argument("--interactive", action="store_true", help="Show interactive menu")
    parser.add_argument("--list",        action="store_true", help="List all models and compositions")
    parser.add_argument("--epochs",    type=int,   default=2)
    parser.add_argument("--fast_demo", action="store_true", default=True)
    parser.add_argument("--no_fast",   action="store_true", help="Disable fast_demo")
    parser.add_argument("--training_tier", choices=["S", "M", "L"], default="M")
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--data_root", default="/tmp/mini_networks_data")
    parser.add_argument("--checkpoint_root", default=os.path.join(os.getcwd(), "runs"))
    parser.add_argument("--no_resume", action="store_true", help="Disable auto-resume for single-model runs")

    args = parser.parse_args()

    if args.list:
        list_models()
        list_compositions()
    elif args.interactive or (not args.model and not args.composition):
        interactive_menu()
    elif args.model:
        run_model(
            args.model,
            epochs=args.epochs,
            fast_demo=not args.no_fast,
            training_tier="S" if not args.no_fast else args.training_tier,
            data_root=args.data_root,
            device=args.device,
            checkpoint_root=args.checkpoint_root,
            resume=not args.no_resume,
        )
    else:
        run_composition(
            args.composition,
            fast_demo=not args.no_fast,
            training_tier="S" if not args.no_fast else args.training_tier,
            data_root=args.data_root,
            device=args.device,
            checkpoint_root=args.checkpoint_root,
        )


if __name__ == "__main__":
    main()
