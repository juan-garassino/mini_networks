#!/usr/bin/env python
"""mini_networks — single entry point for all environments.

Sub-commands
------------
  python main.py serve    [--host 0.0.0.0] [--port 8000] [--reload]
  python main.py train    --model <name> [--fast_demo] [--epochs N] [--device cpu]
  python main.py compose  --composition <name> [--fast_demo] [--device cpu]
  python main.py evaluate --model <name> --checkpoint <dir>
  python main.py menu     # interactive rich TUI
  python main.py list     # list all models and compositions

Colab auto-detection
--------------------
  When no sub-command is given and running in Colab, defaults to 'menu'.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------------------------
# Colab detection
# ---------------------------------------------------------------------------

def _in_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_JUPYTER_TOKEN"))


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        sys.exit("uvicorn not installed — run: uv sync")

    from rich.console import Console
    Console().print(
        f"[bold green]Starting mini_networks API[/bold green]  "
        f"http://{args.host}:{args.port}/docs"
    )
    uvicorn.run(
        "mini_networks.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train a single model."""
    from mini_networks.colab.launcher import run_model
    from rich.console import Console

    console = Console()
    console.print(f"[bold]Training:[/bold] {args.model}  fast_demo={args.fast_demo}")

    extra: dict = {}
    if args.epochs is not None:
        extra["epochs"] = args.epochs
    if args.device:
        extra["device"] = args.device
    if args.data_root:
        extra["data_root"] = args.data_root
    if args.checkpoint_root:
        extra["checkpoint_root"] = args.checkpoint_root
    extra["training_tier"] = "S" if args.fast_demo else args.training_tier
    extra["resume"] = args.resume

    logger = run_model(args.model, fast_demo=args.fast_demo, **extra)
    console.print(f"[green]Done.[/green] Artifacts: {logger.artifacts_dir}")


def cmd_compose(args: argparse.Namespace) -> None:
    """Run a multi-model composition."""
    from mini_networks.colab.launcher import run_composition
    from rich.console import Console

    console = Console()
    console.print(f"[bold]Composition:[/bold] {args.composition}  fast_demo={args.fast_demo}")
    device = args.device or "cpu"
    result = run_composition(
        args.composition,
        fast_demo=args.fast_demo,
        training_tier="S" if args.fast_demo else args.training_tier,
        device=device,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
    )
    if isinstance(result, dict) and "config" in result:
        console.print("[green]Done.[/green]")

def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a model from a checkpoint directory."""
    from rich.console import Console
    from mini_networks.api.dependencies import get_model_registry

    console = Console()
    registry = get_model_registry()

    if args.model not in registry:
        console.print(f"[red]Unknown model: {args.model}[/red]")
        console.print(f"Available: {', '.join(registry)}")
        sys.exit(1)

    ConfigClass, TrainerClass, DataloaderFn = registry[args.model]
    config = ConfigClass(fast_demo=True)
    if args.device:
        config = config.model_copy(update={"device": args.device})

    trainer = TrainerClass()
    trainer.load_checkpoint(config, args.checkpoint)

    # Build a small eval dataloader
    dl = DataloaderFn(config, split="test")

    import tempfile
    from mini_networks.core.logging.logger import Logger
    with tempfile.TemporaryDirectory() as tmp:
        logger = Logger(tmp, "eval")
        metrics = trainer.evaluate(config, dl, logger)

    console.print(f"[bold]Evaluation results for {args.model}:[/bold]")
    for k, v in metrics.items():
        console.print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def cmd_menu(args: argparse.Namespace) -> None:
    """Launch the interactive rich TUI."""
    from mini_networks.colab.launcher import interactive_menu
    interactive_menu()


def cmd_list(args: argparse.Namespace) -> None:
    """List all available models and compositions."""
    from mini_networks.colab.launcher import list_models, list_compositions
    list_models()
    list_compositions()


def _parse_targets(raw: str, available: list[str]) -> list[str]:
    if raw == "all":
        return available
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(f"Unknown targets: {', '.join(unknown)}")
    return selected


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run a sweep across models and/or compositions."""
    from rich.console import Console
    from rich.table import Table
    from mini_networks.colab.launcher import MODELS, COMPOSITIONS, run_model, run_composition

    console = Console()
    training_tier = "S" if args.fast_demo else args.training_tier

    models = _parse_targets(args.models, MODELS) if args.include_models else []
    compositions = _parse_targets(args.compositions, COMPOSITIONS) if args.include_compositions else []

    if not models and not compositions:
        console.print("[red]Sweep has nothing to run. Enable models and/or compositions.[/red]")
        sys.exit(1)

    total = len(models) + len(compositions)
    console.print(
        f"[bold]Sweep:[/bold] tier={training_tier} items={total} "
        f"checkpoint_root={args.checkpoint_root}"
    )

    results: list[tuple[str, str, str]] = []

    for model in models:
        try:
            run_model(
                model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                fast_demo=args.fast_demo,
                training_tier=training_tier,
                data_root=args.data_root,
                device=args.device,
                checkpoint_root=args.checkpoint_root,
                resume=args.resume,
                validate_inference=True,
            )
            results.append(("model", model, "ok"))
        except Exception as exc:
            results.append(("model", model, f"failed: {exc}"))
            if args.fail_fast:
                raise

    for composition in compositions:
        try:
            run_composition(
                composition,
                fast_demo=args.fast_demo,
                training_tier=training_tier,
                data_root=args.data_root,
                device=args.device,
                checkpoint_root=args.checkpoint_root,
                validate_inference=True,
            )
            results.append(("composition", composition, "ok"))
        except Exception as exc:
            results.append(("composition", composition, f"failed: {exc}"))
            if args.fail_fast:
                raise

    table = Table(title="Sweep Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Status")
    for item_type, name, status in results:
        style = "green" if status == "ok" else "red"
        table.add_row(item_type, name, f"[{style}]{status}[/{style}]")
    console.print(table)

    failures = [row for row in results if row[2] != "ok"]
    if failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini_networks",
        description="mini_networks — educational ML framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Start FastAPI server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    # train
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--model", required=True)
    p_train.add_argument("--fast_demo", action="store_true", default=False)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--data_root", default=None)
    p_train.add_argument("--checkpoint_root", default=os.path.join(os.getcwd(), "runs"))
    p_train.add_argument("--training_tier", choices=["S", "M", "L"], default="M")
    p_train.add_argument("--no-resume", dest="resume", action="store_false")
    p_train.set_defaults(resume=True)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate from checkpoint")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--checkpoint", required=True,
                        help="Path to artifacts/ directory")
    p_eval.add_argument("--device", default=None)

    # menu
    sub.add_parser("menu", help="Interactive TUI menu")

    # compose
    p_comp = sub.add_parser("compose", help="Run a composition")
    p_comp.add_argument("--composition", required=True)
    p_comp.add_argument("--fast_demo", action="store_true", default=False)
    p_comp.add_argument("--device", default=None)
    p_comp.add_argument("--data_root", default=os.path.join(os.getcwd(), "data"))
    p_comp.add_argument("--checkpoint_root", default=os.path.join(os.getcwd(), "runs"))
    p_comp.add_argument("--training_tier", choices=["S", "M", "L"], default="M")

    # sweep
    p_sweep = sub.add_parser("sweep", help="Run many models/compositions in sequence")
    p_sweep.add_argument("--fast_demo", action="store_true", default=False)
    p_sweep.add_argument("--training_tier", choices=["S", "M", "L"], default="S")
    p_sweep.add_argument("--epochs", type=int, default=2)
    p_sweep.add_argument("--batch_size", type=int, default=32)
    p_sweep.add_argument("--device", default="cpu")
    p_sweep.add_argument("--data_root", default=os.path.join(os.getcwd(), "data"))
    p_sweep.add_argument("--checkpoint_root", default=os.path.join(os.getcwd(), "runs"))
    p_sweep.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    p_sweep.add_argument("--compositions", default="all", help="Comma-separated composition names or 'all'")
    p_sweep.add_argument("--skip-models", dest="include_models", action="store_false")
    p_sweep.add_argument("--skip-compositions", dest="include_compositions", action="store_false")
    p_sweep.add_argument("--fail-fast", action="store_true")
    p_sweep.add_argument("--no-resume", dest="resume", action="store_false")
    p_sweep.set_defaults(include_models=True, include_compositions=True, resume=True)

    # list
    sub.add_parser("list", help="List all models and compositions")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Colab fallback: no sub-command → interactive menu
    if args.command is None:
        if _in_colab():
            args.command = "menu"
        else:
            parser.print_help()
            sys.exit(0)

    dispatch = {
        "serve":    cmd_serve,
        "train":    cmd_train,
        "compose":  cmd_compose,
        "sweep":    cmd_sweep,
        "evaluate": cmd_evaluate,
        "menu":     cmd_menu,
        "list":     cmd_list,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
