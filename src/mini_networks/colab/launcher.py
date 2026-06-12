"""Facade for the colab package — import surface kept stable.

Implementation now lives in catalog.py (names/descriptions), probes.py
(inference probes), runners.py (run_model/run_composition), menu.py (TUI).
"""
from __future__ import annotations

from mini_networks.colab.catalog import COMPOSITIONS, DESCRIPTIONS, MODELS  # noqa: F401
from mini_networks.colab.menu import (  # noqa: F401
    install_deps,
    interactive_menu,
    list_compositions,
    list_models,
    main,
)
from mini_networks.colab.probes import (  # noqa: F401
    _run_model_inference_probe,
    _validate_probe_output,
)
from mini_networks.colab.runners import (  # noqa: F401
    COMPOSITION_RUNNERS,
    _run_base,
    run_composition,
    run_model,
)

if __name__ == "__main__":
    main()
