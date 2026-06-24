"""FastAPI app factory for mini_networks."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mini_networks.api.routers.inference import router as inference_router
from mini_networks.api.routers.training import router as training_router
from mini_networks.api.routers.compositions import router as compositions_router
from mini_networks.api.routers.web import router as web_router

# Repo-root frontend/ (no-build SPA). Absent in some installs → skip the mount.
FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(
        title="mini_networks",
        description=(
            "Unified ML training, inference, and composition API: "
            "vision, language, RL, and multimodal pipelines"
        ),
        version="0.1.0",
    )
    app.include_router(training_router, prefix="/train", tags=["training"])
    app.include_router(inference_router, prefix="/infer", tags=["inference"])
    app.include_router(compositions_router, prefix="/compose", tags=["composition"])
    app.include_router(web_router, prefix="/web", tags=["playground"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # The Observatory SPA. Mounted LAST so every API route wins over the catch-all.
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="playground")

    return app


app = create_app()
