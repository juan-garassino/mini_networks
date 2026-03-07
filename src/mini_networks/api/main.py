"""FastAPI app factory for mini_networks."""
from __future__ import annotations

from fastapi import FastAPI

from mini_networks.api.routers.inference import router as inference_router
from mini_networks.api.routers.training import router as training_router
from mini_networks.api.routers.compositions import router as compositions_router


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

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
