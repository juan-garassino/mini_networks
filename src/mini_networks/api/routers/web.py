"""Read-layer endpoints: the Observatory reads runs through these.

Backed by a pluggable RunSource (local runs/ or MLflow) selected by env; the
frontend never knows which. Run ids may contain ``/`` (local ids are
``model/timestamp``), hence the ``:path`` converter.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from mini_networks.api.schemas.web import (
    ConfigResponse,
    MetricsResponse,
    ModelInfo,
    RunListResponse,
    SummaryResponse,
)
from mini_networks.web.lessons import list_lessons, read_lesson
from mini_networks.web.model_catalog import list_model_infos
from mini_networks.web.sources import RunNotFound, RunSource, get_run_source

router = APIRouter()


@router.get("/runs", response_model=RunListResponse)
async def list_runs(source: RunSource = Depends(get_run_source)):
    return RunListResponse(runs=source.list_runs())


@router.get("/runs/{run_id:path}/metrics", response_model=MetricsResponse)
async def run_metrics(run_id: str, since: int | None = None, source: RunSource = Depends(get_run_source)):
    try:
        return source.get_metrics(run_id, since)
    except RunNotFound:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")


@router.get("/runs/{run_id:path}/config", response_model=ConfigResponse)
async def run_config(run_id: str, source: RunSource = Depends(get_run_source)):
    try:
        return source.get_config(run_id)
    except RunNotFound:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")


@router.get("/runs/{run_id:path}/summary", response_model=SummaryResponse)
async def run_summary(run_id: str, source: RunSource = Depends(get_run_source)):
    try:
        return source.get_summary(run_id)
    except RunNotFound:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")


@router.get("/runs/{run_id:path}/artifacts/{name}")
async def run_artifact(run_id: str, name: str, source: RunSource = Depends(get_run_source)):
    try:
        path, media = source.open_artifact(run_id, name)
    except RunNotFound:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {run_id}/{name}")
    return FileResponse(path, media_type=media, filename=name)


@router.get("/models", response_model=list[ModelInfo])
async def models():
    return list_model_infos()


@router.get("/lessons")
async def lessons():
    return list_lessons()


@router.get("/lessons/{lesson_id}")
async def lesson(lesson_id: str):
    try:
        return {"id": lesson_id, "markdown": read_lesson(lesson_id)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Lesson not found: {lesson_id}")
