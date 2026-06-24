// Client for the FastAPI read-layer. Same-origin in production (FastAPI serves
// the static export at /), configurable for dev via NEXT_PUBLIC_API_BASE.
import type {
  RunSummary, MetricsResponse, ConfigResponse, SummaryResponse,
  ModelInfo, Lesson, TrainResponse, InferResponse,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return (await r.json()) as T;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 160)}`);
  return (await r.json()) as T;
}

export const listRuns = () => getJSON<{ runs: RunSummary[] }>("/web/runs");
export const getMetrics = (id: string, since?: number) =>
  getJSON<MetricsResponse>(`/web/runs/${id}/metrics${since != null ? `?since=${since}` : ""}`);
export const getConfig = (id: string) => getJSON<ConfigResponse>(`/web/runs/${id}/config`);
export const getSummary = (id: string) => getJSON<SummaryResponse>(`/web/runs/${id}/summary`);
export const listModels = () => getJSON<ModelInfo[]>("/web/models");
export const listLessons = () => getJSON<Lesson[]>("/web/lessons");
export const getLesson = (id: string) => getJSON<{ id: string; markdown: string }>(`/web/lessons/${id}`);
export const artifactUrl = (id: string, name: string) => `${BASE}/web/runs/${id}/artifacts/${name}`;

export const startTrain = (model: string, body: unknown) => postJSON<TrainResponse>(`/train/${model}`, body);
export const infer = (model: string, body: unknown) => postJSON<InferResponse>(`/infer/${model}`, body);
