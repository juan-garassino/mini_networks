// Mirrors the FastAPI /web read-layer Pydantic schemas.
export type RunStatus =
  | "pending" | "running" | "done" | "failed" | "dispatched" | "unknown";

export interface RunSummary {
  id: string;
  model: string;
  source: "local" | "mlflow" | "jobstore";
  status: RunStatus;
  run_name?: string | null;
  created_at?: string | null;
  last_step?: number | null;
  last_metrics: Record<string, number>;
  artifact_names: string[];
}

export interface MetricSeries { key: string; points: [number, number][]; }
export interface MetricsResponse { run_id: string; series: MetricSeries[]; latest_step?: number | null; }
export interface ConfigResponse { run_id: string; config: Record<string, unknown>; }
export interface SummaryResponse { run_id: string; summary: Record<string, unknown>; }
export interface ModelInfo {
  name: string;
  family?: string | null;
  config_schema: Record<string, unknown>;
  defaults: Record<string, unknown>;
}
export interface Lesson { id: string; num: string; title: string; }
export interface TrainResponse { job_id: string; status: string; output_dir: string; }
export interface InferResponse { model: string; outputs: Record<string, unknown>; }
