variable "project_id" {
  type    = string
  default = "garassino-ml"
}

variable "region" {
  type    = string
  default = "europe-west1"
}

# Shared artifact bucket (created out-of-band; referenced as a data source).
variable "bucket_name" {
  type    = string
  default = "garassino-ml-artifacts"
}

variable "bucket_prefix" {
  type    = string
  default = "mini-networks"
}

variable "ar_repo" {
  type    = string
  default = "ml-images"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "mlflow_experiment" {
  type    = string
  default = "mini-networks"
}

# Global garassino MLflow tracker (Cloud Run + Cloud SQL, --serve-artifacts).
# Public URL, so plain env — no secret. All workspace projects share this UI;
# mini-networks writes under experiment var.mlflow_experiment.
variable "mlflow_tracking_url" {
  type    = string
  default = "https://garassino-mlflow-mjz4n7eeia-ew.a.run.app"
}

variable "runtime_sa_id" {
  type    = string
  default = "mini-networks-train"
}

variable "trigger_sa_id" {
  type    = string
  default = "mini-networks-trigger"
}

# Workload Identity Federation (lives in garassino-op). The pool's full provider
# name and the GitHub repo that may impersonate the runtime SA for CI image push.
variable "wif_provider" {
  type        = string
  default     = ""
  description = "projects/<op-number>/locations/global/workloadIdentityPools/gh-actions/providers/github"
}

variable "github_repo" {
  type        = string
  default     = "juan-garassino/mini-networks"
  description = "owner/repo allowed to impersonate the runtime SA via WIF"
}

# Parallel gate sweep (mini-networks-sweep job). task_count is a default
# ceiling — `make sweep` overrides --tasks per execution to match ITEMS.
# Parallelism is bounded by the regional Cloud Run L4 quota; tasks queue
# beyond it, so a larger count is safe but slower per wall-clock.
variable "sweep_task_count" {
  type    = number
  default = 51 # 32 models + 19 compositions
}

variable "sweep_parallelism" {
  type    = number
  default = 3 # NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion quota is 3 in garassino-ml/europe-west1
}

# Heavy models that warrant a GPU; routed by the trigger function to the GPU job.
variable "gpu_models" {
  type    = string
  default = "diffusion,transformer,vit"
}

# Cloud Run GPU is region-gated; keep the GPU job off until europe-west1 support
# is confirmed (else the documented GCE-L4 fallback). Default CPU covers the
# 31 tiny models.
variable "enable_gpu_job" {
  type    = bool
  default = false
}

variable "cpu_limit" {
  type    = string
  default = "2"
}

variable "memory_limit" {
  type    = string
  default = "4Gi"
}

# Budget alerts need billing-account admin; default off (account-level alerts
# already exist on garassino-ml per the workspace cost policy).
variable "manage_budget" {
  type    = bool
  default = false
}

variable "billing_account" {
  type    = string
  default = ""
}

variable "budget_amount_eur" {
  type    = string
  default = "25"
}

variable "alert_email" {
  type    = string
  default = "juan.garassino@hotmail.com"
}
