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

variable "neon_secret_id" {
  type    = string
  default = "mini-networks-neon-dsn"
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
