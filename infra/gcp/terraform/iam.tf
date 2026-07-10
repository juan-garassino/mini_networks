# Runtime SA — the training job runs as this; GCS + Neon access via ADC.
resource "google_service_account" "runtime" {
  account_id   = var.runtime_sa_id
  display_name = "mini-networks ephemeral training runtime"
}

# Trigger SA — the Cloud Function that launches job executions.
resource "google_service_account" "trigger" {
  account_id   = var.trigger_sa_id
  display_name = "mini-networks train-request trigger"
}

# Runtime SA may write artifacts under the mini-networks/ prefix only.
resource "google_storage_bucket_iam_member" "runtime_objects" {
  bucket = data.google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runtime.email}"
  condition {
    title      = "mini-networks-prefix"
    expression = "resource.name.startsWith(\"projects/_/buckets/${var.bucket_name}/objects/${var.bucket_prefix}/\")"
  }
}

# Runtime SA writes run artifacts + registry checkpoints to the tracker's
# bucket (direct gs:// artifact URIs — see gcs.tf). Scoped by IAM condition to
# the mini-networks EXPERIMENT prefix (id 10 on the shared tracker) so this SA
# can never touch the desktop projects' artifacts. If the experiment is ever
# recreated it gets a new id — update the prefix here.
resource "google_storage_bucket_iam_member" "runtime_mlflow_objects" {
  bucket = data.google_storage_bucket.mlflow_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runtime.email}"
  condition {
    title      = "mini-networks-experiment-prefix"
    expression = "resource.name.startsWith(\"projects/_/buckets/garassino-ml-mlflow-artifacts/objects/10/\")"
  }
}

# Trigger SA: launch Cloud Run Jobs WITH OVERRIDES — needs run.developer
# (run.invoker is NOT sufficient for runJob-with-overrides).
resource "google_project_iam_member" "trigger_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.trigger.email}"
}

# Trigger SA must actAs the runtime SA (the job runs as the runtime SA).
resource "google_service_account_iam_member" "trigger_act_as_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.trigger.email}"
}

# WIF: let the GitHub repo impersonate the runtime SA for CI image push
# (no service-account JSON keys). Skipped when wif_provider is unset.
resource "google_service_account_iam_member" "wif_runtime" {
  count              = var.wif_provider == "" ? 0 : 1
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${var.wif_provider}/attribute.repository/${var.github_repo}"
}
