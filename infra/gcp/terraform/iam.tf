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
    title = "mini-networks-experiment-prefix"
    # Two resource shapes: objects under the experiment prefix (read/write
    # content) and the bucket itself — storage.objects.list is authorized
    # against the BUCKET resource, so without that arm the prefix-scoped
    # grant can't list (champion pulls + the /web read-layer both 403'd).
    # List exposes object NAMES bucket-wide; content stays 10/-only.
    expression = "resource.name.startsWith(\"projects/_/buckets/garassino-ml-mlflow-artifacts/objects/10/\") || resource.name == \"projects/_/buckets/garassino-ml-mlflow-artifacts\""
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
# Attribute-scoped principalSet members bind to the POOL (provider stripped) —
# the provider path is only for the workflow's workload_identity_provider input.
resource "google_service_account_iam_member" "wif_runtime" {
  count              = var.wif_provider == "" ? 0 : 1
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${replace(var.wif_provider, "//providers/.*$/", "")}/attribute.repository/${var.github_repo}"
}

# token_format: access_token in the workflow mints an OAuth token via
# generateAccessToken — that needs TokenCreator on top of workloadIdentityUser.
resource "google_service_account_iam_member" "wif_runtime_token" {
  count              = var.wif_provider == "" ? 0 : 1
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "principalSet://iam.googleapis.com/${replace(var.wif_provider, "//providers/.*$/", "")}/attribute.repository/${var.github_repo}"
}

# CI pushes train images to GAR ml-images as the runtime SA.
resource "google_project_iam_member" "runtime_ar_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}
