# Reuse the shared artifact bucket (created out-of-band); never create/destroy it.
data "google_storage_bucket" "artifacts" {
  name = var.bucket_name
}

# The global tracker's artifact bucket (owned by the desktop garassino-ml
# workspace). The tracker hands out DIRECT gs:// artifact URIs (its
# --default-artifact-root is a plain GCS path, so nothing routes through the
# --serve-artifacts proxy) — every artifact/model upload goes straight to
# this bucket and the runtime SA needs write access (403s otherwise, m-full-3).
data "google_storage_bucket" "mlflow_artifacts" {
  name = "garassino-ml-mlflow-artifacts"
}
