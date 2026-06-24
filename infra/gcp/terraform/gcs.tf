# Reuse the shared artifact bucket (created out-of-band); never create/destroy it.
data "google_storage_bucket" "artifacts" {
  name = var.bucket_name
}
