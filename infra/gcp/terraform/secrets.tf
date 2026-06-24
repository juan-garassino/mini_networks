# Neon DSN for the MLflow tracking DB. The VALUE is added out-of-band — never in
# Terraform/git/state:
#   gcloud secrets versions add mini-networks-neon-dsn --data-file=- <<<"$DSN"
# DSN must include ?sslmode=require (Neon rejects non-TLS).
resource "google_secret_manager_secret" "neon_dsn" {
  secret_id = var.neon_secret_id
  replication {
    auto {}
  }
}
