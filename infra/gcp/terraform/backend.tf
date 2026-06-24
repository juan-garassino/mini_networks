# Terraform state lives in garassino-op (per the workspace GCP layout).
# `make validate` runs `init -backend=false`, so this is inert for static checks.
terraform {
  backend "gcs" {
    bucket = "garassino-op-tf-state"
    prefix = "mini-networks"
  }
}
