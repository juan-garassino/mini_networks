terraform {
  required_version = ">= 1.6"
  required_providers {
    # >=6: node_selector + gpu_zonal_redundancy_disabled on cloud_run_v2_job
    # (the L4 sweep job needs both).
    google  = { source = "hashicorp/google", version = "~> 6.0" }
    archive = { source = "hashicorp/archive", version = "~> 2.4" }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# APIs the stack needs (idempotent).
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "pubsub.googleapis.com",
    "cloudfunctions.googleapis.com",
    "eventarc.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

locals {
  train_image     = "${var.region}-docker.pkg.dev/${var.project_id}/${var.ar_repo}/mini-networks-train:${var.image_tag}"
  train_gpu_image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.ar_repo}/mini-networks-train-gpu:${var.image_tag}"
  app_image       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.ar_repo}/mini-networks-app:${var.image_tag}"
  artifact_root   = "gs://${var.bucket_name}/${var.bucket_prefix}/artifacts"
}
