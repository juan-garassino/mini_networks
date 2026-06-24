# Ephemeral CPU training job. Zero-cost at rest (jobs run only when executed).
# The trigger function launches executions with per-run env overrides.
resource "google_cloud_run_v2_job" "train" {
  name     = "mini-networks-train"
  location = var.region

  template {
    template {
      service_account = google_service_account.runtime.email
      max_retries     = 1
      timeout         = "3600s"

      containers {
        image = local.train_image

        env {
          name  = "MODE"
          value = "train"
        }
        env {
          name  = "CHECKPOINT_ROOT"
          value = "/tmp/runs"
        }
        env {
          name  = "MN_MLFLOW_ARTIFACT_ROOT"
          value = local.artifact_root
        }
        env {
          name  = "MN_MLFLOW_EXPERIMENT"
          value = var.mlflow_experiment
        }
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        # Neon DSN injected from Secret Manager at runtime — never in the image.
        env {
          name = "MN_MLFLOW_TRACKING_URI"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.neon_dsn.secret_id
              version = "latest"
            }
          }
        }

        resources {
          limits = {
            cpu    = var.cpu_limit
            memory = var.memory_limit
          }
        }
      }
    }
  }

  depends_on = [google_secret_manager_secret_iam_member.runtime_secret]
}

# Optional L4 GPU variant for the heavy few (diffusion/transformer/vit).
# Disabled by default: confirm europe-west1 Cloud Run GPU support before enabling
# (else use the GCE-L4 ephemeral fallback documented in the README). The GPU
# node shape (node_selector / launch_stage) is finalized when this is turned on.
resource "google_cloud_run_v2_job" "train_gpu" {
  count    = var.enable_gpu_job ? 1 : 0
  name     = "mini-networks-train-gpu"
  location = var.region

  template {
    template {
      service_account = google_service_account.runtime.email
      max_retries     = 1
      timeout         = "3600s"

      containers {
        image = local.train_image

        env {
          name  = "MODE"
          value = "train"
        }
        env {
          name  = "DEVICE"
          value = "cuda"
        }
        env {
          name  = "CHECKPOINT_ROOT"
          value = "/tmp/runs"
        }
        env {
          name  = "MN_MLFLOW_ARTIFACT_ROOT"
          value = local.artifact_root
        }
        env {
          name  = "MN_MLFLOW_EXPERIMENT"
          value = var.mlflow_experiment
        }
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        env {
          name = "MN_MLFLOW_TRACKING_URI"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.neon_dsn.secret_id
              version = "latest"
            }
          }
        }

        resources {
          limits = {
            cpu              = "4"
            memory           = "16Gi"
            "nvidia.com/gpu" = "1"
          }
        }
      }
    }
  }

  depends_on = [google_secret_manager_secret_iam_member.runtime_secret]
}
