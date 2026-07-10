# ONE parallel gate sweep job: every model + composition runs as its own task
# on an L4 (MODE=sweep-task shards the catalog via CLOUD_RUN_TASK_INDEX; each
# task trains + gates its item, mirrors to MLflow, registers gate-passing M/L
# checkpoints, and uploads a report shard to
# gs://<bucket>/<prefix>/sweeps/<SWEEP_ID>/shards/). Zero cost at rest.
# GPU shape cribbed from the live ppe-train job (gen2 + nvidia-l4 +
# zonal-redundancy off — europe-west1 support is proven by that job).
resource "google_cloud_run_v2_job" "sweep" {
  name     = "mini-networks-sweep"
  location = var.region

  template {
    task_count  = var.sweep_task_count
    parallelism = var.sweep_parallelism

    template {
      service_account               = google_service_account.runtime.email
      max_retries                   = 1
      timeout                       = "3600s"
      execution_environment         = "EXECUTION_ENVIRONMENT_GEN2"
      gpu_zonal_redundancy_disabled = true

      node_selector {
        accelerator = "nvidia-l4"
      }

      containers {
        image = local.train_gpu_image

        env {
          name  = "MODE"
          value = "sweep-task"
        }
        env {
          name  = "TRAINING_TIER"
          value = "M"
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
          name  = "MN_SWEEP_BUCKET"
          value = var.bucket_name
        }
        env {
          name  = "MN_SWEEP_PREFIX"
          value = var.bucket_prefix
        }
        env {
          name  = "MN_MLFLOW_EXPERIMENT"
          value = var.mlflow_experiment
        }
        env {
          name  = "MN_MLFLOW_TRACKING_URI"
          value = var.mlflow_tracking_url
        }
        env {
          name  = "MN_MLFLOW_REGISTER"
          value = "1"
        }
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }

        resources {
          limits = {
            cpu              = "8"
            memory           = "32Gi"
            "nvidia.com/gpu" = "1"
          }
        }
      }
    }
  }
}
