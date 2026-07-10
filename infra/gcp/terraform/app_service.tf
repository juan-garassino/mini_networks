# Public showcase: the playground UI + read-layer + champion inference.
# min-instances 0 (≈€0 idle); training endpoints are hard-disabled
# (MN_DISABLE_TRAIN) — anonymous visitors browse runs and run inference on
# the registry champions, nothing more. Champions are pulled at container
# start by entrypoint-app.sh (runtime SA has scoped read on the tracker
# bucket's experiment prefix).
resource "google_cloud_run_v2_service" "app" {
  name     = "mini-networks-app"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.runtime.email

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = local.app_image

      env {
        name  = "MN_RUN_SOURCE"
        value = "mlflow"
      }
      env {
        name  = "MN_MLFLOW_TRACKING_URI"
        value = var.mlflow_tracking_url
      }
      env {
        name  = "MN_MLFLOW_EXPERIMENT"
        value = var.mlflow_experiment
      }
      env {
        name  = "MN_DISABLE_TRAIN"
        value = "1"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      # The champions pull runs before uvicorn binds the port — allow a
      # generous cold start (32 small checkpoints from GCS).
      startup_probe {
        tcp_socket {
          port = 8080
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        failure_threshold     = 24
      }
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "app_public" {
  name     = google_cloud_run_v2_service.app.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
