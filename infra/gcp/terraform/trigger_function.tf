# Pub/Sub → Cloud Function (2nd gen) → Cloud Run Job execution.
data "archive_file" "function" {
  type        = "zip"
  source_dir  = "${path.module}/../function"
  output_path = "${path.module}/.build/function.zip"
}

resource "google_storage_bucket_object" "function_src" {
  name   = "${var.bucket_prefix}/function-src/${data.archive_file.function.output_md5}.zip"
  bucket = var.bucket_name
  source = data.archive_file.function.output_path
}

resource "google_cloudfunctions2_function" "trigger" {
  name     = "mini-networks-train-trigger"
  location = var.region

  build_config {
    runtime     = "python312"
    entry_point = "on_train_request"
    source {
      storage_source {
        bucket = var.bucket_name
        object = google_storage_bucket_object.function_src.name
      }
    }
  }

  service_config {
    min_instance_count    = 0
    max_instance_count    = 3
    available_memory      = "256Mi"
    timeout_seconds       = 120
    service_account_email = google_service_account.trigger.email
    environment_variables = {
      GCP_PROJECT  = var.project_id
      GCP_REGION   = var.region
      CPU_JOB_NAME = google_cloud_run_v2_job.train.name
      GPU_JOB_NAME = var.enable_gpu_job ? "mini-networks-train-gpu" : ""
      GPU_MODELS   = var.gpu_models
    }
  }

  event_trigger {
    trigger_region = var.region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.train_requests.id
    retry_policy   = "RETRY_POLICY_RETRY"
  }
}
