output "train_job" {
  value = google_cloud_run_v2_job.train.name
}

output "app_url" {
  value = google_cloud_run_v2_service.app.uri
}

output "topic" {
  value = google_pubsub_topic.train_requests.id
}

output "train_image" {
  value = local.train_image
}

output "artifact_root" {
  value = local.artifact_root
}

output "runtime_sa" {
  value = google_service_account.runtime.email
}

output "trigger_sa" {
  value = google_service_account.trigger.email
}

output "next_steps" {
  value = <<-EOT
    1. Publish a test run:
         gcloud pubsub topics publish ${google_pubsub_topic.train_requests.name} \
           --message '{"model":"classifier","training_tier":"S","run_name":"smoke-1"}'
    2. Point the playground at the global tracker and watch it:
         MN_RUN_SOURCE=mlflow MN_MLFLOW_TRACKING_URI=${var.mlflow_tracking_url} python main.py serve
    3. Full parallel gate sweep (all models + compositions on L4 tasks):
         make -C infra/gcp sweep TIER=M
  EOT
}
