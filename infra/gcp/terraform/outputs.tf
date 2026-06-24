output "train_job" {
  value = google_cloud_run_v2_job.train.name
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
    1. Set the Neon DSN (out-of-band — never in git/state):
         printf '%s' "$NEON_DSN" | gcloud secrets versions add ${var.neon_secret_id} --data-file=-
    2. Publish a test run:
         gcloud pubsub topics publish ${google_pubsub_topic.train_requests.name} \
           --message '{"model":"classifier","training_tier":"S","run_name":"smoke-1"}'
    3. Point the playground at MLflow and watch it:
         MN_RUN_SOURCE=mlflow MN_MLFLOW_TRACKING_URI="$NEON_DSN" python main.py serve
  EOT
}
