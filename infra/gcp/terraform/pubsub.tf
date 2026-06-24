# Durable train-request queue. Free-tier volume → ~€0. Nothing runs at rest.
resource "google_pubsub_topic" "train_requests" {
  name = "mini-networks-train-requests"
}
