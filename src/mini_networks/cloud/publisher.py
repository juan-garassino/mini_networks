"""Train-request publisher.

The Lab view / CLI publishes a JobSpec to a Pub/Sub topic; an out-of-process
Cloud Function launches the ephemeral Cloud Run Job (see infra/gcp/). The
google-cloud-pubsub import is lazy so the base package imports without the
``cloud`` extra installed. When unconfigured, ``NullPublisher`` fails loudly
rather than silently dropping a request.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class JobSpec(BaseModel):
    model: str
    tier: str = "M"
    hparams: dict[str, Any] = Field(default_factory=dict)
    run_name: str


@runtime_checkable
class Publisher(Protocol):
    def publish(self, spec: JobSpec) -> str: ...


class NullPublisher:
    def publish(self, spec: JobSpec) -> str:
        raise RuntimeError(
            "Cloud train backend not configured: set MN_PUBSUB_TOPIC and GOOGLE_CLOUD_PROJECT"
        )


class PubSubPublisher:
    def __init__(self, topic: str | None = None, project: str | None = None):
        self.topic = topic or os.environ.get("MN_PUBSUB_TOPIC")
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")

    def publish(self, spec: JobSpec) -> str:
        from google.cloud import pubsub_v1  # lazy; needs the `cloud` extra

        client = pubsub_v1.PublisherClient()
        topic_path = self.topic if "/" in (self.topic or "") else client.topic_path(self.project, self.topic)
        future = client.publish(topic_path, spec.model_dump_json().encode("utf-8"))
        msg_id = future.result(timeout=30)
        log.info("Published train request %s for %s (msg %s)", spec.run_name, spec.model, msg_id)
        return msg_id


def get_publisher() -> Publisher:
    if os.environ.get("MN_PUBSUB_TOPIC"):
        return PubSubPublisher()
    return NullPublisher()
