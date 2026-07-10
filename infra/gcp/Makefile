.PHONY: help validate fmt plan show destroy build-train build-train-gpu push-images dry-run dry-run-sweep sweep sweep-report

TF := terraform -chdir=terraform

PROJECT  ?= garassino-ml
REGION   ?= europe-west1
REGISTRY ?= $(REGION)-docker.pkg.dev/$(PROJECT)/ml-images
TAG      ?= latest

# Parallel sweep knobs. ITEMS defaults to the full catalog (models +
# compositions, catalog order — the same list the container resolves, so the
# task index mapping always agrees). Pass ITEMS=a,b,c for targeted reruns.
TIER     ?= M
ITEMS    ?= $(shell cd ../.. && uv run python -c "from mini_networks.cloud.sweep_shard import default_items; print(','.join(default_items()))")
N_ITEMS   = $(shell echo "$(ITEMS)" | tr ',' '\n' | grep -c .)
SWEEP_ID ?= sweep-$(shell date -u +%Y%m%d-%H%M%S)

help:
	@echo "validate       - terraform fmt -check + validate (static, no cloud)"
	@echo "fmt            - terraform fmt -recursive"
	@echo "plan           - terraform plan"
	@echo "show           - terraform apply (provision the stack)"
	@echo "destroy        - terraform destroy (tear down; images persist)"
	@echo "build-train    - docker build the CPU training image"
	@echo "build-train-gpu- docker build the cu121 image for the L4 sweep job"
	@echo "push-images    - push both images to $(REGISTRY)"
	@echo "dry-run        - run the train image against a local sqlite MLflow (no cloud)"
	@echo "dry-run-sweep  - run ONE sweep-task shard locally (no cloud)"
	@echo "sweep          - execute the parallel gate sweep [TIER=M] [ITEMS=a,b,c]"
	@echo "sweep-report   - merge a sweep's shards: make sweep-report SWEEP=<id>"

validate:
	terraform -chdir=terraform fmt -check -recursive
	$(TF) init -backend=false -input=false >/dev/null
	$(TF) validate

fmt:
	terraform -chdir=terraform fmt -recursive

plan:
	$(TF) init -input=false
	$(TF) plan

show:
	$(TF) init -input=false
	$(TF) apply

destroy:
	$(TF) destroy

build-train:
	cd ../.. && docker build -f infra/gcp/Dockerfile.train -t mini-networks-train:dev .

build-train-gpu:
	cd ../.. && docker build -f infra/gcp/Dockerfile.train \
	  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
	  -t mini-networks-train-gpu:dev .

push-images: build-train build-train-gpu
	docker tag mini-networks-train:dev $(REGISTRY)/mini-networks-train:$(TAG)
	docker tag mini-networks-train-gpu:dev $(REGISTRY)/mini-networks-train-gpu:$(TAG)
	docker push $(REGISTRY)/mini-networks-train:$(TAG)
	docker push $(REGISTRY)/mini-networks-train-gpu:$(TAG)

dry-run: build-train
	mkdir -p /tmp/mlruns
	docker run --rm \
	  -e MODE=train -e MODEL=classifier -e TRAINING_TIER=S -e RUN_NAME=local-smoke -e DEVICE=cpu \
	  -e MN_MLFLOW_TRACKING_URI=sqlite:////tmp/mlruns/mlflow.db \
	  -e MN_MLFLOW_ARTIFACT_ROOT=/tmp/mlruns/artifacts \
	  -e HPARAMS='{"batch_size":16}' \
	  -v /tmp/mlruns:/tmp/mlruns \
	  mini-networks-train:dev

dry-run-sweep: build-train
	mkdir -p /tmp/mlruns
	docker run --rm \
	  -e MODE=sweep-task -e CLOUD_RUN_TASK_INDEX=0 -e ITEMS=classifier \
	  -e SWEEP_ID=local-smoke -e TRAINING_TIER=S -e DEVICE=cpu \
	  -e MN_MLFLOW_TRACKING_URI=sqlite:////tmp/mlruns/mlflow.db \
	  -e MN_MLFLOW_ARTIFACT_ROOT=/tmp/mlruns/artifacts \
	  -v /tmp/mlruns:/tmp/mlruns \
	  mini-networks-train:dev

# One execution = one sweep. --tasks matches ITEMS; env overrides use the ^|^
# delimiter because ITEMS itself is comma-separated. Async: a full M sweep is
# ~1-2h wall at parallelism 4 — watch with `gcloud run jobs executions list`.
sweep:
	gcloud run jobs execute mini-networks-sweep \
	  --project $(PROJECT) --region $(REGION) \
	  --tasks $(N_ITEMS) \
	  --update-env-vars "^|^ITEMS=$(ITEMS)|SWEEP_ID=$(SWEEP_ID)|TRAINING_TIER=$(TIER)" \
	  --async
	@echo ""
	@echo "sweep '$(SWEEP_ID)' dispatched: $(N_ITEMS) tasks, tier $(TIER)"
	@echo "watch:  gcloud run jobs executions list --job mini-networks-sweep --region $(REGION) --project $(PROJECT)"
	@echo "report: make sweep-report SWEEP=$(SWEEP_ID)"

sweep-report:
	@if [ -z "$(SWEEP)" ]; then echo "Usage: make sweep-report SWEEP=<sweep-id> [ITEMS=a,b,c]"; exit 1; fi
	cd ../.. && uv run python main.py sweep-report --sweep_id $(SWEEP) --items "$(ITEMS)"
