.PHONY: help validate fmt plan show destroy build-train dry-run

TF := terraform -chdir=terraform

help:
	@echo "validate     - terraform fmt -check + validate (static, no cloud)"
	@echo "fmt          - terraform fmt -recursive"
	@echo "plan         - terraform plan"
	@echo "show         - terraform apply (provision the stack)"
	@echo "destroy      - terraform destroy (tear down; images persist)"
	@echo "build-train  - docker build the training image locally"
	@echo "dry-run      - run the train image against a local sqlite MLflow (no cloud)"

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

dry-run: build-train
	mkdir -p /tmp/mlruns
	docker run --rm \
	  -e MODE=train -e MODEL=classifier -e TRAINING_TIER=S -e RUN_NAME=local-smoke -e DEVICE=cpu \
	  -e MN_MLFLOW_TRACKING_URI=sqlite:////tmp/mlruns/mlflow.db \
	  -e MN_MLFLOW_ARTIFACT_ROOT=/tmp/mlruns/artifacts \
	  -e HPARAMS='{"batch_size":16}' \
	  -v /tmp/mlruns:/tmp/mlruns \
	  mini-networks-train:dev
