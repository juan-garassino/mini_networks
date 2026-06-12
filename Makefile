.PHONY: help install install-dev test test-ci clean list validate-s validate-m validate-l validate-models-s validate-compositions-s

UV_RUN = uv run
PYTHON ?= $(UV_RUN) python
CHECKPOINT_ROOT ?= runs
DATA_ROOT ?= data
DEVICE ?= cpu
BATCH_SIZE ?= 32
EPOCHS ?= 2

help:
	@echo "Available targets:"
	@echo "  install               Install runtime dependencies with uv"
	@echo "  install-dev           Install runtime + dev dependencies with uv"
	@echo "  test                  Run fast test suite (slow tests deselected)"
	@echo "  test-ci               Run tests for CI"
	@echo "  clean                 Remove caches and stray run output"
	@echo "  list                  List available models and compositions"
	@echo "  validate-s            Run full sweep at tier S"
	@echo "  validate-m            Run full sweep at tier M"
	@echo "  validate-l            Run full sweep at tier L"
	@echo "  validate-models-s     Run models-only sweep at tier S"
	@echo "  validate-compositions-s Run compositions-only sweep at tier S"
	@echo ""
	@echo "Variables:"
	@echo "  CHECKPOINT_ROOT=$(CHECKPOINT_ROOT)"
	@echo "  DATA_ROOT=$(DATA_ROOT)"
	@echo "  DEVICE=$(DEVICE)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  EPOCHS=$(EPOCHS)"

install:
	uv sync

install-dev:
	uv sync --dev

test:
	$(UV_RUN) pytest -q

test-ci:
	$(UV_RUN) pytest -q

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name "__pycache__" -not -path "./legacy/*" -not -path "./.venv/*" -type d -prune -exec rm -rf {} +

list:
	$(PYTHON) main.py list

validate-s:
	$(PYTHON) main.py sweep --check --training_tier S --batch_size $(BATCH_SIZE) --epochs $(EPOCHS) --device $(DEVICE) --data_root $(DATA_ROOT) --checkpoint_root $(CHECKPOINT_ROOT)

validate-m:
	$(PYTHON) main.py sweep --check --training_tier M --batch_size $(BATCH_SIZE) --epochs $(EPOCHS) --device $(DEVICE) --data_root $(DATA_ROOT) --checkpoint_root $(CHECKPOINT_ROOT)

validate-l:
	$(PYTHON) main.py sweep --check --training_tier L --batch_size $(BATCH_SIZE) --epochs $(EPOCHS) --device $(DEVICE) --data_root $(DATA_ROOT) --checkpoint_root $(CHECKPOINT_ROOT)

validate-models-s:
	$(PYTHON) main.py sweep --check --training_tier S --skip-compositions --batch_size $(BATCH_SIZE) --epochs $(EPOCHS) --device $(DEVICE) --data_root $(DATA_ROOT) --checkpoint_root $(CHECKPOINT_ROOT)

validate-compositions-s:
	$(PYTHON) main.py sweep --check --training_tier S --skip-models --batch_size $(BATCH_SIZE) --epochs $(EPOCHS) --device $(DEVICE) --data_root $(DATA_ROOT) --checkpoint_root $(CHECKPOINT_ROOT)
