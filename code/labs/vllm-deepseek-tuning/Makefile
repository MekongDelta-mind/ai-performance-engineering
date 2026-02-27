PYTHON ?= python3
ROOT := $(abspath .)
CONFIG ?= configs/benchmark_matrix.yaml
RESULTS ?= $(ROOT)/results
PLOTS ?= $(ROOT)/plots
REPORTS ?= $(ROOT)/reports

.PHONY: smoke smoke-tiny smoke-docker full plot report artifacts teardown purge tidy

smoke:
	$(PYTHON) scripts/run_matrix.py --config $(CONFIG) --profile smoke --run-filter ds_v32_nvfp4_tp2

smoke-tiny:
	$(PYTHON) scripts/run_matrix.py --config configs/smoke_tiny.yaml --profile smoke

smoke-docker:
	$(PYTHON) scripts/run_matrix.py --config configs/smoke_tiny.yaml --profile smoke --vllm-cmd ./scripts/vllm_docker.sh

full:
	$(PYTHON) scripts/run_matrix.py --config $(CONFIG) --profile full

plot:
	$(PYTHON) scripts/plot_results.py --results-dir $(RESULTS) --out-dir $(PLOTS)

report:
	$(PYTHON) scripts/report_results.py --results-dir $(RESULTS) --out-dir $(REPORTS)

artifacts: plot report

teardown:
	bash scripts/teardown.sh

purge:
	bash scripts/teardown.sh --purge-results

# Remove transient local files but keep benchmark artifacts
tidy:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
