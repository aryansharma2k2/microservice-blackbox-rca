.PHONY: setup deploy inject-fault run-rca evaluate eval replay vllm-up vllm-down check-metrics test clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt

deploy:
	bash infra/deploy-boutique.sh
	bash infra/deploy-monitoring.sh

# Prefer the project venv when it exists, else whatever python3 is on PATH.
PYTHON := $(shell test -x .venv/bin/python3 && echo $(abspath .venv/bin/python3) || echo python3)

inject-fault:
	$(PYTHON) fault_injection/inject.py $(ARGS)

run-rca:
	$(PYTHON) -m rca_engine $(ARGS)

evaluate:
	$(PYTHON) eval/run_experiment.py $(ARGS)

# --- vLLM domain -----------------------------------------------------------

# Score the pipeline and every baseline over the committed traces. Needs no
# server and no GPU: this is what makes the published numbers checkable.
eval:
	$(PYTHON) -m eval.run_eval traces/vllm --detail $(ARGS)

# Re-diagnose captured runs from disk.
replay:
	$(PYTHON) -m eval.replay traces/vllm --all $(ARGS)

vllm-up:
	cd deploy/vllm && docker compose --profile cpu up -d

vllm-down:
	cd deploy/vllm && docker compose --profile cpu down

# Fail loudly if the domain references metrics the server does not expose.
check-metrics:
	$(PYTHON) -m rca_engine.scripts.discover_metrics check vllm \
		--url http://localhost:8000/metrics --url http://localhost:8080/metrics

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf experiments/run_*
