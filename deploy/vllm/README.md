# vLLM experiment stack

Brings up a vLLM server and a Prometheus scraping it at 1s, for the TTFT
root-cause experiments.

## Status

**Unvalidated against a running server.** The mechanism graph in
[`rca_engine/domains/vllm.py`](../../rca_engine/domains/vllm.py) is a
hypothesis until `discover_metrics` confirms it against a real `/metrics`
endpoint. Every metric name it references does resolve against the recorded V1
surface in [`tests/fixtures/vllm_metrics_v1.txt`](../../tests/fixtures/vllm_metrics_v1.txt),
but that fixture was written by hand from vLLM's documentation, not captured.

Confirming it is the first thing to do once a server is up — see step 3.

## 1. Start the stack

Local development, tiny model, CPU:

```bash
cd deploy/vllm
docker compose --profile cpu up
```

Trace capture, realistic dynamics, one GPU:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct docker compose --profile gpu up
```

The CPU backend runs the same scheduler, so queueing, preemption, and KV cache
accounting are all real. Only the absolute latencies are distorted, because
the cache lives in host RAM. Use it to check that every metric resolves and
every fault scenario moves the component it claims to — not to draw
conclusions.

On Apple Silicon the default CPU image tag is `-arm64`. On x86 set
`VLLM_CPU_IMAGE=vllm/vllm-openai-cpu:latest-x86_64`, which additionally
assumes `avx512f`.

## 2. Confirm it is serving

```bash
curl -s localhost:8000/health && echo ok
curl -s localhost:9090/-/healthy && echo ok
curl -s localhost:8000/metrics | grep -c '^vllm:'
```

## 3. Record and verify the metric surface

This is the step that turns the domain spec from a guess into something
checked. vLLM renamed metrics between its V0 and V1 engines
(`gpu_cache_usage_perc` → `kv_cache_usage_perc`, `time_in_queue_requests` →
`request_queue_time_seconds`), and published sources disagree about `_total`
suffixes on the prefix-cache counters.

```bash
# Record what this server actually exposes
python -m rca_engine.scripts.discover_metrics snapshot \
    --url http://localhost:8000/metrics \
    --out deploy/vllm/metric_surface.json \
    --source "vllm 0.11 cpu, Qwen2.5-0.5B"

# Fail loudly if the domain references anything absent
python -m rca_engine.scripts.discover_metrics check vllm \
    --surface deploy/vllm/metric_surface.json
```

A missing metric is the worst available failure mode: the component it backs
never goes abnormal, and the pipeline confidently blames something else. The
check exits non-zero so CI can gate on it.

If a name has drifted, fix `rca_engine/domains/vllm.py` and pin the server
version in `docker-compose.yml` before capturing any traces.

## 4. Diagnose a window

```bash
python -m rca_engine --domain vllm --baseline 300 --fault 120
```

## Ports

| Service | URL |
|---|---|
| vLLM (OpenAI API + `/metrics`) | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| DCGM exporter (gpu profile only) | http://localhost:9400 |

Under the `cpu` profile there is no DCGM exporter, so the `gpu_saturation`
component has no backing data and never goes abnormal. That is expected.
