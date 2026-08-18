# TTFT-RCA — root-cause localization inside an LLM inference server

When p99 time-to-first-token spikes on a vLLM server, which subsystem caused it?
This is a change-point pipeline that answers that from Prometheus telemetry alone —
no code changes, no profiler, no request traces — and a labelled fault corpus that
measures how often the answer is right.

**[Live dashboard](https://aryansharma2k2.github.io/microservice-blackbox-rca/)** ·
[the engine](rca_engine/fault_chain.py) ·
[the mechanism graph](rca_engine/domains/vllm.py) ·
[the evaluation](eval/run_eval.py)

```bash
git clone https://github.com/aryansharma2k2/microservice-blackbox-rca
cd microservice-blackbox-rca && pip install -e ".[dev]"
make eval          # re-derives every number below. no GPU, no cluster, no server.
```

---

## The problem

TTFT is the SLI every LLM-serving team alerts on, and nobody can attribute it. When
it spikes the on-call engineer stares at a Grafana board and guesses among KV cache
pressure, batch scheduling stalls, prefix-cache collapse, and preemption.

Those mechanisms are **observationally confusable**: they share symptoms and cause
one another. Preemption is simultaneously a cause of TTFT and an effect of cache
pressure, so ranking metrics by correlation with the SLI implicates both and
separates neither. A threshold on `kv_cache_usage_perc` fires during a plain load
ramp, a prefix-cache collapse, and a genuine misconfiguration alike.

## The approach

An inference server is *one* service, so there is nothing to localize *across*. The
reframe is to localize **inside** it: the vLLM scheduler has real internal causal
structure that forms a DAG, and change-point ordering plus graph filtering apply to
it unchanged.

Twelve mechanism nodes, each backed by one or more Prometheus series, with edges
encoding "A can cause B":

```
arrival_load ──┬─→ kv_cache_pressure ──→ preemption ──┬─→ queueing ──→ ttft
(exogenous)    ├─→ batch_composition ──→ prefill_cost ─┘        ↑
               └─→ gpu_saturation                    host_saturation
request_shape ─┬─→ prefill_cost
(exogenous)    └─→ kv_cache_pressure
prefix_cache_efficacy ─→ prefill_cost, kv_cache_pressure
```

The statistical core needed **zero modification** to retarget — only the metric
registry and the graph are domain-specific, expressed as a
[`DomainSpec`](rca_engine/domains/base.py). Two ideas the generic pipeline could
not have:

- **TTFT decomposes natively.** vLLM exposes `request_queue_time_seconds` and
  `request_prefill_time_seconds` as separate histograms, so the queue/prefill split
  partitions the hypothesis space before any inference runs: queue-dominated points
  at admission-side causes, prefill-dominated at compute-side.
- **Exogenous nodes and a capacity verdict.** Load causes everything, so a naive
  onset ranking blames `arrival_load` for every incident. Nodes marked exogenous can
  drive a **capacity** verdict — "you are simply overloaded, nothing is broken" —
  but are never named as a pathology. That distinction is the difference between
  "add a replica" and "fix your config."

## Results

38 labelled runs on one L4 GPU: 5 confusable mechanisms across 7 fault scenarios,
plus 8 clean runs so the false-positive rate is measurable. Every method sees the
same evidence and is scored identically.

| method | top-1 | top-3 | MRR | FPR | what it is |
|---|---|---|---|---|---|
| **pipeline** | 39% | **74%** | **0.564** | **0%** | eight-layer change-point localization |
| threshold | **42%** | 42% | 0.428 | 0% | static runbook rules, in order |
| correlation | 39% | 61% | 0.502 | 0% | rank by \|r\| against the SLI |
| llm | 34% | 66% | 0.496 | 0% | Claude Opus 5, same evidence |
| topology | 13% | 58% | 0.362 | 100% | always blame the graph root |

**Threshold alerting wins top-1.** That is the honest shape of the result, and the
reason the comparison is in the repo rather than a single number. It is also useless
past rank 1 — its top-3 never improves, because after the first rule fires there is
nothing behind it. Against threshold the pipeline is +32 points of top-3 and +0.14
MRR.

Against whichever baseline is strongest on each metric the win is narrower and is
the one I would defend: **+7.9 points of top-3** over the language model, **+0.06
MRR** over correlation, at 0% false positives. The pipeline leads on ranked
candidate quality, consistently, by a real but modest margin.

Claude Opus 5, handed the same per-component before/after evidence, is genuinely
competitive at 34% / 0.496 — but it never sees onset timing, which is the pipeline's
entire edge.

## What does not work, and why

The dominant error mode is that **measurement modality outranks causality**. The
pipeline ranks by onset order, and onset is measured from telemetry rather than from
the event. Gauges (`kv_cache_usage_perc`, `num_requests_running`) report their new
value on the next scrape; windowed rates and histogram quantiles only reach theirs
after the window fills, roughly half a window late. So a gauge-backed *effect*
systematically appears to precede its own rate-backed *cause* — which is why
`kv_cache_pressure` and `batch_composition`, the two gauge-read mechanisms, absorb
most of the off-diagonal mass in the confusion matrix.

Shifting each series back by half its window recovers +2.7 points of top-1 when
A/B'd on this corpus, and is enabled. It is a correction, not a cure: fixing this
properly means changing what is measured, not how it is ranked.

Two further limitations, both visible in the artifacts:

- **9 of 38 runs never had a clean baseline.** The capture protocol runs baseline,
  inject, fault, repeat; when the previous run's load had not drained, the baseline
  window opens with p99 TTFT already 300–600× its own median, and Layer 1 calibrates
  an enormous sigma from it. Top-1 is 11% on those runs versus 48% on the 29 that
  started quiet. The headline number counts all 38 anyway — dropping a quarter of a
  corpus raises the score, and that belongs in the open rather than applied quietly.
- **The SLI is coarsely bucketed.** `time_to_first_token_seconds` is a histogram, and
  p99 spends most of these traces pinned to two bucket edges. Small genuine TTFT
  movements are invisible.

## Reproducibility

Traces are committed as Parquet, so scoring replays from disk with no server and no
hardware. `make eval` regenerates the table above on a clean clone.

The pipeline is deterministic: Layer 1's block bootstrap is seeded per
`(component, metric)`. Before that was fixed, repeated evaluations of the same corpus
produced different top-3 numbers — the threshold is a percentile of 1000 random
resamples, and change points near it flipped between runs.

CI asserts diagnostic accuracy on every push — an accuracy floor and a false-positive
ceiling, both derived from measurement rather than aspiration — alongside a check
that the domain's metric names still resolve against a captured vLLM
`/metrics` surface.

```bash
make eval                                        # score every method over traces/vllm
python -m eval.replay traces/vllm/<run_id>       # re-diagnose one run
python -m rca_engine.scripts.gen_portfolio_data  # regenerate the dashboard's data
make test
```

## Running it live

```bash
bash deploy/vllm/serve_native.sh up   # vLLM + Prometheus, no Docker-in-Docker
make check-metrics                    # domain's metrics vs the live /metrics surface

python -m eval.run_vllm_experiment --scenario prefix_diversity
python -m eval.run_vllm_batch                     # the whole matrix, resumable
```

`deploy/vllm/doctor.sh` diagnoses a stack that will not come up. Faults are mostly
**workload shaping** rather than infrastructure chaos — a prefix-diversity attack, a
long-output burst, a bimodal request mix — which is cheaper and far more reproducible
than killing containers. Config-side faults restart the server degraded
(`--kv-cache-memory-bytes`, `--max-num-seqs`), and each restart is a labelled fault.

## Second substrate

The engine began as microservice RCA over Google Online Boutique — cAdvisor
telemetry, Chaos Mesh fault injection, eleven services on Kubernetes. Retargeting it
at vLLM meant writing a domain adapter, not a new pipeline; layers 1–8 are untouched,
and the Boutique test suite was the regression gate for that refactor.

It is kept as evidence that the core generalizes across two unrelated causal
substrates — **not** as a second result. There is no labelled fault corpus for that
domain and therefore no accuracy number.

## The pipeline

| Layer | Purpose |
|---|---|
| 1. CUSUM + block bootstrap | Candidate change points, with the threshold calibrated empirically from resamples of the baseline rather than assumed. |
| 2. Markov normal model | Drop changes a learned-normal model still predicts. Detectable ≠ abnormal. |
| 3. FFT burst filter | Remove periodic workload bursts expected at that frequency. |
| 4. Tangent rollback | A detector fires *after* the trend starts; roll back along the tangent to estimate the true onset. |
| 5. Multi-metric aggregation | Collapse per-metric onsets into one onset per mechanism; apply lag compensation. |
| 6. Exogenous check | Capacity versus pathology, from whether the earliest onset is an uncontrolled input. |
| 7. Propagation chain | Order candidates by onset, treating near-simultaneous fires as concurrent. |
| 8. Dependency filter | Demote mechanisms the graph explains as propagation from something earlier. |

An **effect-size gate** sits between layers 1 and 2. Layers 1–4 test whether a change
is *detectable*, never whether it is *large*, so on a quiet server a metric wobbling
by a hundredth of a percent yields a confident change point. The minimum effect size
is refit against this corpus rather than guessed.

## Repository map

```text
rca_engine/          Engine, domain specs (boutique, vllm), metrics client
  domains/           DomainSpec — the only domain-specific code
  scripts/           Metric discovery, matrix generation, dashboard data
workload/            Fault scenarios: workload shaping and config restarts
fault_injection/     Chaos Mesh and kubectl-exec injectors (Boutique)
eval/                Scoring, baselines, replay, batch capture
deploy/vllm/         Native vLLM + Prometheus stack, captured metric surface
traces/vllm/         38 committed runs: metrics.parquet + ground_truth.json
portfolio/           The dashboard, generated from traces/
infra/               Kubernetes, monitoring, Chaos Mesh (Boutique)
```

## Notes

This began as a group research project implementing FChain-style RCA on Online
Boutique. The vLLM domain, the mechanism graph, the fault library, the evaluation
harness, the baselines, and everything reported above are my work on top of that
codebase.
