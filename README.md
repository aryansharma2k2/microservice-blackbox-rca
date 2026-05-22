# microservice-blackbox-rca

Black-box root-cause analysis for microservice failures using metrics, change-point detection, and fault-chain pinpointing.

## Portfolio demo

This repo includes a dependency-free static dashboard in `portfolio/` for free hosting on GitHub Pages or Cloudflare Pages. It replays a real chaos experiment artifact and presents the RCA ranking, telemetry signals, dependency graph, and local reproduction commands.

Local preview:

```bash
python3 -m http.server 4173 --directory portfolio
```

Then open `http://localhost:4173`.

GitHub Pages deployment is configured in `.github/workflows/pages.yml`. In the GitHub repository settings, set **Pages -> Source** to **GitHub Actions**, then push to `main`.

---

## Prerequisites

Install the following tools before anything else.

**Docker Desktop** — https://www.docker.com/products/docker-desktop
Start it and wait for the whale icon in the menu bar to stop animating.

```bash
brew install kubectl kind helm
pip install -r requirements.txt
mkdir -p /tmp/kind-volumes
```

---

## 1. Deploy Online Boutique

Creates the kind cluster (4 nodes), patches all boutique manifests (removes `readOnlyRootFilesystem`, adds `NET_ADMIN`, mounts emptyDir at `/tmp`), and deploys all microservices. Takes 3–5 min on first run due to image pulls.

```bash
bash infra/deploy-boutique.sh
```

Frontend: **http://localhost:8080**

If the port-forward dies:
```bash
kubectl port-forward -n boutique svc/frontend 8080:80 &
```

---

## 2. Deploy Monitoring (Prometheus + Grafana)

```bash
bash infra/deploy-monitoring.sh
```

| Service    | URL                    | Credentials   |
|------------|------------------------|---------------|
| Prometheus | http://localhost:9090  | —             |
| Grafana    | http://localhost:3000  | admin / admin |

If a port-forward dies:
```bash
kubectl port-forward -n monitoring svc/kube-prom-stack-kube-prome-prometheus 9090:9090 &
kubectl port-forward -n monitoring svc/kube-prom-stack-grafana 3000:80 &
```

---

## 3. Deploy Chaos Mesh (fault injection engine)

```bash
bash infra/deploy-chaos-mesh.sh
```

Installs Chaos Mesh 2.7.0 into the `chaos-mesh` namespace via Helm.

---

## 4. Verify Metrics

In Prometheus, run:
```
node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{namespace="boutique"}
```
Should return time-series for each boutique pod.

In Grafana: **Dashboards → Kubernetes / Compute Resources / Pod**, set namespace to `boutique`, time range to last 15 minutes.

---

## 5. Run Load Generator

```bash
python infra/loadgen.py --duration 300 --rps 5 --pattern sine
```

Options: `--pattern` accepts `sine`, `step`, or `constant`. Watch CPU lines move in Grafana.

---

## 6. Fault Injection

Fault injection is handled by two injectors depending on fault type:

| Injector | Faults | Target services |
|---|---|---|
| `chaos_inject.py` (Chaos Mesh) | `cpu_hog`, `mem_leak`, `net_delay`, `packet_loss` | **All 11 services** — works on distroless Go containers |
| `inject.py` (kubectl exec) | `disk_hog` | Shell-only: `adservice`, `currencyservice`, `emailservice`, `paymentservice`, `recommendationservice` |

`disk_hog` uses the exec path because IOChaos requires FUSE kernel support not available on kind clusters.

### Inject a single fault manually

```bash
# Chaos Mesh — works on any service including distroless
python fault_injection/chaos_inject.py --fault cpu_hog   --service frontend       --duration 60
python fault_injection/chaos_inject.py --fault net_delay --service cartservice     --duration 60
python fault_injection/chaos_inject.py --fault mem_leak  --service recommendationservice --duration 60

# kubectl exec fallback — disk_hog only, shell-capable services only
python fault_injection/inject.py --fault disk_hog --service currencyservice --duration 60
```

Concurrent faults (two services at once):
```bash
python fault_injection/chaos_inject.py --fault cpu_hog --service emailservice --concurrent currencyservice --duration 60
```

---

## 7. Run a Single Experiment (inject → collect → RCA → score)

There are two experiment runners with different RCA trigger strategies:

| Script | RCA trigger | Best for |
|---|---|---|
| `eval/run_experiment.py` | Fixed: after full fault duration | Faults that don't spike frontend latency (mem_leak, off-critical-path services) |
| `eval/run_experiment_slo.py` | Reactive: on first SLO violation + 30 s buffer | Faults on the critical latency path (net_delay, cpu_hog on checkout/cart/currency) |

Both scripts share identical arguments, output the same artifacts, and fall back to the duration-elapsed trigger when no SLO violation is detected (so `run_experiment_slo.py` is a strict superset of `run_experiment.py`).

### Fixed-duration trigger (`run_experiment.py`)

Pipeline:
1. Start load generator
2. Wait 60 s baseline
3. Inject fault
4. **Sleep full fault duration** — SLO monitor records violation time as metadata only
5. Collect metrics window anchored to injection time
6. Run RCA
7. Cleanup + recovery

### SLO-violation trigger (`run_experiment_slo.py`)

Pipeline:
1. Start load generator
2. Wait 60 s baseline
3. Inject fault
4. **Block on `violation_event.wait(timeout=duration)`** — unblocks the moment the SLO fires
   - *Violation path*: wait 30 s observation buffer, then proceed to RCA (early)
   - *No-violation path*: duration expires, proceed to RCA (identical to fixed-duration)
5. Collect metrics window anchored to injection time (fault window end = now, not original duration end)
6. Run RCA — `timeline.json` records `triggered_by: slo_violation | duration_elapsed`
7. Cleanup + recovery (inject subprocess runs its full duration uninterrupted)

### Which faults reliably trigger SLO violations

The SLO measures frontend p95 across all loadgen journey steps (homepage → product → cart → checkout). Only faults on the **blocking call path** spike p95:

| Service | Fault | Why it works |
|---|---|---|
| `checkoutservice` | `net_delay`, `cpu_hog` | Checkout blocks frontend directly |
| `currencyservice` | `net_delay` | Called on every page — highest request hit rate |
| `cartservice` | `net_delay` | Blocks add-to-cart and checkout cart lookup |
| `productcatalogservice` | `net_delay` | Blocks homepage and product pages |
| `emailservice` | `net_delay` | checkoutservice makes a synchronous gRPC call to emailservice before returning |
| `paymentservice` | `net_delay` | Blocking call inside checkoutservice |

Faults that **will not** trigger SLO violations: `mem_leak` on any service (Python/Go GC has no latency pauses), anything on `recommendationservice` or `adservice` (frontend doesn't wait for them), `disk_hog` (disk I/O not on hot path).

### cpu_hog experiments

```bash
python eval/run_experiment.py --fault cpu_hog --service frontend           --duration 120
python eval/run_experiment.py --fault cpu_hog --service checkoutservice    --duration 120
python eval/run_experiment.py --fault cpu_hog --service adservice          --duration 120
python eval/run_experiment.py --fault cpu_hog --service recommendationservice --duration 120
python eval/run_experiment.py --fault cpu_hog --service currencyservice    --duration 120
python eval/run_experiment.py --fault cpu_hog --service emailservice       --duration 120
```

### mem_leak experiments

```bash
python eval/run_experiment.py --fault mem_leak --service recommendationservice --duration 120
python eval/run_experiment.py --fault mem_leak --service emailservice           --duration 120
python eval/run_experiment.py --fault mem_leak --service frontend               --duration 120
```

### net_delay experiments

```bash
# Fixed-duration trigger
python eval/run_experiment.py --fault net_delay --service frontend              --duration 120
python eval/run_experiment.py --fault net_delay --service cartservice           --duration 120
python eval/run_experiment.py --fault net_delay --service checkoutservice       --duration 120
python eval/run_experiment.py --fault net_delay --service productcatalogservice --duration 120

# SLO-violation trigger (RCA fires as soon as p95 spikes, not after full 120 s)
python eval/run_experiment_slo.py --fault net_delay --service checkoutservice       --duration 120
python eval/run_experiment_slo.py --fault net_delay --service currencyservice       --duration 120
python eval/run_experiment_slo.py --fault net_delay --service cartservice           --duration 120
python eval/run_experiment_slo.py --fault net_delay --service emailservice          --duration 120
python eval/run_experiment_slo.py --fault cpu_hog   --service checkoutservice       --duration 120
```

### disk_hog experiments (exec fallback, shell-capable services only)

```bash
python eval/run_experiment.py --fault disk_hog --service currencyservice --duration 120
python eval/run_experiment.py --fault disk_hog --service emailservice    --duration 120
```

### Concurrent fault experiments

```bash
python eval/run_experiment.py --fault cpu_hog  --service emailservice         --concurrent currencyservice --duration 120
python eval/run_experiment.py --fault net_delay --service frontend             --concurrent checkoutservice --duration 120
```

Each experiment takes roughly **4 minutes** with the fixed-duration runner (60 s baseline + fault duration + recovery). With the SLO-triggered runner, total time is shorter when a violation fires quickly. Artifacts land in `experiments/<run_id>/`:

```
experiments/20260412_201315/
├── ground_truth.json   # injected fault + target services
├── timeline.json       # timestamps, SLO violation metadata, triggered_by field
├── metrics.parquet     # per-service metric matrix (all 11 services)
└── rca_results.json    # RCA engine output
```

The `timeline.json` from `run_experiment_slo.py` includes an extra field:
```json
"triggered_by": "slo_violation"   // or "duration_elapsed" if no violation fired
```

---

## 8. Propagation Delay Calibration

The RCA engine defaults to a 2-second global threshold when deciding whether two services' fault onsets are "concurrent" or a propagation chain. On a kind cluster with sub-millisecond gRPC, detected onset gaps are typically 0–3 s and the global threshold misclassifies propagation victims.

Calibration measures per-edge delays empirically and stores them so that Layer 7 of FChain uses edge-specific thresholds.

### What it does

For each downstream service (the callee), the script:
1. Starts the load generator at normal RPS
2. Waits a 60 s baseline
3. Injects a `cpu_hog` fault into the callee
4. Waits 120 s for full propagation to all callers
5. Collects metrics and runs `fault_chain.pinpoint()` to measure per-service onset times
6. Computes `delay_s = onset(caller) - onset(callee)` for every direct caller
7. After N trials, stores the per-edge median delay and derives a threshold

**10 unique callee targets × 3 trials ≈ 2 hours total.**

### Quick single-target run (~4 min)

```bash
python calibration/calibrate.py --trials 1 --services checkoutservice
```

### Full calibration (~2 hours)

```bash
python calibration/calibrate.py --trials 3 --output calibration/propagation_delays.json
```

Options:
- `--trials N` — number of injection trials per target service (default 3)
- `--fault FAULT` — fault type for calibration experiments (default `cpu_hog`)
- `--services svc1,svc2` — comma-separated subset of callee targets (default: all 10)
- `--rps N` — load generator RPS during calibration (default 5.0)
- `--output PATH` — where to write the JSON map (default `calibration/propagation_delays.json`)
- `--dry-run` — print the experiment plan without running anything

### Resulting JSON

```json
{
  "version": 1,
  "default_threshold_s": 2.0,
  "edges": {
    "frontend->checkoutservice": {
      "observed_delays_s": [1.0, 2.0, 1.0],
      "median_delay_s": 1.0,
      "threshold_s": 1.5
    },
    "checkoutservice->paymentservice": {
      "observed_delays_s": [0.0, 1.0, 0.0],
      "median_delay_s": 0.0,
      "threshold_s": 1.0
    }
  }
}
```

Threshold formula: `threshold = median_delay + max(1.0, median_delay × 0.5)`. The `+1.0` floor accounts for 1-second Prometheus scrape resolution.

### Using the map in experiments

Pass `--propagation-map` to either experiment runner:

```bash
# SLO-triggered run with edge-aware Layer 7
python eval/run_experiment_slo.py \
    --fault net_delay --service paymentservice --duration 120 \
    --propagation-map calibration/propagation_delays.json

# Fixed-duration run with edge-aware Layer 7
python eval/run_experiment.py \
    --fault cpu_hog --service checkoutservice --duration 120 \
    --propagation-map calibration/propagation_delays.json
```

When the map is loaded, `rca_results.json` will correctly classify propagation victims (e.g. frontend/checkoutservice affected by a paymentservice fault) instead of pinpointing them as concurrent independent faults.

Without the flag the engine falls back to the original 2 s global threshold — fully backward compatible.

---

## 9. Run the Full Experiment Batch

```bash
python eval/run_batch.py --matrix experiments/experiment_matrix.yaml
```

Options:
- `--cooldown 60` — override the cooldown between experiments
- `--dry-run` — print all commands without running any experiment

The batch runner exits non-zero if any experiment fails and prints a summary at the end.

---

## 10. Fetch Metrics Directly

```bash
python -m rca_engine.metrics_client
```

Prints a per-service, per-metric summary table for the last 5 minutes.

---

## Teardown

```bash
kind delete cluster --name fchain-rca
```
