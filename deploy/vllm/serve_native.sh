#!/usr/bin/env bash
# Bring up vLLM + Prometheus as plain processes — no Docker.
#
# Needed because container-based GPU hosts (RunPod Pods, many Kubernetes-based
# notebook services) are themselves containers and cannot run a nested Docker
# daemon: dockerd needs host-level iptables and network-namespace privileges
# that a Pod does not have. Nothing about the experiment requires containers,
# so this path runs the same stack directly.
#
#   bash deploy/vllm/serve_native.sh up      [MODEL]
#   bash deploy/vllm/serve_native.sh status
#   bash deploy/vllm/serve_native.sh down
#
# Writes deploy/vllm/native_state.json so the experiment runner can restart
# the server with degraded flags for config-kind scenarios.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/.run}"
STATE="$HERE/native_state.json"
PROM_VERSION="${PROM_VERSION:-3.1.0}"

MODEL="${2:-${MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
PORT="${PORT:-8000}"
PROM_PORT="${PROM_PORT:-9090}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

mkdir -p "$RUN_DIR"

_arch() {
  case "$(uname -m)" in
    x86_64) echo amd64 ;;
    aarch64|arm64) echo arm64 ;;
    *) echo "unsupported arch $(uname -m)" >&2; exit 1 ;;
  esac
}

install_prometheus() {
  if [ -x "$RUN_DIR/prometheus/prometheus" ]; then return; fi
  local arch tarball
  arch="$(_arch)"
  tarball="prometheus-${PROM_VERSION}.linux-${arch}"
  echo "[native] downloading Prometheus ${PROM_VERSION} (${arch})"
  curl -fsSL -o "$RUN_DIR/prom.tgz" \
    "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/${tarball}.tar.gz"
  tar xzf "$RUN_DIR/prom.tgz" -C "$RUN_DIR"
  mv "$RUN_DIR/${tarball}" "$RUN_DIR/prometheus"
  rm -f "$RUN_DIR/prom.tgz"
}

write_prom_config() {
  # Only vLLM. There is no cAdvisor here, so the host_saturation component has
  # no data — it is marked optional in the domain precisely for this case.
  cat > "$RUN_DIR/prometheus.yml" <<EOF
global:
  scrape_interval: 1s
  evaluation_interval: 1s
  scrape_timeout: 900ms

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["localhost:${PORT}"]
    metrics_path: /metrics
EOF
}

start_vllm() {
  local extra="${VLLM_EXTRA_ARGS:-}"
  echo "[native] starting vLLM: $MODEL ${extra}"
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    $extra \
    > "$RUN_DIR/vllm.log" 2>&1 &
  echo $! > "$RUN_DIR/vllm.pid"

  cat > "$STATE" <<EOF
{
  "backend": "native",
  "model": "$MODEL",
  "port": $PORT,
  "run_dir": "$RUN_DIR",
  "base_args": ["--host", "0.0.0.0", "--port", "$PORT",
                "--gpu-memory-utilization", "$GPU_MEM_UTIL",
                "--max-model-len", "$MAX_MODEL_LEN"],
  "extra_args": "$extra"
}
EOF
}

report_failure() {
  # vLLM wraps every startup failure in "Engine core initialization failed.
  # See root cause above." Printing the tail of the log therefore shows the
  # wrapper and hides the cause, which is what made the first several failures
  # here take a round trip each to diagnose. Dig out the real line instead.
  echo >&2
  echo "[native] vLLM did not come up. ROOT CAUSE:" >&2
  echo "------------------------------------------------------------" >&2
  grep -E "(Error|Exception|error:|assert)" "$RUN_DIR/vllm.log" 2>/dev/null \
    | grep -viE "Engine core initialization failed|^\s*File |\^\^\^" \
    | tail -6 >&2 || true
  echo "------------------------------------------------------------" >&2
  echo "[native] full log : $RUN_DIR/vllm.log" >&2
  echo "[native] diagnose : bash deploy/vllm/doctor.sh" >&2
  return 1
}

wait_healthy() {
  local waited=0 last_line=""
  echo "[native] waiting for vLLM (up to 20 min; first start downloads weights)"
  for _ in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
      echo "[native] healthy after ${waited}s"; return 0
    fi
    # If the process is gone, stop waiting — it crashed, and every further
    # second of dots is a second of a paid GPU doing nothing.
    if ! pgrep -f "vllm serve" >/dev/null 2>&1; then
      echo "[native] the vLLM process exited after ${waited}s." >&2
      report_failure
      return 1
    fi
    # Show real progress instead of dots, so a slow download is visibly
    # distinguishable from a hang.
    local line
    line="$(tail -1 "$RUN_DIR/vllm.log" 2>/dev/null | cut -c1-100)"
    if [ -n "$line" ] && [ "$line" != "$last_line" ]; then
      printf '  [%3ds] %s\n' "$waited" "$line"
      last_line="$line"
    fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "[native] timed out after ${waited}s." >&2
  report_failure
}

case "${1:-up}" in
  up)
    install_prometheus
    write_prom_config
    if ! curl -sf "http://localhost:${PROM_PORT}/-/healthy" >/dev/null 2>&1; then
      echo "[native] starting Prometheus on :${PROM_PORT}"
      nohup "$RUN_DIR/prometheus/prometheus" \
        --config.file="$RUN_DIR/prometheus.yml" \
        --storage.tsdb.path="$RUN_DIR/prom-data" \
        --storage.tsdb.retention.time=6h \
        --web.listen-address="0.0.0.0:${PROM_PORT}" \
        > "$RUN_DIR/prometheus.log" 2>&1 &
      echo $! > "$RUN_DIR/prom.pid"
    fi
    start_vllm
    wait_healthy
    echo "[native] up.  vLLM :${PORT}   Prometheus :${PROM_PORT}"
    ;;

  down)
    for p in vllm prom; do
      if [ -f "$RUN_DIR/$p.pid" ]; then
        kill "$(cat "$RUN_DIR/$p.pid")" 2>/dev/null || true
        rm -f "$RUN_DIR/$p.pid"
      fi
    done
    pkill -f "vllm serve" 2>/dev/null || true
    rm -f "$STATE"
    echo "[native] down."
    ;;

  status)
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 \
      && echo "vLLM      : healthy" || echo "vLLM      : down"
    curl -sf "http://localhost:${PROM_PORT}/-/healthy" >/dev/null 2>&1 \
      && echo "Prometheus: healthy" || echo "Prometheus: down"
    ;;

  *)
    echo "usage: $0 {up|down|status} [MODEL]" >&2
    exit 1
    ;;
esac
