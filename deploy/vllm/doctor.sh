#!/usr/bin/env bash
# One command that answers "why won't vLLM start here?"
#
#   bash deploy/vllm/doctor.sh            # environment report
#   bash deploy/vllm/doctor.sh --serve    # also try to start vLLM and capture
#                                         # the real error, not the wrapper
#
# Paste the whole output. It is designed to contain everything needed to
# diagnose a failure without another round trip: driver/CUDA pairing, shared
# memory, disk, ports, and — crucially — the ROOT CAUSE line from vLLM rather
# than the "Engine core initialization failed" wrapper that hides it.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/.run}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"

hr() { printf '%s\n' "------------------------------------------------------------"; }
sec() { echo; hr; echo "## $1"; hr; }

sec "host"
echo "uname   : $(uname -a)"
echo "distro  : $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || echo unknown)"
echo "cpus    : $(nproc 2>/dev/null || echo '?')"
echo "memory  : $(free -h 2>/dev/null | awk '/^Mem:/{print $2" total, "$7" available"}' || echo '?')"
echo "in container: $([ -f /.dockerenv ] && echo yes || echo 'probably not')"

sec "gpu driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
  echo
  echo "Driver -> max CUDA it can run:"
  echo "  535.x -> 12.2   |  550.x -> 12.4   |  570.x -> 12.8   |  580.x+ -> 13.0"
else
  echo "nvidia-smi NOT FOUND — no GPU visible to this shell"
fi

sec "python / torch / vllm"
echo "python  : $(python3 --version 2>&1)  ($(command -v python3))"
python3 - <<'PY' 2>&1
try:
    import torch
    print(f"torch   : {torch.__version__}  built for CUDA {torch.version.cuda}")
    print(f"cuda ok : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device  : {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"torch   : NOT IMPORTABLE — {type(e).__name__}: {e}")
try:
    import vllm
    print(f"vllm    : {vllm.__version__}")
except Exception as e:
    print(f"vllm    : NOT IMPORTABLE — {type(e).__name__}: {e}")
for mod in ("msgspec", "numpy", "pandas", "pyarrow", "click"):
    try:
        m = __import__(mod)
        print(f"{mod:<8}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod:<8}: MISSING ({type(e).__name__})")
PY

sec "the usual suspects"
shm=$(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
echo "/dev/shm size : ${shm:-unknown}   (vLLM needs >=256M; 64M WILL fail)"
echo "disk free     : $(df -h "$ROOT" 2>/dev/null | awk 'NR==2{print $4}')"
echo "hf cache      : $(du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | cut -f1 || echo 'none')"
echo -n "port ${PORT}      : "
(curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && echo "IN USE (something already serving)") || echo "free"
echo -n "port 9090     : "
(curl -sf "http://localhost:9090/-/healthy" >/dev/null 2>&1 && echo "IN USE (prometheus up)") || echo "free"

sec "root cause from the last vLLM run"
LOG="$RUN_DIR/vllm.log"
if [ -f "$LOG" ]; then
  echo "log: $LOG  ($(wc -l < "$LOG") lines)"
  echo
  # vLLM prints "Engine core initialization failed. See root cause above." —
  # the wrapper. The real error is the LAST exception line that is not that
  # wrapper, so search for it specifically.
  echo "--- real error (first non-wrapper exception) ---"
  grep -nE "^[^ ]*(Error|Exception|RuntimeError|TypeError|ValueError|ImportError|OSError):" "$LOG" \
    | grep -viE "Engine core initialization failed" | head -5
  grep -nE "(Error|Exception|RuntimeError|TypeError|ValueError|ImportError|OSError|assert)" "$LOG" \
    | grep -viE "Engine core initialization failed|^\s*File " | tail -8
  echo
  echo "--- 15 lines before the first ERROR marker ---"
  first=$(grep -n "ERROR" "$LOG" | head -1 | cut -d: -f1)
  if [ -n "${first:-}" ]; then
    start=$(( first > 15 ? first - 15 : 1 ))
    sed -n "${start},$((first + 10))p" "$LOG"
  else
    echo "(no ERROR marker found)"
  fi
else
  echo "no log at $LOG — vLLM has not been started via serve_native.sh yet"
fi

if [ "${1:-}" = "--serve" ]; then
  sec "live start attempt (90s, foreground, unfiltered)"
  echo "model: $MODEL"
  echo "Anything below is vLLM's own output. The first traceback IS the cause."
  hr
  timeout 90 vllm serve "$MODEL" --port "$PORT" --max-model-len 2048 2>&1 \
    | tail -60
  echo
  echo "(90s cap — if you only see download/loading progress, that is fine:"
  echo " it means nothing crashed. Re-run serve_native.sh and be patient.)"
fi

sec "verdict"
python3 - <<'PY' 2>/dev/null || echo "could not evaluate"
import shutil, subprocess, re, os
problems = []
try:
    out = subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().split(".")[0]
    drv = int(out)
except Exception:
    drv = None
    problems.append("no nvidia-smi: this shell cannot see a GPU")
try:
    import torch
    cu = float(".".join((torch.version.cuda or "0").split(".")[:2]))
    if drv is not None:
        cap = 13.0 if drv >= 580 else 12.8 if drv >= 570 else 12.4 if drv >= 550 else 12.2
        if cu > cap:
            problems.append(
                f"DRIVER MISMATCH: torch needs CUDA {cu} but driver {drv} caps at {cap}. "
                "Pick a host with a newer driver, or install torch/vllm built for that CUDA.")
    if not torch.cuda.is_available():
        problems.append("torch.cuda.is_available() is False")
except Exception as e:
    problems.append(f"torch not importable: {e}")
try:
    st = os.statvfs("/dev/shm")
    mb = st.f_blocks * st.f_frsize / 1e6
    if mb < 256:
        problems.append(f"/dev/shm is only {mb:.0f}MB — vLLM needs >=256MB. "
                        "Try: mount -o remount,size=8G /dev/shm  "
                        "or run with VLLM_ENABLE_V1_MULTIPROCESSING=0")
except Exception:
    pass
print("\n".join(f"  [X] {p}" for p in problems) if problems
      else "  [OK] nothing obviously wrong — if vLLM still fails, the --serve output above has it")
PY
echo
