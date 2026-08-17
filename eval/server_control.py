"""Start, restart, and perturb the vLLM server, whatever it is running on.

Two backends, auto-detected:

``compose``
    The docker-compose stack in ``deploy/vllm``. Config faults recreate the
    service with extra flags; CPU faults throttle the container's quota.

``native``
    vLLM and Prometheus as plain processes, brought up by
    ``deploy/vllm/serve_native.sh``. Required on container-based GPU hosts
    (RunPod Pods and similar), which are themselves containers and cannot run
    a nested Docker daemon — dockerd needs host iptables and network-namespace
    privileges a Pod does not have. Config faults kill and relaunch the
    process; CPU faults spawn competing busy loops.

Nothing about the experiment needs containers, so the native path is not a
downgrade. The one real loss is cAdvisor, which means the ``host_saturation``
component has no data — it is marked optional in the domain for exactly this
reason.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Protocol

import requests

ROOT = Path(__file__).parent.parent
COMPOSE_DIR = ROOT / "deploy" / "vllm"
NATIVE_STATE = COMPOSE_DIR / "native_state.json"
SERVE_NATIVE = COMPOSE_DIR / "serve_native.sh"

#: CPU allowance the server keeps under `cpu_hog`, as a fraction of one core
#: (compose) or the number of competing busy loops (native).
CPU_HOG_QUOTA = "0.5"
CPU_HOG_WORKERS = max(2, (os.cpu_count() or 4) - 1)

#: A 7B on an L4 becomes healthy in ~3.5 minutes with warm weights. 420s
#: leaves generous margin while ensuring a server that cannot start costs
#: one scenario's worth of time, not fifteen minutes per repeat.
HEALTH_TIMEOUT_S = 420


class ServerControl(Protocol):
    name: str

    def restart_with(self, server_args: tuple[str, ...], server_url: str) -> None: ...
    def restore(self, server_url: str) -> None: ...
    def start_cpu_hog(self) -> object: ...
    def stop_cpu_hog(self, token: object) -> None: ...


def wait_for_health(server_url: str, timeout_s: float = HEALTH_TIMEOUT_S) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            requests.get(f"{server_url}/health", timeout=5).raise_for_status()
            return
        except requests.exceptions.RequestException:
            time.sleep(5)
    raise RuntimeError(
        f"Server did not become healthy within {timeout_s:.0f}s of restart."
    )


# ---------------------------------------------------------------------------
# docker-compose
# ---------------------------------------------------------------------------


class ComposeControl:
    name = "compose"

    def _active(self) -> tuple[str, str]:
        """(profile, service) of the running vLLM container."""
        out = subprocess.run(
            ["docker", "ps", "--filter", "name=vllm", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        if any("vllm-gpu" in n for n in out):
            return "gpu", "vllm-gpu"
        if any("vllm-cpu" in n for n in out):
            return "cpu", "vllm-cpu"
        raise RuntimeError("No vLLM container is running.")

    def _recreate(self, extra: str, server_url: str) -> None:
        profile, service = self._active()
        subprocess.run(
            ["docker", "compose", "--profile", profile, "up", "-d",
             "--force-recreate", service],
            cwd=COMPOSE_DIR,
            env={**os.environ, "VLLM_EXTRA_ARGS": extra},
            check=True, capture_output=True,
        )
        wait_for_health(server_url)

    def restart_with(self, server_args, server_url) -> None:
        self._recreate(" ".join(server_args), server_url)

    def restore(self, server_url) -> None:
        self._recreate("", server_url)

    def start_cpu_hog(self):
        _, service = self._active()
        container = subprocess.run(
            ["docker", "ps", "--filter", f"name={service}", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True,
        ).stdout.split()[0]
        subprocess.run(
            ["docker", "update", "--cpus", CPU_HOG_QUOTA, container],
            check=True, capture_output=True,
        )
        return container

    def stop_cpu_hog(self, token) -> None:
        subprocess.run(
            ["docker", "update", "--cpus", "0", str(token)],
            check=False, capture_output=True,
        )


# ---------------------------------------------------------------------------
# native processes
# ---------------------------------------------------------------------------


class NativeControl:
    name = "native"

    def _state(self) -> dict:
        if not NATIVE_STATE.exists():
            raise RuntimeError(
                f"{NATIVE_STATE} not found. Start the server with:\n"
                f"  bash {SERVE_NATIVE} up"
            )
        return json.loads(NATIVE_STATE.read_text())

    def _relaunch(self, extra: str, server_url: str) -> None:
        state = self._state()
        run_dir = Path(state["run_dir"])

        # Stop the current server and wait for the port to actually free, so
        # the replacement does not fail to bind.
        subprocess.run(["pkill", "-f", "vllm serve"], check=False)
        for _ in range(60):
            try:
                requests.get(f"{server_url}/health", timeout=2)
                time.sleep(1)
            except requests.exceptions.RequestException:
                break

        cmd = ["vllm", "serve", state["model"], *state["base_args"], *extra.split()]
        with open(run_dir / "vllm.log", "ab") as log:
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        (run_dir / "vllm.pid").write_text(str(proc.pid))
        NATIVE_STATE.write_text(json.dumps({**state, "extra_args": extra}, indent=2))
        wait_for_health(server_url)

    def restart_with(self, server_args, server_url) -> None:
        self._relaunch(" ".join(server_args), server_url)

    def restore(self, server_url) -> None:
        self._relaunch("", server_url)

    def start_cpu_hog(self):
        """Spawn busy loops that compete with the server for CPU.

        vLLM's API server, tokenizer, and detokenizer are CPU-bound, so
        starving them spikes TTFT while the GPU and the KV cache stay idle —
        the same mechanism the compose backend produces by lowering the
        container's quota.
        """
        workers = [
            subprocess.Popen(
                ["python3", "-c", "while True: pass"], start_new_session=True
            )
            for _ in range(CPU_HOG_WORKERS)
        ]
        return workers

    def stop_cpu_hog(self, token) -> None:
        for proc in token or []:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def detect(prefer: str | None = None) -> ServerControl:
    """Pick a backend. Native wins when its state file exists."""
    if prefer == "compose":
        return ComposeControl()
    if prefer == "native":
        return NativeControl()

    if NATIVE_STATE.exists():
        return NativeControl()
    try:
        subprocess.run(["docker", "ps"], check=True, capture_output=True, timeout=10)
        return ComposeControl()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return NativeControl()
