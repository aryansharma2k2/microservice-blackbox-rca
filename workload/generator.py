"""Workload generator for a vLLM OpenAI-compatible server.

Separate from ``infra/loadgen.py``, which drives Online Boutique storefront
journeys — the request shape, the pacing policy, and the SLI are all different
here.  It deliberately exposes the same ``current_p95(window_seconds)``
interface, so ``eval``'s ``SLOMonitor`` works against either without changes.

Three things this does that the Boutique generator does not:

**Constant rate, not a sine wave.**  A periodic component would put a strong
frequency into every series and hand the FFT burst filter something to reject
that has nothing to do with the injected fault.  Scenarios create a clean step
change instead.

**Client-measured TTFT.**  Requests stream, and time-to-first-chunk is
recorded per request.  That is the SLI the experiments trigger on, and it
independently cross-checks the server's own histogram.

**Controlled request shape.**  Prompt length, output length, and prefix reuse
are all parameters, because for an inference server the workload shape *is*
the fault.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import requests

from workload.scenarios import Phase

logger = logging.getLogger(__name__)

#: Rough tokens-per-word for common English under a typical BPE tokenizer.
#: Only an estimate, and deliberately so: scenarios are defined by the *ratio*
#: between their baseline and fault phases, which survives a sloppy constant.
#: Calibrate against the server's /tokenize endpoint if exact counts ever
#: matter.
TOKENS_PER_WORD = 1.3

#: HTTP connection pool size. Must comfortably exceed the in-flight request
#: count at the highest scenario rate (24 rps against multi-second responses),
#: or the client throttles itself and the fault under test is masked.
CONNECTION_POOL_SIZE = 256

#: Common short words, most of which are a single token. Keeps the estimate
#: above honest and makes generated prompts compress poorly, so the server
#: cannot shortcut them.
_VOCAB = (
    "the of and to in a is that it for on with as was at by from or an be this "
    "have has had not are were you we they he she but all can her his one our "
    "out so if no up do what when which who will would there their about into "
    "over then them these two more some time only just also any how its said"
).split()


def tokens_to_words(n_tokens: int) -> int:
    """Approximate word count for a target token count."""
    return max(1, int(n_tokens / TOKENS_PER_WORD))


def make_text(n_tokens: int, rng: random.Random) -> str:
    """Generate roughly *n_tokens* tokens of filler text."""
    return " ".join(rng.choice(_VOCAB) for _ in range(tokens_to_words(n_tokens)))


@dataclass
class RequestResult:
    """One completed (or failed) request."""

    completed_at: float
    ttft_s: float | None
    total_s: float | None
    prompt_tokens: int
    ok: bool
    error: str = ""


@dataclass
class WorkloadStats:
    #: Requests handed to a worker thread.
    dispatched: int = 0
    #: Requests that finished, successfully or not. `dispatched - sent` is the
    #: in-flight backlog, which is the clearest saturation signal available
    #: client-side: if it grows without bound the offered rate exceeds what
    #: the server can absorb.
    sent: int = 0
    ok: int = 0
    failed: int = 0
    errors: dict[str, int] = field(default_factory=dict)


class VllmWorkloadGenerator:
    """Drives an OpenAI-compatible completions endpoint at a controlled shape.

    Parameters
    ----------
    base_url:
        Server root, e.g. ``http://localhost:8000``.
    model:
        Model name to request. Discovered from ``/v1/models`` when omitted.
    seed:
        Seeds prompt generation and the shared prefix pool, so a scenario
        replays identically.
    """

    #: Prefix length in tokens. Long enough to span several KV blocks so that
    #: reusing one is a meaningful prefix-cache hit.
    PREFIX_TOKENS = 512

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str | None = None,
        seed: int = 0,
        quiet: bool = False,
        request_timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.seed = seed
        self._quiet = quiet
        self._timeout = request_timeout

        self._rng = random.Random(seed)
        self._session = requests.Session()
        # requests' default pool holds 10 connections. Requests are fired
        # concurrently, so at the rates the scenarios use the client would
        # queue on its own pool before the server ever became the bottleneck
        # — and the experiment would be measuring the load generator.
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=CONNECTION_POOL_SIZE,
            pool_maxsize=CONNECTION_POOL_SIZE,
            max_retries=0,
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self._phase: Phase | None = None
        self._phase_lock = threading.Lock()
        self._prefixes: list[str] = []
        self._prefix_seed_used: int | None = None

        self._results: deque[RequestResult] = deque()
        self._results_lock = threading.Lock()
        self.stats = WorkloadStats()
        #: (timestamp, in-flight count) sampled by the pacing loop. A rising
        #: trend means backlog is accumulating even when the completion rate
        #: still looks acceptable — the system is not in steady state, so the
        #: baseline is not a valid control period.
        self._backlog: deque[tuple[float, int]] = deque(maxlen=10_000)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- setup -----------------------------------------------------------

    def discover_model(self) -> str:
        """Return the served model name, querying the server if needed."""
        if self.model:
            return self.model
        resp = self._session.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data:
            raise RuntimeError(f"{self.base_url}/v1/models returned no models")
        self.model = data[0]["id"]
        logger.info("Discovered model %s", self.model)
        return self.model

    def _prefix_pool(self, count: int) -> list[str]:
        """Deterministic pool of long shared prefixes, built once."""
        if self._prefix_seed_used != count:
            pool_rng = random.Random(self.seed)
            self._prefixes = [
                make_text(self.PREFIX_TOKENS, pool_rng) for _ in range(count)
            ]
            self._prefix_seed_used = count
        return self._prefixes

    # -- request shaping -------------------------------------------------

    def build_prompt(self, phase: Phase, rng: random.Random) -> tuple[str, int]:
        """Build one prompt for *phase*. Returns (text, approx_token_count)."""
        target = rng.randint(*phase.prompt_tokens)

        if phase.shared_prefixes > 0:
            pool = self._prefix_pool(phase.shared_prefixes)
            prefix = rng.choice(pool)
        else:
            # A unique prefix per request drives the prefix cache hit rate to
            # zero without changing prompt length at all.
            prefix = make_text(self.PREFIX_TOKENS, rng)

        remaining = max(1, target - self.PREFIX_TOKENS)
        return f"{prefix}\n\n{make_text(remaining, rng)}", target

    # -- one request -----------------------------------------------------

    def _send_one(self, phase: Phase, rng: random.Random) -> None:
        prompt, approx_tokens = self.build_prompt(phase, rng)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": rng.randint(*phase.max_tokens),
            # Greedy, so output length is driven by max_tokens rather than by
            # sampling luck. Keeps the workload reproducible.
            "temperature": 0.0,
            "stream": True,
        }

        started = time.perf_counter()
        ttft: float | None = None
        ok = False
        error = ""

        try:
            with self._session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                stream=True,
                timeout=self._timeout,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8", "replace")
                    if not line.startswith("data:"):
                        continue
                    body = line[len("data:"):].strip()
                    if body == "[DONE]":
                        break
                    if ttft is None:
                        # First token off the wire. This is the SLI.
                        ttft = time.perf_counter() - started
                    # Drain the rest so total latency is meaningful, but do not
                    # bother parsing every chunk.
                    del body
                ok = True
        except requests.exceptions.RequestException as exc:
            error = type(exc).__name__
        except (ValueError, json.JSONDecodeError) as exc:  # pragma: no cover
            error = type(exc).__name__

        total = time.perf_counter() - started if ok else None
        result = RequestResult(
            completed_at=time.time(),
            ttft_s=ttft,
            total_s=total,
            prompt_tokens=approx_tokens,
            ok=ok,
            error=error,
        )

        with self._results_lock:
            self._results.append(result)
            self.stats.sent += 1
            if ok:
                self.stats.ok += 1
            else:
                self.stats.failed += 1
                self.stats.errors[error] = self.stats.errors.get(error, 0) + 1
            # Bound memory: keep a couple of minutes of history.
            cutoff = result.completed_at - 120.0
            while self._results and self._results[0].completed_at < cutoff:
                self._results.popleft()

    # -- SLI accessors ---------------------------------------------------

    def _recent_ttfts(self, window_seconds: float) -> list[float]:
        cutoff = time.time() - window_seconds
        with self._results_lock:
            return sorted(
                r.ttft_s
                for r in self._results
                if r.completed_at >= cutoff and r.ttft_s is not None
            )

    @staticmethod
    def _percentile(sorted_values: list[float], q: float) -> float:
        if not sorted_values:
            raise ValueError("empty")
        idx = min(len(sorted_values) - 1, int(len(sorted_values) * q))
        return sorted_values[idx]

    def current_p95(self, window_seconds: float = 10.0) -> float | None:
        """p95 TTFT in seconds over the recent window, or None if no data.

        Named to match ``infra.loadgen.WorkloadGenerator`` so ``SLOMonitor``
        works against either generator unchanged.
        """
        recent = self._recent_ttfts(window_seconds)
        return self._percentile(recent, 0.95) if recent else None

    def current_p99(self, window_seconds: float = 10.0) -> float | None:
        recent = self._recent_ttfts(window_seconds)
        return self._percentile(recent, 0.99) if recent else None

    # -- lifecycle -------------------------------------------------------

    def set_phase(self, phase: Phase) -> None:
        """Switch the workload shape without restarting the generator.

        This is how a fault is injected for ``workload``-kind scenarios: the
        baseline runs, then the phase changes, producing a clean step change
        rather than the ramp a restart would create.
        """
        with self._phase_lock:
            self._phase = phase
        if not self._quiet:
            logger.info("Workload phase -> %s", phase.describe())

    def _current_phase(self) -> Phase:
        with self._phase_lock:
            assert self._phase is not None, "run() sets the phase before looping"
            return self._phase

    def _run_loop(self, duration_seconds: float) -> None:
        start = time.time()
        next_req = start
        rng = random.Random(self.seed + 1)

        while not self._stop_event.is_set():
            if time.time() - start >= duration_seconds:
                break

            now = time.time()
            if now < next_req:
                # Cap the sleep so a phase change takes effect promptly even at
                # very low rates.
                self._stop_event.wait(min(next_req - now, 0.5))
                continue

            phase = self._current_phase()
            with self._results_lock:
                self.stats.dispatched += 1
                self._backlog.append(
                    (time.time(), self.stats.dispatched - self.stats.sent)
                )
            threading.Thread(
                target=self._send_one, args=(phase, random.Random(rng.random())),
                daemon=True,
            ).start()
            next_req += 1.0 / max(0.1, phase.rps)

            # If we have fallen far behind (server slow, thread churn), resync
            # rather than firing a burst to catch up.
            if next_req < now - 1.0:
                next_req = now

        self._stop_event.set()

    def run(
        self,
        duration_seconds: float,
        phase: Phase,
        block: bool = False,
    ) -> threading.Thread:
        """Start generating at *phase* for up to *duration_seconds*."""
        self.discover_model()
        self.set_phase(phase)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(duration_seconds,),
            daemon=True,
            name="vllm-loadgen",
        )
        if not self._quiet:
            logger.info(
                "Starting workload: model=%s duration=%.0fs %s",
                self.model, duration_seconds, phase.describe(),
            )
        self._thread.start()
        if block:
            self._thread.join()
        return self._thread

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def health(self, offered_rps: float, window_seconds: float) -> dict:
        """Is the server keeping up with the offered load?

        A run whose *baseline* is already saturated cannot support any
        conclusion: the queue is growing before the fault is injected, so the
        two windows are not a controlled comparison, and onset ordering across
        a monotonically drifting system is meaningless. This is the check that
        distinguishes "the pipeline was wrong" from "the experiment was
        invalid", which otherwise look identical in the output.

        Returns the measurements plus a ``saturated`` verdict.
        """
        now = time.time()
        cutoff = now - window_seconds
        with self._results_lock:
            dispatched = self.stats.dispatched
            completed = self.stats.sent
            failed = self.stats.failed
            recent = [r for r in self._results if r.completed_at >= cutoff]
            backlog = [(t, n) for t, n in self._backlog if t >= cutoff]

        in_flight = max(0, dispatched - completed)
        completed_rps = len(recent) / window_seconds if window_seconds > 0 else 0.0
        failure_rate = failed / completed if completed else 0.0
        keeping_up = completed_rps >= 0.9 * offered_rps if offered_rps > 0 else True

        # Backlog trend across the window. A completion rate that merely looks
        # close to the offered rate can still hide slow accumulation, and that
        # accumulation shows up later as a monotonic latency drift the change
        # point detector will (correctly) flag — turning a "clean" run into a
        # fault it was never supposed to contain.
        backlog_growth = 0.0
        if len(backlog) >= 4:
            mid = len(backlog) // 2
            first = sum(n for _, n in backlog[:mid]) / mid
            second = sum(n for _, n in backlog[mid:]) / (len(backlog) - mid)
            backlog_growth = second - first
        drifting = backlog_growth > max(1.0, 0.5 * offered_rps)

        return {
            "offered_rps": round(offered_rps, 2),
            "completed_rps": round(completed_rps, 2),
            "in_flight": in_flight,
            "backlog_growth": round(backlog_growth, 2),
            "failure_rate": round(failure_rate, 3),
            "saturated": (not keeping_up) or failure_rate > 0.05 or drifting,
        }

    def summary(self) -> dict:
        with self._results_lock:
            done = [r for r in self._results if r.ttft_s is not None]
        ttfts = sorted(r.ttft_s for r in done)  # type: ignore[misc]
        return {
            "sent": self.stats.sent,
            "ok": self.stats.ok,
            "failed": self.stats.failed,
            "errors": dict(self.stats.errors),
            "ttft_p50_ms": round(self._percentile(ttfts, 0.50) * 1000, 1) if ttfts else None,
            "ttft_p95_ms": round(self._percentile(ttfts, 0.95) * 1000, 1) if ttfts else None,
            "ttft_p99_ms": round(self._percentile(ttfts, 0.99) * 1000, 1) if ttfts else None,
        }
