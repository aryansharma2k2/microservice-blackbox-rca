"""Fault injection CLI.

Usage:
    # Single service
    python inject.py --fault cpu_hog --service cartservice --duration 60

    # Multiple services simultaneously
    python inject.py --fault mem_leak --service cartservice --concurrent checkoutservice,frontend --duration 60
"""

import subprocess
import sys
import threading
from pathlib import Path

import click

# Project root on sys.path so the qualified import below works both when this
# file is run as a script and when it is imported as part of the package.
# Matches the pattern in chaos_inject.py.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import fault_injection.ground_truth as gt

NAMESPACE = "boutique"
FAULTS_DIR = Path(__file__).parent / "faults"
EXPERIMENTS_DIR = ROOT / "experiments"


# ---------------------------------------------------------------------------
# kubectl helpers
# ---------------------------------------------------------------------------

def find_pods(service: str, namespace: str = NAMESPACE) -> list[str]:
    """Return names of Running pods for *service* (matched via app= label)."""
    result = subprocess.run(
        [
            "kubectl", "get", "pods",
            "-n", namespace,
            "-l", f"app={service}",
            "--field-selector=status.phase=Running",
            "-o", "jsonpath={.items[*].metadata.name}",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kubectl get pods failed: {result.stderr.strip()}")

    pods = result.stdout.strip().split()
    if not pods:
        raise RuntimeError(f"No running pods found for service '{service}' in namespace '{namespace}'")
    return pods


def _get_container_name(pod: str, namespace: str = NAMESPACE) -> str:
    """Return the name of the first container in *pod*."""
    result = subprocess.run(
        [
            "kubectl", "get", "pod", pod,
            "-n", namespace,
            "-o", "jsonpath={.spec.containers[0].name}",
        ],
        capture_output=True, text=True,
    )
    name = result.stdout.strip()
    if not name:
        raise RuntimeError(f"Could not determine container name for pod '{pod}'")
    return name


def probe_shell(pod: str, container: str, namespace: str = NAMESPACE) -> bool:
    """Return True if the container has /bin/sh available via kubectl exec."""
    result = subprocess.run(
        ["kubectl", "exec", "-n", namespace, pod, "-c", container,
         "--", "sh", "-c", "echo ok"],
        capture_output=True, text=True, timeout=10,
    )
    return result.returncode == 0 and "ok" in result.stdout


def exec_script(pod: str, script: str, env: dict[str, str], namespace: str = NAMESPACE) -> None:
    """Run *script* inside *pod* via kubectl exec -- sh -c.

    Env vars in *env* are prepended as shell assignments.
    The script runs inside the actual service container (same cgroup that
    cAdvisor monitors), so resource usage is visible in Prometheus metrics.
    """
    container = _get_container_name(pod, namespace)
    assignments = "\n".join(f"{k}={v}" for k, v in env.items())
    full_script = f"{assignments}\n{script}"

    result = subprocess.run(
        ["kubectl", "exec", "-n", namespace, pod, "-c", container, "--", "sh", "-c", full_script],
        text=True,
        capture_output=True,
    )

    if result.stdout:
        click.echo(result.stdout.rstrip())
    if result.stderr:
        click.echo(result.stderr.rstrip(), err=True)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "executable file not found" in stderr or "exec:" in stderr:
            raise RuntimeError(
                f"No shell in pod '{pod}' (distroless container). "
                "Boutique services with a shell: adservice, currencyservice, emailservice, "
                "paymentservice, recommendationservice. "
                "Distroless (no shell): cartservice, checkoutservice, frontend, "
                "productcatalogservice, shippingservice."
            )
        raise RuntimeError(f"Script exited {result.returncode} in pod '{pod}': {stderr}")


# ---------------------------------------------------------------------------
# Injection logic
# ---------------------------------------------------------------------------

def _inject_one(fault: str, service: str, duration: int, namespace: str, dry_run: bool) -> None:
    """Inject *fault* into one pod of *service*."""
    pods = find_pods(service, namespace)
    pod = pods[0]   # target the first running pod
    container = _get_container_name(pod, namespace)
    click.echo(f"  on {service}  (pod: {pod}, container: {container})")

    script_path = FAULTS_DIR / f"{fault}.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Fault script not found: {script_path}")
    script = script_path.read_text()

    env = {"DURATION": str(duration)}

    if dry_run:
        click.echo(f"    [dry-run] would exec {script_path.name} in {pod} with env {env}")
        return

    if not probe_shell(pod, container, namespace):
        raise RuntimeError(
            f"Cannot inject into '{service}': container '{container}' has no shell "
            "(distroless image). Injectable services (have /bin/sh): "
            "adservice, currencyservice, emailservice, paymentservice, recommendationservice."
        )

    exec_script(pod, script, env, namespace)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--fault", required=True,
    type=click.Choice(["cpu_hog", "mem_leak", "net_delay", "disk_hog"]),
    help="Type of fault to inject.",
)
@click.option("--service", required=True, help="Primary target service name.")
@click.option("--duration", default=60, show_default=True, help="Duration in seconds.")
@click.option(
    "--concurrent", default=None,
    help="Comma-separated additional services to fault simultaneously.",
)
@click.option("--namespace", default=NAMESPACE, show_default=True)
@click.option("--run-id", default=None, help="Run ID to use (generated if omitted).")
@click.option("--dry-run", is_flag=True)
def inject(fault: str, service: str, duration: int, concurrent: str | None, namespace: str, run_id: str | None, dry_run: bool) -> None:
    """Inject a fault into one or more running microservice pods."""
    # Collect all target services
    services = [service]
    if concurrent:
        services += [s.strip() for s in concurrent.split(",") if s.strip()]
    services = list(dict.fromkeys(services))   # deduplicate, preserve order

    # Record ground truth before injecting
    if run_id is None:
        run_id = gt.make_run_id()
    out_dir = EXPERIMENTS_DIR / run_id
    if not dry_run:
        gt_path = gt.write(
            run_id=run_id,
            fault_type=fault,
            target_services=services,
            duration_seconds=duration,
            output_dir=out_dir,
        )
        click.echo(f"Ground truth in {gt_path}")

    click.echo(f"Injecting '{fault}' into {services} for {duration}s  (run_id={run_id})")

    if len(services) == 1:
        try:
            _inject_one(fault, services[0], duration, namespace, dry_run)
        except (RuntimeError, FileNotFoundError) as exc:
            click.echo(f"ERROR: {exc}", err=True)
            sys.exit(1)
    else:
        # Concurrent injection — one thread per service
        errors: list[str] = []
        lock = threading.Lock()

        def _worker(svc: str) -> None:
            try:
                _inject_one(fault, svc, duration, namespace, dry_run)
            except Exception as exc:
                with lock:
                    errors.append(f"{svc}: {exc}")

        threads = [threading.Thread(target=_worker, args=(svc,), daemon=True) for svc in services]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            for err in errors:
                click.echo(f"ERROR: {err}", err=True)
            sys.exit(1)

    click.echo("Injection complete.")


if __name__ == "__main__":
    inject()
