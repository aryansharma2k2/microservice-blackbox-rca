"""The committed LLM cache must replay without the optional [llm] extra.

The published comparison table includes an LLM baseline row. That number is
only reproducible because every response is cached under traces/llm_cache and
committed. If reading the cache requires the extra, then anyone who clones the
repo and runs `make eval` — and CI, which installs only [dev] — silently scores
the LLM baseline as an empty ranking and gets a different table.

That is exactly what happened: eval/llm_baseline.py imported pydantic at module
scope, so the whole module was unimportable without the extra and the baseline
degraded to nothing. These tests run in a subprocess with the optional
dependencies blocked, because the failure only exists in an interpreter where
they were never importable.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BLOCK = """
import sys
from importlib.abc import MetaPathFinder
BLOCKED = ("pydantic", "anthropic")
class Block(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in BLOCKED or any(name.startswith(b + ".") for b in BLOCKED):
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, Block())
"""


def _run(body: str) -> str:
    """Execute *body* in a subprocess where pydantic and anthropic are absent."""
    result = subprocess.run(
        [sys.executable, "-c", BLOCK + textwrap.dedent(body)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\n{result.stdout}\n{result.stderr}"
    )
    return result.stdout.strip()


def test_module_imports_without_the_optional_extra():
    out = _run(
        """
        import eval.llm_baseline as m
        print(m.HAVE_PYDANTIC)
        """
    )
    assert out == "False", "the guard should report pydantic as absent"


def test_a_cached_diagnosis_is_readable_without_pydantic():
    out = _run(
        """
        from eval.llm_baseline import Diagnosis
        d = Diagnosis(ranked_components=["kv_cache_pressure"],
                      verdict="pathology", reasoning="cached")
        print(d.ranked_components[0], d.verdict)
        """
    )
    assert out == "kv_cache_pressure pathology"


def test_llm_baseline_still_scores_from_the_committed_cache():
    """The regression guard proper: the row must not collapse to zero."""
    out = _run(
        """
        from pathlib import Path
        from eval.run_eval import evaluate
        card = evaluate(Path("traces/vllm"))["llm"]
        print(f"{card.top1:.4f} {card.top3:.4f} {card.runs}")
        """
    )
    top1, top3, runs = out.split()
    assert int(runs) > 0
    assert float(top1) > 0.0, (
        "LLM baseline scored nothing without the extra — the committed cache "
        "is not being read, so the published table is not reproducible"
    )
    assert float(top3) > float(top1)
