"""
Observability — pipeline timing, token usage, cost estimates, LangSmith tracing.

Every node in the pipeline is wrapped with instrument_node() which:
  - Times the node execution
  - Extracts token usage from the node's _telemetry output
  - Records success/failure
  - Optionally emits a LangSmith trace span

At the end, PipelineObserver.summary() returns a dict for the UI metrics panel,
and persist() appends the run to logs/pipeline_metrics.jsonl.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# LangSmith is optional — gracefully degrade if not installed / not configured
try:
    from langsmith import traceable
except Exception:
    def traceable(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator


# Path to append metrics JSONL
METRICS_PATH = Path(__file__).resolve().parent.parent / "logs" / "pipeline_metrics.jsonl"
_FILE_LOCK = threading.Lock()


# ── Pricing table (input / output USD per 1M tokens) ────────────────────────
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":         (0.15,  0.60),
    "gpt-4o":              (2.50, 10.00),
    "gpt-4-turbo":        (10.00, 30.00),
    "claude-haiku-4-5":    (1.00,  5.00),
    "claude-sonnet-4-6":   (3.00, 15.00),
}


def estimate_tokens(text: str) -> int:
    """Rough token count estimate: ~4 chars per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate LLM call cost in USD from token counts."""
    in_rate, out_rate = MODEL_PRICING_PER_1M.get(model, (0.0, 0.0))
    cost = (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
    return round(cost, 8)


def setup_langsmith() -> None:
    """
    Enable LangSmith tracing when LANGSMITH_API_KEY is present in .env.
    Safe to call multiple times — idempotent.
    """
    key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT",
                              os.environ.get("LANGSMITH_PROJECT", "jobmatch-ai"))
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


# ── Per-node metric record ───────────────────────────────────────────────────

@dataclass
class NodeMetric:
    node: str
    success: bool
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error: str | None = None


# ── Pipeline observer ────────────────────────────────────────────────────────

class PipelineObserver:
    """Collects per-node metrics and computes an aggregated pipeline summary."""

    def __init__(self) -> None:
        self.node_metrics: list[NodeMetric] = []
        self.started_at: str = datetime.now(timezone.utc).isoformat()

    def record(self, metric: NodeMetric) -> None:
        self.node_metrics.append(metric)

    def _aggregate(self) -> dict[str, Any]:
        total = len(self.node_metrics)
        success_count = sum(1 for m in self.node_metrics if m.success)
        total_latency  = sum(m.latency_ms for m in self.node_metrics)
        total_in_tok   = sum(m.input_tokens for m in self.node_metrics)
        total_out_tok  = sum(m.output_tokens for m in self.node_metrics)
        total_cost     = round(sum(m.estimated_cost_usd for m in self.node_metrics), 8)
        success_rate   = round((success_count / total) * 100, 2) if total else 0.0

        return {
            "started_at":                self.started_at,
            "finished_at":               datetime.now(timezone.utc).isoformat(),
            "total_nodes":               total,
            "success_count":             success_count,
            "failure_count":             total - success_count,
            "success_rate_pct":          success_rate,
            "total_latency_ms":          total_latency,
            "total_input_tokens":        total_in_tok,
            "total_output_tokens":       total_out_tok,
            "estimated_total_cost_usd":  total_cost,
            "nodes":                     [asdict(m) for m in self.node_metrics],
        }

    def summary(self) -> dict[str, Any]:
        """Return the aggregated metrics dict (used by app.py for the metrics panel)."""
        return self._aggregate()

    def persist(self) -> None:
        """Append this run's metrics as a JSONL line to logs/pipeline_metrics.jsonl."""
        payload = self._aggregate()
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=True)
        with _FILE_LOCK:
            with METRICS_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


# ── Node instrumentation wrapper ─────────────────────────────────────────────

def instrument_node(
    node_name: str,
    node_fn: Callable[[dict], dict],
    observer: PipelineObserver,
) -> Callable[[dict], dict]:
    """
    Wrap a pipeline node with:
      - Wall-clock timing
      - Success / error detection
      - Token + cost extraction from the node's _telemetry output
      - Optional LangSmith trace span (if langsmith is installed & API key set)
    """

    @traceable(name=node_name, run_type="chain")
    def _wrapped(state: dict) -> dict:
        start = time.perf_counter()
        success = True
        error: str | None = None
        result: dict[str, Any] = {}

        try:
            result = node_fn(state)
            if result.get("error"):
                success = False
                error = str(result["error"])
            return result
        except Exception as exc:
            success = False
            error = str(exc)
            raise
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            telemetry: dict = {}
            if isinstance(result, dict):
                telemetry = result.get("_telemetry", {}).get(node_name, {})
            observer.record(
                NodeMetric(
                    node=node_name,
                    success=success,
                    latency_ms=latency_ms,
                    input_tokens=int(telemetry.get("input_tokens", 0)),
                    output_tokens=int(telemetry.get("output_tokens", 0)),
                    estimated_cost_usd=float(telemetry.get("estimated_cost_usd", 0.0)),
                    error=error,
                )
            )

    return _wrapped
