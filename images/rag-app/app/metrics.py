"""Prometheus-backed live metrics analysis."""

import re

from .config import MAX_METRICS_NAMESPACES, PROMETHEUS_URL, client
from .log_questions import _dedupe, extract_namespaces
from .log_retrieval import discover_prometheus_namespaces


METRICS_KEYWORDS = (
    "cpu", "memory", "ram", "restart", "restarts", "usage", "utilization",
    "node", "nodes", "pod status", "running", "healthy", "metric", "metrics",
    "prometheus", "live", "current", "now",
)


def is_metrics_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in METRICS_KEYWORDS)


def prom_query(promql: str) -> list[dict]:
    """Run a single instant PromQL query, return result vector entries."""
    resp = client.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": promql},
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        return []
    return data.get("data", {}).get("result", [])


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _per_namespace_queries(ns: str) -> dict:
    return {
        "Pods Running": (
            f'kube_pod_status_phase{{namespace="{ns}",phase="Running"}} == 1',
            lambda r: r["metric"].get("pod", "?"),
            lambda v: "Running",
        ),
        "Pod CPU (cores, 5m avg)": (
            f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{ns}",pod!="",container!=""}}[5m]))',
            lambda r: r["metric"].get("pod", "?"),
            lambda v: f"{float(v):.3f}",
        ),
        "Pod Memory (working set)": (
            f'sum by (pod) (container_memory_working_set_bytes{{namespace="{ns}",pod!="",container!=""}})',
            lambda r: r["metric"].get("pod", "?"),
            lambda v: _fmt_bytes(float(v)),
        ),
        "Container Restarts (1h)": (
            f'sum by (pod) (increase(kube_pod_container_status_restarts_total{{namespace="{ns}"}}[1h]))',
            lambda r: r["metric"].get("pod", "?"),
            lambda v: f"{float(v):.0f}",
        ),
    }


_GLOBAL_QUERIES = {
    "Node CPU Utilization": (
        '1 - avg by (kubernetes_node) (rate(node_cpu_seconds_total{job="node-exporter",mode="idle"}[5m]))',
        lambda r: r["metric"].get("kubernetes_node", "?"),
        lambda v: f"{float(v) * 100:.1f}%",
    ),
}


def _render_queries(queries: dict) -> list[str]:
    out = []
    for title, (promql, label_fn, val_fn) in queries.items():
        try:
            results = prom_query(promql)
        except Exception as e:
            out.append(f"{title}: query failed ({type(e).__name__}: {e})")
            continue
        if not results:
            out.append(f"{title}: (no data)")
            continue
        seen = {}
        for r in results:
            val = r.get("value", [None, None])[1]
            if val is None:
                continue
            try:
                seen[label_fn(r)] = val_fn(val)
            except Exception:
                continue
        rows = sorted(seen.items())
        body = "\n".join(f"  - {name}: {value}" for name, value in rows) or "  (empty)"
        out.append(f"{title}:\n{body}")
    return out


def metrics_snapshot(namespaces: list[str]) -> str:
    """Run PromQL snapshot queries for each namespace plus cluster-wide metrics."""
    sections: list[str] = []
    for ns in namespaces:
        sections.append(f"=== Live metrics from Prometheus (namespace={ns}) ===")
        sections.extend(_render_queries(_per_namespace_queries(ns)))
    sections.append("=== Cluster-wide metrics ===")
    sections.extend(_render_queries(_GLOBAL_QUERIES))
    return "\n".join(sections)


def select_metrics_namespaces(question: str, configured: list[str]) -> list[str]:
    """Select namespaces for Prometheus queries from labels plus configured defaults."""
    available = _dedupe(configured + discover_prometheus_namespaces())
    matched = extract_namespaces(question, available)
    if matched:
        return matched[:MAX_METRICS_NAMESPACES]
    if re.search(r"\b(?:all|every)\s+namespaces?\b", question.lower()):
        return available[:MAX_METRICS_NAMESPACES]
    if configured:
        return configured[:MAX_METRICS_NAMESPACES]
    return available[:MAX_METRICS_NAMESPACES] or ["default"]


def build_metrics_prompt(question: str, metrics_text: str) -> str:
    """User-message body for the metrics path. Caller pairs with METRICS_SYSTEM_PROMPT."""
    return f"METRICS:\n{metrics_text}\n\nQUESTION: {question}"


def fast_metrics_analysis(question: str, metrics_text: str) -> str:
    return (
        f"Live metrics snapshot:\n\n{metrics_text}\n\n"
        "(Set USE_LLM=true on the rag-app deployment for an LLM-generated summary.)"
    )

