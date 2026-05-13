"""Runtime configuration shared by the RAG app modules."""

import os

import httpx

QWEN3_URL = os.getenv("QWEN3_URL", "http://qwen3-server:8080")
EMBED_URL = os.getenv("EMBED_URL", "http://embedding-server:8080")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus.monitoring.svc.cluster.local:9090")


def _csv_env(name: str, default: str = "") -> list[str]:
    values = [
        v.strip()
        for v in os.getenv(name, default).split(",")
        if v.strip() and v.strip().lower() not in {"*", "all"}
    ]
    return values


METRICS_NAMESPACES = _csv_env("METRICS_NAMESPACE", "ai-platform,monitoring")
LOG_NAMESPACES = _csv_env("LOG_NAMESPACE")
COLLECTION = os.getenv("COLLECTION", "logs")
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "600"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))
MAX_NAMESPACE_LOG_POINTS = int(os.getenv("MAX_NAMESPACE_LOG_POINTS", "5000"))
MAX_DISCOVERY_POINTS = int(os.getenv("MAX_DISCOVERY_POINTS", "1000"))
MAX_METRICS_NAMESPACES = int(os.getenv("MAX_METRICS_NAMESPACES", "12"))
DEFAULT_NAMESPACE_LOOKBACK_HOURS = int(os.getenv("DEFAULT_NAMESPACE_LOOKBACK_HOURS", "24"))
QWEN3_TIMEOUT = float(os.getenv("QWEN3_TIMEOUT", "180"))
USE_LLM = os.getenv("USE_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}


SYSTEM_PROMPT = (
    "You are a DevOps log analyst. Use only LOGS. Reply in at most three "
    "short bullets: recurring error, likely cause, next fix. Mention source "
    "or timestamp when present. If unclear, say so. /no_think"
)

METRICS_SYSTEM_PROMPT = (
    "You are a DevOps SRE. Use ONLY the METRICS below to answer. "
    "Reply in at most three short bullets: current state, anomaly, next action. "
    "If a value is missing, say so. /no_think"
)


# Shared HTTP client. Keep dependency checks bounded; give CPU inference its own
# longer request timeout in llm.generate_analysis().
client = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0))
