"""Qdrant and embedding retrieval helpers for log analysis."""

import re
from datetime import datetime, timedelta, timezone

from .config import (
    COLLECTION,
    EMBED_URL,
    LOG_NAMESPACES,
    MAX_CONTEXT_CHARS,
    MAX_DISCOVERY_POINTS,
    MAX_NAMESPACE_LOG_POINTS,
    PROMETHEUS_URL,
    QDRANT_URL,
    client,
)
from .log_questions import _dedupe, extract_namespaces


def embed_text(text: str) -> list[float]:
    """Send text to embedding service, get back a vector."""
    resp = client.post(f"{EMBED_URL}/embed", json={"texts": [text]})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def search_logs(vector: list[float], top_k: int, lookback: timedelta | None = None) -> list[dict]:
    """Query Qdrant for the most similar log chunks."""
    limit = top_k if lookback is None else min(max(top_k * 8, top_k), 64)
    resp = client.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        json={
            "vector": vector,
            "limit": limit,
            "with_payload": True,
        },
    )
    resp.raise_for_status()
    results = resp.json().get("result", [])
    chunks = [
        {
            "text": r["payload"]["text"],
            "source": r["payload"].get("source", "unknown"),
            "level": r["payload"].get("level", "info"),
            "timestamp": r["payload"].get("timestamp", ""),
            "score": r["score"],
        }
        for r in results
    ]
    if lookback is None:
        return chunks

    since = datetime.now(timezone.utc) - lookback
    filtered = []
    for chunk in chunks:
        ts = _parse_timestamp(chunk.get("timestamp"))
        if ts and ts >= since:
            filtered.append(chunk)
    return filtered[:top_k]


def build_prompt(question: str, log_chunks: list[dict]) -> str:
    """Build the user-message body. Caller pairs this with SYSTEM_PROMPT.

    CPU-only llama.cpp under VirtualBox is very sensitive to prompt size, so
    this keeps retrieved context bounded.
    """
    remaining = MAX_CONTEXT_CHARS
    context_parts = []
    for c in log_chunks:
        if remaining <= 0:
            break
        text = c["text"].strip()
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "\n...[truncated]"
        remaining -= len(text)
        context_parts.append(
            f"Source={c['source']} Level={c['level']} Time={c['timestamp']}\n{text}"
        )
    context_block = "\n\n".join(context_parts)
    return f"LOGS:\n{context_block}\n\nQUESTION: {question}"


def scroll_log_payloads(
    qdrant_filter: dict | None = None,
    max_points: int = MAX_NAMESPACE_LOG_POINTS,
) -> list[dict]:
    """Read payloads from Qdrant without vector search for exact log analysis."""
    payloads: list[dict] = []
    offset = None

    while len(payloads) < max_points:
        body: dict = {
            "limit": min(256, max_points - len(payloads)),
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        if qdrant_filter:
            body["filter"] = qdrant_filter

        resp = client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            json=body,
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})
        points = result.get("points", [])
        for point in points:
            payload = point.get("payload") or {}
            if payload.get("text"):
                payloads.append(payload)

        offset = result.get("next_page_offset")
        if not offset or not points:
            break

    return payloads


def _payload_namespace(payload: dict) -> str:
    ns = payload.get("namespace")
    if isinstance(ns, str) and ns:
        return ns
    source = str(payload.get("source", ""))
    if "/" in source:
        return source.split("/", 1)[0]
    return ""


def discover_log_namespaces() -> list[str]:
    """Discover namespaces from Qdrant log payloads."""
    try:
        payloads = scroll_log_payloads(max_points=MAX_DISCOVERY_POINTS)
    except Exception:
        return []
    namespaces = []
    for payload in payloads:
        ns = _payload_namespace(payload)
        if ns:
            namespaces.append(ns)
    return _dedupe(namespaces)


def discover_prometheus_namespaces() -> list[str]:
    """Discover namespaces from Prometheus label values."""
    namespaces = []
    for label in ("namespace", "kubernetes_namespace"):
        try:
            resp = client.get(f"{PROMETHEUS_URL}/api/v1/label/{label}/values", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "success":
                namespaces.extend(str(ns) for ns in data.get("data", []) if ns)
        except Exception:
            continue
    return _dedupe(namespaces)


def available_log_namespaces() -> list[str]:
    return _dedupe(LOG_NAMESPACES + discover_log_namespaces() + discover_prometheus_namespaces())


def requested_log_namespaces(question: str) -> list[str]:
    return extract_namespaces(question, available_log_namespaces())


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _payload_level(payload: dict) -> str:
    level = str(payload.get("level", "")).upper()
    if level == "WARNING":
        return "WARN"
    if level == "FATAL":
        return "ERROR"
    if level in {"ERROR", "WARN"}:
        return level

    text = str(payload.get("text", "")).upper()
    if any(marker in text for marker in ("FATAL", "ERROR", "FAIL", "TIMEOUT", "OOM", "EXCEPTION", "CRASH")):
        return "ERROR"
    if re.match(r"^W\d{4}\b", text) or "WARN" in text or "WARNING" in text:
        return "WARN"
    if level in {"INFO", "DEBUG"}:
        return level
    return "INFO"


def fetch_namespace_logs(namespaces: list[str], lookback: timedelta) -> list[dict]:
    """Fetch logs for namespaces and time window from Qdrant payloads."""
    since = datetime.now(timezone.utc) - lookback
    wanted = {ns.lower() for ns in namespaces}
    entries: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for ns in namespaces:
        payloads = scroll_log_payloads({
            "must": [{"key": "namespace", "match": {"value": ns}}]
        })

        for payload in payloads:
            if _payload_namespace(payload).lower() not in wanted:
                continue
            ts = _parse_timestamp(payload.get("timestamp"))
            if ts is None or ts < since:
                continue
            key = (
                str(payload.get("source", "")),
                str(payload.get("timestamp", "")),
                str(payload.get("text", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            entries.append({
                "text": str(payload.get("text", "")),
                "source": str(payload.get("source", "unknown")),
                "level": _payload_level(payload),
                "timestamp": str(payload.get("timestamp", "")),
                "dt": ts,
            })

    entries.sort(key=lambda e: e["dt"])
    return entries

