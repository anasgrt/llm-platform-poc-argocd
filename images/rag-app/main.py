"""Log Analysis RAG Application.

Flow:
  1. User asks a question about logs
  2. Embed the question via embedding service
  3. Search Qdrant for similar log chunks
  4. Return a fast deterministic summary by default
  5. Optionally send compact context to Qwen when USE_LLM=true

Environment variables:
  QWEN3_URL     — llama.cpp server (default: http://qwen3-server:8080)
  EMBED_URL     — embedding service (default: http://embedding-server:8080)
  QDRANT_URL    — Qdrant REST API (default: http://qdrant:6333)
  COLLECTION    — Qdrant collection name (default: logs)
  USE_LLM       — set true to use Qwen generation instead of Fast RAG
"""

import json
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import traceback

app = FastAPI(title="Log Analysis Platform")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"{type(exc).__name__}: {str(exc)}"
    if isinstance(exc, httpx.RequestError):
        error_msg += f" (Request URL: {exc.request.url})"
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            error_msg += f" - Response: {exc.response.text}"
        except:
            pass
    print(f"500 Internal Server Error: {error_msg}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg}
    )

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

# Shared HTTP client. Keep dependency checks bounded; give CPU inference its own
# longer request timeout in generate_analysis().
client = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0))

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


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 reasoning blocks and any trailing reflection that leaks."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    for marker in ("\nWait,", "\nBut wait", "\nHmm,", "\nActually,"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    history: list[ChatMessage] = Field(default_factory=list)


def bound_history(history: list[ChatMessage]) -> list[dict]:
    """Sanitize and cap the client-supplied turn history.

    Drops anything that isn't a user/assistant message (defends against a
    client trying to inject a 'system' role) and keeps only the last N turns
    so CPU inference doesn't blow up on long sessions.
    """
    valid = [m for m in history if m.role in ("user", "assistant") and m.content]
    keep = valid[-(2 * MAX_HISTORY_TURNS):]
    return [{"role": m.role, "content": m.content} for m in keep]


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    num_chunks_used: int


# ── Step 1: Embed the query ──────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """Send text to embedding service, get back a vector."""
    resp = client.post(f"{EMBED_URL}/embed", json={"texts": [text]})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


# ── Step 2: Search Qdrant for similar log chunks ─────────────────────────────

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


# ── Step 3: Build prompt with log context ────────────────────────────────────

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


# ── Step 4: Call Qwen3 via llama.cpp OpenAI API ──────────────────────────────

def generate_analysis(system: str, user: str, history: list[dict] | None = None) -> str:
    """Send a chat-formatted request to Qwen3 and return the full completion.

    Uses /v1/chat/completions so the Qwen3 chat template is applied properly:
    the model-specific chat envelope is added by the server, stop tokens fire
    correctly, and the /no_think directive in the system message is honored.
    chat_template_kwargs disables thinking for builds that support it.

    `history` carries prior {role, content} turns so follow-up questions
    ("what about the second one?") have the context they need. Already
    bounded by bound_history() at the API layer — assumed safe here.
    """
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    resp = client.post(
        f"{QWEN3_URL}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.3,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=QWEN3_TIMEOUT,
    )
    resp.raise_for_status()
    return strip_thinking(resp.json()["choices"][0]["message"]["content"])


def _visible_token(token: str, in_think: bool) -> tuple[str, bool]:
    """Drop streamed <think> blocks while preserving visible answer text."""
    out: list[str] = []
    text = token
    while text:
        lower = text.lower()
        if in_think:
            end = lower.find("</think>")
            if end == -1:
                return "".join(out), True
            text = text[end + len("</think>"):]
            in_think = False
            continue

        start = lower.find("<think>")
        if start == -1:
            out.append(text)
            break
        out.append(text[:start])
        text = text[start + len("<think>"):]
        in_think = True
    return "".join(out), in_think


def generate_analysis_stream(system: str, user: str, history: list[dict] | None = None):
    """Stream tokens from Qwen3 via llama.cpp OpenAI API.

    Yields each content delta as it arrives so the caller can push it to
    the client immediately.  Uses SSE (text/event-stream) format that
    llama-cpp-python's OpenAI-compatible server emits.
    """
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    with client.stream(
        "POST",
        f"{QWEN3_URL}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.3,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=QWEN3_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        in_think = False
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]  # strip "data: " prefix
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    visible, in_think = _visible_token(token, in_think)
                    if visible:
                        yield visible
            except (json.JSONDecodeError, IndexError, KeyError):
                continue


def answer_is_incomplete(answer: str) -> bool:
    """Detect failed/aborted LLM replies that should use deterministic fallback."""
    text = strip_thinking(answer).strip()
    if len(text) < 40:
        return True
    tail = text.rsplit(maxsplit=1)[-1].lower().strip("`*_.,")
    return tail in {
        "in", "of", "from", "to", "for", "with", "and", "or", "the", "a", "an",
        "by", "at", "on", "as", "is", "are", "was", "were",
    }


def fast_log_analysis(question: str, log_chunks: list[dict]) -> str:
    """Return a bounded deterministic summary without blocking on CPU LLM inference."""
    priority = {"FATAL": 0, "ERROR": 1, "WARN": 2, "INFO": 3, "DEBUG": 4}
    ranked = sorted(log_chunks, key=lambda c: priority.get(c["level"], 9))
    error_lines = []

    for chunk in ranked:
        for line in chunk["text"].splitlines():
            upper = line.upper()
            if chunk["level"] in {"FATAL", "ERROR", "WARN"} or any(
                marker in upper for marker in ("FATAL", "ERROR", "WARN", "CRASH", "FAIL", "TIMEOUT", "OOM")
            ):
                error_lines.append((chunk, line.strip()))

    if not error_lines:
        sources = ", ".join(sorted({c["source"] for c in log_chunks}))
        return (
            f"- No explicit ERROR/FATAL lines were found in the top retrieved chunks from {sources}.\n"
            "- The retrieved context is mostly informational or warning-level; broaden the query or re-run ingestion if expected errors are missing.\n"
            "- Next check: inspect the source logs around the returned timestamps for adjacent failures."
        )

    top_chunk, top_line = error_lines[0]
    affected_sources = ", ".join(sorted({chunk["source"] for chunk, _ in error_lines}))
    levels = ", ".join(sorted({chunk["level"] for chunk, _ in error_lines}, key=lambda x: priority.get(x, 9)))
    evidence = "\n".join(f"  - {line[:180]}" for _, line in error_lines[:3])

    return (
        f"- Recurring severity in retrieved logs: {levels} from {affected_sources}.\n"
        f"- Strongest match: {top_chunk['source']} {top_chunk['level']} {top_chunk['timestamp']} -> {top_line[:220]}\n"
        f"- Evidence:\n{evidence}\n"
        "- Next fix: inspect the named service/pod around these timestamps, then correlate adjacent WARN/FATAL lines for the first failure in the chain."
    )


# ── Namespace log pattern path ───────────────────────────────────────────────

ERROR_PATTERN_KEYWORDS = (
    "error", "errors", "warn", "warning", "warnings", "failed", "failure",
    "fail", "fatal", "crash", "crashloop", "exception", "timeout", "oom",
    "pattern", "recurring",
)
TIME_WINDOW_RE = re.compile(
    r"\b(?:last|past)\s+(\d+)\s*(minute|minutes|min|m|hour|hours|hr|hrs|h|day|days|d)\b",
    re.IGNORECASE,
)
NAMESPACE_NAME_RE = re.compile(r"\b[a-z0-9]([-a-z0-9]*[a-z0-9])?\b")
NAMESPACE_STOP_WORDS = {
    "a", "about", "all", "an", "and", "are", "by", "current", "error",
    "errors", "for", "from", "hours", "in", "is", "last", "logs", "namespace",
    "namespaces", "of", "past", "pattern", "prometheus", "retrieved", "the",
    "what", "warn", "warning", "warnings",
}


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = value.lower()
        if key not in seen:
            out.append(value)
            seen.add(key)
    return out


def _namespace_word_match(question: str, namespace: str) -> bool:
    return re.search(
        rf"(?<![a-z0-9-]){re.escape(namespace.lower())}(?![a-z0-9-])",
        question.lower(),
    ) is not None


def extract_namespaces(question: str, available: list[str] | None = None) -> list[str]:
    """Return Kubernetes namespaces explicitly named in the question.

    The available list should come from Qdrant payloads, Prometheus labels, or
    optional deployment defaults. Phrases like "the foo namespace" are accepted
    even before that namespace has been discovered, so missing data returns a
    clear "no logs found" answer instead of falling back to stale semantic hits.
    """
    q = question.lower()
    matches = [ns for ns in (available or []) if _namespace_word_match(q, ns)]

    for pattern in (
        r"\bnamespace\s+([a-z0-9]([-a-z0-9]*[a-z0-9])?)\b",
        r"\b([a-z0-9]([-a-z0-9]*[a-z0-9])?)\s+namespace\b",
    ):
        for match in re.finditer(pattern, q):
            candidate = match.group(1)
            if NAMESPACE_NAME_RE.fullmatch(candidate) and candidate not in NAMESPACE_STOP_WORDS:
                matches.append(candidate)

    return _dedupe(matches)


def parse_lookback(question: str) -> timedelta | None:
    q = question.lower()
    match = TIME_WINDOW_RE.search(q)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith(("minute", "min")) or unit == "m":
            return timedelta(minutes=amount)
        if unit.startswith(("hour", "hr")) or unit == "h":
            return timedelta(hours=amount)
        if unit.startswith("day") or unit == "d":
            return timedelta(days=amount)
    if re.search(r"\b(?:last|past)\s+hour\b", q):
        return timedelta(hours=1)
    if re.search(r"\b(?:last|past)\s+day\b", q):
        return timedelta(days=1)
    return None


def is_log_problem_question(question: str) -> bool:
    q = question.lower()
    concrete = [k for k in ERROR_PATTERN_KEYWORDS if k not in {"pattern", "recurring"}]
    if any(k in q for k in concrete):
        return True
    return bool(re.search(r"\blogs?\b", q)) and any(k in q for k in ("pattern", "recurring"))


def wants_error_only_analysis(question: str) -> bool:
    q = question.lower()
    asks_for_error = any(
        re.search(rf"\b{word}\b", q)
        for word in ("error", "errors", "fatal", "exception", "crash", "crashloop", "oom")
    )
    asks_for_warning = any(re.search(rf"\b{word}\b", q) for word in ("warn", "warning", "warnings"))
    return asks_for_error and not asks_for_warning


def is_namespace_log_pattern_question(question: str) -> bool:
    namespaces = requested_log_namespaces(question)
    if not namespaces:
        return False
    return is_log_problem_question(question)


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
        filtered = scroll_log_payloads({
            "must": [{"key": "namespace", "match": {"value": ns}}]
        })
        candidates = filtered + scroll_log_payloads()

        for payload in candidates:
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


def _field_value(text: str, key: str) -> str:
    quoted = re.search(rf"\b{re.escape(key)}=\"([^\"]+)\"", text)
    if quoted:
        return quoted.group(1)
    bare = re.search(rf"\b{re.escape(key)}=([^ ]+)", text)
    return bare.group(1) if bare else ""


def _signature(text: str) -> str:
    msg = _field_value(text, "msg")
    err = _field_value(text, "error") or _field_value(text, "err")
    quoted_msg = re.search(r'\]\s+"([^"]+)"', text)
    if quoted_msg and err:
        return f"{quoted_msg.group(1)}: {err}"
    if msg and err:
        return f"{msg}: {err}"
    if msg:
        return msg
    if err:
        return f"error: {err}"

    normalized = re.sub(r"\d{4}-\d{2}-\d{2}T[^\s]+", "<ts>", text)
    normalized = re.sub(r"\b\d+(?:\.\d+)?(?:ms|s|m|h|B|KiB|MiB|GiB)?\b", "<n>", normalized)
    normalized = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:160] or "unclassified log line"


def _is_problem_entry(entry: dict) -> bool:
    if entry["level"] in {"ERROR", "WARN"}:
        return True
    upper = entry["text"].upper()
    return any(marker in upper for marker in ("ERROR", "WARN", "FAIL", "FATAL", "CRASH", "TIMEOUT", "OOM"))


def _fmt_dt(dt: datetime | None) -> str:
    if not dt:
        return "unknown time"
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _lookback_label(lookback: timedelta) -> str:
    seconds = int(lookback.total_seconds())
    if seconds % 86400 == 0:
        days = seconds // 86400
        return f"last {days} day" + ("" if days == 1 else "s")
    if seconds % 3600 == 0:
        hours = seconds // 3600
        return f"last {hours} hour" + ("" if hours == 1 else "s")
    minutes = max(1, seconds // 60)
    return f"last {minutes} minute" + ("" if minutes == 1 else "s")


def _next_action(top_signature: str, sources: list[str]) -> str:
    sig = top_signature.lower()
    source_text = " ".join(sources).lower()
    if "user token not found" in sig or "unauthorized" in sig or "authenticate request" in sig:
        return "Check Grafana auth/session traffic first: invalid or missing browser/API tokens are generating the repeated warnings."
    if "broken pipe" in sig or "failed to write metrics" in sig:
        return "Check the Prometheus scrape path and kube-state-metrics load; the scrape client is closing while metrics are being written."
    if "prometheus" in source_text:
        return "Check the Prometheus target, scrape, or query path named in the source before looking at the model stack."
    return "Check the emitting pod/service for the repeated signature and correlate with pod restarts or recent config changes."


def namespace_log_analysis(
    namespaces: list[str],
    entries: list[dict],
    lookback: timedelta,
    strict_errors: bool = False,
) -> str:
    ns_label = ", ".join(namespaces)
    window = _lookback_label(lookback)
    if not entries:
        return (
            f"- No logs were found for namespace {ns_label} in the {window}.\n"
            "- Fluent Bit may not have shipped matching records yet, or the Qdrant retention window has no data for that namespace.\n"
            "- Next check: verify matching `/var/log/containers/*.log` files exist and Fluent Bit is ingesting into `/api/ingest`."
        )

    problems = [entry for entry in entries if _is_problem_entry(entry)]
    error_entries = [entry for entry in problems if entry["level"] == "ERROR"]
    warn_entries = [entry for entry in problems if entry["level"] == "WARN"]
    first_ts = _fmt_dt(entries[0]["dt"])
    last_ts = _fmt_dt(entries[-1]["dt"])
    if not problems:
        sources = ", ".join(sorted({e["source"] for e in entries})[:5])
        return (
            f"- Namespace {ns_label}, {window}: no WARN/ERROR/FATAL pattern found in {len(entries)} logs ({first_ts} to {last_ts}).\n"
            f"- Sources seen: {sources or 'none'}.\n"
            "- Next check: ask for pod status or restarts if you want a metrics-based health view instead of log errors."
        )

    if strict_errors and not error_entries:
        return (
            f"- Namespace {ns_label}, {window}: no ERROR/FATAL logs found in {len(entries)} logs scanned ({first_ts} to {last_ts}).\n"
            "- No error pattern was detected for the requested namespace and time window.\n"
            "- Next check: inspect pod restarts/events if you expected actual errors, or ask separately for WARN-level patterns."
        )

    pattern_entries = error_entries if (strict_errors or error_entries) else problems
    level_counts = Counter(e["level"] for e in pattern_entries)
    pattern_counts = Counter(_signature(e["text"]) for e in pattern_entries)
    top_patterns = pattern_counts.most_common(3)
    top_signature = top_patterns[0][0]
    top_sources = sorted({e["source"] for e in pattern_entries if _signature(e["text"]) == top_signature})[:3]

    levels = ", ".join(f"{level}={count}" for level, count in sorted(level_counts.items()))
    patterns = "; ".join(f"{count}x {sig}" for sig, count in top_patterns)
    if error_entries:
        summary = f"{len(pattern_entries)} error logs out of {len(entries)} scanned"
        pattern_label = "Dominant ERROR pattern"
    else:
        summary = f"no ERROR/FATAL logs found; {len(problems)} warning logs out of {len(entries)} scanned"
        pattern_label = "Dominant WARN pattern"
    sample = next(e for e in pattern_entries if _signature(e["text"]) == top_signature)
    sample_text = sample["text"][:220]

    return (
        f"- Namespace {ns_label}, {window}: {summary} ({first_ts} to {last_ts}); levels: {levels}.\n"
        f"- {pattern_label}: {patterns}. Main source(s): {', '.join(top_sources)}.\n"
        f"- Example: {_fmt_dt(sample['dt'])} {sample['source']} -> {sample_text}. Next action: {_next_action(top_signature, top_sources)}"
    )


def namespace_sources(entries: list[dict]) -> list[dict]:
    problem_entries = [entry for entry in entries if _is_problem_entry(entry)]
    selected = (problem_entries or entries)[-20:]
    return [
        {
            "text": e["text"],
            "source": e["source"],
            "level": e["level"],
            "timestamp": e["timestamp"],
        }
        for e in selected
    ]


# ── Prometheus: live metrics path ────────────────────────────────────────────

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


# ── API Endpoints ────────────────────────────────────────────────────────────

def _ndjson(event_type: str, **fields) -> str:
    return json.dumps({"type": event_type, **fields}) + "\n"


@app.post("/api/analyze")
def analyze_logs(req: QueryRequest):
    """Streaming RAG endpoint — emits NDJSON events as the answer is generated."""

    bounded_hist = bound_history(req.history)

    def event_stream():
        try:
            queried_log_namespaces = requested_log_namespaces(req.question)
            if queried_log_namespaces and is_log_problem_question(req.question):
                queried = queried_log_namespaces
                lookback = parse_lookback(req.question) or timedelta(hours=DEFAULT_NAMESPACE_LOOKBACK_HOURS)
                strict_errors = wants_error_only_analysis(req.question)
                ns_label = ", ".join(queried)
                yield _ndjson("status", data=f"Scanning Qdrant logs for namespace {ns_label} ({_lookback_label(lookback)})...")
                entries = fetch_namespace_logs(queried, lookback)
                answer = namespace_log_analysis(
                    queried,
                    entries,
                    lookback,
                    strict_errors=strict_errors,
                )
                result_sources = namespace_sources(entries)
                if strict_errors:
                    result_sources = namespace_sources([e for e in entries if e["level"] == "ERROR"])
                yield _ndjson("token", data=answer)
                yield _ndjson(
                    "done",
                    sources=result_sources,
                    num_chunks_used=len(result_sources) if strict_errors else len(entries),
                )
                return

            if is_metrics_question(req.question):
                queried = select_metrics_namespaces(req.question, METRICS_NAMESPACES)
                ns_label = ", ".join(queried)
                yield _ndjson("status", data=f"Querying Prometheus for live metrics ({ns_label})...")
                metrics_text = metrics_snapshot(queried)
                if USE_LLM:
                    yield _ndjson("status", data="Generating analysis from live metrics...")
                    try:
                        answer_parts = []
                        for token in generate_analysis_stream(
                            METRICS_SYSTEM_PROMPT,
                            build_metrics_prompt(req.question, metrics_text),
                            bounded_hist,
                        ):
                            answer_parts.append(token)
                        answer = strip_thinking("".join(answer_parts))
                        if answer_is_incomplete(answer):
                            answer = fast_metrics_analysis(req.question, metrics_text)
                    except Exception:
                        answer = fast_metrics_analysis(req.question, metrics_text)
                    yield _ndjson("token", data=answer)
                else:
                    answer = fast_metrics_analysis(req.question, metrics_text)
                    yield _ndjson("token", data=answer)
                yield _ndjson(
                    "done",
                    sources=[{"source": f"prometheus:{ns}"} for ns in queried],
                    num_chunks_used=len(queried),
                )
                return

            yield _ndjson("status", data="Retrieving relevant log chunks...")
            query_vector = embed_text(req.question)
            lookback = parse_lookback(req.question)
            log_chunks = search_logs(query_vector, req.top_k, lookback)

            if not log_chunks:
                if lookback:
                    yield _ndjson("token", data=f"No relevant log entries found in the {_lookback_label(lookback)}. Check whether live log ingestion is running for the namespace/source you asked about.")
                else:
                    yield _ndjson("token", data="No relevant log entries found. Have you run the ingestion job?")
                yield _ndjson("done", sources=[], num_chunks_used=0)
                return

            if USE_LLM:
                yield _ndjson("status", data=f"Generating analysis from {len(log_chunks)} retrieved chunks...")
                try:
                    answer_parts = []
                    for token in generate_analysis_stream(
                        SYSTEM_PROMPT,
                        build_prompt(req.question, log_chunks),
                        bounded_hist,
                    ):
                        answer_parts.append(token)
                    answer = strip_thinking("".join(answer_parts))
                    if answer_is_incomplete(answer):
                        answer = fast_log_analysis(req.question, log_chunks)
                except Exception:
                    answer = fast_log_analysis(req.question, log_chunks)
                yield _ndjson("token", data=answer)
            else:
                yield _ndjson("status", data=f"Summarizing {len(log_chunks)} retrieved chunks...")
                answer = fast_log_analysis(req.question, log_chunks)
                yield _ndjson("token", data=answer)

            yield _ndjson("done", sources=log_chunks, num_chunks_used=len(log_chunks))
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            yield _ndjson("error", data=f"{type(e).__name__}: {e} - {body}")
        except httpx.RequestError as e:
            yield _ndjson("error", data=f"{type(e).__name__}: {e} (Request URL: {e.request.url})")
        except Exception as e:
            traceback.print_exc()
            yield _ndjson("error", data=f"{type(e).__name__}: {e}")

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ── Live log ingestion (Fluent Bit / Vector / similar) ──────────────────────

VECTOR_DIM = 384
_collection_ready = False
_LEVEL_RE = re.compile(r"\b(FATAL|ERROR|WARN|WARNING|INFO|DEBUG)\b", re.IGNORECASE)


def ensure_collection() -> None:
    global _collection_ready
    if _collection_ready:
        return
    r = client.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5.0)
    if r.status_code != 200:
        client.put(
            f"{QDRANT_URL}/collections/{COLLECTION}",
            json={"vectors": {"size": VECTOR_DIM, "distance": "Cosine"}},
            timeout=10.0,
        )
    _collection_ready = True


def parse_shipper_record(rec: dict) -> dict | None:
    """Normalize a Fluent Bit / Vector record into our log payload shape."""
    text = rec.get("log") or rec.get("message") or rec.get("text")
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) > 2000:
        text = text[:2000] + "...[truncated]"

    k8s = rec.get("kubernetes") or {}
    ns = k8s.get("namespace_name", "")
    pod = k8s.get("pod_name", "")
    container = k8s.get("container_name", "")
    if ns or pod:
        source = "/".join(p for p in (ns, pod, container) if p)
    else:
        source = rec.get("source") or "live-stream"

    level = "INFO"
    m = _LEVEL_RE.search(text)
    if m:
        lvl = m.group(1).upper()
        level = "WARN" if lvl == "WARNING" else ("ERROR" if lvl == "FATAL" else lvl)

    return {
        "text": text,
        "source": source,
        "namespace": ns,
        "pod": pod,
        "container": container,
        "level": level,
        "timestamp": rec.get("@timestamp") or rec.get("time") or rec.get("date") or "",
    }


@app.post("/api/ingest")
def ingest(records: list[dict] = Body(...)):
    """Receive a batch of log records, embed them, upsert to Qdrant.

    Designed for Fluent Bit's `http` output (format: json) — the body is a
    JSON array of records. Returns a small JSON status so the shipper can
    confirm acceptance.
    """
    ensure_collection()
    parsed = [p for p in (parse_shipper_record(r) for r in records) if p]
    if not parsed:
        return {"ingested": 0, "skipped": len(records)}

    texts = [p["text"] for p in parsed]
    resp = client.post(f"{EMBED_URL}/embed", json={"texts": texts}, timeout=30.0)
    resp.raise_for_status()
    vectors = resp.json()["embeddings"]

    points = [
        {"id": str(uuid.uuid4()), "vector": v, "payload": p}
        for p, v in zip(parsed, vectors)
    ]
    r = client.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json={"points": points},
        timeout=30.0,
    )
    r.raise_for_status()
    return {"ingested": len(points), "skipped": len(records) - len(points)}


@app.get("/health")
def health():
    """Health check — verifies connectivity to all backends."""
    status = {"rag_app": "ok", "mode": "llm" if USE_LLM else "fast_rag"}
    checks = [
        ("qwen3", f"{QWEN3_URL}/v1/models"),
        ("embed", f"{EMBED_URL}/health"),
        ("qdrant", f"{QDRANT_URL}/healthz"),
        ("prometheus", f"{PROMETHEUS_URL}/-/ready"),
    ]
    for name, url in checks:
        try:
            r = client.get(url, timeout=5.0)
            status[name] = "ok" if r.status_code == 200 else f"status:{r.status_code}"
        except Exception as e:
            status[name] = f"error: {type(e).__name__}"
    return status


# ── Chat UI ──────────────────────────────────────────────────────────────────

_MODEL_NAME_CACHE: str | None = None


def get_model_name() -> str:
    """Fetch the served model id from qwen3-server and cache it.

    llama.cpp returns the model file path (e.g. /models/Qwen3-4B-Q4_K_M.gguf);
    we strip the directory and the .gguf suffix so the badge stays compact.
    """
    global _MODEL_NAME_CACHE
    if _MODEL_NAME_CACHE:
        return _MODEL_NAME_CACHE
    try:
        r = client.get(f"{QWEN3_URL}/v1/models", timeout=5.0)
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            mid = str(data[0].get("id", ""))
            name = os.path.basename(mid)
            if name.endswith(".gguf"):
                name = name[: -len(".gguf")]
            if name:
                _MODEL_NAME_CACHE = name
                return name
    except Exception:
        pass
    return "model unavailable"


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Log Analysis — __MODEL_NAME__</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f1117;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{padding:16px 24px;border-bottom:1px solid #1e2130;display:flex;align-items:center;gap:12px}
header h1{font-size:18px;font-weight:500;color:#fff}
header span{font-size:12px;background:#2a4a3a;color:#4ade80;padding:2px 10px;border-radius:99px}
.chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:720px;padding:14px 18px;border-radius:12px;font-size:14px;line-height:1.7;white-space:pre-wrap}
.msg.user{background:#1e3a5f;align-self:flex-end;color:#bfdbfe}
.msg.bot{background:#1e2130;align-self:flex-start;border:1px solid #2a2d3a}
.msg.bot .sources{margin-top:12px;padding-top:10px;border-top:1px solid #2a2d3a;font-size:12px;color:#888}
.input-bar{padding:16px 72px 16px 24px;border-top:1px solid #1e2130;display:flex;gap:10px}
.input-bar input{flex:1;background:#1a1d2e;border:1px solid #2a2d3a;color:#fff;padding:12px 16px;border-radius:10px;font-size:14px;outline:none}
.input-bar input:focus{border-color:#3b82f6}
.input-bar button{background:#3b82f6;color:#fff;border:none;padding:12px 24px;border-radius:10px;cursor:pointer;font-size:14px;font-weight:500;min-width:118px}
.input-bar button:disabled{opacity:.4;cursor:not-allowed}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #444;border-top-color:#3b82f6;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header>
  <h1>Log Analysis Platform</h1>
  <span>__MODEL_NAME__</span>
  <button id="clearBtn" style="margin-left:auto;background:#2a2d3a;color:#888;border:1px solid #3a3d4a;padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px">Clear</button>
</header>
<div class="chat" id="chat">
  <div class="msg bot">Ready. Ask me about your logs — errors, patterns, root causes, correlations.<br><br>
Examples:<br>• What errors keep recurring?<br>• Why are pods crashlooping?<br>• Summarize the NGINX 5xx errors<br>• What happened around 03:14 UTC?</div>
</div>
<form class="input-bar" id="form" autocomplete="off">
  <input id="q" placeholder="Ask about your logs..." autofocus>
  <button id="btn" type="submit">Analyze</button>
</form>
<script>
const chat=document.getElementById('chat'),form=document.getElementById('form'),q=document.getElementById('q'),btn=document.getElementById('btn');
const STORAGE_KEY='log_chat_history';
const MAX_HISTORY=50;
let busy=false;

function loadHistory(){
  try{
    const raw=localStorage.getItem(STORAGE_KEY);
    if(!raw) return;
    const history=JSON.parse(raw);
    if(!Array.isArray(history)||!history.length) return;
    chat.innerHTML='';
    for(const item of history){
      chat.innerHTML+=`<div class="msg user">${esc(item.q)}</div>`;
      const bubble=document.createElement('div');
      bubble.className='msg bot';
      const body=document.createElement('span');
      body.textContent=item.a;
      bubble.appendChild(body);
      if(item.src){
        const src=document.createElement('div');
        src.className='sources';
        src.textContent=item.src;
        bubble.appendChild(src);
      }
      chat.appendChild(bubble);
    }
    chat.scrollTop=chat.scrollHeight;
  }catch(_){}
}

function saveHistory(){
  const msgs=Array.from(chat.children);
  const history=[];
  let lastUser=null;
  for(const el of msgs){
    if(el.classList.contains('user')){
      lastUser=el.textContent;
    }else if(el.classList.contains('bot')&&lastUser){
      const body=el.querySelector('span')||el;
      const srcEl=el.querySelector('.sources');
      history.push({q:lastUser,a:body.textContent,src:srcEl?srcEl.textContent:''});
      lastUser=null;
    }
  }
  if(history.length>MAX_HISTORY) history.splice(0,history.length-MAX_HISTORY);
  localStorage.setItem(STORAGE_KEY,JSON.stringify(history));
}

// Walk the visible chat to build the {role, content} list the server expects.
// Only completed user→bot pairs count; the just-appended in-progress user msg
// has no bot pair yet, so it's correctly excluded.
function buildHistoryPayload(){
  const out=[];
  let pendingUser=null;
  for(const el of chat.children){
    if(el.classList.contains('user')){
      pendingUser=el.textContent;
    }else if(el.classList.contains('bot')&&pendingUser){
      const body=el.querySelector('span')||el;
      const answer=body.textContent.trim();
      if(answer){
        out.push({role:'user',content:pendingUser});
        out.push({role:'assistant',content:answer});
      }
      pendingUser=null;
    }
  }
  return out;
}

form.addEventListener('submit',send);
btn.addEventListener('click',send);
q.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){
    e.preventDefault();
    if(form.requestSubmit) form.requestSubmit(); else send(e);
  }
});
async function send(e){
  if(e) e.preventDefault();
  if(busy)return;
  const text=q.value.trim(); if(!text)return;
  busy=true; q.value=''; btn.disabled=true;
  chat.innerHTML+=`<div class="msg user">${esc(text)}</div>`;
  const bubble=document.createElement('div');
  bubble.className='msg bot';
  const body=document.createElement('span');
  let status=document.createElement('span');
  status.innerHTML='<div class="spinner"></div> Retrieving logs...';
  bubble.appendChild(body); bubble.appendChild(status);
  chat.appendChild(bubble); chat.scrollTop=chat.scrollHeight;
  let answer='';
  let sourcesText='';
  try{
    const history=buildHistoryPayload();
    const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text,history})});
    if(!r.ok){
      let detail=''; try{ detail=(await r.json()).detail||''; }catch(_){}
      throw new Error(detail||('HTTP '+r.status));
    }
    const reader=r.body.getReader(), decoder=new TextDecoder();
    let buf='';
    while(true){
      const {value,done}=await reader.read();
      if(done) break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\\n'); buf=lines.pop();
      for(const line of lines){
        if(!line.trim()) continue;
        let evt; try{ evt=JSON.parse(line); }catch(_){ continue; }
        if(evt.type==='token'){
          if(status){ status.remove(); status=null; }
          answer+=evt.data;
          body.textContent=answer;
        } else if(evt.type==='status'){
          status.textContent=evt.data;
        } else if(evt.type==='done'){
          if(evt.sources&&evt.sources.length){
            const src=document.createElement('div');
            src.className='sources';
            sourcesText=`Based on ${evt.num_chunks_used} log chunks from: ${[...new Set(evt.sources.map(s=>s.source))].join(', ')}`;
            src.textContent=sourcesText;
            bubble.appendChild(src);
          }
        } else if(evt.type==='error'){
          bubble.innerHTML='Error: '+esc(evt.data);
        }
      }
      chat.scrollTop=chat.scrollHeight;
    }
  }catch(e){bubble.innerHTML='Error: '+esc(e.message)}
  busy=false; btn.disabled=false; q.focus(); chat.scrollTop=chat.scrollHeight;
  saveHistory();
}
window.send=send;
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
loadHistory();

document.getElementById('clearBtn').addEventListener('click',()=>{
  if(!confirm('Clear chat history?')) return;
  localStorage.removeItem(STORAGE_KEY);
  chat.innerHTML=`<div class="msg bot">Ready. Ask me about your logs — errors, patterns, root causes, correlations.<br><br>Examples:<br>• What errors keep recurring?<br>• Why are pods crashlooping?<br>• Summarize the NGINX 5xx errors<br>• What happened around 03:14 UTC?</div>`;
});
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    html = CHAT_HTML.replace("__MODEL_NAME__", get_model_name())
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )
