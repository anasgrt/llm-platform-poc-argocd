"""FastAPI routes for the log analysis platform."""

import json
from datetime import timedelta
import traceback

import httpx
from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import (
    DEFAULT_NAMESPACE_LOOKBACK_HOURS,
    EMBED_URL,
    METRICS_NAMESPACES,
    METRICS_SYSTEM_PROMPT,
    QDRANT_URL,
    QWEN3_URL,
    PROMETHEUS_URL,
    SYSTEM_PROMPT,
    USE_LLM,
    client,
)
from .ingestion import ingest_records
from .llm import answer_is_incomplete, generate_analysis_stream, strip_thinking
from .logs import (
    _lookback_label,
    build_prompt,
    embed_text,
    fast_log_analysis,
    fetch_namespace_logs,
    is_log_problem_question,
    namespace_log_analysis,
    namespace_sources,
    parse_lookback,
    requested_log_namespaces,
    search_logs,
    wants_error_only_analysis,
)
from .metrics import (
    build_metrics_prompt,
    fast_metrics_analysis,
    is_metrics_question,
    metrics_snapshot,
    select_metrics_namespaces,
)
from .models import QueryRequest, bound_history
from .ui import render_ui_html

app = FastAPI(title="Log Analysis Platform")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"{type(exc).__name__}: {str(exc)}"
    if isinstance(exc, httpx.RequestError):
        error_msg += f" (Request URL: {exc.request.url})"
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            error_msg += f" - Response: {exc.response.text}"
        except Exception:
            pass
    print(f"500 Internal Server Error: {error_msg}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": error_msg})


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


@app.post("/api/ingest")
def ingest(records: list[dict] = Body(...)):
    """Receive a batch of log records, embed them, and upsert them to Qdrant."""
    return ingest_records(records)


@app.get("/health")
def health():
    """Lightweight liveness/readiness probe — never blocks on downstream services."""
    return {"rag_app": "ok", "mode": "llm" if USE_LLM else "fast_rag"}


@app.get("/health/full")
def health_full():
    """Deep health check — verifies connectivity to all backends."""
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


@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse(
        render_ui_html(),
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )
