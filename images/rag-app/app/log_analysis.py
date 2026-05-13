"""Deterministic log analysis and response shaping."""

import re
from collections import Counter
from datetime import datetime, timedelta


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
