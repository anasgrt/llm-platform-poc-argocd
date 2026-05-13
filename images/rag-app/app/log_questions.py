"""Question parsing helpers for log analysis."""

import re
from datetime import timedelta

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
