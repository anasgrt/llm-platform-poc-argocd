"""Compatibility exports for log analysis modules."""

from .log_analysis import (
    _lookback_label,
    fast_log_analysis,
    namespace_log_analysis,
    namespace_sources,
)
from .log_questions import (
    _dedupe,
    extract_namespaces,
    is_log_problem_question,
    parse_lookback,
    wants_error_only_analysis,
)
from .log_retrieval import (
    available_log_namespaces,
    build_prompt,
    discover_log_namespaces,
    discover_prometheus_namespaces,
    embed_text,
    fetch_namespace_logs,
    requested_log_namespaces,
    scroll_log_payloads,
    search_logs,
)


def is_namespace_log_pattern_question(question: str) -> bool:
    namespaces = requested_log_namespaces(question)
    if not namespaces:
        return False
    return is_log_problem_question(question)
