"""Fail-open-for-business, allowlisted OpenTelemetry spans.

This module does not configure an exporter by default. In particular it must
not capture prompt, result, exception text, paths, or identity tokens.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import re
from threading import Lock
from typing import Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode


_SPAN_NAMES = frozenset({
    "robotops.run.initial", "robotops.run.resume", "robotops.graph.invoke",
    "robotops.tool.execute",
    "robotops.node.risk",
    "robotops.node.process_verifier",
    "robotops.retrieval.lookup",
    "robotops.retrieval.wait",
    "robotops.retrieval.rerank",
    "robotops.node.draft",
    "robotops.node.verification",
    "robotops.persistence.finalize",
    "robotops.effect.readback",
    "robotops.effect.execute",
    "robotops.approval.consumed",
    "robotops.approval.issued",
    "robotops.approval.expired",
    "robotops.approval.cancelled",
    "robotops.llm.planner",
    "robotops.llm.tool_decision",
    "robotops.llm.generate",
    "robotops.llm.parse",
    "robotops.llm.identify",
    "robotops.llm.classify",
    "robotops.llm.ask",
    "robotops.llm.claim_semantics",
    "robotops.llm.request",
    "robotops.session.load",
    "robotops.checkpoint.get_tuple",
    "robotops.checkpoint.put",
    "robotops.checkpoint.put_writes",
    "robotops.checkpoint.restore",
    "robotops.persistence.run_state",
    "robotops.persistence.orphan_state",
    "robotops.ticket.draft_committed",
})
_OUTCOMES = frozenset({
    "QUEUED", "RUNNING", "INTERRUPTED", "CANCEL_REQUESTED",
    "COMPLETED", "CANCELLED", "FAILED", "ERROR", "UNKNOWN",
})
_SERVER_RUN_ID = re.compile(r"[A-Za-z0-9_-]{32}\Z")
_CORRELATION_DOMAIN = b"robotops.run-correlation.v1\0"
_READONLY_TOOL_NAMES = frozenset({
    "search_manual", "search_fault_cases", "query_device_information",
    "query_simulated_telemetry", "analyze_synthetic_ros_log",
})
_RETRIEVAL_WAIT_RESULTS = {
    "NONE": "READY",
    "RETRIEVAL_TIMEOUT": "TIMEOUT",
    "RETRIEVAL_BUSY": "BUSY",
}
_DRAFT_KINDS = frozenset({"DIAGNOSTIC", "SAFETY"})
_VERIFICATION_STATUSES = frozenset({
    "VERIFIED", "UNVERIFIED", "SKIPPED_SAFETY", "SKIPPED_FOLLOW_UP",
})
_PROCESS_VERIFICATION_STATUSES = frozenset({"PASSED", "FAILED", "ERROR"})
_PERSISTED_EVENTS = frozenset({
    "RUN_QUEUED", "RUN_RETRY_QUEUED", "RUN_RUNNING", "RUN_INTERRUPTED",
    "RUN_RESUMED", "RUN_FAILED", "RUN_CANCEL_REQUESTED", "RUN_CANCELLED",
    "RUN_COMPLETED",
})
_APPROVAL_DECISIONS = frozenset({"APPROVE", "REJECT"})
_PRIVATE_PROVIDER = None
_PRIVATE_PROVIDER_LOCK = Lock()


def _install_private_provider(provider: object) -> None:
    """Route only this application's explicit spans to a local provider."""
    global _PRIVATE_PROVIDER
    with _PRIVATE_PROVIDER_LOCK:
        if _PRIVATE_PROVIDER is not None:
            raise RuntimeError("LOCAL_OTLP_ALREADY_CONFIGURED")
        _PRIVATE_PROVIDER = provider


def _clear_private_provider(provider: object) -> None:
    global _PRIVATE_PROVIDER
    with _PRIVATE_PROVIDER_LOCK:
        if _PRIVATE_PROVIDER is provider:
            _PRIVATE_PROVIDER = None


def _get_tracer():
    if _PRIVATE_PROVIDER is not None:
        return _PRIVATE_PROVIDER.get_tracer("robotops.diagnostic")
    return trace.get_tracer("robotops.diagnostic")


def set_safe_outcome(span: Span | None, outcome: str) -> None:
    if span is None or outcome not in _OUTCOMES:
        return
    try:
        span.set_attribute("robotops.outcome", outcome)
        if outcome in {"FAILED", "ERROR"}:
            span.set_status(Status(StatusCode.ERROR))
    except Exception:
        # Telemetry is deliberately never allowed to change the diagnosis.
        pass


def set_safe_run_correlation(span: Span | None, run_id: str) -> None:
    """Add only a pseudonymous, stable run lookup key to a real root span."""
    if span is None or type(run_id) is not str or _SERVER_RUN_ID.fullmatch(run_id) is None:
        return
    try:
        key = hashlib.sha256(_CORRELATION_DOMAIN + run_id.encode("ascii")).hexdigest()[:32]
        span.set_attribute("robotops.run_correlation", key)
    except Exception:
        # Even a broken observer must not change service-authoritative execution.
        pass


def set_safe_tool_name(span: Span | None, tool_name: str) -> None:
    """Record only an exact read-only registry name, never caller-supplied text."""
    if span is None or type(tool_name) is not str or tool_name not in _READONLY_TOOL_NAMES:
        return
    try:
        span.set_attribute("robotops.tool_name", tool_name)
    except Exception:
        pass


def set_safe_handoff_required(span: Span | None, required: bool) -> None:
    """Record only the deterministic gate's strict Boolean decision."""
    if span is None or type(required) is not bool:
        return
    try:
        span.set_attribute("robotops.handoff_required", required)
    except Exception:
        pass


def set_safe_retrieval_wait_result(span: Span | None, source_error: object) -> None:
    """Map a caller-wait result to a closed vocabulary; never record source data."""
    if span is None:
        return
    result = (
        _RETRIEVAL_WAIT_RESULTS.get(source_error, "FAILED")
        if type(source_error) is str else "FAILED"
    )
    try:
        span.set_attribute("robotops.retrieval_wait_result", result)
    except Exception:
        pass


def set_safe_draft_kind(span: Span | None, draft_kind: object) -> None:
    """Record only the validated draft category, never its content."""
    if span is None or type(draft_kind) is not str or draft_kind not in _DRAFT_KINDS:
        return
    try:
        span.set_attribute("robotops.draft_kind", draft_kind)
    except Exception:
        pass


def set_safe_verification_status(span: Span | None, status: object) -> None:
    """Record only an accepted verification state, never claims or evidence."""
    if span is None or type(status) is not str or status not in _VERIFICATION_STATUSES:
        return
    try:
        span.set_attribute("robotops.verification_status", status)
    except Exception:
        pass


def set_safe_process_verification_status(span: Span | None, status: object) -> None:
    """Separate deterministic process results from claim-semantic results."""
    if span is None or type(status) is not str or status not in _PROCESS_VERIFICATION_STATUSES:
        return
    try:
        span.set_attribute("robotops.process_verification_status", status)
    except Exception:
        pass


def set_safe_persisted_event(span: Span | None, event_type: object) -> None:
    """Record only a transaction-returned run event category, never its keys."""
    if span is None or type(event_type) is not str or event_type not in _PERSISTED_EVENTS:
        return
    try:
        span.set_attribute("robotops.persisted_event", event_type)
    except Exception:
        pass


def set_safe_readback_matched(span: Span | None, matched: object) -> None:
    """Record only the strict Boolean comparison, not the device state."""
    if span is None or type(matched) is not bool:
        return
    try:
        span.set_attribute("robotops.readback_matched", matched)
    except Exception:
        pass


def set_safe_approval_decision(span: Span | None, decision: object) -> None:
    """Expose only a committed V1/V2 decision, never its approval identity."""
    if span is None or type(decision) is not str or decision not in _APPROVAL_DECISIONS:
        return
    try:
        span.set_attribute("robotops.approval_decision", decision)
    except Exception:
        pass


def set_safe_llm_attempt(span: Span | None, attempt: object) -> None:
    """Record only a bounded one-based ordinal for a real SDK request."""
    if span is None or type(attempt) is not int or not 1 <= attempt <= 1_000_000:
        return
    try:
        span.set_attribute("robotops.attempt", attempt)
    except Exception:
        pass


def set_safe_provider_usage(span: Span | None, response: object | None) -> None:
    """Trust only complete, consistent int64 usage on a returned provider response."""
    if span is None:
        return
    try:
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens")
        output_tokens = getattr(usage, "completion_tokens")
        total_tokens = getattr(usage, "total_tokens")
        known = all(
            type(value) is int and 0 <= value <= (2**63 - 1)
            for value in (input_tokens, output_tokens, total_tokens)
        ) and total_tokens >= input_tokens + output_tokens
    except Exception:
        known = False
    if not known:
        try:
            span.set_attribute("robotops.usage_status", "UNKNOWN")
            span.set_attribute("robotops.cache_status", "UNKNOWN")
        except Exception:
            pass
        return
    try:
        span.set_attribute("robotops.input_tokens", input_tokens)
        span.set_attribute("robotops.output_tokens", output_tokens)
        span.set_attribute("robotops.total_tokens", total_tokens)
        span.set_attribute("robotops.usage_source", "PROVIDER")
        span.set_attribute("robotops.usage_status", "KNOWN")
    except Exception:
        # A broken exporter must not change the provider call or its response.
        pass
    try:
        cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens")
        cache_miss_tokens = getattr(usage, "prompt_cache_miss_tokens")
        cache_known = all(
            type(value) is int and 0 <= value <= (2**63 - 1)
            for value in (cache_hit_tokens, cache_miss_tokens)
        ) and cache_hit_tokens + cache_miss_tokens == input_tokens
    except Exception:
        cache_known = False
    if not cache_known:
        try:
            span.set_attribute("robotops.cache_status", "UNKNOWN")
        except Exception:
            pass
        return
    try:
        span.set_attribute("robotops.cache_hit_tokens", cache_hit_tokens)
        span.set_attribute("robotops.cache_miss_tokens", cache_miss_tokens)
        span.set_attribute("robotops.cache_source", "PROVIDER")
        span.set_attribute("robotops.cache_status", "KNOWN")
    except Exception:
        pass


@contextmanager
def safe_span(name: str) -> Iterator[Span | None]:
    """Start a fixed-name span without automatic exception text or stack traces."""
    if name not in _SPAN_NAMES:
        raise ValueError("unsupported span name")
    try:
        manager = _get_tracer().start_as_current_span(
            name, record_exception=False, set_status_on_exception=False,
        )
        span = manager.__enter__()
    except Exception:
        yield None
        return
    try:
        yield span
    except BaseException:
        set_safe_outcome(span, "ERROR")
        raise
    finally:
        try:
            manager.__exit__(None, None, None)
        except Exception:
            pass
