"""Bounded acquisition of frozen claim sources; no graph-state mutation."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import CoroutineType

from opentelemetry import context as otel_context, trace
from opentelemetry.context import Context

from app.agent.claim_sources import (
    CONTEXT_FIELDS,
    TOOL_FIELDS,
    MAX_SOURCE_BYTES,
    _json_copy,
    _validate_context,
    freeze_claim_sources,
)
from app.rag.evidence_reranker import rerank_evidence
from app.observability.tracing import safe_span, set_safe_outcome


WAIT_SECONDS = 5.0
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="claim-retrieval")
_SLOTS = threading.BoundedSemaphore(2)
_INPUT_FIELDS = tuple(dict.fromkeys((*CONTEXT_FIELDS, *TOOL_FIELDS)))


def _failure(reason: str) -> dict:
    return {"sources": None, "source_error": reason}


def _worker(state: dict) -> dict:
    """The entire retrieval coroutine, including synchronous work, runs here."""
    try:
        with safe_span("robotops.retrieval.rerank") as span:
            result = asyncio.run(rerank_evidence(
                query=state["user_query"], doc_type=None, top_k=10,
            ))
            set_safe_outcome(span, "COMPLETED")
    except BaseException:
        # Retrieval-side cancellation is a call failure, not caller cancellation.
        return _failure("RETRIEVAL_ERROR")
    if type(result) is CoroutineType:
        try:
            result.close()
        except BaseException:
            pass
        return _failure("SOURCE_INVALID")
    try:
        sources = freeze_claim_sources(state, result)
    except BaseException:
        return _failure("SOURCE_INVALID")
    return {"sources": sources, "source_error": "NONE"}


def _wait_parent_context() -> Context | None:
    """Transfer only the active OTel parent, never the request or baggage."""
    try:
        parent = trace.get_current_span().get_span_context()
        if not parent.is_valid:
            return None
        return trace.set_span_in_context(trace.NonRecordingSpan(parent), Context())
    except Exception:
        return None


def _worker_with_parent(state: dict, parent: Context) -> dict:
    try:
        token = otel_context.attach(parent)
    except Exception:
        return _worker(state)
    try:
        return _worker(state)
    finally:
        try:
            otel_context.detach(token)
        except Exception:
            pass


async def collect_claim_sources(state: object) -> dict:
    """Copy bounded inputs, admit once, and discard all late worker results."""
    try:
        if type(state) is not dict or any(type(key) is not str for key in state):
            return _failure("SOURCE_INVALID")
        detached = _json_copy(
            {key: state[key] for key in _INPUT_FIELDS}, maximum=MAX_SOURCE_BYTES,
        )
        _validate_context({key: detached[key] for key in CONTEXT_FIELDS})
    except Exception:
        return _failure("SOURCE_INVALID")
    if not _SLOTS.acquire(blocking=False):
        return _failure("RETRIEVAL_BUSY")
    try:
        parent = _wait_parent_context()
        future = (
            _EXECUTOR.submit(_worker, detached)
            if parent is None
            else _EXECUTOR.submit(_worker_with_parent, detached, parent)
        )
    except BaseException:
        _SLOTS.release()
        return _failure("RETRIEVAL_BUSY")
    # Only completion of the real thread releases capacity. Waiting never cancels
    # it, enqueues a replacement, or gives its late value access to graph state.
    future.add_done_callback(lambda _finished: _SLOTS.release())
    waiter = asyncio.wrap_future(future)
    done, _pending = await asyncio.wait({waiter}, timeout=WAIT_SECONDS)
    if not done:
        return _failure("RETRIEVAL_TIMEOUT")
    return waiter.result()
