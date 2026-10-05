"""MOD-025B traceable rerank over the frozen hybrid retrieval snapshot."""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import unicodedata
from typing import Any, Protocol

from app.rag import hybrid_retriever
from app.schemas.evidence_search import EvidenceSearchResponse


SCHEMA_VERSION = 1
RERANK_PROFILE = "robotops_evidence_rerank_v1"
TRUST_BOUNDARY = "UNTRUSTED_DATA"
CANDIDATE_K = 20
RERANK_TIMEOUT_MS = 1_000
DEFAULT_MODEL = "robotops-token-overlap-reranker"
DEFAULT_VERSION = "1.0.0"
_SAFE_DESCRIPTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~/-]{0,127}$")


class Reranker(Protocol):
    """A non-blocking async scorer; identity and evidence remain server-owned."""

    model: str
    version: str

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...


class TokenOverlapReranker:
    """Offline deterministic baseline frozen by the MOD-025B profile."""

    model = DEFAULT_MODEL
    version = DEFAULT_VERSION

    @staticmethod
    def _rerank_sync(
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        query_tokens = list(dict.fromkeys(hybrid_retriever.analyze(query)))
        query_token_set = set(query_tokens)
        normalized_query = unicodedata.normalize("NFKC", query).lower()
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            if query_tokens:
                document_tokens = set(hybrid_retriever.analyze(candidate["text"]))
                coverage = len(query_token_set.intersection(document_tokens)) / len(
                    query_tokens
                )
                normalized_text = unicodedata.normalize(
                    "NFKC", candidate["text"]
                ).lower()
                exact_bonus = 0.25 if normalized_query in normalized_text else 0.0
                score = float(coverage + exact_bonus)
            else:
                score = 0.0
            results.append({"chunk_id": candidate["chunk_id"], "score": score})
        return results

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._rerank_sync, query, candidates)


DEFAULT_RERANKER: Reranker = TokenOverlapReranker()


class _InvalidRerankResult(ValueError):
    pass


class _RerankDeadlineExceeded(RuntimeError):
    pass


class _RerankAdapterCancelled(RuntimeError):
    pass


class RetrievalUnavailableError(RuntimeError):
    """The hybrid retrieval call itself did not complete."""


class InvalidRetrievalResultError(ValueError):
    """The hybrid retrieval adapter returned a non-contract result."""


def _canonical_sha256(value: Any) -> str:
    return hybrid_retriever.canonical_sha256(value)


def _safe_descriptor(engine: Reranker) -> tuple[str | None, str | None]:
    try:
        model = engine.model
        version = engine.version
    except Exception as exc:  # pragma: no cover - exact exception is untrusted
        raise _InvalidRerankResult("descriptor unavailable") from exc
    if not isinstance(model, str) or not isinstance(version, str):
        raise _InvalidRerankResult("descriptor invalid")
    if any(
        _SAFE_DESCRIPTOR_RE.fullmatch(value) is None
        or any(segment in {"", ".", ".."} for segment in value.split("/"))
        for value in (model, version)
    ):
        raise _InvalidRerankResult("descriptor invalid")
    return model, version


def _retrieval_model(raw: object) -> EvidenceSearchResponse:
    result = EvidenceSearchResponse.model_validate(raw)
    if result.top_k != CANDIDATE_K or result.candidate_k != CANDIDATE_K:
        raise ValueError("retrieval snapshot does not use the frozen candidate window")
    return result


def _input_projection(
    retrieval: EvidenceSearchResponse,
) -> tuple[list[dict[str, Any]], list[str], str]:
    items: list[dict[str, Any]] = []
    for fused_rank, result in enumerate(retrieval.results, start=1):
        items.append(
            {
                "chunk_id": result.chunk_id,
                "content_sha256": result.content_sha256,
                "identity_sha256": result.identity_sha256,
                "fused_rank": fused_rank,
                "fused_score": result.fused_score,
            }
        )
    input_order = [item["chunk_id"] for item in items]
    digest = _canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "query_sha256": retrieval.query_sha256,
            "retrieval_profile": retrieval.retrieval_profile,
            "retrieval_results_sha256": retrieval.results_sha256,
            "candidates": items,
        }
    )
    return items, input_order, digest


def _adapter_candidates(
    retrieval: EvidenceSearchResponse,
) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item.chunk_id,
            "text": item.text,
            "content_sha256": item.content_sha256,
            "identity_sha256": item.identity_sha256,
            "fused_rank": rank,
            "fused_score": item.fused_score,
        }
        for rank, item in enumerate(retrieval.results, start=1)
    ]


def _normalize_scores(
    raw: object,
    *,
    input_order: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    if type(raw) is not list or len(raw) != len(input_order):
        raise _InvalidRerankResult("reranker must return the full input permutation")
    input_ids = set(input_order)
    scores_by_id: dict[str, float] = {}
    for item in raw:
        if type(item) is not dict or set(item) != {"chunk_id", "score"}:
            raise _InvalidRerankResult("reranker output shape invalid")
        chunk_id = item.get("chunk_id")
        score = item.get("score")
        if not isinstance(chunk_id, str) or chunk_id not in input_ids:
            raise _InvalidRerankResult("reranker returned an unknown chunk")
        if chunk_id in scores_by_id:
            raise _InvalidRerankResult("reranker returned a duplicate chunk")
        if type(score) is not float or not math.isfinite(score):
            raise _InvalidRerankResult("reranker score must be a finite float")
        scores_by_id[chunk_id] = score
    if set(scores_by_id) != input_ids:
        raise _InvalidRerankResult("reranker omitted an input chunk")

    # The score array is canonicalized to input order, never adapter order.
    scores = [
        {"chunk_id": chunk_id, "score": scores_by_id[chunk_id]}
        for chunk_id in input_order
    ]
    fused_ranks = {chunk_id: rank for rank, chunk_id in enumerate(input_order, start=1)}
    reranked_order = sorted(
        input_order,
        key=lambda chunk_id: (
            -scores_by_id[chunk_id],
            fused_ranks[chunk_id],
            chunk_id,
        ),
    )
    return scores, reranked_order


def _discard_returned_awaitables(value: object) -> None:
    """Close/cancel awaitables returned as data without executing untrusted work."""
    pending = [value]
    seen: set[int] = set()
    inspected = 0
    while pending and inspected < 100:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        inspected += 1
        if inspect.iscoroutine(current):
            try:
                current.close()
            except BaseException:
                pass
        elif isinstance(current, asyncio.Future):
            try:
                current.cancel()
            except BaseException:
                pass
        elif type(current) is list:
            pending.extend(current)
        elif type(current) is dict:
            pending.extend(current.values())


def _discard_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except BaseException:
        # A timed-out adapter is untrusted. Its late result/exception is discarded.
        return


async def _call_with_deadline(
    engine: Reranker,
    *,
    query: str,
    candidates: list[dict[str, Any]],
) -> object:
    task = asyncio.create_task(engine.rerank(query=query, candidates=candidates))
    try:
        done, _pending = await asyncio.wait(
            {task}, timeout=RERANK_TIMEOUT_MS / 1_000
        )
    except BaseException:
        task.cancel()
        task.add_done_callback(_discard_task_result)
        raise
    if task not in done:
        task.cancel()
        task.add_done_callback(_discard_task_result)
        raise _RerankDeadlineExceeded("reranker deadline exceeded")
    try:
        return task.result()
    except asyncio.CancelledError as exc:
        raise _RerankAdapterCancelled("reranker cancelled itself") from exc


def _top_status(retrieval_status: str, rerank_status: str) -> str:
    if retrieval_status == "FAILED":
        return "FAILED"
    if retrieval_status != "HYBRID_SUCCESS" or rerank_status in {"TIMEOUT", "FAILED"}:
        return "DEGRADED"
    return "SUCCESS"


def _build_response(
    *,
    query: str,
    doc_type: str | None,
    requested_top_k: int,
    retrieval: EvidenceSearchResponse,
    rerank_status: str,
    rerank_error: str | None,
    model: str | None,
    version: str | None,
    scores: list[dict[str, Any]],
    reranked_order: list[str],
    final_mode: str,
) -> dict[str, Any]:
    retrieval_dump = retrieval.model_dump(mode="python")
    retrieval_sha256 = _canonical_sha256(retrieval_dump)
    _input_items, input_order, input_sha256 = _input_projection(retrieval)
    output_sha256 = _canonical_sha256(
        {"scores": scores, "reranked_order": reranked_order}
    )

    if final_mode == "RERANKED":
        final_order = reranked_order[:requested_top_k]
        tie_break = [
            "RERANK_SCORE_DESC",
            "FUSED_RANK_ASC",
            "CHUNK_ID_ASC",
        ]
    elif final_mode == "FUSION_FALLBACK":
        final_order = input_order[:requested_top_k]
        tie_break = ["FUSED_RANK_ASC", "CHUNK_ID_ASC"]
    else:
        final_order = []
        tie_break = []
    final_order_sha256 = _canonical_sha256(final_order)

    sources = {
        item.chunk_id: item.model_dump(mode="python") for item in retrieval.results
    }
    fused_ranks = {
        chunk_id: rank for rank, chunk_id in enumerate(input_order, start=1)
    }
    score_map = {item["chunk_id"]: item["score"] for item in scores}
    results: list[dict[str, Any]] = []
    for final_rank, chunk_id in enumerate(final_order, start=1):
        source = sources[chunk_id]
        results.append(
            {
                **source,
                "fused_rank": fused_ranks[chunk_id],
                "final_rank": final_rank,
                "rerank_score": (
                    score_map[chunk_id] if final_mode == "RERANKED" else None
                ),
            }
        )
    results_sha256 = _canonical_sha256(results)

    retrieval_reasons = list(retrieval.reason_codes)
    reason_codes = retrieval_reasons + ([rerank_error] if rerank_error else [])
    top_status = _top_status(retrieval.status, rerank_status)
    request_sha256 = _canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "query_sha256": retrieval.query_sha256,
            "doc_type": doc_type,
            "requested_top_k": requested_top_k,
        }
    )
    trace_projection = {
        "schema_version": SCHEMA_VERSION,
        "rerank_profile": RERANK_PROFILE,
        "query_sha256": retrieval.query_sha256,
        "doc_type": doc_type,
        "requested_top_k": requested_top_k,
        "candidate_k": CANDIDATE_K,
        "retrieval_sha256": retrieval_sha256,
        "model": model,
        "version": version,
        "timeout_ms": RERANK_TIMEOUT_MS,
        "rerank_status": rerank_status,
        "rerank_error": rerank_error,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "final_order_sha256": final_order_sha256,
        "results_sha256": results_sha256,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "rerank_profile": RERANK_PROFILE,
        "trust_boundary": TRUST_BOUNDARY,
        "query": query,
        "query_sha256": retrieval.query_sha256,
        "request_sha256": request_sha256,
        "doc_type": doc_type,
        "requested_top_k": requested_top_k,
        "candidate_k": CANDIDATE_K,
        "status": top_status,
        "reason_codes": reason_codes,
        "retrieval": retrieval_dump,
        "retrieval_sha256": retrieval_sha256,
        "rerank": {
            "status": rerank_status,
            "error_code": rerank_error,
            "model": model,
            "version": version,
            "timeout_ms": RERANK_TIMEOUT_MS,
            "input_count": len(input_order),
            "input_order": input_order,
            "input_sha256": input_sha256,
            "output_count": len(scores),
            "scores": scores,
            "reranked_order": reranked_order,
            "output_sha256": output_sha256,
        },
        "finalization": {
            "mode": final_mode,
            "tie_break": tie_break,
            "final_order": final_order,
            "final_order_sha256": final_order_sha256,
        },
        "count": len(results),
        "results": results,
        "results_sha256": results_sha256,
        "trace_sha256": _canonical_sha256(trace_projection),
    }


async def rerank_evidence(
    *,
    query: str,
    doc_type: str | None = None,
    top_k: int = 5,
    reranker: Reranker | None = None,
) -> dict[str, Any]:
    """Retrieve 20 candidates, rerank once, and return a traceable envelope."""
    try:
        raw_retrieval = hybrid_retriever.search(
            query=query,
            doc_type=doc_type,
            top_k=CANDIDATE_K,
        )
    except Exception as exc:
        raise RetrievalUnavailableError("hybrid retrieval unavailable") from exc
    try:
        retrieval = _retrieval_model(raw_retrieval)
    except Exception as exc:
        raise InvalidRetrievalResultError("hybrid retrieval result invalid") from exc
    if retrieval.query != query or retrieval.doc_type != doc_type:
        raise InvalidRetrievalResultError(
            "retrieval snapshot is not bound to the request"
        )

    if retrieval.status == "FAILED":
        return _build_response(
            query=query,
            doc_type=doc_type,
            requested_top_k=top_k,
            retrieval=retrieval,
            rerank_status="SKIPPED_RETRIEVAL_FAILED",
            rerank_error=None,
            model=None,
            version=None,
            scores=[],
            reranked_order=[],
            final_mode="NONE",
        )
    if not retrieval.results:
        return _build_response(
            query=query,
            doc_type=doc_type,
            requested_top_k=top_k,
            retrieval=retrieval,
            rerank_status="SKIPPED_EMPTY",
            rerank_error=None,
            model=None,
            version=None,
            scores=[],
            reranked_order=[],
            final_mode="EMPTY",
        )

    engine = reranker if reranker is not None else DEFAULT_RERANKER
    try:
        model, version = _safe_descriptor(engine)
    except _InvalidRerankResult:
        return _build_response(
            query=query,
            doc_type=doc_type,
            requested_top_k=top_k,
            retrieval=retrieval,
            rerank_status="FAILED",
            rerank_error="RERANK_INVALID_RESULT",
            model=None,
            version=None,
            scores=[],
            reranked_order=[],
            final_mode="FUSION_FALLBACK",
        )

    candidates = _adapter_candidates(retrieval)
    input_order = [item["chunk_id"] for item in candidates]
    try:
        raw_scores = await _call_with_deadline(
            engine,
            query=query,
            candidates=[dict(item) for item in candidates],
        )
    except _RerankDeadlineExceeded:
        rerank_status = "TIMEOUT"
        rerank_error = "RERANK_TIMEOUT"
        scores: list[dict[str, Any]] = []
        reranked_order: list[str] = []
        final_mode = "FUSION_FALLBACK"
    except Exception:
        rerank_status = "FAILED"
        rerank_error = "RERANK_FAILED"
        scores = []
        reranked_order = []
        final_mode = "FUSION_FALLBACK"
    else:
        try:
            scores, reranked_order = _normalize_scores(
                raw_scores,
                input_order=input_order,
            )
        except Exception:
            _discard_returned_awaitables(raw_scores)
            rerank_status = "FAILED"
            rerank_error = "RERANK_INVALID_RESULT"
            scores = []
            reranked_order = []
            final_mode = "FUSION_FALLBACK"
        else:
            rerank_status = "SUCCESS"
            rerank_error = None
            final_mode = "RERANKED"

    return _build_response(
        query=query,
        doc_type=doc_type,
        requested_top_k=top_k,
        retrieval=retrieval,
        rerank_status=rerank_status,
        rerank_error=rerank_error,
        model=model,
        version=version,
        scores=scores,
        reranked_order=reranked_order,
        final_mode=final_mode,
    )


def failure_response(
    *,
    query: str,
    doc_type: str | None,
    top_k: int,
    invalid_result: bool,
) -> dict[str, Any]:
    """Build the frozen request-bound typed 503 safety envelope."""
    retrieval = _retrieval_model(
        hybrid_retriever.failure_response(
            query=query,
            doc_type=doc_type,
            top_k=CANDIDATE_K,
            invalid_result=invalid_result,
        )
    )
    return _build_response(
        query=query,
        doc_type=doc_type,
        requested_top_k=top_k,
        retrieval=retrieval,
        rerank_status="SKIPPED_RETRIEVAL_FAILED",
        rerank_error=None,
        model=None,
        version=None,
        scores=[],
        reranked_order=[],
        final_mode="NONE",
    )
