"""MOD-026A document-only claim/evidence binding over a strict rerank snapshot."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import inspect
import itertools
import math
import re
from typing import Any, Protocol

from pydantic import BaseModel

from app.rag import evidence_reranker, hybrid_retriever
from app.schemas.evidence_claims import (
    BINDER_TIMEOUT_MS,
    CLAIM_PROFILE,
    MAX_BINDER_INPUT_BYTES,
    MAX_EVIDENCE_TEXT_BYTES,
    MAX_INVENTORY_BYTES,
    MAX_RERANK_SNAPSHOT_BYTES,
    MAX_RESPONSE_BYTES,
    SCHEMA_VERSION,
    SUPPORT_THRESHOLD,
    TRUST_BOUNDARY,
    EvidenceClaimsDraft,
    EvidenceClaimsRequest,
    EvidenceClaimsResponse,
    binder_input_projection,
    canonical_sha256,
    canonical_size,
    citation_from_inventory,
    claims_from_draft,
    inventory_from_rerank,
)
from app.schemas.evidence_rerank import EvidenceRerankResponse


DEFAULT_MODEL = "robotops-token-overlap-claim-binder"
DEFAULT_VERSION = "1.0.0"
_SAFE_DESCRIPTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~/-]{0,127}$")


class ClaimBinder(Protocol):
    """A bounded async adapter that scores the full claim/evidence product."""

    model: str
    version: str

    async def bind(
        self,
        *,
        claims: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...


class TokenOverlapClaimBinder:
    """Offline deterministic contract baseline; it is not semantic entailment."""

    model = DEFAULT_MODEL
    version = DEFAULT_VERSION

    @staticmethod
    def _bind_sync(
        claims: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        matrix: list[dict[str, Any]] = []
        evidence_tokens = {
            item["evidence_id"]: set(hybrid_retriever.analyze(item["text"]))
            for item in evidence
        }
        for claim in claims:
            ordered_tokens = list(
                dict.fromkeys(hybrid_retriever.analyze(claim["text"]))
            )
            token_set = set(ordered_tokens)
            for item in evidence:
                score = (
                    float(
                        len(token_set.intersection(evidence_tokens[item["evidence_id"]]))
                        / len(ordered_tokens)
                    )
                    if ordered_tokens
                    else 0.0
                )
                matrix.append(
                    {
                        "claim_id": claim["claim_id"],
                        "evidence_id": item["evidence_id"],
                        "support_score": score,
                    }
                )
        return matrix

    async def bind(
        self,
        *,
        claims: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._bind_sync, claims, evidence)


DEFAULT_BINDER: ClaimBinder = TokenOverlapClaimBinder()


class _InvalidBinderResult(ValueError):
    pass


class _BinderDeadlineExceeded(RuntimeError):
    pass


class _BinderAdapterCancelled(RuntimeError):
    pass


class _InvalidClaimResponse(ValueError):
    pass


def _safe_descriptor(engine: ClaimBinder) -> tuple[str, str]:
    try:
        model = engine.model
        version = engine.version
        if type(model) is not str or type(version) is not str:
            raise _InvalidBinderResult("binder descriptor invalid")
        if any(
            _SAFE_DESCRIPTOR_RE.fullmatch(value) is None
            or any(segment in {"", ".", ".."} for segment in value.split("/"))
            for value in (model, version)
        ):
            raise _InvalidBinderResult("binder descriptor invalid")
    except asyncio.CancelledError as exc:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
        raise _InvalidBinderResult("binder descriptor cancelled itself") from exc
    except _InvalidBinderResult:
        raise
    except Exception as exc:
        raise _InvalidBinderResult("binder descriptor unavailable") from exc
    return model, version


def _discard_returned_awaitables(value: object) -> None:
    # Cleanup must not replace the request outcome with an adapter's exception.
    # This is synchronous: actual caller cancellation still propagates at the
    # awaiting boundary, rather than being consumed by this best-effort walk.
    try:
        _discard_returned_awaitables_bounded(value)
    except BaseException:
        pass


def _discard_returned_awaitables_bounded(value: object) -> None:
    # Covers a 100-row matrix even when every field is an invalid awaitable,
    # while retaining a hard bound for arbitrary/cyclic adapter object graphs.
    object_budget = 1_024
    pending = [value]
    seen: set[int] = set()
    inspected = 0
    owner = asyncio.current_task()
    while pending and inspected < object_budget:
        current = pending.pop()
        # Scalar cells cannot own awaitables. They must not exhaust the object
        # budget before all 100 permitted matrix rows have been inspected.
        if type(current) in (str, int, float, bool, bytes, type(None)):
            continue
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
            if current is owner:
                continue
            try:
                if current.done():
                    try:
                        pending.append(current.result())
                    except BaseException:
                        pass
                else:
                    current.cancel()
                    current.add_done_callback(_discard_task_result)
            except BaseException:
                pass
        elif isinstance(current, dict):
            # Invoke builtin traversal directly, never rejected subclasses'
            # overridable keys/values/iteration hooks. Bound the worklist too.
            pending.extend(itertools.islice(dict.keys(current), object_budget - inspected))
            pending.extend(itertools.islice(dict.values(current), object_budget - inspected))
        elif isinstance(current, (list, tuple, set, frozenset)):
            container_type = next(
                kind for kind in (list, tuple, set, frozenset)
                if isinstance(current, kind)
            )
            pending.extend(
                itertools.islice(container_type.__iter__(current), object_budget - inspected)
            )
        elif inspect.isawaitable(current):
            try:
                future = asyncio.ensure_future(current)
                if future is not owner:
                    if future.done():
                        try:
                            pending.append(future.result())
                        except BaseException:
                            pass
                    else:
                        future.cancel()
                        future.add_done_callback(_discard_task_result)
            except BaseException:
                try:
                    close = getattr(current, "close", None)
                    if callable(close):
                        close()
                except BaseException:
                    pass


def _discard_task_result(task: asyncio.Future[Any]) -> None:
    try:
        value = task.result()
    except BaseException:
        return
    _discard_returned_awaitables(value)


async def _call_with_deadline(
    engine: ClaimBinder,
    *,
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> object:
    async def invoke_and_await() -> object:
        bind_callable = engine.bind
        if inspect.iscoroutinefunction(bind_callable):
            awaitable = bind_callable(claims=claims, evidence=evidence)
        else:
            # A synchronous adapter is outside the async protocol.  Run it away
            # from the request loop so it cannot block or self-cancel the caller.
            context = contextvars.copy_context()
            worker_result = asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    context.run, bind_callable, claims=claims, evidence=evidence
                ),
            )
            try:
                # Cancelling the wait cannot stop a running thread.  Preserve
                # its result channel so a late coroutine is still closed.
                awaitable = await asyncio.shield(worker_result)
            except BaseException:
                worker_result.add_done_callback(_discard_task_result)
                raise
        if not inspect.isawaitable(awaitable):
            _discard_returned_awaitables(awaitable)
            raise _InvalidBinderResult("binder did not return an awaitable")
        try:
            result_task = asyncio.ensure_future(awaitable)
        except BaseException:
            _discard_returned_awaitables(awaitable)
            raise
        if result_task is asyncio.current_task():
            raise _InvalidBinderResult("binder returned its invocation task")
        try:
            return await result_task
        except BaseException:
            if not result_task.done():
                result_task.cancel()
                result_task.add_done_callback(_discard_task_result)
            raise

    task = asyncio.create_task(invoke_and_await())
    try:
        done, _pending = await asyncio.wait(
            {task}, timeout=BINDER_TIMEOUT_MS / 1_000
        )
    except asyncio.CancelledError:
        task.cancel()
        task.add_done_callback(_discard_task_result)
        raise
    except BaseException:
        task.cancel()
        task.add_done_callback(_discard_task_result)
        raise
    if task not in done:
        task.cancel()
        task.add_done_callback(_discard_task_result)
        raise _BinderDeadlineExceeded("binder deadline exceeded")
    try:
        return task.result()
    except asyncio.CancelledError as exc:
        raise _BinderAdapterCancelled("binder cancelled itself") from exc


def _normalize_matrix(
    raw: object,
    *,
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_pairs = [
        (claim["claim_id"], item["evidence_id"])
        for claim in claims
        for item in evidence
    ]
    if type(raw) is not list or len(raw) != len(expected_pairs):
        raise _InvalidBinderResult("binder must return the complete pair matrix")
    expected_pair_set = set(expected_pairs)
    scores: dict[tuple[str, str], float] = {}
    for item in raw:
        if (
            type(item) is not dict
            or any(type(key) is not str for key in item)
            or set(item) != {
                "claim_id",
                "evidence_id",
                "support_score",
            }
        ):
            raise _InvalidBinderResult("binder result shape invalid")
        claim_id = item.get("claim_id")
        evidence_id = item.get("evidence_id")
        score = item.get("support_score")
        pair = (claim_id, evidence_id)
        if (
            type(claim_id) is not str
            or type(evidence_id) is not str
            or pair not in expected_pair_set
            or pair in scores
        ):
            raise _InvalidBinderResult("binder returned an unknown or duplicate pair")
        if (
            type(score) is not float
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise _InvalidBinderResult("binder support score invalid")
        scores[(claim_id, evidence_id)] = score
    if set(scores) != expected_pair_set:
        raise _InvalidBinderResult("binder omitted a required pair")
    return [
        {
            "claim_id": claim_id,
            "evidence_id": evidence_id,
            "support_score": scores[(claim_id, evidence_id)],
        }
        for claim_id, evidence_id in expected_pairs
    ]


def _build_results(
    *,
    claims: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    binder_status: str,
    matrix: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    inventory_by_id = {item["evidence_id"]: item for item in inventory}
    supported_by_claim: dict[str, list[tuple[dict[str, Any], float]]] = {
        item["claim_id"]: [] for item in claims
    }
    for pair in matrix:
        if pair["support_score"] >= SUPPORT_THRESHOLD:
            supported_by_claim[pair["claim_id"]].append(
                (inventory_by_id[pair["evidence_id"]], pair["support_score"])
            )
    trusted_for_coverage = binder_status in {"SUCCESS", "SKIPPED_EMPTY"}
    results: list[dict[str, Any]] = []
    for claim in claims:
        supported = sorted(
            supported_by_claim[claim["claim_id"]],
            key=lambda pair: (
                -pair[1], pair[0]["final_rank"], pair[0]["evidence_id"]
            ),
        )[:3]
        citations = [
            citation_from_inventory(evidence, score)
            for evidence, score in supported
        ]
        if binder_status == "SUCCESS" and citations:
            verdict = "SUPPORTED"
            disposition = "PUBLISH"
        elif trusted_for_coverage:
            verdict = "INSUFFICIENT_EVIDENCE"
            disposition = {
                "CAUSE": "MARK_UNCERTAIN",
                "STEP": "OMIT",
                "RISK": "ESCALATE",
            }[claim["kind"]]
        else:
            verdict = "UNVERIFIED"
            disposition = "BLOCK"
            citations = []
        base = {
            **claim,
            "verdict": verdict,
            "disposition": disposition,
            "citations": citations,
            "citations_sha256": canonical_sha256(citations),
        }
        results.append({**base, "claim_result_sha256": canonical_sha256(base)})
    return results


def _build_response(
    *,
    request: EvidenceClaimsRequest,
    rerank: EvidenceRerankResponse,
    inventory: list[dict[str, Any]],
    binder_status: str,
    binder_error: str | None,
    model: str | None,
    version: str | None,
    descriptor_read_count: int | None,
    invocation_count: int | None,
    matrix: list[dict[str, Any]],
    top_error: str | None = None,
) -> dict[str, Any]:
    query_sha256 = canonical_sha256(request.query)
    draft_dump = request.draft.model_dump(mode="python")
    claims = claims_from_draft(request.draft)
    claim_set_sha256 = canonical_sha256(claims)
    rerank_dump = rerank.model_dump(mode="python")
    rerank_snapshot_sha256 = canonical_sha256(rerank_dump)
    evidence_inventory_sha256 = canonical_sha256(inventory)
    input_projection = binder_input_projection(
        claim_set_sha256=claim_set_sha256,
        rerank=rerank,
        evidence_inventory_sha256=evidence_inventory_sha256,
        claims=claims,
        evidence_inventory=inventory,
    )
    input_sha256 = canonical_sha256(input_projection)
    output_sha256 = canonical_sha256(matrix)
    binder_trace = {
        "status": binder_status,
        "error_code": binder_error,
        "model": model,
        "version": version,
        "threshold": float(SUPPORT_THRESHOLD),
        "timeout_ms": BINDER_TIMEOUT_MS,
        "descriptor_read_count": descriptor_read_count,
        "invocation_count": invocation_count,
        "input_claim_count": len(claims),
        "input_evidence_count": len(inventory),
        "expected_pair_count": len(claims) * len(inventory),
        "output_count": len(matrix),
        "matrix": matrix,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
    }
    results = _build_results(
        claims=claims,
        inventory=inventory,
        binder_status=binder_status,
        matrix=matrix,
    )
    results_sha256 = canonical_sha256(results)
    counts = {
        "total": len(results),
        "supported": sum(item["disposition"] == "PUBLISH" for item in results),
        "uncertain": sum(
            item["disposition"] == "MARK_UNCERTAIN" for item in results
        ),
        "omitted": sum(item["disposition"] == "OMIT" for item in results),
        "escalated": sum(item["disposition"] == "ESCALATE" for item in results),
        "blocked": sum(item["disposition"] == "BLOCK" for item in results),
    }
    need_human = any(
        item["disposition"] in {"ESCALATE", "BLOCK"} for item in results
    )
    reason_codes = list(rerank.reason_codes)
    if binder_error is not None:
        reason_codes.append(binder_error)
    insufficient_kinds = {
        item["kind"] for item in results if item["verdict"] == "INSUFFICIENT_EVIDENCE"
    }
    for kind, code in (
        ("CAUSE", "CAUSE_INSUFFICIENT_EVIDENCE"),
        ("STEP", "STEP_INSUFFICIENT_EVIDENCE"),
        ("RISK", "RISK_INSUFFICIENT_EVIDENCE"),
    ):
        if kind in insufficient_kinds:
            reason_codes.append(code)

    if rerank.status == "FAILED" or top_error is not None:
        status = "FAILED"
    elif (
        rerank.status == "SUCCESS"
        and binder_status == "SUCCESS"
        and all(item["verdict"] == "SUPPORTED" for item in results)
    ):
        status = "SUCCESS"
    else:
        status = "DEGRADED"
    request_sha256 = canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "query_sha256": query_sha256,
            "doc_type": request.doc_type,
            "draft": draft_dump,
        }
    )
    trace_projection = {
        "schema_version": SCHEMA_VERSION,
        "claim_profile": CLAIM_PROFILE,
        "query_sha256": query_sha256,
        "doc_type": request.doc_type,
        "request_sha256": request_sha256,
        "claim_set_sha256": claim_set_sha256,
        "rerank_snapshot_sha256": rerank_snapshot_sha256,
        "evidence_inventory_sha256": evidence_inventory_sha256,
        "binder": {
            key: binder_trace[key]
            for key in (
                "status",
                "error_code",
                "model",
                "version",
                "threshold",
                "timeout_ms",
                "descriptor_read_count",
                "invocation_count",
                "input_claim_count",
                "input_evidence_count",
                "expected_pair_count",
                "output_count",
                "input_sha256",
                "output_sha256",
            )
        },
        "results_sha256": results_sha256,
        "counts": counts,
        "need_human": need_human,
        "side_effects_allowed": False,
        "status": status,
        "error_code": top_error,
        "reason_codes": reason_codes,
    }
    response = {
        "schema_version": SCHEMA_VERSION,
        "claim_profile": CLAIM_PROFILE,
        "trust_boundary": TRUST_BOUNDARY,
        "query": request.query,
        "query_sha256": query_sha256,
        "doc_type": request.doc_type,
        "draft": draft_dump,
        "request_sha256": request_sha256,
        "claims": claims,
        "claim_set_sha256": claim_set_sha256,
        "rerank": rerank_dump,
        "rerank_snapshot_sha256": rerank_snapshot_sha256,
        "evidence_inventory": inventory,
        "evidence_inventory_sha256": evidence_inventory_sha256,
        "binder": binder_trace,
        "results": results,
        "results_sha256": results_sha256,
        "counts": counts,
        "need_human": need_human,
        "side_effects_allowed": False,
        "status": status,
        "error_code": top_error,
        "reason_codes": reason_codes,
        "trace_sha256": canonical_sha256(trace_projection),
    }
    if canonical_size(response) > MAX_RESPONSE_BYTES:
        raise _InvalidClaimResponse("claim response exceeds the frozen byte boundary")
    try:
        EvidenceClaimsResponse.model_validate(response)
    except Exception as exc:
        raise _InvalidClaimResponse("claim response failed strict validation") from exc
    return response


def _safe_failed_rerank(
    *, request: EvidenceClaimsRequest, invalid_result: bool
) -> EvidenceRerankResponse:
    return EvidenceRerankResponse.model_validate(
        evidence_reranker.failure_response(
            query=request.query,
            doc_type=request.doc_type,
            top_k=10,
            invalid_result=invalid_result,
        )
    )


def failure_response(
    *,
    query: str,
    doc_type: str | None,
    draft: EvidenceClaimsDraft | dict[str, Any],
    error_code: str,
    discarded_response: bool = False,
) -> dict[str, Any]:
    """Build a request-bound typed 503 without exposing underlying exceptions."""
    draft_payload = (
        draft.model_dump(mode="python") if isinstance(draft, BaseModel) else draft
    )
    request = EvidenceClaimsRequest.model_validate(
        {"query": query, "doc_type": doc_type, "draft": draft_payload}
    )
    rerank = _safe_failed_rerank(
        request=request,
        invalid_result=error_code
        in {"CLAIM_RETRIEVAL_INVALID_RESULT", "CLAIM_RESPONSE_INVALID"},
    )
    return _build_response(
        request=request,
        rerank=rerank,
        inventory=[],
        binder_status=(
            "DISCARDED_RESPONSE_INVALID"
            if discarded_response
            else "SKIPPED_RETRIEVAL_FAILED"
        ),
        binder_error="BINDER_STATE_UNKNOWN" if discarded_response else None,
        model=None,
        version=None,
        descriptor_read_count=None if discarded_response else 0,
        invocation_count=None if discarded_response else 0,
        matrix=[],
        top_error=error_code,
    )


def _validated_rerank(
    raw: object, *, request: EvidenceClaimsRequest
) -> EvidenceRerankResponse:
    raw_payload = raw.model_dump(mode="python") if isinstance(raw, BaseModel) else raw
    rerank = EvidenceRerankResponse.model_validate(raw_payload)
    rerank_dump = rerank.model_dump(mode="python")
    if (
        rerank.query != request.query
        or rerank.doc_type != request.doc_type
        or rerank.requested_top_k != 10
        or rerank.candidate_k != 20
        or [item.final_rank for item in rerank.results]
        != list(range(1, len(rerank.results) + 1))
        or canonical_size(rerank_dump) > MAX_RERANK_SNAPSHOT_BYTES
    ):
        raise ValueError("rerank snapshot escaped the claim request boundary")
    return rerank


def _validated_inventory(rerank: EvidenceRerankResponse) -> list[dict[str, Any]]:
    inventory = inventory_from_rerank(rerank)
    if (
        any(len(item["text"].encode("utf-8")) > MAX_EVIDENCE_TEXT_BYTES for item in inventory)
        or canonical_size(inventory) > MAX_INVENTORY_BYTES
    ):
        raise ValueError("evidence inventory exceeds the frozen byte boundary")
    return inventory


async def verify_claims(
    *,
    query: str,
    doc_type: str | None,
    draft: EvidenceClaimsDraft | dict[str, Any],
    binder: ClaimBinder | None = None,
) -> dict[str, Any]:
    """Bind every normalized claim to the server-owned top-10 rerank evidence."""
    draft_payload = (
        draft.model_dump(mode="python") if isinstance(draft, BaseModel) else draft
    )
    request = EvidenceClaimsRequest.model_validate(
        {"query": query, "doc_type": doc_type, "draft": draft_payload}
    )
    try:
        raw_rerank = await evidence_reranker.rerank_evidence(
            query=request.query,
            doc_type=request.doc_type,
            top_k=10,
        )
    except evidence_reranker.InvalidRetrievalResultError:
        return failure_response(
            query=request.query,
            doc_type=request.doc_type,
            draft=request.draft,
            error_code="CLAIM_RETRIEVAL_INVALID_RESULT",
        )
    except evidence_reranker.RetrievalUnavailableError:
        return failure_response(
            query=request.query,
            doc_type=request.doc_type,
            draft=request.draft,
            error_code="CLAIM_RETRIEVAL_UNAVAILABLE",
        )
    except Exception:
        return failure_response(
            query=request.query,
            doc_type=request.doc_type,
            draft=request.draft,
            error_code="CLAIM_RETRIEVAL_UNAVAILABLE",
        )

    try:
        rerank = _validated_rerank(raw_rerank, request=request)
        inventory = _validated_inventory(rerank)
    except Exception:
        return failure_response(
            query=request.query,
            doc_type=request.doc_type,
            draft=request.draft,
            error_code="CLAIM_RETRIEVAL_INVALID_RESULT",
        )

    if rerank.status == "FAILED":
        try:
            return _build_response(
                request=request,
                rerank=rerank,
                inventory=[],
                binder_status="SKIPPED_RETRIEVAL_FAILED",
                binder_error=None,
                model=None,
                version=None,
                descriptor_read_count=0,
                invocation_count=0,
                matrix=[],
                top_error="CLAIM_RETRIEVAL_FAILED",
            )
        except Exception:
            return failure_response(
                query=request.query,
                doc_type=request.doc_type,
                draft=request.draft,
                error_code="CLAIM_RESPONSE_INVALID",
            )

    if not inventory:
        try:
            return _build_response(
                request=request,
                rerank=rerank,
                inventory=[],
                binder_status="SKIPPED_EMPTY",
                binder_error=None,
                model=None,
                version=None,
                descriptor_read_count=0,
                invocation_count=0,
                matrix=[],
            )
        except Exception:
            return failure_response(
                query=request.query,
                doc_type=request.doc_type,
                draft=request.draft,
                error_code="CLAIM_RESPONSE_INVALID",
            )

    claims = claims_from_draft(request.draft)
    claim_set_sha256 = canonical_sha256(claims)
    inventory_sha256 = canonical_sha256(inventory)
    input_projection = binder_input_projection(
        claim_set_sha256=claim_set_sha256,
        rerank=rerank,
        evidence_inventory_sha256=inventory_sha256,
        claims=claims,
        evidence_inventory=inventory,
    )
    if canonical_size(input_projection) > MAX_BINDER_INPUT_BYTES:
        return failure_response(
            query=request.query,
            doc_type=request.doc_type,
            draft=request.draft,
            error_code="CLAIM_RETRIEVAL_INVALID_RESULT",
        )

    engine = binder if binder is not None else DEFAULT_BINDER
    try:
        model, version = _safe_descriptor(engine)
    except _InvalidBinderResult:
        binder_status = "INVALID"
        binder_error = "BINDER_INVALID_RESULT"
        model = None
        version = None
        descriptor_read_count = 1
        invocation_count = 0
        matrix: list[dict[str, Any]] = []
    else:
        descriptor_read_count = 1
        invocation_count = 1
        raw_matrix: object | None = None
        try:
            raw_matrix = await _call_with_deadline(
                engine,
                claims=copy.deepcopy(claims),
                evidence=copy.deepcopy(inventory),
            )
        except _BinderDeadlineExceeded:
            binder_status = "TIMEOUT"
            binder_error = "BINDER_TIMEOUT"
            matrix = []
        except _InvalidBinderResult:
            binder_status = "INVALID"
            binder_error = "BINDER_INVALID_RESULT"
            matrix = []
        except Exception:
            binder_status = "FAILED"
            binder_error = "BINDER_FAILED"
            matrix = []
        else:
            try:
                matrix = _normalize_matrix(
                    raw_matrix,
                    claims=claims,
                    evidence=inventory,
                )
            except Exception:
                _discard_returned_awaitables(raw_matrix)
                binder_status = "INVALID"
                binder_error = "BINDER_INVALID_RESULT"
                matrix = []
            else:
                binder_status = "SUCCESS"
                binder_error = None

    try:
        return _build_response(
            request=request,
            rerank=rerank,
            inventory=inventory,
            binder_status=binder_status,
            binder_error=binder_error,
            model=model,
            version=version,
            descriptor_read_count=descriptor_read_count,
            invocation_count=invocation_count,
            matrix=matrix,
        )
    except Exception:
        return failure_response(
            query=request.query,
            doc_type=request.doc_type,
            draft=request.draft,
            error_code="CLAIM_RESPONSE_INVALID",
            discarded_response=True,
        )
