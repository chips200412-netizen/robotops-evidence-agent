"""MOD-025B exact API contract for traceable rerank and fallback."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from app.schemas.evidence_search import (
    EvidenceDocType,
    EvidenceReasonCode,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
    EvidenceSearchResultResponse,
    ExactEvidenceResponse,
)


RerankStatus = Literal[
    "SUCCESS",
    "TIMEOUT",
    "FAILED",
    "SKIPPED_EMPTY",
    "SKIPPED_RETRIEVAL_FAILED",
]
RerankErrorCode = Literal[
    "RERANK_TIMEOUT",
    "RERANK_FAILED",
    "RERANK_INVALID_RESULT",
]
RerankTopStatus = Literal["SUCCESS", "DEGRADED", "FAILED"]
RerankReasonCode = EvidenceReasonCode | RerankErrorCode
FinalizationMode = Literal["RERANKED", "FUSION_FALLBACK", "EMPTY", "NONE"]
TieBreakValue = Literal[
    "RERANK_SCORE_DESC",
    "FUSED_RANK_ASC",
    "CHUNK_ID_ASC",
]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EvidenceRerankRequest(EvidenceSearchRequest):
    """Same strict request surface as evidence-search, with an independent name."""


class ExactRerankResponse(ExactEvidenceResponse):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class EvidenceRerankScoreResponse(ExactRerankResponse):
    chunk_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._~-]+$")
    score: float = Field(strict=True, allow_inf_nan=False)

    @field_validator("score", mode="before")
    @classmethod
    def score_is_an_exact_float(cls, value: object) -> object:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("rerank score must be a finite float")
        return value


class EvidenceRerankTraceResponse(ExactRerankResponse):
    status: RerankStatus
    error_code: RerankErrorCode | None
    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~/-]{0,127}$",
    )
    version: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~/-]{0,127}$",
    )
    timeout_ms: int = Field(strict=True, ge=1_000, le=1_000)
    input_count: int = Field(strict=True, ge=0, le=20)
    input_order: list[str]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_count: int = Field(strict=True, ge=0, le=20)
    scores: list[EvidenceRerankScoreResponse]
    reranked_order: list[str]
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("model", "version")
    @classmethod
    def descriptor_has_safe_namespace_segments(
        cls, value: str | None
    ) -> str | None:
        if value is not None and any(
            segment in {"", ".", ".."} for segment in value.split("/")
        ):
            raise ValueError("descriptor contains an unsafe namespace segment")
        return value

    @field_validator("input_order", "reranked_order")
    @classmethod
    def validate_order_ids(cls, value: list[str]) -> list[str]:
        if len(value) > 20 or len(value) != len(set(value)):
            raise ValueError("rerank order must contain at most 20 unique ids")
        if any(
            not isinstance(item, str)
            or not item
            or len(item) > 256
            or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-" for character in item)
            for item in value
        ):
            raise ValueError("rerank order contains an invalid id")
        return value

    @model_validator(mode="after")
    def validate_trace_shape(self) -> "EvidenceRerankTraceResponse":
        if self.input_count != len(self.input_order):
            raise ValueError("rerank input count mismatch")
        if self.output_count != len(self.scores):
            raise ValueError("rerank output count mismatch")
        if self.output_sha256 != _canonical_sha256(
            {
                "scores": [item.model_dump(mode="python") for item in self.scores],
                "reranked_order": self.reranked_order,
            }
        ):
            raise ValueError("rerank output digest mismatch")
        if self.status == "SUCCESS":
            if self.error_code is not None or self.model is None or self.version is None:
                raise ValueError("successful rerank requires a safe descriptor")
        elif self.status == "TIMEOUT":
            if (
                self.error_code != "RERANK_TIMEOUT"
                or self.model is None
                or self.version is None
            ):
                raise ValueError("timed out rerank has an invalid descriptor or code")
        elif self.status == "FAILED":
            if self.error_code not in {"RERANK_FAILED", "RERANK_INVALID_RESULT"}:
                raise ValueError("failed rerank requires a stable error code")
            if (self.model is None) != (self.version is None):
                raise ValueError("rerank descriptor must be both present or both absent")
            if self.error_code == "RERANK_FAILED" and self.model is None:
                raise ValueError("execution failure requires the selected descriptor")
        else:
            if (
                self.error_code is not None
                or self.model is not None
                or self.version is not None
            ):
                raise ValueError("skipped rerank must not read a descriptor")
        if self.status != "SUCCESS":
            if self.output_count != 0 or self.scores or self.reranked_order:
                raise ValueError("non-successful rerank must publish an empty output")
        return self


class EvidenceFinalizationResponse(ExactRerankResponse):
    mode: FinalizationMode
    tie_break: list[TieBreakValue]
    final_order: list[str]
    final_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("final_order")
    @classmethod
    def validate_final_ids(cls, value: list[str]) -> list[str]:
        if len(value) > 20 or len(value) != len(set(value)):
            raise ValueError("final order must contain at most 20 unique ids")
        if any(
            not isinstance(item, str)
            or not item
            or len(item) > 256
            or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-" for character in item)
            for item in value
        ):
            raise ValueError("final order contains an invalid id")
        return value

    @model_validator(mode="after")
    def validate_finalization(self) -> "EvidenceFinalizationResponse":
        expected_tie_break = {
            "RERANKED": [
                "RERANK_SCORE_DESC",
                "FUSED_RANK_ASC",
                "CHUNK_ID_ASC",
            ],
            "FUSION_FALLBACK": ["FUSED_RANK_ASC", "CHUNK_ID_ASC"],
            "EMPTY": [],
            "NONE": [],
        }[self.mode]
        if self.tie_break != expected_tie_break:
            raise ValueError("finalization tie break does not match its mode")
        if self.final_order_sha256 != _canonical_sha256(self.final_order):
            raise ValueError("final order digest mismatch")
        return self


class EvidenceRerankResultResponse(EvidenceSearchResultResponse):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    fused_rank: int = Field(strict=True, ge=1, le=20)
    final_rank: int = Field(strict=True, ge=1, le=20)
    rerank_score: float | None = Field(default=None, strict=True, allow_inf_nan=False)

    @field_validator("rerank_score", mode="before")
    @classmethod
    def rerank_score_is_an_exact_float_or_none(cls, value: object) -> object:
        if value is not None and (
            type(value) is not float or not math.isfinite(value)
        ):
            raise ValueError("final rerank score must be a finite float or null")
        return value


class EvidenceRerankResponse(ExactRerankResponse):
    schema_version: int = Field(strict=True, ge=1, le=1)
    rerank_profile: Literal["robotops_evidence_rerank_v1"]
    trust_boundary: Literal["UNTRUSTED_DATA"]
    query: str = Field(min_length=1, max_length=2_000)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    doc_type: EvidenceDocType | None
    requested_top_k: int = Field(strict=True, ge=1, le=20)
    candidate_k: int = Field(strict=True, ge=20, le=20)
    status: RerankTopStatus
    reason_codes: list[RerankReasonCode]
    retrieval: EvidenceSearchResponse
    retrieval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rerank: EvidenceRerankTraceResponse
    finalization: EvidenceFinalizationResponse
    count: int = Field(strict=True, ge=0, le=20)
    results: list[EvidenceRerankResultResponse]
    results_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("query")
    @classmethod
    def query_is_utf8_encodable(cls, value: str) -> str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("query must not contain isolated Unicode surrogates")
        return value

    @model_validator(mode="after")
    def validate_response(self) -> "EvidenceRerankResponse":
        retrieval = self.retrieval
        retrieval_dump = retrieval.model_dump(mode="python")
        if (
            retrieval.query != self.query
            or retrieval.query_sha256 != self.query_sha256
            or retrieval.doc_type != self.doc_type
            or retrieval.top_k != 20
            or retrieval.candidate_k != self.candidate_k
        ):
            raise ValueError("retrieval snapshot is not bound to the rerank request")
        if self.query_sha256 != _canonical_sha256(self.query):
            raise ValueError("query digest mismatch")
        if self.request_sha256 != _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "query_sha256": self.query_sha256,
                "doc_type": self.doc_type,
                "requested_top_k": self.requested_top_k,
            }
        ):
            raise ValueError("request digest mismatch")
        if self.retrieval_sha256 != _canonical_sha256(retrieval_dump):
            raise ValueError("retrieval digest mismatch")

        input_items = [
            {
                "chunk_id": item.chunk_id,
                "content_sha256": item.content_sha256,
                "identity_sha256": item.identity_sha256,
                "fused_rank": rank,
                "fused_score": item.fused_score,
            }
            for rank, item in enumerate(retrieval.results, start=1)
        ]
        input_order = [item["chunk_id"] for item in input_items]
        expected_input_sha256 = _canonical_sha256(
            {
                "schema_version": 1,
                "query_sha256": self.query_sha256,
                "retrieval_profile": retrieval.retrieval_profile,
                "retrieval_results_sha256": retrieval.results_sha256,
                "candidates": input_items,
            }
        )
        if (
            self.rerank.input_count != len(input_order)
            or self.rerank.input_order != input_order
            or self.rerank.input_sha256 != expected_input_sha256
        ):
            raise ValueError("rerank input projection mismatch")

        retrieval_reasons = list(retrieval.reason_codes)
        if retrieval.status == "FAILED":
            expected_status = "FAILED"
            expected_rerank_status = "SKIPPED_RETRIEVAL_FAILED"
            expected_mode = "NONE"
            expected_reasons = retrieval_reasons
        elif not input_order:
            expected_status = (
                "SUCCESS" if retrieval.status == "HYBRID_SUCCESS" else "DEGRADED"
            )
            expected_rerank_status = "SKIPPED_EMPTY"
            expected_mode = "EMPTY"
            expected_reasons = retrieval_reasons
        elif self.rerank.status == "SUCCESS":
            expected_status = (
                "SUCCESS" if retrieval.status == "HYBRID_SUCCESS" else "DEGRADED"
            )
            expected_rerank_status = "SUCCESS"
            expected_mode = "RERANKED"
            expected_reasons = retrieval_reasons
        elif self.rerank.status in {"TIMEOUT", "FAILED"}:
            expected_status = "DEGRADED"
            expected_rerank_status = self.rerank.status
            expected_mode = "FUSION_FALLBACK"
            expected_reasons = retrieval_reasons + [self.rerank.error_code]
        else:
            raise ValueError("rerank skip status is not valid for non-empty retrieval")
        if (
            self.status != expected_status
            or self.rerank.status != expected_rerank_status
            or self.finalization.mode != expected_mode
            or self.reason_codes != expected_reasons
        ):
            raise ValueError("top-level rerank state is inconsistent")

        if self.rerank.status == "SUCCESS":
            score_ids = [item.chunk_id for item in self.rerank.scores]
            if (
                self.rerank.output_count != len(input_order)
                or score_ids != input_order
                or set(self.rerank.reranked_order) != set(input_order)
                or len(self.rerank.reranked_order) != len(input_order)
            ):
                raise ValueError("successful rerank is not a complete permutation")
            score_map = {item.chunk_id: item.score for item in self.rerank.scores}
            fused_rank_map = {
                chunk_id: rank for rank, chunk_id in enumerate(input_order, start=1)
            }
            expected_reranked_order = sorted(
                input_order,
                key=lambda chunk_id: (
                    -score_map[chunk_id],
                    fused_rank_map[chunk_id],
                    chunk_id,
                ),
            )
            if self.rerank.reranked_order != expected_reranked_order:
                raise ValueError("reranked order does not follow the frozen tie break")
            expected_final_order = expected_reranked_order[: self.requested_top_k]
        elif self.finalization.mode == "FUSION_FALLBACK":
            score_map = {}
            expected_final_order = input_order[: self.requested_top_k]
        else:
            score_map = {}
            expected_final_order = []
        if self.finalization.final_order != expected_final_order:
            raise ValueError("final order does not match the selected mode")

        sources = {
            item.chunk_id: item.model_dump(mode="python") for item in retrieval.results
        }
        expected_results: list[dict] = []
        fused_rank_map = {
            chunk_id: rank for rank, chunk_id in enumerate(input_order, start=1)
        }
        for final_rank, chunk_id in enumerate(expected_final_order, start=1):
            expected_results.append(
                {
                    **sources[chunk_id],
                    "fused_rank": fused_rank_map[chunk_id],
                    "final_rank": final_rank,
                    "rerank_score": (
                        score_map[chunk_id]
                        if self.finalization.mode == "RERANKED"
                        else None
                    ),
                }
            )
        actual_results = [item.model_dump(mode="python") for item in self.results]
        if (
            self.count != len(actual_results)
            or self.count > self.requested_top_k
            or actual_results != expected_results
            or self.results_sha256 != _canonical_sha256(actual_results)
        ):
            raise ValueError("final result projection mismatch")

        trace_projection = {
            "schema_version": 1,
            "rerank_profile": self.rerank_profile,
            "query_sha256": self.query_sha256,
            "doc_type": self.doc_type,
            "requested_top_k": self.requested_top_k,
            "candidate_k": self.candidate_k,
            "retrieval_sha256": self.retrieval_sha256,
            "model": self.rerank.model,
            "version": self.rerank.version,
            "timeout_ms": self.rerank.timeout_ms,
            "rerank_status": self.rerank.status,
            "rerank_error": self.rerank.error_code,
            "input_sha256": self.rerank.input_sha256,
            "output_sha256": self.rerank.output_sha256,
            "final_order_sha256": self.finalization.final_order_sha256,
            "results_sha256": self.results_sha256,
        }
        if self.trace_sha256 != _canonical_sha256(trace_projection):
            raise ValueError("rerank trace digest mismatch")
        return self
