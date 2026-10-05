"""MOD-026A strict contract for document-only claim/evidence binding."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.evidence_rerank import EvidenceRerankResponse, RerankReasonCode
from app.schemas.evidence_search import EvidenceDocType


ClaimKind = Literal["CAUSE", "STEP", "RISK"]
EvidenceSourceKind = Literal["MANUAL_CHUNK", "FAULT_CASE_CHUNK"]
BinderStatus = Literal[
    "SUCCESS",
    "TIMEOUT",
    "FAILED",
    "INVALID",
    "SKIPPED_EMPTY",
    "SKIPPED_RETRIEVAL_FAILED",
    "DISCARDED_RESPONSE_INVALID",
]
BinderErrorCode = Literal[
    "BINDER_TIMEOUT",
    "BINDER_FAILED",
    "BINDER_INVALID_RESULT",
    "BINDER_STATE_UNKNOWN",
]
ClaimVerdict = Literal["SUPPORTED", "INSUFFICIENT_EVIDENCE", "UNVERIFIED"]
ClaimDisposition = Literal[
    "PUBLISH", "MARK_UNCERTAIN", "OMIT", "ESCALATE", "BLOCK"
]
ClaimsTopStatus = Literal["SUCCESS", "DEGRADED", "FAILED"]
ClaimsTopErrorCode = Literal[
    "CLAIM_RETRIEVAL_FAILED",
    "CLAIM_RETRIEVAL_UNAVAILABLE",
    "CLAIM_RETRIEVAL_INVALID_RESULT",
    "CLAIM_RESPONSE_INVALID",
]
ClaimReasonCode = RerankReasonCode | BinderErrorCode | Literal[
    "CAUSE_INSUFFICIENT_EVIDENCE",
    "STEP_INSUFFICIENT_EVIDENCE",
    "RISK_INSUFFICIENT_EVIDENCE",
]

SCHEMA_VERSION = 1
CLAIM_PROFILE = "robotops_document_claim_evidence_v1"
TRUST_BOUNDARY = "UNTRUSTED_DATA"
SUPPORT_THRESHOLD = 0.70
BINDER_TIMEOUT_MS = 1_000
MAX_DRAFT_BYTES = 16_384
MAX_EVIDENCE_TEXT_BYTES = 32_768
MAX_INVENTORY_BYTES = 262_144
MAX_BINDER_INPUT_BYTES = 393_216
MAX_RERANK_SNAPSHOT_BYTES = 524_288
MAX_RESPONSE_BYTES = 1_048_576


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_size(value: object) -> int:
    return len(canonical_json(value).encode("utf-8"))


def normalize_claim_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not 1 <= len(normalized) <= 1_000:
        raise ValueError("claim text length is outside the frozen boundary")
    if any(
        0xD800 <= ord(character) <= 0xDFFF
        or unicodedata.category(character).startswith("C")
        for character in normalized
    ):
        raise ValueError("claim text contains a forbidden Unicode character")
    return normalized


class StrictClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ExactClaimResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class EvidenceClaimsDraft(StrictClaimRequest):
    causes: list[str] = Field(default_factory=list, max_length=4)
    steps: list[str] = Field(default_factory=list, max_length=5)
    risk_level: Literal["低", "中", "高"]

    @field_validator("causes", "steps", mode="before")
    @classmethod
    def normalize_text_items(cls, value: object) -> object:
        if type(value) is not list:
            return value
        normalized: list[str] = []
        for item in value:
            if type(item) is not str:
                raise ValueError("claim entries must be strings")
            normalized.append(normalize_claim_text(item))
        if len(normalized) != len(set(normalized)):
            raise ValueError("duplicate normalized claim text within one kind")
        return normalized

    @model_validator(mode="after")
    def draft_is_bounded(self) -> "EvidenceClaimsDraft":
        if canonical_size(self.model_dump(mode="python")) > MAX_DRAFT_BYTES:
            raise ValueError("normalized draft exceeds the frozen byte boundary")
        return self


class EvidenceClaimsRequest(StrictClaimRequest):
    query: str = Field(min_length=1, max_length=2_000)
    doc_type: EvidenceDocType | None = None
    draft: EvidenceClaimsDraft

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: object) -> object:
        if type(value) is str:
            return value.strip()
        return value

    @field_validator("query")
    @classmethod
    def query_is_utf8_encodable(cls, value: str) -> str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("query must not contain isolated Unicode surrogates")
        return value


class EvidenceClaimsDraftResponse(ExactClaimResponse):
    causes: list[str] = Field(max_length=4)
    steps: list[str] = Field(max_length=5)
    risk_level: Literal["低", "中", "高"]

    @field_validator("causes", "steps")
    @classmethod
    def claim_texts_are_already_normalized(cls, value: list[str]) -> list[str]:
        if any(normalize_claim_text(item) != item for item in value):
            raise ValueError("response draft must already be normalized")
        if len(value) != len(set(value)):
            raise ValueError("response draft contains a duplicate claim")
        return value

    @model_validator(mode="after")
    def draft_is_bounded(self) -> "EvidenceClaimsDraftResponse":
        if canonical_size(self.model_dump(mode="python")) > MAX_DRAFT_BYTES:
            raise ValueError("normalized draft exceeds the frozen byte boundary")
        return self

class EvidenceClaimResponse(ExactClaimResponse):
    claim_id: str = Field(pattern=r"^(cause|step|risk)-[0-9]{3}$")
    kind: ClaimKind
    ordinal: int = Field(strict=True, ge=1, le=5)
    text: str = Field(min_length=1, max_length=1_000)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_claim(self) -> "EvidenceClaimResponse":
        prefix = {"CAUSE": "cause", "STEP": "step", "RISK": "risk"}[self.kind]
        if self.claim_id != f"{prefix}-{self.ordinal:03d}":
            raise ValueError("claim id does not match kind and ordinal")
        if self.kind != "RISK" and normalize_claim_text(self.text) != self.text:
            raise ValueError("claim text is not normalized")
        if self.text_sha256 != text_sha256(self.text):
            raise ValueError("claim text digest mismatch")
        if self.kind == "RISK" and self.ordinal != 1:
            raise ValueError("risk claim must have ordinal one")
        return self


class EvidenceInventoryItemResponse(ExactClaimResponse):
    evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: EvidenceSourceKind
    chunk_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._~-]+$")
    doc_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._~-]+$")
    case_id: str | None = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9._~-]+$",
    )
    filename: str = Field(min_length=1, max_length=512)
    doc_type: EvidenceDocType
    chunk_index: int = Field(strict=True, ge=0)
    text: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_rank: int = Field(strict=True, ge=1, le=10)

    @field_validator("filename")
    @classmethod
    def filename_is_basename(cls, value: str) -> str:
        if (
            "\x00" in value
            or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
            or PurePosixPath(value).name != value
            or PureWindowsPath(value).name != value
            or PureWindowsPath(value).drive
        ):
            raise ValueError("filename must be a basename")
        return value

    @field_validator("text")
    @classmethod
    def evidence_text_is_bounded(cls, value: str) -> str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("evidence text must be UTF-8 encodable")
        if len(value.encode("utf-8")) > MAX_EVIDENCE_TEXT_BYTES:
            raise ValueError("evidence text exceeds the frozen byte boundary")
        return value

    @model_validator(mode="after")
    def validate_source_mapping(self) -> "EvidenceInventoryItemResponse":
        if self.evidence_id != self.identity_sha256:
            raise ValueError("evidence id must be the server-owned identity digest")
        if self.doc_type == "manual":
            if self.source_kind != "MANUAL_CHUNK" or self.case_id is not None:
                raise ValueError("manual source mapping is invalid")
        elif self.source_kind != "FAULT_CASE_CHUNK" or self.case_id != self.doc_id:
            raise ValueError("fault-case source mapping is invalid")
        return self


class EvidenceCitationResponse(ExactClaimResponse):
    evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: EvidenceSourceKind
    chunk_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._~-]+$")
    doc_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._~-]+$")
    case_id: str | None = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9._~-]+$",
    )
    filename: str = Field(min_length=1, max_length=512)
    doc_type: EvidenceDocType
    chunk_index: int = Field(strict=True, ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_rank: int = Field(strict=True, ge=1, le=10)
    support_score: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("support_score", mode="before")
    @classmethod
    def score_is_an_exact_float(cls, value: object) -> object:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("support score must be a finite exact float")
        return value


class ClaimBindingScoreResponse(ExactClaimResponse):
    claim_id: str = Field(pattern=r"^(cause|step|risk)-[0-9]{3}$")
    evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_score: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("support_score", mode="before")
    @classmethod
    def score_is_an_exact_float(cls, value: object) -> object:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("support score must be a finite exact float")
        return value


class ClaimBinderTraceResponse(ExactClaimResponse):
    status: BinderStatus
    error_code: BinderErrorCode | None
    model: str | None = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~/-]{0,127}$",
    )
    version: str | None = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~/-]{0,127}$",
    )
    threshold: float = Field(
        strict=True, ge=SUPPORT_THRESHOLD, le=SUPPORT_THRESHOLD
    )
    timeout_ms: int = Field(strict=True, ge=BINDER_TIMEOUT_MS, le=BINDER_TIMEOUT_MS)
    descriptor_read_count: int | None = Field(strict=True, ge=0, le=1)
    invocation_count: int | None = Field(strict=True, ge=0, le=1)
    input_claim_count: int = Field(strict=True, ge=1, le=10)
    input_evidence_count: int = Field(strict=True, ge=0, le=10)
    expected_pair_count: int = Field(strict=True, ge=0, le=100)
    output_count: int = Field(strict=True, ge=0, le=100)
    matrix: list[ClaimBindingScoreResponse] = Field(max_length=100)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("model", "version")
    @classmethod
    def descriptor_has_safe_segments(cls, value: str | None) -> str | None:
        if value is not None and any(
            segment in {"", ".", ".."} for segment in value.split("/")
        ):
            raise ValueError("descriptor contains an unsafe namespace segment")
        return value

    @model_validator(mode="after")
    def validate_trace_state(self) -> "ClaimBinderTraceResponse":
        if self.threshold != SUPPORT_THRESHOLD:
            raise ValueError("binder threshold mismatch")
        if self.expected_pair_count != self.input_claim_count * self.input_evidence_count:
            raise ValueError("binder pair count mismatch")
        if self.output_sha256 != canonical_sha256(
            [item.model_dump(mode="python") for item in self.matrix]
        ):
            raise ValueError("binder output digest mismatch")

        if self.status == "SUCCESS":
            valid = (
                self.error_code is None
                and self.model is not None
                and self.version is not None
                and self.descriptor_read_count == 1
                and self.invocation_count == 1
                and self.output_count == self.expected_pair_count
                and len(self.matrix) == self.expected_pair_count
            )
        elif self.status == "TIMEOUT":
            valid = self._ordinary_failure("BINDER_TIMEOUT")
        elif self.status == "FAILED":
            valid = self._ordinary_failure("BINDER_FAILED")
        elif self.status == "INVALID":
            descriptor_invalid = (
                self.error_code == "BINDER_INVALID_RESULT"
                and self.model is None
                and self.version is None
                and self.descriptor_read_count == 1
                and self.invocation_count == 0
            )
            output_invalid = (
                self.error_code == "BINDER_INVALID_RESULT"
                and self.model is not None
                and self.version is not None
                and self.descriptor_read_count == 1
                and self.invocation_count == 1
            )
            valid = (descriptor_invalid or output_invalid) and self._empty_output()
        elif self.status in {"SKIPPED_EMPTY", "SKIPPED_RETRIEVAL_FAILED"}:
            valid = (
                self.error_code is None
                and self.model is None
                and self.version is None
                and self.descriptor_read_count == 0
                and self.invocation_count == 0
                and self._empty_output()
            )
        else:
            valid = (
                self.error_code == "BINDER_STATE_UNKNOWN"
                and self.model is None
                and self.version is None
                and self.descriptor_read_count is None
                and self.invocation_count is None
                and self.input_evidence_count == 0
                and self.expected_pair_count == 0
                and self._empty_output()
            )
        if not valid:
            raise ValueError("binder trace state is inconsistent")
        return self

    def _empty_output(self) -> bool:
        return self.output_count == 0 and not self.matrix

    def _ordinary_failure(self, error_code: str) -> bool:
        return (
            self.error_code == error_code
            and self.model is not None
            and self.version is not None
            and self.descriptor_read_count == 1
            and self.invocation_count == 1
            and self._empty_output()
        )


class EvidenceClaimResultResponse(ExactClaimResponse):
    claim_id: str = Field(pattern=r"^(cause|step|risk)-[0-9]{3}$")
    kind: ClaimKind
    ordinal: int = Field(strict=True, ge=1, le=5)
    text: str = Field(min_length=1, max_length=1_000)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verdict: ClaimVerdict
    disposition: ClaimDisposition
    citations: list[EvidenceCitationResponse] = Field(max_length=3)
    citations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_local_digests(self) -> "EvidenceClaimResultResponse":
        citation_dump = [item.model_dump(mode="python") for item in self.citations]
        if self.citations_sha256 != canonical_sha256(citation_dump):
            raise ValueError("citation digest mismatch")
        projection = self.model_dump(mode="python", exclude={"claim_result_sha256"})
        if self.claim_result_sha256 != canonical_sha256(projection):
            raise ValueError("claim result digest mismatch")
        return self


class EvidenceClaimCountsResponse(ExactClaimResponse):
    total: int = Field(strict=True, ge=1, le=10)
    supported: int = Field(strict=True, ge=0, le=10)
    uncertain: int = Field(strict=True, ge=0, le=4)
    omitted: int = Field(strict=True, ge=0, le=5)
    escalated: int = Field(strict=True, ge=0, le=1)
    blocked: int = Field(strict=True, ge=0, le=10)


def claims_from_draft(
    draft: EvidenceClaimsDraft | EvidenceClaimsDraftResponse,
) -> list[dict[str, object]]:
    claims: list[dict[str, object]] = []
    for kind, prefix, values in (
        ("CAUSE", "cause", draft.causes),
        ("STEP", "step", draft.steps),
    ):
        for ordinal, value in enumerate(values, start=1):
            claims.append(
                {
                    "claim_id": f"{prefix}-{ordinal:03d}",
                    "kind": kind,
                    "ordinal": ordinal,
                    "text": value,
                    "text_sha256": text_sha256(value),
                }
            )
    risk_text = f"风险等级：{draft.risk_level}"
    claims.append(
        {
            "claim_id": "risk-001",
            "kind": "RISK",
            "ordinal": 1,
            "text": risk_text,
            "text_sha256": text_sha256(risk_text),
        }
    )
    return claims


def inventory_from_rerank(rerank: EvidenceRerankResponse) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for item in rerank.results:
        inventory.append(
            {
                "evidence_id": item.identity_sha256,
                "source_kind": (
                    "MANUAL_CHUNK" if item.doc_type == "manual" else "FAULT_CASE_CHUNK"
                ),
                "chunk_id": item.chunk_id,
                "doc_id": item.doc_id,
                "case_id": item.doc_id if item.doc_type == "fault_case" else None,
                "filename": item.filename,
                "doc_type": item.doc_type,
                "chunk_index": item.chunk_index,
                "text": item.text,
                "content_sha256": item.content_sha256,
                "identity_sha256": item.identity_sha256,
                "final_rank": item.final_rank,
            }
        )
    return inventory


def binder_input_projection(
    *,
    claim_set_sha256: str,
    rerank: EvidenceRerankResponse,
    evidence_inventory_sha256: str,
    claims: list[dict[str, object]],
    evidence_inventory: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_set_sha256": claim_set_sha256,
        "rerank_trace_sha256": rerank.trace_sha256,
        "rerank_results_sha256": rerank.results_sha256,
        "evidence_inventory_sha256": evidence_inventory_sha256,
        "claims": [
            {
                "claim_id": item["claim_id"],
                "text": item["text"],
                "text_sha256": item["text_sha256"],
            }
            for item in claims
        ],
        "evidence": [
            {
                "evidence_id": item["evidence_id"],
                "text": item["text"],
                "content_sha256": item["content_sha256"],
                "identity_sha256": item["identity_sha256"],
                "final_rank": item["final_rank"],
            }
            for item in evidence_inventory
        ],
    }


def citation_from_inventory(
    item: dict[str, object], support_score: float
) -> dict[str, object]:
    return {
        key: value
        for key, value in item.items()
        if key != "text"
    } | {"support_score": support_score}


class EvidenceClaimsResponse(ExactClaimResponse):
    schema_version: int = Field(strict=True, ge=SCHEMA_VERSION, le=SCHEMA_VERSION)
    claim_profile: Literal["robotops_document_claim_evidence_v1"]
    trust_boundary: Literal["UNTRUSTED_DATA"]
    query: str = Field(min_length=1, max_length=2_000)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    doc_type: EvidenceDocType | None
    draft: EvidenceClaimsDraftResponse
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claims: list[EvidenceClaimResponse] = Field(min_length=1, max_length=10)
    claim_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rerank: EvidenceRerankResponse
    rerank_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_inventory: list[EvidenceInventoryItemResponse] = Field(max_length=10)
    evidence_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binder: ClaimBinderTraceResponse
    results: list[EvidenceClaimResultResponse] = Field(min_length=1, max_length=10)
    results_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counts: EvidenceClaimCountsResponse
    need_human: bool
    side_effects_allowed: Literal[False]
    status: ClaimsTopStatus
    error_code: ClaimsTopErrorCode | None
    reason_codes: list[ClaimReasonCode]
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("query")
    @classmethod
    def query_is_utf8_encodable(cls, value: str) -> str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("query must not contain isolated Unicode surrogates")
        return value

    @field_validator("rerank", mode="before")
    @classmethod
    def rerank_instance_is_dumped_for_fresh_validation(cls, value: object) -> object:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="python")
        return value

    @field_validator("side_effects_allowed", mode="before")
    @classmethod
    def side_effects_flag_is_exact_false(cls, value: object) -> object:
        if type(value) is not bool or value is not False:
            raise ValueError("side effects flag must be the exact false boolean")
        return value

    @model_validator(mode="after")
    def validate_complete_contract(self) -> "EvidenceClaimsResponse":
        draft_dump = self.draft.model_dump(mode="python")
        if self.query_sha256 != canonical_sha256(self.query):
            raise ValueError("query digest mismatch")
        if self.request_sha256 != canonical_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "query_sha256": self.query_sha256,
                "doc_type": self.doc_type,
                "draft": draft_dump,
            }
        ):
            raise ValueError("request digest mismatch")

        expected_claims = claims_from_draft(self.draft)
        actual_claims = [item.model_dump(mode="python") for item in self.claims]
        if actual_claims != expected_claims:
            raise ValueError("claim set is not server-derived from the draft")
        if self.claim_set_sha256 != canonical_sha256(actual_claims):
            raise ValueError("claim set digest mismatch")

        rerank_dump = self.rerank.model_dump(mode="python")
        if (
            self.rerank.query != self.query
            or self.rerank.query_sha256 != self.query_sha256
            or self.rerank.doc_type != self.doc_type
            or self.rerank.requested_top_k != 10
            or self.rerank.candidate_k != 20
            or [item.final_rank for item in self.rerank.results]
            != list(range(1, len(self.rerank.results) + 1))
        ):
            raise ValueError("rerank snapshot is not bound to the claim request")
        if canonical_size(rerank_dump) > MAX_RERANK_SNAPSHOT_BYTES:
            raise ValueError("rerank snapshot exceeds the frozen byte boundary")
        if self.rerank_snapshot_sha256 != canonical_sha256(rerank_dump):
            raise ValueError("rerank snapshot digest mismatch")

        expected_inventory = inventory_from_rerank(self.rerank)
        actual_inventory = [
            item.model_dump(mode="python") for item in self.evidence_inventory
        ]
        if actual_inventory != expected_inventory:
            raise ValueError("evidence inventory is not derived from rerank results")
        if canonical_size(actual_inventory) > MAX_INVENTORY_BYTES:
            raise ValueError("evidence inventory exceeds the frozen byte boundary")
        if self.evidence_inventory_sha256 != canonical_sha256(actual_inventory):
            raise ValueError("evidence inventory digest mismatch")

        input_projection = binder_input_projection(
            claim_set_sha256=self.claim_set_sha256,
            rerank=self.rerank,
            evidence_inventory_sha256=self.evidence_inventory_sha256,
            claims=actual_claims,
            evidence_inventory=actual_inventory,
        )
        if canonical_size(input_projection) > MAX_BINDER_INPUT_BYTES:
            raise ValueError("binder input exceeds the frozen byte boundary")
        if (
            self.binder.input_claim_count != len(actual_claims)
            or self.binder.input_evidence_count != len(actual_inventory)
            or self.binder.expected_pair_count
            != len(actual_claims) * len(actual_inventory)
            or self.binder.input_sha256 != canonical_sha256(input_projection)
        ):
            raise ValueError("binder input projection mismatch")

        matrix = [item.model_dump(mode="python") for item in self.binder.matrix]
        if self.binder.status == "SUCCESS":
            expected_pairs = [
                (claim["claim_id"], evidence["evidence_id"])
                for claim in actual_claims
                for evidence in actual_inventory
            ]
            actual_pairs = [
                (item["claim_id"], item["evidence_id"]) for item in matrix
            ]
            if actual_pairs != expected_pairs:
                raise ValueError("binder matrix is not the canonical Cartesian product")
        elif matrix:
            raise ValueError("untrusted binder state must not publish a matrix")

        expected_results = self._expected_results(
            actual_claims=actual_claims,
            actual_inventory=actual_inventory,
            matrix=matrix,
        )
        actual_results = [item.model_dump(mode="python") for item in self.results]
        if actual_results != expected_results:
            raise ValueError("claim result projection mismatch")
        if self.results_sha256 != canonical_sha256(actual_results):
            raise ValueError("claim results digest mismatch")

        expected_counts = {
            "total": len(actual_results),
            "supported": sum(item["disposition"] == "PUBLISH" for item in actual_results),
            "uncertain": sum(
                item["disposition"] == "MARK_UNCERTAIN" for item in actual_results
            ),
            "omitted": sum(item["disposition"] == "OMIT" for item in actual_results),
            "escalated": sum(
                item["disposition"] == "ESCALATE" for item in actual_results
            ),
            "blocked": sum(item["disposition"] == "BLOCK" for item in actual_results),
        }
        if self.counts.model_dump(mode="python") != expected_counts:
            raise ValueError("claim counts mismatch")
        expected_need_human = any(
            item["disposition"] in {"ESCALATE", "BLOCK"}
            for item in actual_results
        )
        if self.need_human != expected_need_human:
            raise ValueError("human escalation flag mismatch")

        expected_reasons = list(self.rerank.reason_codes)
        if self.binder.error_code is not None:
            expected_reasons.append(self.binder.error_code)
        insufficient = {item["kind"] for item in actual_results if item["verdict"] == "INSUFFICIENT_EVIDENCE"}
        for kind, code in (
            ("CAUSE", "CAUSE_INSUFFICIENT_EVIDENCE"),
            ("STEP", "STEP_INSUFFICIENT_EVIDENCE"),
            ("RISK", "RISK_INSUFFICIENT_EVIDENCE"),
        ):
            if kind in insufficient:
                expected_reasons.append(code)
        if self.reason_codes != expected_reasons:
            raise ValueError("claim reason ordering mismatch")

        if self.rerank.status == "FAILED":
            expected_status = "FAILED"
            if self.error_code is None:
                raise ValueError("failed claim response requires a stable error")
            safe_failure_suffix = {
                "CLAIM_RETRIEVAL_UNAVAILABLE": "UNAVAILABLE",
                "CLAIM_RETRIEVAL_INVALID_RESULT": "INVALID_RESULT",
                "CLAIM_RESPONSE_INVALID": "INVALID_RESULT",
            }.get(self.error_code)
            if safe_failure_suffix is not None and (
                self.rerank.retrieval.reason_codes != [
                    f"LEXICAL_LANE_{safe_failure_suffix}",
                    f"VECTOR_LANE_{safe_failure_suffix}",
                ]
            ):
                raise ValueError("claim failure kind does not match its safe snapshot")
        else:
            all_supported = all(
                item["verdict"] == "SUPPORTED" for item in actual_results
            )
            expected_status = (
                "SUCCESS"
                if self.rerank.status == "SUCCESS"
                and self.binder.status == "SUCCESS"
                and all_supported
                else "DEGRADED"
            )
            if self.error_code is not None:
                raise ValueError("200 claim response cannot expose a 503 error")
        if self.status != expected_status:
            raise ValueError("top-level claim status mismatch")

        if self.binder.status == "DISCARDED_RESPONSE_INVALID":
            if self.status != "FAILED" or self.error_code != "CLAIM_RESPONSE_INVALID":
                raise ValueError("discarded response state lost its priority")
        elif self.rerank.status == "FAILED":
            if self.binder.status != "SKIPPED_RETRIEVAL_FAILED":
                raise ValueError("failed rerank must skip binding")
        elif not actual_inventory:
            if self.binder.status != "SKIPPED_EMPTY":
                raise ValueError("empty evidence must skip binding")
        elif self.binder.status in {
            "SKIPPED_EMPTY",
            "SKIPPED_RETRIEVAL_FAILED",
            "DISCARDED_RESPONSE_INVALID",
        }:
            raise ValueError("non-empty evidence has an invalid skip state")

        trace_projection = {
            "schema_version": SCHEMA_VERSION,
            "claim_profile": self.claim_profile,
            "query_sha256": self.query_sha256,
            "doc_type": self.doc_type,
            "request_sha256": self.request_sha256,
            "claim_set_sha256": self.claim_set_sha256,
            "rerank_snapshot_sha256": self.rerank_snapshot_sha256,
            "evidence_inventory_sha256": self.evidence_inventory_sha256,
            "binder": {
                key: getattr(self.binder, key)
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
            "results_sha256": self.results_sha256,
            "counts": expected_counts,
            "need_human": self.need_human,
            "side_effects_allowed": self.side_effects_allowed,
            "status": self.status,
            "error_code": self.error_code,
            "reason_codes": self.reason_codes,
        }
        if self.trace_sha256 != canonical_sha256(trace_projection):
            raise ValueError("claim trace digest mismatch")
        if canonical_size(self.model_dump(mode="python")) > MAX_RESPONSE_BYTES:
            raise ValueError("claim response exceeds the frozen byte boundary")
        return self

    def _expected_results(
        self,
        *,
        actual_claims: list[dict[str, object]],
        actual_inventory: list[dict[str, object]],
        matrix: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        inventory_by_id = {
            item["evidence_id"]: item for item in actual_inventory
        }
        scores_by_claim: dict[str, list[tuple[dict[str, object], float]]] = {
            str(item["claim_id"]): [] for item in actual_claims
        }
        for pair in matrix:
            if pair["support_score"] >= SUPPORT_THRESHOLD:
                evidence = inventory_by_id[str(pair["evidence_id"])]
                scores_by_claim[str(pair["claim_id"])].append(
                    (evidence, float(pair["support_score"]))
                )

        trusted_for_coverage = self.binder.status in {"SUCCESS", "SKIPPED_EMPTY"}
        expected_results: list[dict[str, object]] = []
        for claim in actual_claims:
            supported = sorted(
                scores_by_claim[str(claim["claim_id"])],
                key=lambda pair: (
                    -pair[1],
                    pair[0]["final_rank"],
                    pair[0]["evidence_id"],
                ),
            )[:3]
            citations = [
                citation_from_inventory(evidence, score)
                for evidence, score in supported
            ]
            if citations and self.binder.status == "SUCCESS":
                verdict = "SUPPORTED"
                disposition = "PUBLISH"
            elif trusted_for_coverage:
                verdict = "INSUFFICIENT_EVIDENCE"
                disposition = {
                    "CAUSE": "MARK_UNCERTAIN",
                    "STEP": "OMIT",
                    "RISK": "ESCALATE",
                }[str(claim["kind"])]
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
            expected_results.append(
                {**base, "claim_result_sha256": canonical_sha256(base)}
            )
        return expected_results
