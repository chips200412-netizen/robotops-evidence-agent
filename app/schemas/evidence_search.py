"""MOD-025A typed API contract for traceable hybrid retrieval."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EvidenceDocType = Literal["manual", "fault_case"]
EvidenceLaneStatus = Literal["SUCCESS", "FAILED"]
EvidenceSearchStatus = Literal[
    "HYBRID_SUCCESS",
    "DEGRADED_LEXICAL_ONLY",
    "DEGRADED_VECTOR_ONLY",
    "FAILED",
]
EvidenceReasonCode = Literal[
    "LEXICAL_LANE_UNAVAILABLE",
    "LEXICAL_LANE_INVALID_RESULT",
    "VECTOR_LANE_UNAVAILABLE",
    "VECTOR_LANE_INVALID_RESULT",
    "CROSS_LANE_IDENTITY_CONFLICT",
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


class StrictEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ExactEvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceSearchRequest(StrictEvidenceRequest):
    query: str = Field(min_length=1, max_length=2_000)
    doc_type: EvidenceDocType | None = None
    top_k: int = Field(default=5, strict=True, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_is_utf8_encodable(cls, value: str) -> str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("query must not contain isolated Unicode surrogates")
        return value


class EvidenceChunkResponse(ExactEvidenceResponse):
    chunk_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._~-]+$")
    doc_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._~-]+$")
    filename: str = Field(min_length=1, max_length=512)
    doc_type: EvidenceDocType
    chunk_index: int = Field(strict=True, ge=0)
    text: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

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
    def text_is_utf8_encodable(cls, value: str) -> str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("text must not contain isolated Unicode surrogates")
        return value

    @model_validator(mode="after")
    def validate_hashes(self) -> "EvidenceChunkResponse":
        content_sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        identity = {
            "schema_version": 1,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "doc_type": self.doc_type,
            "chunk_index": self.chunk_index,
            "content_sha256": content_sha256,
        }
        if self.content_sha256 != content_sha256:
            raise ValueError("content digest mismatch")
        if self.identity_sha256 != _canonical_sha256(identity):
            raise ValueError("identity digest mismatch")
        return self


class LexicalCandidateResponse(EvidenceChunkResponse):
    rank: int = Field(strict=True, ge=1)
    raw_score: float = Field(strict=True, gt=0.0)


class VectorCandidateResponse(EvidenceChunkResponse):
    rank: int = Field(strict=True, ge=1)
    raw_score: float = Field(strict=True, ge=0.0, le=1.0)
    distance: float = Field(strict=True, ge=0.0)


class LexicalLaneResponse(ExactEvidenceResponse):
    status: EvidenceLaneStatus
    error_code: Literal[
        "LEXICAL_LANE_UNAVAILABLE", "LEXICAL_LANE_INVALID_RESULT"
    ] | None
    count: int = Field(strict=True, ge=0, le=20)
    candidates: list[LexicalCandidateResponse] = Field(max_length=20)
    candidates_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_lane(self) -> "LexicalLaneResponse":
        if self.count != len(self.candidates):
            raise ValueError("lexical count does not match candidates")
        if [item.rank for item in self.candidates] != list(range(1, self.count + 1)):
            raise ValueError("lexical ranks must be contiguous and one-based")
        if len({item.chunk_id for item in self.candidates}) != self.count:
            raise ValueError("lexical chunk ids must be unique")
        if self.candidates != sorted(
            self.candidates, key=lambda item: (-item.raw_score, item.chunk_id)
        ):
            raise ValueError("lexical candidates are not in frozen rank order")
        if self.candidates_sha256 != _canonical_sha256(
            [item.model_dump(mode="python") for item in self.candidates]
        ):
            raise ValueError("lexical candidates digest mismatch")
        if self.status == "SUCCESS" and self.error_code is not None:
            raise ValueError("successful lexical lane cannot have an error")
        if self.status == "FAILED" and (
            self.error_code is None or self.count != 0
        ):
            raise ValueError("failed lexical lane must expose one code and no candidates")
        return self


class VectorLaneResponse(ExactEvidenceResponse):
    status: EvidenceLaneStatus
    error_code: Literal["VECTOR_LANE_UNAVAILABLE", "VECTOR_LANE_INVALID_RESULT"] | None
    count: int = Field(strict=True, ge=0, le=20)
    candidates: list[VectorCandidateResponse] = Field(max_length=20)
    candidates_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_lane(self) -> "VectorLaneResponse":
        if self.count != len(self.candidates):
            raise ValueError("vector count does not match candidates")
        if [item.rank for item in self.candidates] != list(range(1, self.count + 1)):
            raise ValueError("vector ranks must be contiguous and one-based")
        if len({item.chunk_id for item in self.candidates}) != self.count:
            raise ValueError("vector chunk ids must be unique")
        if self.candidates != sorted(
            self.candidates, key=lambda item: (-item.raw_score, item.chunk_id)
        ):
            raise ValueError("vector candidates are not in frozen rank order")
        for item in self.candidates:
            expected_similarity = max(0.0, min(1.0, 1.0 - item.distance / 2.0))
            if item.raw_score != expected_similarity:
                raise ValueError("vector similarity does not match raw distance")
        if self.candidates_sha256 != _canonical_sha256(
            [item.model_dump(mode="python") for item in self.candidates]
        ):
            raise ValueError("vector candidates digest mismatch")
        if self.status == "SUCCESS" and self.error_code is not None:
            raise ValueError("successful vector lane cannot have an error")
        if self.status == "FAILED" and (
            self.error_code is None or self.count != 0
        ):
            raise ValueError("failed vector lane must expose one code and no candidates")
        return self


class EvidenceLanesResponse(ExactEvidenceResponse):
    lexical: LexicalLaneResponse
    vector: VectorLaneResponse


class EvidenceFusionResponse(ExactEvidenceResponse):
    method: Literal["RRF"]
    rrf_k: int = Field(strict=True, ge=60, le=60)
    lexical_weight: float = Field(strict=True, ge=1.0, le=1.0)
    vector_weight: float = Field(strict=True, ge=1.0, le=1.0)
    candidate_k: int = Field(strict=True, ge=20, le=20)
    tie_break: list[Literal["FUSED_SCORE_DESC", "CHUNK_ID_ASC"]]

    @model_validator(mode="after")
    def exact_tie_break(self) -> "EvidenceFusionResponse":
        if self.tie_break != ["FUSED_SCORE_DESC", "CHUNK_ID_ASC"]:
            raise ValueError("fusion tie break does not match the frozen profile")
        return self


class EvidenceSearchResultResponse(EvidenceChunkResponse):
    fused_score: float = Field(strict=True, gt=0.0)
    lexical_rank: int | None = Field(default=None, strict=True, ge=1)
    vector_rank: int | None = Field(default=None, strict=True, ge=1)
    lexical_raw_score: float | None = Field(default=None, strict=True, gt=0.0)
    vector_raw_score: float | None = Field(
        default=None, strict=True, ge=0.0, le=1.0
    )
    vector_distance: float | None = Field(default=None, strict=True, ge=0.0)
    matched_lanes: list[Literal["lexical", "vector"]]

    @model_validator(mode="after")
    def validate_lane_projection(self) -> "EvidenceSearchResultResponse":
        expected_lanes: list[str] = []
        if self.lexical_rank is not None:
            expected_lanes.append("lexical")
            if self.lexical_raw_score is None:
                raise ValueError("lexical rank requires its raw score")
        elif self.lexical_raw_score is not None:
            raise ValueError("lexical raw score requires its rank")
        if self.vector_rank is not None:
            expected_lanes.append("vector")
            if self.vector_raw_score is None or self.vector_distance is None:
                raise ValueError("vector rank requires raw score and distance")
        elif self.vector_raw_score is not None or self.vector_distance is not None:
            raise ValueError("vector score and distance require its rank")
        if not expected_lanes or self.matched_lanes != expected_lanes:
            raise ValueError("matched_lanes does not match lane rank projection")
        return self


class EvidenceSearchResponse(ExactEvidenceResponse):
    schema_version: int = Field(strict=True, ge=1, le=1)
    retrieval_profile: Literal["robotops_hybrid_retrieval_v1"]
    trust_boundary: Literal["UNTRUSTED_DATA"]
    query: str = Field(min_length=1, max_length=2_000)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    doc_type: EvidenceDocType | None
    top_k: int = Field(strict=True, ge=1, le=20)
    candidate_k: int = Field(strict=True, ge=20, le=20)
    status: EvidenceSearchStatus
    reason_codes: list[EvidenceReasonCode]
    fusion: EvidenceFusionResponse
    lanes: EvidenceLanesResponse
    count: int = Field(strict=True, ge=0)
    results: list[EvidenceSearchResultResponse]
    results_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("query")
    @classmethod
    def query_is_utf8_encodable(cls, value: str) -> str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("query must not contain isolated Unicode surrogates")
        return value

    @model_validator(mode="after")
    def validate_response(self) -> "EvidenceSearchResponse":
        if self.query_sha256 != _canonical_sha256(self.query):
            raise ValueError("query digest mismatch")
        if self.candidate_k != self.fusion.candidate_k:
            raise ValueError("candidate window does not match fusion profile")
        if self.count != len(self.results) or self.count > self.top_k:
            raise ValueError("result count is inconsistent")
        if self.results_sha256 != _canonical_sha256(
            [item.model_dump(mode="python") for item in self.results]
        ):
            raise ValueError("results digest mismatch")

        lexical = self.lanes.lexical
        vector = self.lanes.vector
        lexical_by_id = {item.chunk_id: item for item in lexical.candidates}
        vector_by_id = {item.chunk_id: item for item in vector.candidates}
        if self.doc_type is not None and any(
            item.doc_type != self.doc_type
            for item in [*lexical.candidates, *vector.candidates, *self.results]
        ):
            raise ValueError("lane candidate escaped the requested document filter")
        shared_ids = set(lexical_by_id).intersection(vector_by_id)
        conflicting_ids = {
            chunk_id
            for chunk_id in shared_ids
            if (
                lexical_by_id[chunk_id].identity_sha256
                != vector_by_id[chunk_id].identity_sha256
                or lexical_by_id[chunk_id].content_sha256
                != vector_by_id[chunk_id].content_sha256
            )
        }
        if self.status == "HYBRID_SUCCESS":
            valid = (
                lexical.status == "SUCCESS"
                and vector.status == "SUCCESS"
                and self.reason_codes == []
                and not conflicting_ids
            )
        elif self.status == "DEGRADED_LEXICAL_ONLY":
            valid = (
                lexical.status == "SUCCESS"
                and vector.status == "FAILED"
                and self.reason_codes == [vector.error_code]
            )
        elif self.status == "DEGRADED_VECTOR_ONLY":
            valid = (
                lexical.status == "FAILED"
                and vector.status == "SUCCESS"
                and self.reason_codes == [lexical.error_code]
            )
        else:
            both_failed = (
                lexical.status == "FAILED"
                and vector.status == "FAILED"
                and self.reason_codes == [lexical.error_code, vector.error_code]
            )
            identity_conflict = (
                lexical.status == "SUCCESS"
                and vector.status == "SUCCESS"
                and self.reason_codes == ["CROSS_LANE_IDENTITY_CONFLICT"]
                and bool(conflicting_ids)
            )
            valid = (both_failed or identity_conflict) and not self.results
        if not valid:
            raise ValueError("overall status does not match lane states")

        if self.status != "FAILED":
            expected_results: list[dict] = []
            for chunk_id in set(lexical_by_id).union(vector_by_id):
                lexical_item = lexical_by_id.get(chunk_id)
                vector_item = vector_by_id.get(chunk_id)
                source = lexical_item or vector_item
                if source is None:  # pragma: no cover - union guarantees a source
                    raise ValueError("result source is missing")
                lexical_rank = lexical_item.rank if lexical_item else None
                vector_rank = vector_item.rank if vector_item else None
                score = 0.0
                matched_lanes: list[str] = []
                if lexical_item is not None:
                    score += 1.0 / (60 + lexical_rank)
                    matched_lanes.append("lexical")
                if vector_item is not None:
                    score += 1.0 / (60 + vector_rank)
                    matched_lanes.append("vector")
                expected_results.append(
                    {
                        "chunk_id": source.chunk_id,
                        "doc_id": source.doc_id,
                        "filename": source.filename,
                        "doc_type": source.doc_type,
                        "chunk_index": source.chunk_index,
                        "text": source.text,
                        "content_sha256": source.content_sha256,
                        "identity_sha256": source.identity_sha256,
                        "fused_score": score,
                        "lexical_rank": lexical_rank,
                        "vector_rank": vector_rank,
                        "lexical_raw_score": (
                            lexical_item.raw_score if lexical_item else None
                        ),
                        "vector_raw_score": vector_item.raw_score if vector_item else None,
                        "vector_distance": vector_item.distance if vector_item else None,
                        "matched_lanes": matched_lanes,
                    }
                )
            expected_results.sort(
                key=lambda item: (-item["fused_score"], item["chunk_id"])
            )
            expected_results = expected_results[: self.top_k]
            actual_results = [item.model_dump(mode="python") for item in self.results]
            if actual_results != expected_results:
                raise ValueError("fused results do not match lane candidates")
        return self
