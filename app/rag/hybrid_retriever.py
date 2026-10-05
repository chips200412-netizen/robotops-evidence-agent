"""MOD-025A deterministic lexical + vector retrieval with traceable RRF."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
import unicodedata
from typing import Any

from app.rag import vector_store
from app.rag.embeddings import embed_query


SCHEMA_VERSION = 1
RETRIEVAL_PROFILE = "robotops_hybrid_retrieval_v1"
ANALYZER = "unicode_nfkc_lower_cjk_bigram_v1"
TRUST_BOUNDARY = "UNTRUSTED_DATA"
CANDIDATE_K = 20
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60
LEXICAL_WEIGHT = 1.0
VECTOR_WEIGHT = 1.0
_SPAN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


class _InvalidLaneResult(ValueError):
    pass


class _IdentityConflict(ValueError):
    pass


def _contains_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def canonical_json(value: Any) -> str:
    """Version-local canonical JSON; rejects non-finite numbers."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def analyze(text: str) -> list[str]:
    """NFKC/lower plus ASCII tokens and adjacent BMP Han bigrams."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _SPAN_RE.finditer(normalized):
        span = match.group(0)
        if span.isascii():
            tokens.append(span)
        elif len(span) == 1:
            tokens.append(span)
        else:
            tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
    return tokens


def _normalize_chunk(item: object) -> dict[str, Any]:
    if not isinstance(item, dict) or not {"id", "text", "metadata"}.issubset(item):
        raise _InvalidLaneResult("chunk envelope is invalid")
    chunk_id = item.get("id")
    text = item.get("text")
    metadata = item.get("metadata")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise _InvalidLaneResult("chunk id is invalid")
    if _contains_surrogate(chunk_id):
        raise _InvalidLaneResult("chunk id is not UTF-8 encodable")
    if not isinstance(text, str) or not isinstance(metadata, dict):
        raise _InvalidLaneResult("chunk content is invalid")
    if _contains_surrogate(text):
        raise _InvalidLaneResult("chunk text is not UTF-8 encodable")

    doc_id = metadata.get("doc_id")
    filename = metadata.get("filename")
    doc_type = metadata.get("doc_type")
    chunk_index = metadata.get("chunk_index")
    if not isinstance(doc_id, str) or not doc_id:
        raise _InvalidLaneResult("document id is invalid")
    if _contains_surrogate(doc_id):
        raise _InvalidLaneResult("document id is not UTF-8 encodable")
    if not isinstance(filename, str) or not filename:
        raise _InvalidLaneResult("filename is invalid")
    if _contains_surrogate(filename):
        raise _InvalidLaneResult("filename is not UTF-8 encodable")
    if (
        PurePosixPath(filename).name != filename
        or PureWindowsPath(filename).name != filename
        or PureWindowsPath(filename).drive
    ):
        raise _InvalidLaneResult("filename must not expose a path")
    if doc_type not in {"manual", "fault_case"}:
        raise _InvalidLaneResult("document type is invalid")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise _InvalidLaneResult("chunk index is invalid")

    content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "filename": filename,
        "doc_type": doc_type,
        "chunk_index": chunk_index,
        "content_sha256": content_sha256,
    }
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "filename": filename,
        "doc_type": doc_type,
        "chunk_index": chunk_index,
        "text": text,
        "content_sha256": content_sha256,
        "identity_sha256": canonical_sha256(identity),
    }


def _empty_lane(status: str, error_code: str | None = None) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    return {
        "status": status,
        "error_code": error_code,
        "count": 0,
        "candidates": candidates,
        "candidates_sha256": canonical_sha256(candidates),
    }


def _lexical_candidates(doc_type: str | None, query: str) -> list[dict[str, Any]]:
    raw_chunks = vector_store.list_chunks(doc_type=doc_type)
    normalized: list[tuple[dict[str, Any], list[str]]] = []
    seen: set[str] = set()
    for raw in raw_chunks:
        chunk = _normalize_chunk(raw)
        if doc_type is not None and chunk["doc_type"] != doc_type:
            raise _InvalidLaneResult("lexical adapter escaped document filter")
        if chunk["chunk_id"] in seen:
            raise _InvalidLaneResult("duplicate chunk id")
        seen.add(chunk["chunk_id"])
        normalized.append((chunk, analyze(chunk["text"])))

    query_terms = list(dict.fromkeys(analyze(query)))
    if not normalized or not query_terms:
        return []

    document_count = len(normalized)
    lengths = [len(tokens) for _, tokens in normalized]
    average_length = sum(lengths) / document_count
    if average_length <= 0:
        return []
    counters = [Counter(tokens) for _, tokens in normalized]
    document_frequency = {
        term: sum(1 for counter in counters if counter.get(term, 0) > 0)
        for term in query_terms
    }

    scored: list[tuple[float, dict[str, Any]]] = []
    for (chunk, _tokens), counter, document_length in zip(
        normalized, counters, lengths, strict=True
    ):
        score = 0.0
        for term in query_terms:
            frequency = counter.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * document_length / average_length
            )
            score += idf * frequency * (BM25_K1 + 1.0) / denominator
        if not math.isfinite(score):
            raise _InvalidLaneResult("non-finite lexical score")
        if score > 0.0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
    candidates: list[dict[str, Any]] = []
    for rank, (score, chunk) in enumerate(scored[:CANDIDATE_K], start=1):
        candidates.append({**chunk, "rank": rank, "raw_score": float(score)})
    return candidates


def _vector_candidates(doc_type: str | None, query: str) -> list[dict[str, Any]]:
    embedding = embed_query(query)
    where = {"doc_type": doc_type} if doc_type else None
    raw_hits = vector_store.query(
        embedding=embedding,
        top_k=CANDIDATE_K,
        where=where,
    )
    prepared: list[tuple[float, float, dict[str, Any]]] = []
    seen: set[str] = set()
    for raw in raw_hits:
        chunk = _normalize_chunk(raw)
        if doc_type is not None and chunk["doc_type"] != doc_type:
            raise _InvalidLaneResult("vector adapter escaped document filter")
        if chunk["chunk_id"] in seen:
            raise _InvalidLaneResult("duplicate chunk id")
        seen.add(chunk["chunk_id"])
        distance = raw.get("distance") if isinstance(raw, dict) else None
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not math.isfinite(float(distance))
            or float(distance) < 0.0
        ):
            raise _InvalidLaneResult("vector distance is invalid")
        distance_float = float(distance)
        similarity = max(0.0, min(1.0, 1.0 - distance_float / 2.0))
        prepared.append((similarity, distance_float, chunk))

    prepared.sort(key=lambda item: (-item[0], item[2]["chunk_id"]))
    return [
        {
            **chunk,
            "rank": rank,
            "raw_score": float(similarity),
            "distance": float(distance),
        }
        for rank, (similarity, distance, chunk) in enumerate(
            prepared[:CANDIDATE_K], start=1
        )
    ]


def _run_lexical_lane(doc_type: str | None, query: str) -> dict[str, Any]:
    try:
        candidates = _lexical_candidates(doc_type, query)
    except _InvalidLaneResult:
        return _empty_lane("FAILED", "LEXICAL_LANE_INVALID_RESULT")
    except Exception:
        return _empty_lane("FAILED", "LEXICAL_LANE_UNAVAILABLE")
    return {
        "status": "SUCCESS",
        "error_code": None,
        "count": len(candidates),
        "candidates": candidates,
        "candidates_sha256": canonical_sha256(candidates),
    }


def _run_vector_lane(doc_type: str | None, query: str) -> dict[str, Any]:
    try:
        candidates = _vector_candidates(doc_type, query)
    except _InvalidLaneResult:
        return _empty_lane("FAILED", "VECTOR_LANE_INVALID_RESULT")
    except Exception:
        return _empty_lane("FAILED", "VECTOR_LANE_UNAVAILABLE")
    return {
        "status": "SUCCESS",
        "error_code": None,
        "count": len(candidates),
        "candidates": candidates,
        "candidates_sha256": canonical_sha256(candidates),
    }


def _identity_projection(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "chunk_id",
            "doc_id",
            "filename",
            "doc_type",
            "chunk_index",
            "text",
            "content_sha256",
            "identity_sha256",
        )
    }


def _fuse(
    lexical: dict[str, Any],
    vector: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    union: dict[str, dict[str, Any]] = {}
    for lane_name, lane in (("lexical", lexical), ("vector", vector)):
        if lane["status"] != "SUCCESS":
            continue
        for candidate in lane["candidates"]:
            chunk_id = candidate["chunk_id"]
            existing = union.get(chunk_id)
            identity = _identity_projection(candidate)
            if existing is None:
                existing = {
                    **identity,
                    "lexical_rank": None,
                    "vector_rank": None,
                    "lexical_raw_score": None,
                    "vector_raw_score": None,
                    "vector_distance": None,
                }
                union[chunk_id] = existing
            elif _identity_projection(existing) != identity:
                raise _IdentityConflict("cross-lane identity mismatch")

            if lane_name == "lexical":
                existing["lexical_rank"] = candidate["rank"]
                existing["lexical_raw_score"] = candidate["raw_score"]
            else:
                existing["vector_rank"] = candidate["rank"]
                existing["vector_raw_score"] = candidate["raw_score"]
                existing["vector_distance"] = candidate["distance"]

    results: list[dict[str, Any]] = []
    for item in union.values():
        score = 0.0
        matched_lanes: list[str] = []
        if item["lexical_rank"] is not None:
            score += LEXICAL_WEIGHT / (RRF_K + item["lexical_rank"])
            matched_lanes.append("lexical")
        if item["vector_rank"] is not None:
            score += VECTOR_WEIGHT / (RRF_K + item["vector_rank"])
            matched_lanes.append("vector")
        results.append(
            {
                **item,
                "fused_score": float(score),
                "matched_lanes": matched_lanes,
            }
        )
    results.sort(key=lambda item: (-item["fused_score"], item["chunk_id"]))
    return results[:top_k]


def search(
    *,
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> dict[str, Any]:
    """Execute both lanes independently and return a stable, typed-ready envelope."""
    lexical = _run_lexical_lane(doc_type, query)
    vector = _run_vector_lane(doc_type, query)

    if lexical["status"] == "FAILED" and vector["status"] == "FAILED":
        status = "FAILED"
        reason_codes = [lexical["error_code"], vector["error_code"]]
        results: list[dict[str, Any]] = []
    else:
        try:
            results = _fuse(lexical, vector, top_k)
        except _IdentityConflict:
            status = "FAILED"
            reason_codes = ["CROSS_LANE_IDENTITY_CONFLICT"]
            results = []
        else:
            if lexical["status"] == "SUCCESS" and vector["status"] == "SUCCESS":
                status = "HYBRID_SUCCESS"
                reason_codes = []
            elif lexical["status"] == "SUCCESS":
                status = "DEGRADED_LEXICAL_ONLY"
                reason_codes = [vector["error_code"]]
            else:
                status = "DEGRADED_VECTOR_ONLY"
                reason_codes = [lexical["error_code"]]

    return {
        "schema_version": SCHEMA_VERSION,
        "retrieval_profile": RETRIEVAL_PROFILE,
        "trust_boundary": TRUST_BOUNDARY,
        "query": query,
        "query_sha256": canonical_sha256(query),
        "doc_type": doc_type,
        "top_k": top_k,
        "candidate_k": CANDIDATE_K,
        "status": status,
        "reason_codes": reason_codes,
        "fusion": {
            "method": "RRF",
            "rrf_k": RRF_K,
            "lexical_weight": LEXICAL_WEIGHT,
            "vector_weight": VECTOR_WEIGHT,
            "candidate_k": CANDIDATE_K,
            "tie_break": ["FUSED_SCORE_DESC", "CHUNK_ID_ASC"],
        },
        "lanes": {"lexical": lexical, "vector": vector},
        "count": len(results),
        "results": results,
        "results_sha256": canonical_sha256(results),
    }


def failure_response(
    *,
    query: str,
    top_k: int,
    doc_type: str | None,
    invalid_result: bool,
) -> dict[str, Any]:
    """Build a request-bound, typed failure without exposing an internal exception."""
    lexical_code = (
        "LEXICAL_LANE_INVALID_RESULT"
        if invalid_result
        else "LEXICAL_LANE_UNAVAILABLE"
    )
    vector_code = (
        "VECTOR_LANE_INVALID_RESULT" if invalid_result else "VECTOR_LANE_UNAVAILABLE"
    )
    lexical = _empty_lane("FAILED", lexical_code)
    vector = _empty_lane("FAILED", vector_code)
    results: list[dict[str, Any]] = []
    return {
        "schema_version": SCHEMA_VERSION,
        "retrieval_profile": RETRIEVAL_PROFILE,
        "trust_boundary": TRUST_BOUNDARY,
        "query": query,
        "query_sha256": canonical_sha256(query),
        "doc_type": doc_type,
        "top_k": top_k,
        "candidate_k": CANDIDATE_K,
        "status": "FAILED",
        "reason_codes": [lexical_code, vector_code],
        "fusion": {
            "method": "RRF",
            "rrf_k": RRF_K,
            "lexical_weight": LEXICAL_WEIGHT,
            "vector_weight": VECTOR_WEIGHT,
            "candidate_k": CANDIDATE_K,
            "tie_break": ["FUSED_SCORE_DESC", "CHUNK_ID_ASC"],
        },
        "lanes": {"lexical": lexical, "vector": vector},
        "count": 0,
        "results": results,
        "results_sha256": canonical_sha256(results),
    }
