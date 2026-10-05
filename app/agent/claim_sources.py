"""Frozen, reproducible claim sources; no semantic judgments or publication."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any
import unicodedata

from app.agent import synthetic_evidence, tool_loop
from app.agent.state import MAX_TOOL_ITERATIONS
from app.schemas.evidence_claims import inventory_from_rerank
from app.schemas.evidence_rerank import EvidenceRerankRequest, EvidenceRerankResponse


SOURCE_PROFILE = "robotops_agent_claim_sources_v1"
MAX_SOURCE_BYTES = 524_288
MAX_INVENTORY_BYTES = 524_288
MAX_SNAPSHOT_BYTES = 2_097_152
MAX_INVENTORY_ITEMS = 512
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 100_000
CLAIM_SOURCE_INVALID = "CLAIM_SOURCE_INVALID"
CONTEXT_FIELDS = ("user_query", "device_model", "fault_category", "fault_description")
TOOL_FIELDS = (
    "device_model", "fault_category", "fault_description", "tool_capability_profile",
    "diagnostic_plan", "pending_tool_action", "completed_plan_steps", "tool_trajectory",
    "tool_iterations", "agent_route", "tool_loop_failed", "execution_stop_reason",
    "retrieved_documents", "tool_result_evidence", "verification_status",
    "verifier_result", "verifier_result_sha256", "effect_policy_status",
    "effect_proposal", "effect_proposal_sha256", "risk_level", "urgency", "need_human",
)
_SNAPSHOT_FIELDS = {
    "schema_version", "profile", "context", "rerank", "resource_metadata",
    "tool_snapshot", "inventory", "status", "snapshot_sha256",
}


class ClaimSourceError(ValueError):
    """Stable public failure without source data, paths, or adapter exceptions."""

    def __init__(self) -> None:
        self.code = CLAIM_SOURCE_INVALID
        super().__init__(CLAIM_SOURCE_INVALID)


def _reject() -> None:
    raise ClaimSourceError()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _json_copy(value: object, *, maximum: int) -> Any:
    """Reject hostile types and bound traversal before serialization or copying."""
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]
    ancestors: set[int] = set()
    nodes = 0
    discovered = 1
    while stack:
        current, depth, leaving = stack.pop()
        if leaving:
            ancestors.remove(id(current))
            continue
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            _reject()
        kind = type(current)
        if kind is dict or kind is list:
            identity = id(current)
            if identity in ancestors:
                _reject()
            ancestors.add(identity)
            stack.append((current, depth, True))
            if kind is dict:
                if any(type(key) is not str for key in current):
                    _reject()
                discovered += len(current) * 2
                if discovered > MAX_JSON_NODES:
                    _reject()
                for key, item in current.items():
                    stack.append((item, depth + 1, False))
                    stack.append((key, depth + 1, False))
            else:
                discovered += len(current)
                if discovered > MAX_JSON_NODES:
                    _reject()
                stack.extend((item, depth + 1, False) for item in current)
        elif kind is str:
            if len(current) > maximum:
                _reject()
            current.encode("utf-8", errors="strict")
        elif kind is float:
            if not math.isfinite(current):
                _reject()
        elif kind is not int and kind is not bool and kind is not type(None):
            _reject()

    chunks: list[str] = []
    size = 0
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(value):
        size += len(chunk.encode("utf-8"))
        if size > maximum:
            _reject()
        chunks.append(chunk)
    return json.loads("".join(chunks))


def _exact_fields(value: object, fields: object) -> None:
    if type(value) is not dict or set(value) != set(fields):
        _reject()


def _validate_context(context: object) -> None:
    _exact_fields(context, CONTEXT_FIELDS)
    if any(type(context[name]) is not str for name in CONTEXT_FIELDS):
        _reject()
    request = EvidenceRerankRequest.model_validate(
        {"query": context["user_query"], "doc_type": None, "top_k": 10}
    )
    if request.query != context["user_query"]:
        _reject()
    for name, maximum in (
        ("device_model", 100), ("fault_category", 100), ("fault_description", 1_000)
    ):
        text = context[name]
        if len(text) > maximum or unicodedata.normalize("NFKC", text).strip() != text:
            _reject()


def _validate_process(tool_snapshot: dict, context: dict, metadata: list) -> None:
    _exact_fields(tool_snapshot, TOOL_FIELDS)
    field_types = {
        **{key: str for key in (
            "device_model", "fault_category", "fault_description", "tool_capability_profile",
            "agent_route", "execution_stop_reason", "verification_status",
            "verifier_result_sha256", "effect_policy_status", "effect_proposal_sha256",
            "risk_level", "urgency",
        )},
        **{key: dict for key in (
            "diagnostic_plan", "pending_tool_action", "tool_result_evidence",
            "verifier_result", "effect_proposal",
        )},
        **{key: list for key in (
            "completed_plan_steps", "tool_trajectory", "retrieved_documents",
        )},
        "tool_iterations": int, "tool_loop_failed": bool, "need_human": bool,
    }
    if any(type(tool_snapshot[key]) is not kind for key, kind in field_types.items()):
        _reject()
    if not 0 <= tool_snapshot["tool_iterations"] <= MAX_TOOL_ITERATIONS:
        _reject()
    if any(tool_snapshot[name] != context[name] for name in CONTEXT_FIELDS[1:]):
        _reject()
    if (
        tool_snapshot["tool_capability_profile"] != "synthetic_evidence_v1"
        or tool_snapshot["effect_policy_status"] != "NOT_STARTED"
        or tool_snapshot["effect_proposal"] != {}
        or tool_snapshot["effect_proposal_sha256"] != ""
        or type(tool_snapshot["need_human"]) is not bool
        or tool_snapshot["risk_level"] not in {"", "低", "中", "高"}
        or tool_snapshot["urgency"] not in {"", "低", "中", "高"}
    ):
        _reject()
    normalized = tool_loop.validate_checkpoint_progress(
        tool_snapshot, resource_metadata=metadata
    )
    # The legacy validator may normalize its inputs; stored sources cannot do so.
    for key, value in normalized.items():
        if key not in tool_snapshot or _canonical(value) != _canonical(tool_snapshot[key]):
            _reject()


def _inventory(rerank: EvidenceRerankResponse, process: dict) -> list[dict]:
    inventory: list[dict] = []

    def append(source_kind: str, text: str, locator: dict) -> None:
        if len(inventory) >= MAX_INVENTORY_ITEMS:
            _reject()
        item = {"source_kind": source_kind, "text": text, "locator": locator}
        inventory.append({"evidence_id": _digest(item), **item})

    for source in inventory_from_rerank(rerank):
        append(
            source["source_kind"], source["text"],
            {key: value for key, value in source.items()
             if key not in {"evidence_id", "source_kind", "text"}},
        )
    for event in process["tool_trajectory"]:
        if event["event"] != "TOOL" or event["tool_name"] not in {
            synthetic_evidence.SIMULATED_TELEMETRY_TOOL,
            synthetic_evidence.SYNTHETIC_ROS_LOG_TOOL,
        }:
            continue
        bound = process["tool_result_evidence"][event["step_id"]]
        result = bound["result"]
        telemetry = event["tool_name"] == synthetic_evidence.SIMULATED_TELEMETRY_TOOL
        for index, record in enumerate(result["samples" if telemetry else "entries"]):
            locator = {
                "step_id": event["step_id"], "sequence": event["sequence"],
                "tool_name": event["tool_name"], "device_model": result["device_model"],
                "resource_id": result["resource_id"], "start_ms": result["start_ms"],
                "end_ms": result["end_ms"], "file_sha256": result["resource_sha256"],
                "result_sha256": bound["result_sha256"], "record_index": index,
                "record": record,
            }
            append("TELEMETRY_SAMPLE" if telemetry else "ROS_LOG_ENTRY",
                   _canonical(record), locator)
    return _json_copy(inventory, maximum=MAX_INVENTORY_BYTES)


def _rebuild(context: dict, rerank: dict, metadata: list, process: dict) -> dict:
    _validate_context(context)
    normalized_rerank = EvidenceRerankResponse.model_validate(rerank)
    if (
        normalized_rerank.query != context["user_query"]
        or normalized_rerank.doc_type is not None
        or normalized_rerank.requested_top_k != 10
        or _canonical(normalized_rerank.model_dump(mode="python")) != _canonical(rerank)
    ):
        _reject()
    validated_metadata = synthetic_evidence.validate_resource_metadata(metadata)
    if _canonical(validated_metadata) != _canonical(metadata):
        _reject()
    _validate_process(process, context, metadata)
    inventory = _inventory(normalized_rerank, process)
    snapshot = {
        "schema_version": 1, "profile": SOURCE_PROFILE, "context": context,
        "rerank": rerank, "resource_metadata": metadata, "tool_snapshot": process,
        "inventory": inventory, "status": "READY" if inventory else "EMPTY",
    }
    snapshot["snapshot_sha256"] = _digest(snapshot)
    _json_copy({key: value for key, value in snapshot.items() if key != "inventory"},
               maximum=MAX_SOURCE_BYTES)
    return _json_copy(snapshot, maximum=MAX_SNAPSHOT_BYTES)


def freeze_claim_sources(state: object, rerank: object) -> dict:
    """Freeze only already-acquired sources, with one metadata read."""
    try:
        if type(state) is not dict or any(type(key) is not str for key in state):
            _reject()
        # Unrelated conversation/runtime objects are neither read nor copied.
        source = _json_copy({
            "context": {name: state[name] for name in CONTEXT_FIELDS},
            "tool_snapshot": {name: state[name] for name in TOOL_FIELDS},
            "rerank": rerank,
        }, maximum=MAX_SOURCE_BYTES)
        metadata = _json_copy(synthetic_evidence.freeze_resource_metadata(),
                              maximum=MAX_SOURCE_BYTES)
        return _rebuild(source["context"], source["rerank"], metadata,
                        source["tool_snapshot"])
    except ClaimSourceError:
        raise
    except Exception:
        raise ClaimSourceError() from None


def validate_claim_sources(snapshot: object, *, expected_context: object) -> dict:
    """Rebuild sources without I/O; the caller supplies authoritative context."""
    try:
        raw = _json_copy(snapshot, maximum=MAX_SNAPSHOT_BYTES)
        _exact_fields(raw, _SNAPSHOT_FIELDS)
        _json_copy({key: value for key, value in raw.items() if key != "inventory"},
                   maximum=MAX_SOURCE_BYTES)
        _json_copy(raw["inventory"], maximum=MAX_INVENTORY_BYTES)
        if type(raw["inventory"]) is not list or len(raw["inventory"]) > MAX_INVENTORY_ITEMS:
            _reject()
        expected = _json_copy(expected_context, maximum=MAX_SOURCE_BYTES)
        _validate_context(expected)
        if _canonical(raw["context"]) != _canonical(expected):
            _reject()
        rebuilt = _rebuild(raw["context"], raw["rerank"], raw["resource_metadata"],
                           raw["tool_snapshot"])
        if _canonical(raw) != _canonical(rebuilt):
            _reject()
        return rebuilt
    except ClaimSourceError:
        raise
    except Exception:
        raise ClaimSourceError() from None
