"""Fail-closed browser projection for the public EFFECT_V2 response.

The server remains the authority. This parser only prevents the page from
displaying or submitting controls for an inconsistent public projection.
It never rebuilds private proof or grants an execution capability.
"""

from __future__ import annotations

import json
from typing import Any

from app.schemas.diagnostic_sessions import (
    DiagnosticEffectApprovalPayloadV2Response,
    DiagnosticEffectV2Response,
)


_IDENTITY = ("session_id", "run_id", "request_id", "turn_number")
_MUTABLE_PAYLOAD = {"ttl_ms", "status", "decision", "can_decide", "server_time"}
_APPROVED_TERMINAL = {"VERIFIED", "EXECUTION_FAILED", "VERIFICATION_FAILED", "UNKNOWN"}


def parse_v2_effect_projection(run: dict[str, Any]) -> dict[str, Any]:
    """Return an existing-page-compatible view, or an invalid view with no actions."""
    invalid = {"valid": False, "kind": "EFFECT_V2", "effect": {}, "interrupt": None, "payload": {}}
    try:
        raw = run["effect"]
        if type(raw) is not dict or raw.get("approval_kind") != "EFFECT_V2":
            return invalid
        effect = DiagnosticEffectV2Response.model_validate_json(
            json.dumps(raw, ensure_ascii=False, allow_nan=False)
        ).model_dump(mode="json")
        if effect != raw:
            return invalid
        authority = effect["authority"]
        if any(authority[key] != run.get(key) for key in _IDENTITY):
            return invalid
        if effect["execution_contract"] != "claim_effect_v2":
            return invalid
        status = run.get("status")
        approval = effect["approval_status"]
        decision = effect["decision"]
        action_status = effect["status"]
        interrupt = run.get("interrupt")
        payload: dict[str, Any] = {}
        if status == "INTERRUPTED":
            if approval not in {"PENDING", "EXPIRED"} or type(interrupt) is not dict:
                return invalid
            raw_payload = interrupt.get("payload")
            if type(raw_payload) is not dict:
                return invalid
            payload = DiagnosticEffectApprovalPayloadV2Response.model_validate_json(
                json.dumps(raw_payload, ensure_ascii=False, allow_nan=False)
            ).model_dump(mode="json")
            if payload != raw_payload:
                return invalid
            for key in DiagnosticEffectApprovalPayloadV2Response.model_fields:
                if key not in _MUTABLE_PAYLOAD and payload[key] != effect[key]:
                    return invalid
            if (
                payload["status"] != approval
                or payload["can_decide"] is not effect["can_decide"]
                or interrupt.get("interrupt_id") != effect["interrupt_id"]
                or interrupt.get("approval_id") != effect["approval_id"]
                or action_status != "PREPARED"
                or decision is not None
            ):
                return invalid
        elif interrupt is not None:
            return invalid
        if status == "COMPLETED" and not (
            approval == "CONSUMED"
            and (decision == "REJECT" and action_status == "PREPARED"
                 or decision == "APPROVE" and action_status in _APPROVED_TERMINAL)
        ):
            return invalid
        if approval == "PENDING" and status != "INTERRUPTED":
            return invalid
        if status not in {
            "QUEUED", "RUNNING", "INTERRUPTED", "CANCEL_REQUESTED", "COMPLETED",
            "CANCELLED", "FAILED",
        }:
            return invalid
        return {
            "valid": True, "kind": "EFFECT_V2", "effect": effect,
            "interrupt": interrupt if type(interrupt) is dict else None,
            "payload": payload,
        }
    except (KeyError, TypeError, ValueError):
        return invalid
