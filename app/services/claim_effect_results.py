"""Pure V2 durable replay and projections; callers retain all authorization.

No function acquires identity, time, randomness, storage or device access. A
public action cannot prove a binding by itself: the service must independently
validate its parent interrupt, registry/request relations and durable records.
"""

from __future__ import annotations

from app.agent.claim_sources import CONTEXT_FIELDS, _canonical, _digest, _json_copy
from app.schemas.diagnostic_sessions import (
    DiagnosticEffectApprovalPayloadV2Response, DiagnosticEffectV2Response,
    DiagnosticEffectResultResponse, DiagnosticEffectVerificationResponse,
)
from app.services.claim_sessions import _authorized, validate_claim_response
from app.services.effect_capabilities_v2 import _proof
from app.services.effect_records_v2 import (
    APPROVAL_COLUMNS, LEDGER_COLUMNS, MAX_RECORDS_BYTES, _read_json,
    validate_pending_records,
)


CLAIM_EFFECT_RESULT_INVALID = "CLAIM_EFFECT_RESULT_INVALID"
RUN_PROFILE = "robotops_agent_claim_run_v2"
MAX_RUN_RESULT_BYTES = 4_194_304
_MAX_TIME = 2**63 - 1
_RUN_STATES = {
    "QUEUED", "RUNNING", "INTERRUPTED", "CANCEL_REQUESTED", "COMPLETED", "CANCELLED", "FAILED",
}
_TERMINAL_EFFECTS = {"VERIFIED", "EXECUTION_FAILED", "VERIFICATION_FAILED", "UNKNOWN"}
_IDENTITY_FIELDS = {"session_id", "run_id", "request_id", "turn_number"}


class ClaimEffectResultError(ValueError):
    """Stable, redacted rejection, including failures of pure dependencies."""

    def __init__(self) -> None:
        self.code = CLAIM_EFFECT_RESULT_INVALID
        super().__init__(CLAIM_EFFECT_RESULT_INVALID)


def _reject() -> None:
    raise ClaimEffectResultError()


def _fields(value: object, fields: object) -> None:
    if type(value) is not dict or set(value) != set(fields):
        _reject()


def _equal(actual: object, expected: object) -> None:
    if _canonical(actual) != _canonical(expected):
        _reject()


def _time(value: object) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_TIME:
        _reject()
    return value


def _exact_model(model: type, value: object) -> dict:
    raw = _json_copy(value, maximum=MAX_RECORDS_BYTES)
    normalized = model.model_validate_json(_canonical(raw), strict=True).model_dump(mode="json")
    _equal(raw, normalized)
    return raw


def _optional_result(value: object, digest: object, model: type) -> dict | None:
    if value is None and digest is None:
        return None
    raw = _read_json(value, maximum=32_768)
    parsed = _exact_model(model, raw)
    if type(digest) is not str or digest != _digest(parsed):
        _reject()
    return parsed


def _runtime(ledger: dict) -> dict:
    return {
        "schema_version": 1, "effect_key": ledger["effect_key"],
        "binding_sha256": ledger["binding_sha256"], "status": ledger["status"],
        "attempt_count": ledger["attempt_count"], "result_sha256": ledger["result_sha256"],
        "verification_sha256": ledger["verification_sha256"],
    }


def _validate_mutable(approval: dict, ledger: dict) -> tuple[dict | None, dict | None]:
    """Check the original lifecycle before any initial-candidate reconstruction."""
    issued = _time(approval["issued_at"])
    expires = _time(approval["expires_at"])
    created = _time(ledger["created_at"])
    updated = _time(ledger["updated_at"])
    if expires != issued + 900_000 or created != issued or updated < created:
        _reject()
    approval_status, decision, consumed = (approval[key] for key in ("status", "decision", "consumed_at"))
    if type(approval_status) is not str:
        _reject()
    if approval_status == "PENDING":
        if decision is not None or consumed is not None:
            _reject()
    elif approval_status == "CONSUMED":
        if type(decision) is not str or decision not in {"APPROVE", "REJECT"}:
            _reject()
        if not issued <= _time(consumed) < expires:
            _reject()
    elif approval_status in {"CANCELLED", "EXPIRED"}:
        if decision is not None or _time(consumed) < issued:
            _reject()
        if approval_status == "EXPIRED" and consumed < expires:
            _reject()
    else:
        _reject()
    status, attempt = ledger["status"], ledger["attempt_count"]
    if type(status) is not str or type(attempt) is not int or attempt not in {0, 1}:
        _reject()
    result = _optional_result(ledger["result_json"], ledger["result_sha256"], DiagnosticEffectResultResponse)
    verification = _optional_result(
        ledger["verification_json"], ledger["verification_sha256"], DiagnosticEffectVerificationResponse,
    )
    if status == "PREPARED":
        valid = attempt == 0 and result is None and verification is None
    elif status in {"EXECUTING", "EXECUTION_FAILED"}:
        valid = attempt == 1 and result is None and verification is None
    elif status == "APPLIED":
        valid = attempt == 1 and result is not None and verification is None
    elif status in {"VERIFIED", "VERIFICATION_FAILED"}:
        valid = attempt == 1 and result is not None and verification is not None
    elif status == "UNKNOWN":
        valid = attempt == 1 and (result is not None or verification is None)
    else:
        valid = False
    if not valid or status != "PREPARED" and (
        approval_status != "CONSUMED" or decision != "APPROVE" or updated < consumed
    ):
        _reject()
    if result is not None:
        arguments = _read_json(ledger["arguments_json"], maximum=32_768)
        for key in ("effect_type", "source", "effect_key", "binding_sha256"):
            _equal(result[key], ledger[key])
        _equal(result["device_model"], arguments["device_model"])
    if verification is not None:
        if result is None or verification["status"] != status:
            _reject()
        for key in ("source", "effect_key", "binding_sha256", "device_model"):
            _equal(verification[key], result[key])
        _equal(verification["expected_boot_generation"], result["boot_generation_after"])
    return result, verification


def validate_effect_records(records: object, *, expected_authority: object,
                            execution_contract: object, interrupt_id: object,
                            approval_id: object, expected_binding_sha256: object) -> dict:
    """Replay immutable proof and fully check persisted mutable lifecycle facts."""
    try:
        raw = _json_copy(records, maximum=MAX_RECORDS_BYTES)
        _fields(raw, {"approval", "ledger"})
        _fields(raw["approval"], APPROVAL_COLUMNS)
        _fields(raw["ledger"], LEDGER_COLUMNS)
        approval, ledger = raw["approval"], raw["ledger"]
        result, verification = _validate_mutable(approval, ledger)
        candidate = _json_copy(raw, maximum=MAX_RECORDS_BYTES)
        candidate["approval"].update(status="PENDING", decision=None, consumed_at=None)
        candidate["ledger"].update(
            status="PREPARED", attempt_count=0, result_json=None, result_sha256=None,
            verification_json=None, verification_sha256=None, updated_at=ledger["created_at"],
        )
        material = validate_pending_records(
            candidate, expected_authority=expected_authority, execution_contract=execution_contract,
            interrupt_id=interrupt_id, approval_id=approval_id,
            expected_binding_sha256=expected_binding_sha256,
        )
        return {key: _json_copy(value, maximum=MAX_RECORDS_BYTES) for key, value in {
            "material": material, "approval": approval, "ledger": ledger,
            "runtime": _runtime(ledger), "result": result, "verification": verification,
        }.items()}
    except ClaimEffectResultError:
        raise
    except Exception:
        raise ClaimEffectResultError() from None


def _terminal(action: dict) -> bool:
    return action["approval_status"] == "CONSUMED" and (
        action["decision"] == "REJECT" and action["status"] == "PREPARED"
        or action["decision"] == "APPROVE" and action["status"] in _TERMINAL_EFFECTS
    )


def _run_projection(action: dict, run_status: object) -> None:
    if type(run_status) is not str or run_status not in _RUN_STATES:
        _reject()
    if action["approval_status"] == "PENDING" and run_status != "INTERRUPTED":
        _reject()
    if action["approval_status"] == "EXPIRED" and run_status not in {"INTERRUPTED", "CANCELLED"}:
        _reject()
    if run_status == "INTERRUPTED" and action["approval_status"] not in {"PENDING", "EXPIRED"}:
        _reject()
    if run_status == "COMPLETED" and not _terminal(action):
        _reject()


def build_effect_response(records: object, *, expected_authority: object,
                          execution_contract: object, interrupt_id: object,
                          approval_id: object, expected_binding_sha256: object,
                          now_ms: object, run_status: object) -> dict:
    """Project current GET facts using the caller's independently supplied clock."""
    try:
        checked = validate_effect_records(
            records, expected_authority=expected_authority, execution_contract=execution_contract,
            interrupt_id=interrupt_id, approval_id=approval_id,
            expected_binding_sha256=expected_binding_sha256,
        )
        approval, ledger = checked["approval"], checked["ledger"]
        now = _time(now_ms)
        if now < approval["issued_at"]:
            _reject()
        public = checked["material"]["public_payload"]
        action = {key: value for key, value in public.items() if key != "ttl_ms"}
        status = approval["status"]
        if status == "PENDING" and now >= approval["expires_at"]:
            status = "EXPIRED"
        action.update(
            approval_status=status, decision=approval["decision"],
            can_decide=status == "PENDING" and run_status == "INTERRUPTED",
            status=ledger["status"], attempt_count=ledger["attempt_count"],
            result=checked["result"], result_sha256=ledger["result_sha256"],
            verification=checked["verification"], verification_sha256=ledger["verification_sha256"],
            updated_at=ledger["updated_at"],
        )
        action = _exact_model(DiagnosticEffectV2Response, action)
        _run_projection(action, run_status)
        return action
    except ClaimEffectResultError:
        raise
    except Exception:
        raise ClaimEffectResultError() from None


def build_effect_approval_payload(records: object, *, expected_authority: object,
                                  execution_contract: object, interrupt_id: object,
                                  approval_id: object, expected_binding_sha256: object,
                                  now_ms: object, run_status: object) -> dict:
    """Return the exact PENDING/EXPIRED interrupt view, never the private proof."""
    try:
        action = build_effect_response(
            records, expected_authority=expected_authority, execution_contract=execution_contract,
            interrupt_id=interrupt_id, approval_id=approval_id,
            expected_binding_sha256=expected_binding_sha256, now_ms=now_ms, run_status=run_status,
        )
        if run_status != "INTERRUPTED" or action["approval_status"] not in {"PENDING", "EXPIRED"}:
            _reject()
        payload = {key: action[key] for key in DiagnosticEffectApprovalPayloadV2Response.model_fields
                   if key in action}
        payload.update(ttl_ms=900_000, status=action["approval_status"], server_time=now_ms)
        return _exact_model(DiagnosticEffectApprovalPayloadV2Response, payload)
    except ClaimEffectResultError:
        raise
    except Exception:
        raise ClaimEffectResultError() from None


def _frame_parts(response: object, authority: object, action: object) -> dict:
    authorized = _authorized(authority, run=True)
    validated = validate_claim_response(response, expected_authority=authorized)
    publication = validated["diagnosis"]["claim_evidence"]
    if action is not None:
        action = _exact_model(DiagnosticEffectV2Response, action)
        if not _terminal(action):
            _reject()
        _equal(action["authority"], authorized)
        assessment = publication["verification"]["assessment"]
        proof = _proof(authorized, assessment, action["risk"],
                       {key: authorized[key] for key in CONTEXT_FIELDS})
        _equal(action["proof_sha256"], _digest(proof))
        _equal(action["action_basis"], proof["action_basis"])
        _equal(action["claim_summary"], {
            "status": "VERIFIED", "verification_sha256": assessment["verification_sha256"],
            "source_snapshot_sha256": assessment["sources"]["snapshot_sha256"],
            "verifier_result_sha256": assessment["sources"]["tool_snapshot"]["verifier_result_sha256"],
            "claim_ids": [item["claim_id"] for item in assessment["claims"]],
        })
        _equal(action["effect_key"], _digest({
            "domain": "robotops_effect_key_v2", "authority": authorized,
            "interrupt_id": action["interrupt_id"], "proof_sha256": action["proof_sha256"],
            "proposal_sha256": action["proposal_sha256"],
            "arguments_sha256": _digest(action["proposal"]["arguments"]),
        }))
    return {"authority": authorized, "response": validated,
            "publication": _json_copy(publication, maximum=MAX_RUN_RESULT_BYTES), "action": action}


def encode_claim_effect_run_result(response: object, *, authority: object, action: object) -> str:
    """Encode the separate V2 terminal frame, preserving the readonly response."""
    try:
        parts = _frame_parts(response, authority, action)
        frame = {"schema_version": 2, "profile": RUN_PROFILE,
                 **{key: parts[key] for key in ("authority", "response", "action")}}
        return _canonical(_json_copy(frame, maximum=MAX_RUN_RESULT_BYTES))
    except ClaimEffectResultError:
        raise
    except Exception:
        raise ClaimEffectResultError() from None


def decode_claim_effect_run_result(value: object, *, expected_identity: object) -> dict:
    """Validate exact V2 bytes, registry identity and action/assessment binding."""
    try:
        frame = _read_json(value, maximum=MAX_RUN_RESULT_BYTES)
        _fields(frame, {"schema_version", "profile", "authority", "response", "action"})
        if type(frame["schema_version"]) is not int or frame["schema_version"] != 2 or frame["profile"] != RUN_PROFILE:
            _reject()
        identity = _json_copy(expected_identity, maximum=32_768)
        _fields(identity, _IDENTITY_FIELDS)
        authority = _authorized(frame["authority"], run=True)
        _equal(identity, {key: authority[key] for key in _IDENTITY_FIELDS})
        return _frame_parts(frame["response"], authority, frame["action"])
    except ClaimEffectResultError:
        raise
    except Exception:
        raise ClaimEffectResultError() from None
