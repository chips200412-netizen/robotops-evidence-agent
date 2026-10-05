"""Pure session boundaries for an independently authorized claim publication.

The caller owns database identity, previous-turn facts and sticky safety. These
helpers neither acquire that authority from a publication nor perform any I/O.
"""

from __future__ import annotations

import json

from app.agent.claim_publication import (
    MAX_PUBLICATION_BYTES, PUBLICATION_PROFILE, _authority, _safety,
    finalize_claim_publication, validate_claim_publication,
)
from app.agent.claim_sources import (
    CONTEXT_FIELDS, TOOL_FIELDS, _canonical, _json_copy, _validate_context,
)
from app.agent.claim_verification import validate_claim_verification
from app.schemas.diagnostic_sessions import DiagnosticTurnResponse


RUN_PROFILE = "robotops_agent_claim_run_v1"
MAX_RUN_RESULT_BYTES = 3_145_728
CLAIM_SESSION_INVALID = "CLAIM_SESSION_INVALID"
_IDENTITY_FIELDS = {"session_id", "run_id", "request_id", "turn_number"}
_SAFETY_FIELDS = {
    "risk_level", "urgency", "need_human", "risk_check_failed", "llm_failed", "tool_loop_failed",
}
_RESPONSE_FIELDS = {
    "session_id", "status", "turn_number", "reply", "updated_at", "references",
    "nodes", "follow_up", "diagnosis", "ticket_id", "ticket_note",
}
_DIAGNOSIS_FIELDS = {
    "device_model", "fault_category", "risk_level", "causes", "steps", "need_human",
    "claim_evidence",
}
_STATE_FIELDS = {
    *CONTEXT_FIELDS, *TOOL_FIELDS, *_SAFETY_FIELDS, "claim_evidence_profile",
    "agent_tool_loop_enabled", "ticket_mode", "effect_runtime", "route",
    "follow_up_rounds", "follow_up_questions", "claim_context_seed",
}
_RISK_ORDER = {"": 0, "低": 1, "中": 2, "高": 3}


class ClaimSessionError(ValueError):
    """One stable error without private payloads or dependency exception text."""

    def __init__(self) -> None:
        self.code = CLAIM_SESSION_INVALID
        super().__init__(CLAIM_SESSION_INVALID)


def _reject() -> None:
    raise ClaimSessionError()


def _fields(value: object, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        _reject()


def _equal(actual: object, expected: object) -> None:
    # Python equality would accept True == 1 and 1.0 == 1.
    if _canonical(actual) != _canonical(expected):
        _reject()


def _authorized(value: object, *, run: bool = False) -> dict:
    authority = _json_copy(value, maximum=MAX_RUN_RESULT_BYTES)
    context = _authority(authority)
    _validate_context(context)
    if run and authority["run_id"] is None:
        _reject()
    return authority


def _state_inputs(state: object, authority: dict) -> dict:
    if type(state) is not dict or any(type(key) is not str for key in state):
        _reject()
    if "__interrupt__" in state:
        _reject()
    values = _json_copy({key: state[key] for key in _STATE_FIELDS},
                        maximum=MAX_RUN_RESULT_BYTES)
    if (
        values["claim_evidence_profile"] != PUBLICATION_PROFILE
        or values["agent_tool_loop_enabled"] is not True
        or values["tool_capability_profile"] != "synthetic_evidence_v1"
        or values["ticket_mode"] != "draft"
        or values["effect_policy_status"] != "NOT_STARTED"
        or type(values["effect_proposal"]) is not dict or values["effect_proposal"]
        or values["effect_proposal_sha256"] != ""
        or type(values["effect_runtime"]) is not dict or values["effect_runtime"]
    ):
        _reject()
    context = {key: values[key] for key in CONTEXT_FIELDS}
    _validate_context(context)
    _equal(context, {key: authority[key] for key in CONTEXT_FIELDS})
    seed = values["claim_context_seed"]
    _validate_context(seed)
    if seed["user_query"] != context["user_query"]:
        _reject()
    _safety({key: values[key] for key in _SAFETY_FIELDS})
    if type(values["follow_up_rounds"]) is not int or not 0 <= values["follow_up_rounds"] <= 2:
        _reject()
    return values


def _bind_sources(values: dict, verification: dict, safety: dict) -> None:
    sources = verification["sources"]
    if sources is None:
        return
    source = sources["tool_snapshot"]
    # A publication may conservatively raise safety; it cannot lose the process
    # that the frozen sources describe, or lower that process's safety signals.
    for key in TOOL_FIELDS:
        if key not in {"risk_level", "urgency", "need_human", "tool_loop_failed"}:
            _equal(values[key], source[key])
    _source_safety(verification, safety)


def _source_safety(verification: dict, safety: dict) -> None:
    if verification["sources"] is None:
        return
    source = verification["sources"]["tool_snapshot"]
    for key in ("risk_level", "urgency"):
        if _RISK_ORDER[safety[key]] < _RISK_ORDER[source[key]]:
            _reject()
    for key in ("need_human", "tool_loop_failed"):
        if source[key] and not safety[key]:
            _reject()


def _state_update(publication: dict) -> dict:
    projection = publication["projection"]
    diagnosis = projection["diagnosis"]
    delivery = publication["verification"]["delivery"]
    safety = publication["safety"]
    return {
        "claim_publication": publication,
        "final_answer": projection["reply"], "references": projection["references"],
        "diagnosis": diagnosis["causes"], "solution_steps": diagnosis["steps"],
        "follow_up_questions": projection["follow_up"],
        "follow_up_rounds": delivery["follow_up_rounds"] + (projection["status"] == "AWAITING_USER"),
        "risk_level": diagnosis["risk_level"],
        "urgency": "高" if diagnosis["need_human"] else safety["urgency"],
        "need_human": diagnosis["need_human"], "ticket_id": "", "ticket_note": "",
    }


def publish_claim_turn(state: object, *, authority: object, follow_up_rounds: object) -> dict:
    """Finalize one unpublished active turn using service-owned inputs."""
    try:
        authorized = _authorized(authority)
        values = _state_inputs(state, authorized)
        if type(state["claim_publication"]) is not dict or state["claim_publication"]:
            _reject()
        if type(follow_up_rounds) is not int or not 0 <= follow_up_rounds <= 2:
            _reject()
        if values["follow_up_rounds"] != follow_up_rounds:
            _reject()
        verification = validate_claim_verification(
            state["claim_verification"],
            expected_context={key: authorized[key] for key in CONTEXT_FIELDS},
        )
        draft = _json_copy(state["claim_draft"], maximum=MAX_PUBLICATION_BYTES)
        _equal(draft, verification["draft"])
        safety = {key: values[key] for key in _SAFETY_FIELDS}
        _bind_sources(values, verification, safety)
        publication = finalize_claim_publication(
            authority=authorized, verification=verification, safety=safety,
            delivery={"route": values["route"], "follow_up_rounds": follow_up_rounds,
                      "follow_up_limit": 2, "follow_up_questions": values["follow_up_questions"]},
        )
        # Each return value is detached: callers may prepare a state without
        # accidentally altering its separately retained publication/response.
        return {
            "publication": publication,
            "projection": _json_copy(publication["projection"], maximum=MAX_PUBLICATION_BYTES),
            "state_update": _json_copy(_state_update(publication), maximum=MAX_RUN_RESULT_BYTES),
        }
    except ClaimSessionError:
        raise
    except Exception:
        raise ClaimSessionError() from None


def validate_claim_response(response: object, *, expected_authority: object) -> dict:
    """Validate the full public turn, including exact canonical schema output."""
    try:
        authority = _authorized(expected_authority)
        raw = _json_copy(response, maximum=MAX_RUN_RESULT_BYTES)
        _fields(raw, _RESPONSE_FIELDS)
        normalized = DiagnosticTurnResponse.model_validate_json(
            _canonical(raw), strict=True,
        ).model_dump(mode="json")
        _equal(raw, normalized)
        _fields(raw["diagnosis"], _DIAGNOSIS_FIELDS)
        publication = validate_claim_publication(
            raw["diagnosis"]["claim_evidence"], expected_authority=authority,
        )
        _source_safety(publication["verification"]["assessment"], publication["safety"])
        if raw["session_id"] != authority["session_id"] or raw["turn_number"] != authority["turn_number"]:
            _reject()
        projection = publication["projection"]
        for key in ("status", "reply", "references", "follow_up"):
            _equal(raw[key], projection[key])
        _equal(raw["diagnosis"], {**projection["diagnosis"], "claim_evidence": publication})
        if raw["ticket_id"] is not None or raw["ticket_note"] is not None:
            _reject()
        return raw
    except ClaimSessionError:
        raise
    except Exception:
        raise ClaimSessionError() from None


def validate_claim_terminal_state(state: object, *, expected_authority: object) -> dict:
    """Check published checkpoint fields without feeding projected safety back."""
    try:
        authority = _authorized(expected_authority)
        values = _state_inputs(state, authority)
        publication = validate_claim_publication(
            state["claim_publication"], expected_authority=authority,
        )
        verification = publication["verification"]["assessment"]
        for key, expected in (("claim_draft", verification["draft"]),
                              ("claim_verification", verification)):
            _equal(_json_copy(state[key], maximum=MAX_PUBLICATION_BYTES), expected)
        _bind_sources(values, verification, publication["safety"])
        for key in ("risk_check_failed", "llm_failed", "tool_loop_failed"):
            if values[key] is not publication["safety"][key]:
                _reject()
        _equal(values["route"], publication["verification"]["delivery"]["route"])
        for key, expected in _state_update(publication).items():
            _equal(_json_copy(state[key], maximum=MAX_RUN_RESULT_BYTES), expected)
        return publication
    except ClaimSessionError:
        raise
    except Exception:
        raise ClaimSessionError() from None


def encode_claim_run_result(response: object, *, authority: object) -> str:
    """Encode the active-only frame, keeping service authority independent."""
    try:
        authorized = _authorized(authority, run=True)
        validated = validate_claim_response(response, expected_authority=authorized)
        frame = _json_copy({
            "schema_version": 1, "profile": RUN_PROFILE,
            "authority": authorized, "response": validated,
        }, maximum=MAX_RUN_RESULT_BYTES)
        return _canonical(frame)
    except ClaimSessionError:
        raise
    except Exception:
        raise ClaimSessionError() from None


def _object_pairs(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            _reject()
        value[key] = item
    return value


def _invalid_constant(_value: str) -> None:
    _reject()


def decode_claim_run_result(value: object, *, expected_identity: object) -> dict:
    """Bind registry identity before validating the frozen historical context."""
    try:
        if type(value) is not str or len(value) > MAX_RUN_RESULT_BYTES:
            _reject()
        if len(value.encode("utf-8", errors="strict")) > MAX_RUN_RESULT_BYTES:
            _reject()
        parsed = json.loads(value, object_pairs_hook=_object_pairs,
                            parse_constant=_invalid_constant)
        frame = _json_copy(parsed, maximum=MAX_RUN_RESULT_BYTES)
        _fields(frame, {"schema_version", "profile", "authority", "response"})
        if type(frame["schema_version"]) is not int or frame["schema_version"] != 1 or frame["profile"] != RUN_PROFILE:
            _reject()
        authority = _authorized(frame["authority"], run=True)
        identity = _json_copy(expected_identity, maximum=MAX_RUN_RESULT_BYTES)
        _fields(identity, _IDENTITY_FIELDS)
        _equal(identity, {key: authority[key] for key in _IDENTITY_FIELDS})
        response = validate_claim_response(frame["response"], expected_authority=authority)
        return {"authority": authority, "response": response,
                "publication": _json_copy(response["diagnosis"]["claim_evidence"],
                                           maximum=MAX_PUBLICATION_BYTES)}
    except ClaimSessionError:
        raise
    except Exception:
        raise ClaimSessionError() from None
