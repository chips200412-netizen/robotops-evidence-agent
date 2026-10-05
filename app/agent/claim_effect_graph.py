"""Private V2 graph bridge: propose from frozen claims, delegate every effect.

An intent and a resumed decision are not capabilities. Only the service-owned
authority may consume its durable approval and attempt the synthetic action.
"""
from __future__ import annotations

import re

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.agent import claim_graph, nodes
from app.agent.claim_publication import _authority, _safety
from app.agent.claim_sources import _canonical, _digest, _json_copy, _validate_context
from app.agent.claim_verification import validate_claim_verification
from app.agent.runtime_context import (
    ASYNC_AUTHORITY_SURFACE, SYNTHETIC_RESTART_EFFECT_MODE,
    AgentRuntimeContext, EffectAuthority,
)
from app.agent.state import AgentState


EXECUTION_CONTRACT = "claim_effect_v2"
INTENT_PROFILE = "robotops_claim_effect_intent_v2"
_MAX_BYTES = 3_145_728
_APPROVAL_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)
_SAFETY_FIELDS = (
    "risk_level", "urgency", "need_human", "risk_check_failed", "llm_failed", "tool_loop_failed",
)
_RISK_ORDER = {"": 0, "低": 1, "中": 2, "高": 3}
_EXECUTION_STATUSES = frozenset({
    "APPLIED", "VERIFIED", "EXECUTION_FAILED", "VERIFICATION_FAILED", "UNKNOWN",
})
_VERIFICATION_STATUSES = frozenset({"VERIFIED", "VERIFICATION_FAILED", "UNKNOWN"})
_FAILED_STATUSES = frozenset({"EXECUTION_FAILED", "VERIFICATION_FAILED", "UNKNOWN"})


class ClaimEffectGraphError(ValueError):
    """A damaged graph input cannot be converted to a no-action diagnosis."""

    def __init__(self) -> None:
        self.code = "CLAIM_EFFECT_GRAPH_INVALID"
        super().__init__(self.code)


def build_claim_effect_intent(*, authority: object, verification: object, safety: object) -> dict:
    """Validate first, then return the deterministic eligible intent or no action.

    No IDs, nonce, clock, capability material, live source, or adapter is created
    here. The service must independently bind and authorize the actual request.
    """
    try:
        raw = _json_copy({"authority": authority, "verification": verification,
                          "safety": safety}, maximum=_MAX_BYTES)
        owned, final_safety = raw["authority"], raw["safety"]
        context = _authority(owned)
        _validate_context(context)
        if owned["run_id"] is None or owned["request_id"] is None:
            raise ClaimEffectGraphError()
        _safety(final_safety)
        frozen = validate_claim_verification(raw["verification"], expected_context=context)
        if (
            frozen["mode"] != "VERIFY" or frozen["status"] != "VERIFIED"
            or frozen["failure_reason"] != "NONE"
            or frozen["checks"] != {"injection_ids": [], "conflicts": []}
            or not frozen["draft"]["causes"] or not frozen["draft"]["steps"]
            or any(row["verdict"] != "SUPPORTED" for row in frozen["results"])
            or final_safety["risk_level"] not in {"低", "中"}
            or final_safety["urgency"] not in {"低", "中"}
            or final_safety["risk_level"] != frozen["risk_level"]
            or any(final_safety[key] for key in _SAFETY_FIELDS[2:])
        ):
            return {}
        process = frozen["sources"]["tool_snapshot"]
        if (
            process["tool_capability_profile"] != "synthetic_evidence_v1"
            or process["agent_route"] != "done" or process["verification_status"] != "PASSED"
            or process["tool_loop_failed"] or process["need_human"]
            or any(_RISK_ORDER[final_safety[key]] < _RISK_ORDER[process[key]]
                   for key in ("risk_level", "urgency"))
        ):
            return {}
        eligible = False
        for item in frozen["sources"]["inventory"]:
            if item["source_kind"] != "TELEMETRY_SAMPLE":
                continue
            locator = item["locator"]
            result = process["tool_result_evidence"][locator["step_id"]]["result"]
            if (locator["device_model"] == owned["device_model"]
                    and locator["record"]["quality"] == "BAD"
                    and result["source"] == "SYNTHETIC_LOCAL"):
                eligible = True
        if not eligible:
            return {}
        return {
            "approval_kind": "EFFECT_V2", "intent_profile": INTENT_PROFILE,
            "proposal": {
                "schema_version": 2, "effect_type": "restart_simulated_robot",
                "source": "SYNTHETIC_LOCAL", "arguments": {
                    "device_model": owned["device_model"], "expected_boot_generation_delta": 1,
                },
            },
            "verification_sha256": frozen["verification_sha256"],
        }
    except Exception:
        raise ClaimEffectGraphError() from None


def _v2_values(state: dict) -> dict:
    try:
        values = _json_copy({
            "claim_effect_intent": state.get("claim_effect_intent", {}),
            "claim_effect_runtime": state.get("claim_effect_runtime", {}),
            "claim_effect_decision": state.get("claim_effect_decision", ""),
        }, maximum=_MAX_BYTES)
        if (type(values["claim_effect_intent"]) is not dict
                or type(values["claim_effect_runtime"]) is not dict
                or type(values["claim_effect_decision"]) is not str
                or values["claim_effect_decision"] not in {"", "APPROVE", "REJECT"}):
            raise ClaimEffectGraphError()
        return values
    except Exception:
        raise ClaimEffectGraphError() from None


def _private_context(runtime: Runtime[AgentRuntimeContext] | None) -> AgentRuntimeContext | None:
    context = getattr(runtime, "context", None)
    if (
        type(context) is AgentRuntimeContext
        and context.execution_contract == EXECUTION_CONTRACT
        and context.surface == ASYNC_AUTHORITY_SURFACE
        and context.effect_mode == SYNTHETIC_RESTART_EFFECT_MODE
        and isinstance(context.effect_authority, EffectAuthority)
    ):
        return context
    return None


def _subject(state: dict, runtime: Runtime[AgentRuntimeContext] | None,
             *, approval: bool | None) -> tuple[AgentRuntimeContext, dict]:
    """Check the private identity and frozen claim before touching authority."""
    try:
        context = _private_context(runtime)
        if context is None:
            raise ClaimEffectGraphError()
        approval_id = context.approval_id
        if approval_id is None:
            if approval is True:
                raise ClaimEffectGraphError()
        elif (approval is False or type(approval_id) is not str
              or _APPROVAL_ID.fullmatch(approval_id) is None):
            raise ClaimEffectGraphError()
        facts = claim_graph._readonly(state)
        authority = {
            "session_id": context.session_id, "run_id": context.run_id,
            "request_id": context.request_id, "turn_number": context.turn_number, **facts,
        }
        intent = build_claim_effect_intent(
            authority=authority, verification=state["claim_verification"],
            safety={key: state[key] for key in _SAFETY_FIELDS},
        )
        return context, intent
    except Exception:
        raise ClaimEffectGraphError() from None


def _stored_intent(state: dict, runtime: Runtime[AgentRuntimeContext] | None,
                   *, approval: bool | None) -> tuple[AgentRuntimeContext, dict, dict]:
    values = _v2_values(state)
    context, expected = _subject(state, runtime, approval=approval)
    if not expected or _canonical(values["claim_effect_intent"]) != _canonical(expected):
        raise ClaimEffectGraphError()
    return context, expected, values


def _fail_closed(*, effect_runtime: dict | None = None) -> dict:
    """Only final safety and the independent effect projection may change."""
    update = {"risk_level": "高", "urgency": "高", "need_human": True}
    if effect_runtime is not None:
        update["claim_effect_runtime"] = effect_runtime
    return update


def route_after_claim_verify(
    state: AgentState, runtime: Runtime[AgentRuntimeContext] | None = None,
) -> str:
    values = _v2_values(state)
    if any(values.values()):
        raise ClaimEffectGraphError()
    return "claim_effect_policy" if _private_context(runtime) is not None else "end"


def claim_effect_policy_node(
    state: AgentState, runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    if any(_v2_values(state).values()):
        raise ClaimEffectGraphError()
    _, intent = _subject(state, runtime, approval=False)
    return {"claim_effect_intent": intent, "claim_effect_runtime": {}, "claim_effect_decision": ""}


def route_after_claim_effect_policy(state: AgentState) -> str:
    values = _v2_values(state)
    if values["claim_effect_runtime"] or values["claim_effect_decision"]:
        raise ClaimEffectGraphError()
    return "claim_approval_interrupt" if values["claim_effect_intent"] else "end"


def claim_approval_interrupt_node(
    state: AgentState, runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    _, intent, values = _stored_intent(state, runtime, approval=None)
    if values["claim_effect_runtime"] or values["claim_effect_decision"]:
        raise ClaimEffectGraphError()
    # Interrupt is deliberately outside any exception handler: LangGraph must
    # retain the actual resumable checkpoint and its framework interrupt ID.
    decision = interrupt(intent)
    _subject(state, runtime, approval=True)
    if type(decision) is not str or decision not in {"APPROVE", "REJECT"}:
        raise ClaimEffectGraphError()
    return {"claim_effect_decision": decision, **(_fail_closed() if decision == "REJECT" else {})}


def route_after_claim_approval_interrupt(state: AgentState) -> str:
    values = _v2_values(state)
    if not values["claim_effect_intent"] or values["claim_effect_runtime"]:
        raise ClaimEffectGraphError()
    if values["claim_effect_decision"] == "REJECT":
        return "end"
    if values["claim_effect_decision"] != "APPROVE":
        raise ClaimEffectGraphError()
    return "claim_effect_execution"


def claim_effect_execution_node(
    state: AgentState, runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    context, intent, values = _stored_intent(state, runtime, approval=True)
    if values["claim_effect_decision"] != "APPROVE":
        raise ClaimEffectGraphError()
    if values["claim_effect_runtime"]:
        raise ClaimEffectGraphError()
    proposal = intent["proposal"]
    try:
        outcome = context.effect_authority.execute_effect(
            session_id=context.session_id, run_id=context.run_id,
            approval_id=context.approval_id, proposal=proposal, proposal_sha256=_digest(proposal),
        )
        result = nodes._validated_effect_runtime(outcome, allowed_statuses=_EXECUTION_STATUSES)
    except Exception:
        return _fail_closed(effect_runtime={})
    return {"claim_effect_runtime": result,
            **(_fail_closed() if result["status"] in _FAILED_STATUSES else {})}


def claim_effect_verification_node(
    state: AgentState, runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    values = _v2_values(state)
    if values["claim_effect_decision"] != "APPROVE":
        raise ClaimEffectGraphError()
    try:
        current = nodes._validated_effect_runtime(
            values["claim_effect_runtime"], allowed_statuses=_EXECUTION_STATUSES,
        )
    except Exception:
        return _fail_closed(effect_runtime={})
    if current["status"] in _FAILED_STATUSES:
        return _fail_closed(effect_runtime=current)
    context, intent, _ = _stored_intent(state, runtime, approval=True)
    if current["status"] == "VERIFIED":
        return {"claim_effect_runtime": current}
    proposal = intent["proposal"]
    try:
        outcome = context.effect_authority.verify_effect(
            session_id=context.session_id, run_id=context.run_id,
            approval_id=context.approval_id, proposal=proposal, proposal_sha256=_digest(proposal),
        )
        result = nodes._validated_effect_runtime(
            outcome, allowed_statuses=_VERIFICATION_STATUSES, previous=current,
        )
    except Exception:
        return _fail_closed(effect_runtime=current)
    return {"claim_effect_runtime": result,
            **(_fail_closed() if result["status"] != "VERIFIED" else {})}
