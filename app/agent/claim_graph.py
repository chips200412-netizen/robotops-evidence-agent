"""Opt-in graph bridge: acquire and verify evidence, never publish a response."""
from __future__ import annotations

import asyncio

from app.agent import nodes
from app.agent.claim_sources import (
    CONTEXT_FIELDS, MAX_SOURCE_BYTES, _json_copy, _validate_context,
)
from app.agent.claim_verification import _draft
from app.agent.claim_semantics import verify_claim_draft
from app.agent.diagnostic_facts import merge_authoritative_facts
from app.agent.safety import llm_failure_state, needs_human
from app.observability.tracing import (
    safe_span, set_safe_draft_kind, set_safe_outcome,
    set_safe_retrieval_wait_result, set_safe_verification_status,
)


ACTIVE_PROFILE = "robotops_agent_claim_evidence_v1"
DISABLED_PROFILE = "disabled_v0"
_CLAIM_VALUES = ("claim_draft", "claim_verification", "claim_publication")
_SAFETY_DRAFT = {"kind": "SAFETY", "causes": [], "steps": []}


class ClaimGraphError(ValueError):
    def __init__(self) -> None:
        self.code = "CLAIM_GRAPH_INVALID"
        super().__init__(self.code)


def is_claim_active(state: dict) -> bool:
    """Absence is legacy-only; unknown versions or disabled payloads never opt in."""
    if type(state) is not dict:
        raise ClaimGraphError()
    profile = state.get("claim_evidence_profile", DISABLED_PROFILE)
    if type(profile) is not str or profile not in {ACTIVE_PROFILE, DISABLED_PROFILE}:
        raise ClaimGraphError()
    if profile == DISABLED_PROFILE:
        for name in (*_CLAIM_VALUES, "claim_context_seed"):
            value = state.get(name, {})
            if type(value) is not dict or value:
                raise ClaimGraphError()
        return False
    return True


def validate_claim_context(state: dict) -> dict:
    """No normalization/truncation after the source process has bound its context."""
    try:
        context = _json_copy({name: state[name] for name in CONTEXT_FIELDS},
                             maximum=MAX_SOURCE_BYTES)
        _validate_context(context)
        return context
    except Exception:
        raise ClaimGraphError() from None


def validate_claim_context_seed(state: dict, *, entry: bool = False) -> dict:
    """Require the original, checkpointed context until this turn has finished."""
    try:
        if type(state) is not dict:
            raise ClaimGraphError()
        seed = _json_copy(state["claim_context_seed"], maximum=MAX_SOURCE_BYTES)
        if entry:
            if type(seed) is not dict or seed:
                raise ClaimGraphError()
        else:
            _validate_context(seed)
            if (type(state["user_query"]) is not str
                    or seed["user_query"] != state["user_query"]):
                raise ClaimGraphError()
        return seed
    except Exception:
        raise ClaimGraphError() from None


def _readonly(state: dict, *, entry: bool = False) -> dict:
    if not is_claim_active(state):
        raise ClaimGraphError()
    try:
        fields = ("agent_tool_loop_enabled", "tool_capability_profile", "ticket_mode",
                  "effect_policy_status", "effect_proposal", "effect_proposal_sha256",
                  "effect_runtime", "risk_level", "urgency", "need_human",
                  "risk_check_failed", "llm_failed", "tool_loop_failed", *_CLAIM_VALUES)
        values = _json_copy({name: state[name] for name in fields},
                            maximum=MAX_SOURCE_BYTES)
        if (values["agent_tool_loop_enabled"] is not True
                or values["tool_capability_profile"] != "synthetic_evidence_v1"
                or values["ticket_mode"] != "draft"
                or values["effect_policy_status"] != "NOT_STARTED"
                or values["effect_proposal"] != {}
                or values["effect_proposal_sha256"] != ""
                or values["effect_runtime"] != {}
                or "__interrupt__" in state):
            raise ClaimGraphError()
        for name in ("risk_level", "urgency"):
            if type(values[name]) is not str or values[name] not in {"", "低", "中", "高"}:
                raise ClaimGraphError()
        for name in ("need_human", "risk_check_failed", "llm_failed", "tool_loop_failed"):
            if type(values[name]) is not bool:
                raise ClaimGraphError()
        if any(type(values[name]) is not dict for name in _CLAIM_VALUES):
            raise ClaimGraphError()
        if values["claim_publication"] or (entry and any(values[name] for name in _CLAIM_VALUES)):
            raise ClaimGraphError()
        validate_claim_context_seed(state, entry=entry)
        return validate_claim_context(state)
    except Exception:
        raise ClaimGraphError() from None


def parse_claim_node(state: dict) -> dict:
    if not is_claim_active(state):
        return nodes.parse_node(state)
    seed = _readonly(state, entry=True)
    update = nodes.parse_node(state)
    return {**update, "claim_context_seed": seed}


def classify_claim_node(state: dict) -> dict:
    if not is_claim_active(state):
        return nodes.classify_node(state)
    _readonly(state)
    seed = validate_claim_context_seed(state)
    update = nodes.classify_node(state)
    try:
        merged = merge_authoritative_facts(
            seed, {**state, **update}, cumulative_query=seed["user_query"],
        )
        context = validate_claim_context(merged)
        return {**update, **{name: context[name] for name in CONTEXT_FIELDS
                            if name != "user_query"}}
    except Exception:
        raise ClaimGraphError() from None


def generate_claim_node(state: dict) -> dict:
    if not is_claim_active(state):
        return nodes.generate_node(state)
    _readonly(state)
    with safe_span("robotops.node.draft") as span:
        result = _generate_claim_draft(state)
        draft = result.get("claim_draft")
        set_safe_draft_kind(span, draft.get("kind") if type(draft) is dict else None)
        set_safe_outcome(span, "COMPLETED")
        return result


def _generate_claim_draft(state: dict) -> dict:
    """Preserve the existing claim draft and fail-closed generation behavior."""
    # Even an inconsistent custom upstream cannot let a tool failure call generation.
    subject = {**state, "need_human": True} if state["tool_loop_failed"] else state
    update = nodes.generate_node(subject)
    merged = {**subject, **update}
    if needs_human(merged) or merged["tool_loop_failed"]:
        draft = _draft(_SAFETY_DRAFT, "SAFETY")
    else:
        try:
            draft = _draft({"kind": "DIAGNOSTIC", "causes": update["diagnosis"],
                            "steps": update["solution_steps"]}, "VERIFY")
            update = {**update, "diagnosis": list(draft["causes"]),
                      "solution_steps": list(draft["steps"])}
        except Exception:
            update = llm_failure_state("generate")
            draft = _draft(_SAFETY_DRAFT, "SAFETY")
    return {**update, "claim_draft": draft, "claim_verification": {},
            "claim_publication": {}, "final_answer": "", "references": []}


async def aclaim_verify_node(state: dict) -> dict:
    context = _readonly(state)
    try:
        if state.get("route") == "ask":
            if needs_human(state) or state["tool_loop_failed"]:
                raise ClaimGraphError()
            mode = "FOLLOW_UP"
            draft = _draft(_SAFETY_DRAFT, mode)
        elif state.get("route") == "generate":
            raw_draft = state["claim_draft"]
            mode = "SAFETY" if raw_draft.get("kind") == "SAFETY" else "VERIFY"
            draft = _draft(raw_draft, mode)
        else:
            raise ClaimGraphError()
    except Exception:
        raise ClaimGraphError() from None

    sources, source_error = None, "NONE"
    if mode == "VERIFY":
        from app.agent.claim_retrieval import collect_claim_sources
        with safe_span("robotops.retrieval.wait") as span:
            acquired = await collect_claim_sources(state)
            sources, source_error = acquired["sources"], acquired["source_error"]
            set_safe_retrieval_wait_result(span, source_error)
            set_safe_outcome(span, "COMPLETED")
    try:
        with safe_span("robotops.node.verification") as span:
            verification = await verify_claim_draft(
                draft, sources, context=context, risk_level=state["risk_level"],
                mode=mode, source_error=source_error,
            )
            status = verification.get("status") if type(verification) is dict else None
            set_safe_verification_status(span, status)
            set_safe_outcome(span, "COMPLETED")
    except Exception:
        raise ClaimGraphError() from None
    _readonly(state)
    return {"claim_draft": draft, "claim_verification": verification,
            "claim_publication": {}, "final_answer": "", "references": [],
            "ticket_id": "", "ticket_note": ""}


def claim_verify_node(state: dict) -> dict:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aclaim_verify_node(state))
    raise ClaimGraphError()
