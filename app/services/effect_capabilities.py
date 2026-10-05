"""MOD-024 typed approval material for the local synthetic restart demo.

This module is deliberately pure: it canonicalizes already-authoritative state,
derives hashes, and never performs an effect, I/O, logging, or time lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from collections.abc import Mapping
from typing import Any

from app.agent import tool_loop


APPROVAL_KIND_EFFECT_V1 = "EFFECT_V1"
EFFECT_CAPABILITY_TTL_MS = 900_000
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class EffectCapabilityError(ValueError):
    """Expose only a stable contract code, never source data."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class EffectCapabilityMaterial:
    approval_id: str
    session_id: str
    run_id: str
    interrupt_id: str
    approval_kind: str
    effect_key: str
    effect_type: str
    source: str
    proposal_json: str
    proposal_sha256: str
    arguments_json: str
    arguments_sha256: str
    risk_summary_json: str
    risk_summary_sha256: str
    verifier_summary_json: str
    verifier_summary_sha256: str
    verifier_result_sha256: str
    binding_json: str
    binding_sha256: str
    nonce_digest: str
    issued_at: int
    expires_at: int
    public_payload: dict[str, Any]


def canonical_json(value: Any) -> str:
    try:
        return tool_loop.canonical_json(value)
    except Exception as exc:
        raise EffectCapabilityError("EFFECT_CAPABILITY_INVALID") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _HEX_64.fullmatch(value) is not None


def build_effect_proposal(state: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return tool_loop.build_effect_proposal(state)
    except Exception as exc:
        raise EffectCapabilityError("EFFECT_POLICY_DENIED") from exc


def build_effect_interrupt_request(proposal: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(proposal, Mapping):
        raise EffectCapabilityError("EFFECT_PROPOSAL_INVALID")
    normalized = dict(proposal)
    # Shape-only guard. Full authority binding is repeated with checkpoint state
    # when the service commits the interrupt.
    arguments = normalized.get("arguments")
    if (
        set(normalized) != {"schema_version", "effect_type", "source", "arguments"}
        or type(normalized.get("schema_version")) is not int
        or normalized["schema_version"] != 1
        or normalized.get("effect_type")
        != tool_loop.EFFECT_TYPE_RESTART_SIMULATED_ROBOT
        or normalized.get("source") != tool_loop.EFFECT_SOURCE_SYNTHETIC_LOCAL
        or not isinstance(arguments, dict)
        or set(arguments) != {"device_model", "expected_boot_generation_delta"}
        or not isinstance(arguments.get("device_model"), str)
        or not arguments["device_model"]
        or len(arguments["device_model"]) > 200
        or type(arguments.get("expected_boot_generation_delta")) is not int
        or arguments["expected_boot_generation_delta"] != 1
    ):
        raise EffectCapabilityError("EFFECT_PROPOSAL_INVALID")
    return {
        "approval_kind": APPROVAL_KIND_EFFECT_V1,
        "proposal": normalized,
        "proposal_sha256": canonical_sha256(normalized),
    }


def is_effect_interrupt_request(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("approval_kind") == (
        APPROVAL_KIND_EFFECT_V1
    )


def stable_effect_key(
    *,
    session_id: str,
    run_id: str,
    interrupt_id: str,
    proposal_sha256: str,
    effect_type: str,
    arguments_sha256: str,
) -> str:
    subject = {
        "domain": "robotops_effect_key_v1",
        "session_id": session_id,
        "run_id": run_id,
        "interrupt_id": interrupt_id,
        "proposal_sha256": proposal_sha256,
        "effect_type": effect_type,
        "arguments_sha256": arguments_sha256,
    }
    return canonical_sha256(subject)


def nonce_digest(nonce: str) -> str:
    if not isinstance(nonce, str) or len(nonce) < 22:
        raise EffectCapabilityError("EFFECT_CAPABILITY_INVALID")
    return hashlib.sha256(
        f"robotops_effect_approval_nonce_v1:{nonce}".encode("utf-8")
    ).hexdigest()


def build_binding(
    *,
    session_id: str,
    run_id: str,
    interrupt_id: str,
    proposal_sha256: str,
    effect_key: str,
    effect_type: str,
    source: str,
    arguments_sha256: str,
    risk_summary_sha256: str,
    verifier_summary_sha256: str,
    verifier_result_sha256: str,
    nonce_digest_value: str,
    issued_at: int,
    expires_at: int,
) -> dict[str, Any]:
    if not all(
        _valid_digest(value)
        for value in (
            proposal_sha256,
            effect_key,
            arguments_sha256,
            risk_summary_sha256,
            verifier_summary_sha256,
            verifier_result_sha256,
            nonce_digest_value,
        )
    ):
        raise EffectCapabilityError("EFFECT_BINDING_INVALID")
    return {
        "schema_version": 1,
        "domain": "robotops_effect_approval_binding_v1",
        "session_id": session_id,
        "run_id": run_id,
        "interrupt_id": interrupt_id,
        "proposal_sha256": proposal_sha256,
        "effect_key": effect_key,
        "effect_type": effect_type,
        "source": source,
        "arguments_sha256": arguments_sha256,
        "risk_summary_sha256": risk_summary_sha256,
        "verifier_summary_sha256": verifier_summary_sha256,
        "verifier_result_sha256": verifier_result_sha256,
        "nonce_digest": nonce_digest_value,
        "issued_at": issued_at,
        "expires_at": expires_at,
    }


def _risk_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "risk_level": state.get("risk_level"),
        "urgency": state.get("urgency"),
        "need_human": state.get("need_human"),
        "risk_check_failed": state.get("risk_check_failed"),
        "llm_failed": state.get("llm_failed"),
        "tool_loop_failed": state.get("tool_loop_failed"),
    }
    if (
        summary["risk_level"] not in {"低", "中"}
        or summary["urgency"] not in {"低", "中"}
        or any(type(summary[key]) is not bool for key in (
            "need_human", "risk_check_failed", "llm_failed", "tool_loop_failed"
        ))
        or any(summary[key] for key in (
            "need_human", "risk_check_failed", "llm_failed", "tool_loop_failed"
        ))
    ):
        raise EffectCapabilityError("EFFECT_POLICY_DENIED")
    return summary


def risk_summary_for_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return _risk_summary(state)


def _verifier_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    result = state.get("verifier_result")
    result_sha256 = state.get("verifier_result_sha256")
    try:
        progress = tool_loop.validate_checkpoint_progress(state)
    except Exception as exc:
        raise EffectCapabilityError("EFFECT_VERIFIER_INVALID") from exc
    evidence = progress.get("tool_result_evidence")
    if (
        state.get("verification_status") != "PASSED"
        or not isinstance(result, dict)
        or not _valid_digest(result_sha256)
        or result_sha256 != canonical_sha256(result)
        or not isinstance(result.get("checks"), dict)
        or set(result["checks"]) != set(tool_loop.VERIFIER_CHECK_NAMES)
        or any(type(value) is not bool for value in result["checks"].values())
        or not isinstance(result.get("reason_codes"), list)
        or not _valid_digest(result.get("subject_sha256"))
        or not _valid_digest(result.get("evidence_sha256"))
        or not isinstance(evidence, dict)
        or canonical_sha256(evidence) != result["evidence_sha256"]
    ):
        raise EffectCapabilityError("EFFECT_VERIFIER_INVALID")
    checks = dict(result["checks"])
    return {
        "status": "PASSED",
        "checks": checks,
        "reason_codes": list(result["reason_codes"]),
        "supporting_evidence": sorted(
            name for name, passed in checks.items() if passed
        ),
        "missing_evidence": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "verifier_result_sha256": result_sha256,
        "subject_sha256": result["subject_sha256"],
        "evidence_sha256": result["evidence_sha256"],
        "evidence": evidence,
    }


def verifier_summary_for_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return _verifier_summary(state)


def build_capability_material(
    *,
    session_id: str,
    run_id: str,
    interrupt_id: str,
    approval_id: str,
    nonce: str,
    issued_at_ms: int,
    state: Mapping[str, Any],
    interrupt_request: Mapping[str, Any],
) -> EffectCapabilityMaterial:
    if (
        any(
            not isinstance(value, str) or not value
            for value in (session_id, run_id, interrupt_id, approval_id)
        )
        or len(approval_id) != 32
        or nonce == approval_id
        or type(issued_at_ms) is not int
        or issued_at_ms < 0
        or not isinstance(state, Mapping)
        or not isinstance(interrupt_request, Mapping)
    ):
        raise EffectCapabilityError("EFFECT_CAPABILITY_INVALID")
    try:
        progress = tool_loop.validate_checkpoint_progress(state)
    except Exception as exc:
        raise EffectCapabilityError("EFFECT_CHECKPOINT_INVALID") from exc
    proposal = state.get("effect_proposal")
    try:
        normalized_proposal = tool_loop.validate_effect_proposal(
            proposal,
            state=state,
            progress=progress,
        )
    except Exception as exc:
        raise EffectCapabilityError("EFFECT_PROPOSAL_INVALID") from exc
    expected_request = build_effect_interrupt_request(normalized_proposal)
    if canonical_json(dict(interrupt_request)) != canonical_json(expected_request):
        raise EffectCapabilityError("EFFECT_INTERRUPT_INVALID")
    proposal_sha256 = canonical_sha256(normalized_proposal)
    if (
        state.get("effect_policy_status") != "PROPOSED"
        or state.get("effect_proposal_sha256") != proposal_sha256
    ):
        raise EffectCapabilityError("EFFECT_PROPOSAL_INVALID")

    arguments = normalized_proposal["arguments"]
    arguments_json = canonical_json(arguments)
    arguments_sha256 = canonical_sha256(arguments)
    proposal_json = canonical_json(normalized_proposal)
    risk = _risk_summary(state)
    risk_summary_json = canonical_json(risk)
    risk_summary_sha256 = canonical_sha256(risk)
    verifier = _verifier_summary(state)
    verifier_summary_json = canonical_json(verifier)
    verifier_summary_sha256 = canonical_sha256(verifier)
    digest = nonce_digest(nonce)
    effect_key = stable_effect_key(
        session_id=session_id,
        run_id=run_id,
        interrupt_id=interrupt_id,
        proposal_sha256=proposal_sha256,
        effect_type=normalized_proposal["effect_type"],
        arguments_sha256=arguments_sha256,
    )
    expires_at_ms = issued_at_ms + EFFECT_CAPABILITY_TTL_MS
    binding = build_binding(
        session_id=session_id,
        run_id=run_id,
        interrupt_id=interrupt_id,
        proposal_sha256=proposal_sha256,
        effect_key=effect_key,
        effect_type=normalized_proposal["effect_type"],
        source=normalized_proposal["source"],
        arguments_sha256=arguments_sha256,
        risk_summary_sha256=risk_summary_sha256,
        verifier_summary_sha256=verifier_summary_sha256,
        verifier_result_sha256=verifier["verifier_result_sha256"],
        nonce_digest_value=digest,
        issued_at=issued_at_ms,
        expires_at=expires_at_ms,
    )
    binding_json = canonical_json(binding)
    binding_sha256 = canonical_sha256(binding)
    public_payload = {
        "approval_kind": APPROVAL_KIND_EFFECT_V1,
        "effect_key": effect_key,
        "binding_sha256": binding_sha256,
        "proposal": normalized_proposal,
        "proposal_sha256": proposal_sha256,
        "risk": risk,
        "verifier": verifier,
        "issued_at": issued_at_ms,
        "expires_at": expires_at_ms,
        "ttl_ms": EFFECT_CAPABILITY_TTL_MS,
        "status": "PENDING",
        "decision": None,
        "can_decide": True,
    }
    return EffectCapabilityMaterial(
        approval_id=approval_id,
        session_id=session_id,
        run_id=run_id,
        interrupt_id=interrupt_id,
        approval_kind=APPROVAL_KIND_EFFECT_V1,
        effect_key=effect_key,
        effect_type=normalized_proposal["effect_type"],
        source=normalized_proposal["source"],
        proposal_json=proposal_json,
        proposal_sha256=proposal_sha256,
        arguments_json=arguments_json,
        arguments_sha256=arguments_sha256,
        risk_summary_json=risk_summary_json,
        risk_summary_sha256=risk_summary_sha256,
        verifier_summary_json=verifier_summary_json,
        verifier_summary_sha256=verifier_summary_sha256,
        verifier_result_sha256=verifier["verifier_result_sha256"],
        binding_json=binding_json,
        binding_sha256=binding_sha256,
        nonce_digest=digest,
        issued_at=issued_at_ms,
        expires_at=expires_at_ms,
        public_payload=public_payload,
    )
