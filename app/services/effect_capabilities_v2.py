"""Pure pre-approval subjects for the explicitly synthetic V2 demonstration.

No capability is issued here. Callers must own the identity, execution contract,
safety and independent binding anchor; hashes are not signatures or permissions.
Frozen evidence is replayed without looking up tools, resources, clocks or state.
"""

from __future__ import annotations

import hashlib
import re

from app.agent.claim_publication import PUBLICATION_PROFILE, _authority, _safety
from app.agent.claim_sources import _canonical, _digest, _json_copy, _validate_context
from app.agent.claim_verification import validate_claim_verification


APPROVAL_KIND_EFFECT_V2 = "EFFECT_V2"
EXECUTION_CONTRACT = "claim_effect_v2"
MATERIAL_PROFILE = "robotops_effect_capability_v2"
EFFECT_CAPABILITY_TTL_MS = 900_000
MAX_MATERIAL_BYTES = 3_145_728
EFFECT_CAPABILITY_V2_INVALID = "EFFECT_CAPABILITY_V2_INVALID"
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}", re.ASCII)
_APPROVAL_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)
_NONCE = re.compile(r"[A-Za-z0-9_-]{22,256}", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}", re.ASCII)
_RISK_ORDER = {"": 0, "低": 1, "中": 2, "高": 3}
_MATERIAL_FIELDS = {
    "schema_version", "profile", "approval_kind", "proof", "proof_sha256", "proposal",
    "proposal_sha256", "arguments_sha256", "effect_key", "binding", "binding_sha256",
    "public_payload",
}


class EffectCapabilityV2Error(ValueError):
    """Stable failure without private input or dependency exception details."""

    def __init__(self) -> None:
        self.code = EFFECT_CAPABILITY_V2_INVALID
        super().__init__(EFFECT_CAPABILITY_V2_INVALID)


def _reject() -> None:
    raise EffectCapabilityV2Error()


def _fields(value: object, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        _reject()


def _token(value: object, pattern: re.Pattern) -> None:
    if type(value) is not str or pattern.fullmatch(value) is None:
        _reject()


def _identity(authority: dict, execution_contract: object,
              interrupt_id: object, approval_id: object) -> dict:
    context = _authority(authority)
    _validate_context(context)
    if authority["run_id"] is None or execution_contract != EXECUTION_CONTRACT:
        _reject()
    _token(interrupt_id, _ID)
    _token(approval_id, _APPROVAL_ID)
    return context


def _proof(authority: dict, verification: object, safety: object, context: dict) -> dict:
    frozen = validate_claim_verification(verification, expected_context=context)
    _safety(safety)
    if (
        frozen["mode"] != "VERIFY" or frozen["status"] != "VERIFIED"
        or frozen["failure_reason"] != "NONE"
        or frozen["checks"] != {"injection_ids": [], "conflicts": []}
        or not frozen["draft"]["causes"] or not frozen["draft"]["steps"]
        or any(row["verdict"] != "SUPPORTED" for row in frozen["results"])
        or safety["risk_level"] not in {"低", "中"}
        or safety["urgency"] not in {"低", "中"}
        or safety["risk_level"] != frozen["risk_level"]
        or any(safety[key] for key in (
            "need_human", "risk_check_failed", "llm_failed", "tool_loop_failed",
        ))
    ):
        _reject()
    process = frozen["sources"]["tool_snapshot"]
    if (
        process["agent_route"] != "done" or process["verification_status"] != "PASSED"
        or process["tool_loop_failed"] or process["need_human"]
        or any(_RISK_ORDER[safety[key]] < _RISK_ORDER[process[key]]
               for key in ("risk_level", "urgency"))
    ):
        _reject()
    # All process/evidence/metadata links have already been reconstructed by
    # validate_claim_verification. This policy is not an action-semantic verdict.
    telemetry_ids = []
    for item in frozen["sources"]["inventory"]:
        if item["source_kind"] != "TELEMETRY_SAMPLE":
            continue
        locator = item["locator"]
        result = process["tool_result_evidence"][locator["step_id"]]["result"]
        if (locator["device_model"] == authority["device_model"]
                and locator["record"]["quality"] == "BAD"
                and result["source"] == "SYNTHETIC_LOCAL"):
            telemetry_ids.append(item["evidence_id"])
    if not telemetry_ids:
        _reject()
    return {
        "schema_version": 2, "profile": "robotops_claim_effect_proof_v2",
        "claim_evidence_profile": PUBLICATION_PROFILE,
        "execution_contract": EXECUTION_CONTRACT, "authority": authority,
        "verification": frozen, "safety": safety,
        "action_basis": {
            "kind": "SYNTHETIC_DEMO_POLICY", "policy": "SYNTHETIC_BAD_TELEMETRY_DEMO_V2",
            "telemetry_evidence_ids": telemetry_ids,
        },
    }


def _reconstruct(*, authority: dict, execution_contract: object, verification: object,
                 safety: object, interrupt_id: object, approval_id: object,
                 nonce_digest: object, issued_at_ms: object) -> dict:
    context = _identity(authority, execution_contract, interrupt_id, approval_id)
    _token(nonce_digest, _DIGEST)
    if (type(issued_at_ms) is not int
            or not 0 <= issued_at_ms <= 2**63 - 1 - EFFECT_CAPABILITY_TTL_MS):
        _reject()
    proof = _proof(authority, verification, safety, context)
    proof_sha256 = _digest(proof)
    arguments = {"device_model": authority["device_model"], "expected_boot_generation_delta": 1}
    proposal = {
        "schema_version": 2, "effect_type": "restart_simulated_robot",
        "source": "SYNTHETIC_LOCAL", "arguments": arguments,
    }
    proposal_sha256, arguments_sha256 = _digest(proposal), _digest(arguments)
    effect_key = _digest({
        "domain": "robotops_effect_key_v2", "authority": authority,
        "interrupt_id": interrupt_id, "proof_sha256": proof_sha256,
        "proposal_sha256": proposal_sha256, "arguments_sha256": arguments_sha256,
    })
    expires_at = issued_at_ms + EFFECT_CAPABILITY_TTL_MS
    binding = {
        "schema_version": 2, "domain": "robotops_effect_approval_binding_v2",
        "execution_contract": EXECUTION_CONTRACT, "authority": authority,
        "interrupt_id": interrupt_id, "approval_id": approval_id,
        "proof_sha256": proof_sha256, "proposal_sha256": proposal_sha256,
        "arguments_sha256": arguments_sha256, "effect_key": effect_key,
        "nonce_digest": nonce_digest, "issued_at": issued_at_ms, "expires_at": expires_at,
    }
    binding_sha256 = _digest(binding)
    frozen = proof["verification"]
    public = {
        "approval_kind": APPROVAL_KIND_EFFECT_V2, "execution_contract": EXECUTION_CONTRACT,
        "authority": authority, "interrupt_id": interrupt_id, "approval_id": approval_id,
        "effect_key": effect_key, "binding_sha256": binding_sha256,
        "proof_sha256": proof_sha256, "proposal": proposal,
        "proposal_sha256": proposal_sha256, "risk": safety,
        "claim_summary": {
            "status": "VERIFIED", "verification_sha256": frozen["verification_sha256"],
            "source_snapshot_sha256": frozen["sources"]["snapshot_sha256"],
            "verifier_result_sha256": frozen["sources"]["tool_snapshot"]["verifier_result_sha256"],
            "claim_ids": [claim["claim_id"] for claim in frozen["claims"]],
        },
        "action_basis": proof["action_basis"], "issued_at": issued_at_ms,
        "expires_at": expires_at, "ttl_ms": EFFECT_CAPABILITY_TTL_MS,
    }
    return _json_copy({
        "schema_version": 2, "profile": MATERIAL_PROFILE,
        "approval_kind": APPROVAL_KIND_EFFECT_V2, "proof": proof,
        "proof_sha256": proof_sha256, "proposal": proposal,
        "proposal_sha256": proposal_sha256, "arguments_sha256": arguments_sha256,
        "effect_key": effect_key, "binding": binding, "binding_sha256": binding_sha256,
        "public_payload": public,
    }, maximum=MAX_MATERIAL_BYTES)


def build_capability_material(*, authority: object, execution_contract: object,
                              verification: object, safety: object, interrupt_id: object,
                              approval_id: object, nonce: object, issued_at_ms: object) -> dict:
    """Build an unsigned pre-approval subject from caller-owned authority."""
    try:
        values = _json_copy({
            "authority": authority, "execution_contract": execution_contract,
            "verification": verification, "safety": safety, "interrupt_id": interrupt_id,
            "approval_id": approval_id, "nonce": nonce, "issued_at_ms": issued_at_ms,
        }, maximum=MAX_MATERIAL_BYTES)
        raw_nonce = values.pop("nonce")
        _token(raw_nonce, _NONCE)
        if raw_nonce == values["approval_id"]:
            _reject()
        values["nonce_digest"] = hashlib.sha256(
            f"robotops_effect_approval_nonce_v2:{raw_nonce}".encode("utf-8")
        ).hexdigest()
        return _reconstruct(**values)
    except EffectCapabilityV2Error:
        raise
    except Exception:
        raise EffectCapabilityV2Error() from None


def validate_capability_material(material: object, *, expected_authority: object,
                                 execution_contract: object, interrupt_id: object,
                                 approval_id: object, expected_binding_sha256: object) -> dict:
    """Replay a subject against an independently held binding anchor, not a clock."""
    try:
        raw = _json_copy(material, maximum=MAX_MATERIAL_BYTES)
        _fields(raw, _MATERIAL_FIELDS)
        expected = _json_copy({
            "authority": expected_authority, "execution_contract": execution_contract,
            "interrupt_id": interrupt_id, "approval_id": approval_id,
            "binding_sha256": expected_binding_sha256,
        }, maximum=MAX_MATERIAL_BYTES)
        _identity(expected["authority"], expected["execution_contract"],
                  expected["interrupt_id"], expected["approval_id"])
        anchor = expected.pop("binding_sha256")
        _token(anchor, _DIGEST)
        if type(raw["binding_sha256"]) is not str or raw["binding_sha256"] != anchor:
            _reject()
        rebuilt = _reconstruct(
            **expected, verification=raw["proof"]["verification"],
            safety=raw["proof"]["safety"], nonce_digest=raw["binding"]["nonce_digest"],
            issued_at_ms=raw["binding"]["issued_at"],
        )
        if rebuilt["binding_sha256"] != anchor or _canonical(raw) != _canonical(rebuilt):
            _reject()
        return rebuilt
    except EffectCapabilityV2Error:
        raise
    except Exception:
        raise EffectCapabilityV2Error() from None
