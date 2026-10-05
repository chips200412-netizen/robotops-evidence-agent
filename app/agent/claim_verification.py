"""Pure, replayable claim assessment over already frozen evidence sources.

Model verdicts are untrusted inputs. This module grants no execution authority,
performs no retrieval, and never turns a malformed partial matrix into support.
"""

from __future__ import annotations

import re
import unicodedata

from app.agent.claim_sources import (
    MAX_SNAPSHOT_BYTES, _canonical, _digest, _json_copy, _validate_context,
    validate_claim_sources,
)


VERIFICATION_PROFILE = "robotops_agent_claim_verification_v1"
PROMPT_VERSION = "robotops-agent-semantic-v1"
CLAIM_VERIFICATION_INVALID = "CLAIM_VERIFICATION_INVALID"
_SNAPSHOT_FIELDS = {
    "schema_version", "profile", "prompt_version", "context", "draft", "risk_level",
    "mode", "source_error", "sources", "claims", "assessment", "checks", "results",
    "status", "failure_reason", "verification_sha256",
}
_SOURCE_ERRORS = {
    "NONE", "SOURCE_INVALID", "RETRIEVAL_BUSY", "RETRIEVAL_TIMEOUT", "RETRIEVAL_ERROR",
}
_ASSESSMENT_FAILURES = {
    "NOT_CALLED": "SEMANTIC_NOT_RUN", "INVALID": "SEMANTIC_INVALID",
    "TIMEOUT": "SEMANTIC_TIMEOUT", "ERROR": "SEMANTIC_ERROR",
    "BUSY": "SEMANTIC_BUSY", "ADAPTER_INVALID": "ADAPTER_INVALID",
}
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", re.ASCII)
_INJECTION = re.compile(
    r"忽略系统|无视系统|绕过审批|跳过审批|调用写工具|"
    r"ignore\s+system|ignore\s+previous\s+instructions|"
    r"bypass\s+approval|skip\s+approval|call\s+write\s+tool"
)


class ClaimVerificationError(ValueError):
    """Stable public failure; no model response, exception, or source text leaks."""

    def __init__(self) -> None:
        self.code = CLAIM_VERIFICATION_INVALID
        super().__init__(CLAIM_VERIFICATION_INVALID)


def _reject() -> None:
    raise ClaimVerificationError()


def _fields(value: object, fields: set[str]) -> None:
    if type(value) is not dict or set(value) != fields:
        _reject()


def _draft(value: object, mode: str) -> dict:
    _fields(value, {"kind", "causes", "steps"})
    if type(value["kind"]) is not str or value["kind"] not in {"DIAGNOSTIC", "SAFETY"}:
        _reject()
    result = {"kind": value["kind"]}
    for field in ("causes", "steps"):
        supplied = value[field]
        if type(supplied) is not list or len(supplied) > 20:
            _reject()
        values = []
        for text in supplied:
            if type(text) is not str:
                _reject()
            normalized = unicodedata.normalize("NFKC", text).strip()
            if not normalized or len(normalized) > 1_000 or normalized in values:
                _reject()
            values.append(normalized)
        result[field] = values
    if (
        (mode == "VERIFY" and result["kind"] != "DIAGNOSTIC")
        or (mode != "VERIFY" and result != {"kind": "SAFETY", "causes": [], "steps": []})
    ):
        _reject()
    return _json_copy(result, maximum=16_384)


def _claims(draft: dict, context: dict, risk_level: str, mode: str) -> list[dict]:
    if mode != "VERIFY":
        return []
    claims = []

    def append(kind: str, ordinal: int, text: str) -> None:
        claim = {"kind": kind, "ordinal": ordinal, "text": text}
        claims.append({"claim_id": _digest(claim), **claim})

    for field, kind in (("causes", "CAUSE"), ("steps", "STEP")):
        for ordinal, text in enumerate(draft[field], 1):
            append(kind, ordinal, text)
    append("RISK", 1, f"针对问题「{context['user_query']}」，设备「{context['device_model']}」"
           f"的当前风险等级为「{risk_level}」。")
    return claims


def _source_failure(sources: dict | None, source_error: str) -> str:
    if sources is None:
        return "SOURCE_INVALID" if source_error == "NONE" else source_error
    process = sources["tool_snapshot"]
    if (
        sources["rerank"]["status"] == "FAILED"
        or process["tool_loop_failed"]
        or process["verification_status"] in {"FAILED", "ERROR"}
    ):
        return "SOURCE_FAILED"
    if sources["status"] == "EMPTY":
        return "SOURCE_EMPTY"
    return "NONE"


def _checks(inventory: list[dict]) -> dict:
    injections = []
    groups: dict[tuple, list[dict]] = {}
    for item in inventory:
        text = unicodedata.normalize("NFKC", item["text"]).casefold()
        if _INJECTION.search(text):
            injections.append(item["evidence_id"])
        if item["source_kind"] == "TELEMETRY_SAMPLE":
            locator = item["locator"]
            record = locator["record"]
            key = (locator["device_model"], record["ts_ms"], record["metric"], record["unit"])
            groups.setdefault(key, []).append(item)
    conflicts = []
    for items in groups.values():
        signatures = {
            (item["locator"]["record"]["value"], item["locator"]["record"]["quality"])
            for item in items
        }
        if len(signatures) > 1:
            conflicts.append([item["evidence_id"] for item in items])
    return {"injection_ids": injections, "conflicts": conflicts}


def _ids(value: object, inventory_order: dict[str, int], *, maximum: int) -> list[str]:
    if type(value) is not list or len(value) > maximum:
        _reject()
    if any(type(item) is not str or item not in inventory_order for item in value):
        _reject()
    if len(set(value)) != len(value):
        _reject()
    return sorted(value, key=inventory_order.__getitem__)


def _output(value: object, claims: list[dict], inventory: list[dict], checks: dict) -> dict:
    _fields(value, {"results", "injection_ids"})
    order = {item["evidence_id"]: index for index, item in enumerate(inventory)}
    injections = _ids(value["injection_ids"], order, maximum=512)
    excluded = set(injections) | set(checks["injection_ids"])
    supplied = value["results"]
    if type(supplied) is not list or len(supplied) != len(claims):
        _reject()
    claim_ids = {claim["claim_id"] for claim in claims}
    rows = {}
    for result in supplied:
        _fields(result, {"claim_id", "verdict", "supporting_ids", "contradicting_ids"})
        identifier = result["claim_id"]
        verdict = result["verdict"]
        if type(identifier) is not str or identifier not in claim_ids or identifier in rows:
            _reject()
        if type(verdict) is not str or verdict not in {"SUPPORTED", "INSUFFICIENT", "CONTRADICTED"}:
            _reject()
        supporting = _ids(result["supporting_ids"], order, maximum=3)
        contradicting = _ids(result["contradicting_ids"], order, maximum=3)
        if (
            set(supporting) & set(contradicting)
            or set(supporting) & excluded
            or (verdict == "SUPPORTED" and (not supporting or contradicting))
            or (verdict == "INSUFFICIENT" and (supporting or contradicting))
            or (verdict == "CONTRADICTED" and not contradicting)
        ):
            _reject()
        rows[identifier] = {
            "claim_id": identifier, "verdict": verdict,
            "supporting_ids": supporting, "contradicting_ids": contradicting,
        }
    return {"results": [rows[claim["claim_id"]] for claim in claims], "injection_ids": injections}


def _assessment(value: object, model_input: dict | None, claims: list[dict],
                inventory: list[dict], checks: dict) -> dict:
    if value is None:
        value = {"model": None, "calls": 0, "outcome": "NOT_CALLED",
                 "input": model_input, "output": None}
    _fields(value, {"model", "calls", "outcome", "input", "output"})
    outcome = value["outcome"]
    if (
        type(outcome) is not str
        or outcome not in {*_ASSESSMENT_FAILURES, "RETURNED"}
        or type(value["calls"]) is not int
        or _canonical(value["input"]) != _canonical(model_input)
    ):
        _reject()
    if model_input is None and outcome != "NOT_CALLED":
        _reject()
    if outcome in {"NOT_CALLED", "ADAPTER_INVALID"}:
        if value["model"] is not None or value["calls"] != 0:
            _reject()
    elif (
        type(value["model"]) is not str
        or _MODEL.fullmatch(value["model"]) is None
        or value["model"].casefold().startswith("sk-")
        or value["calls"] != (0 if outcome == "BUSY" else 1)
    ):
        _reject()
    if outcome == "RETURNED":
        output = _output(value["output"], claims, inventory, checks)
    else:
        if value["output"] is not None:
            _reject()
        output = None
    return {"model": value["model"], "calls": value["calls"], "outcome": outcome,
            "input": model_input, "output": output}


def _build(draft: object, sources: object, *, context: object, risk_level: object,
           mode: object, source_error: object, assessment: object) -> dict:
    _validate_context(context)
    if (
        type(mode) is not str or mode not in {"VERIFY", "SAFETY", "FOLLOW_UP"}
        or type(source_error) is not str or source_error not in _SOURCE_ERRORS
        or type(risk_level) is not str or risk_level not in {"", "低", "中", "高"}
        or (mode == "VERIFY" and risk_level == "")
        or (sources is not None and source_error != "NONE")
        or (mode != "VERIFY" and (sources is not None or source_error != "NONE"))
    ):
        _reject()
    normalized_draft = _draft(draft, mode)
    frozen = None if sources is None else validate_claim_sources(sources, expected_context=context)
    claims = _claims(normalized_draft, context, risk_level, mode)
    inventory = [] if frozen is None else frozen["inventory"]
    checks = _checks(inventory)
    reason = _source_failure(frozen, source_error) if mode == "VERIFY" else "NONE"
    model_input = None
    if mode == "VERIFY" and reason == "NONE":
        excluded = set(checks["injection_ids"])
        model_input = {"context": context, "claims": claims,
                       "inventory": [item for item in inventory if item["evidence_id"] not in excluded]}
    checked_assessment = _assessment(assessment, model_input, claims, inventory, checks)
    outcome = checked_assessment["outcome"]
    if mode != "VERIFY":
        status = "SKIPPED_SAFETY" if mode == "SAFETY" else "SKIPPED_FOLLOW_UP"
        results = []
    elif reason == "NONE" and outcome == "RETURNED":
        status = "VERIFIED"
        results = checked_assessment["output"]["results"]
        flagged = set(checks["injection_ids"]) | set(checked_assessment["output"]["injection_ids"])
        checks["injection_ids"] = [item["evidence_id"] for item in inventory if item["evidence_id"] in flagged]
    else:
        status = "UNVERIFIED"
        if reason == "NONE":
            reason = _ASSESSMENT_FAILURES[outcome]
        results = [{"claim_id": claim["claim_id"], "verdict": "UNVERIFIED",
                    "supporting_ids": [], "contradicting_ids": []} for claim in claims]
    result = {
        "schema_version": 1, "profile": VERIFICATION_PROFILE, "prompt_version": PROMPT_VERSION,
        "context": context, "draft": normalized_draft, "risk_level": risk_level,
        "mode": mode, "source_error": source_error, "sources": frozen,
        "claims": claims, "assessment": checked_assessment, "checks": checks,
        "results": results, "status": status, "failure_reason": reason,
    }
    result["verification_sha256"] = _digest(result)
    return _json_copy(result, maximum=MAX_SNAPSHOT_BYTES)


def build_claim_verification(draft: object, sources: object, *, context: object,
                             risk_level: object, mode: object = "VERIFY",
                             source_error: object = "NONE", assessment: object = None) -> dict:
    """Normalize a draft and model ordering, validating every other trusted boundary."""
    try:
        inputs = _json_copy({"draft": draft, "sources": sources, "context": context,
                             "risk_level": risk_level, "mode": mode,
                             "source_error": source_error, "assessment": assessment},
                            maximum=MAX_SNAPSHOT_BYTES)
        return _build(**inputs)
    except ClaimVerificationError:
        raise
    except Exception:
        raise ClaimVerificationError() from None


def validate_claim_verification(snapshot: object, *, expected_context: object) -> dict:
    """Reconstruct all claims, assessments and checks without trusting stored derivations."""
    try:
        raw = _json_copy(snapshot, maximum=MAX_SNAPSHOT_BYTES)
        _fields(raw, _SNAPSHOT_FIELDS)
        expected = _json_copy(expected_context, maximum=MAX_SNAPSHOT_BYTES)
        _validate_context(expected)
        if _canonical(raw["context"]) != _canonical(expected):
            _reject()
        rebuilt = _build(
            raw["draft"], raw["sources"], context=raw["context"], risk_level=raw["risk_level"],
            mode=raw["mode"], source_error=raw["source_error"], assessment=raw["assessment"],
        )
        if _canonical(raw) != _canonical(rebuilt):
            _reject()
        return rebuilt
    except ClaimVerificationError:
        raise
    except Exception:
        raise ClaimVerificationError() from None
