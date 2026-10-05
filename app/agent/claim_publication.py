"""Pure, reproducible claim publication after authoritative safety decisions.

The envelope retains input safety facts. Displayed safety is a deterministic
policy projection and never masquerades as a claim supported by a document.
"""

from __future__ import annotations

import re
import unicodedata

from app.agent.claim_sources import _canonical, _digest, _json_copy
from app.agent.claim_verification import validate_claim_verification
from app.agent.safety import human_guidance


PUBLICATION_PROFILE = "robotops_agent_claim_evidence_v1"
CLAIM_PUBLICATION_INVALID = "CLAIM_PUBLICATION_INVALID"
MAX_PUBLICATION_BYTES = 2_097_152
_CONTEXT_FIELDS = ("user_query", "device_model", "fault_category", "fault_description")
_AUTHORITY_FIELDS = {*_CONTEXT_FIELDS, "session_id", "run_id", "request_id", "turn_number"}
_SAFETY_FIELDS = {
    "risk_level", "urgency", "need_human", "risk_check_failed", "llm_failed", "tool_loop_failed",
}
_DELIVERY_FIELDS = {"route", "follow_up_rounds", "follow_up_limit", "follow_up_questions"}
_PUBLICATION_FIELDS = {
    "schema_version", "profile", "authority", "verification", "safety", "projection",
    "publication_sha256",
}
_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}", re.ASCII)
POLICY_REASON_ORDER = (
    "RISK_CHECK_FAILED", "LLM_FAILED", "TOOL_LOOP_FAILED", "HIGH_RISK", "HIGH_URGENCY",
    "NEED_HUMAN", "SAFETY_DRAFT", "FOLLOW_UP_LIMIT_REACHED", "CLAIM_SOURCE_FAILED",
    "CLAIM_EVIDENCE_EMPTY", "CLAIM_SEMANTIC_FAILED", "CLAIM_INJECTION_DETECTED",
    "CLAIM_CONTRADICTION_DETECTED", "CLAIM_RISK_UNSUPPORTED", "RISK_CHANGED",
)
_SOURCE_FAILURES = {
    "SOURCE_INVALID", "SOURCE_FAILED", "RETRIEVAL_BUSY", "RETRIEVAL_TIMEOUT", "RETRIEVAL_ERROR",
}
_SEMANTIC_FAILURES = {
    "SEMANTIC_BUSY", "SEMANTIC_TIMEOUT", "SEMANTIC_ERROR", "SEMANTIC_INVALID", "ADAPTER_INVALID",
}


class ClaimPublicationError(ValueError):
    """The publication is unavailable; input data and exceptions stay private."""

    def __init__(self) -> None:
        self.code = CLAIM_PUBLICATION_INVALID
        super().__init__(CLAIM_PUBLICATION_INVALID)


def _reject() -> None:
    raise ClaimPublicationError()


def _fields(value: object, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        _reject()


def _authority(value: object) -> dict:
    _fields(value, _AUTHORITY_FIELDS)
    if any(type(value[key]) is not str for key in _CONTEXT_FIELDS):
        _reject()
    for key in ("session_id", "run_id", "request_id"):
        token = value[key]
        if token is None and key != "session_id":
            continue
        if type(token) is not str or _ID_PATTERN.fullmatch(token) is None:
            _reject()
    if (value["run_id"] is None) != (value["request_id"] is None):
        _reject()
    if type(value["turn_number"]) is not int or not 1 <= value["turn_number"] <= 1_000_000:
        _reject()
    return {key: value[key] for key in _CONTEXT_FIELDS}


def _safety(value: object) -> None:
    _fields(value, _SAFETY_FIELDS)
    for key in ("risk_level", "urgency"):
        if type(value[key]) is not str or value[key] not in {"", "低", "中", "高"}:
            _reject()
    for key in ("need_human", "risk_check_failed", "llm_failed", "tool_loop_failed"):
        if type(value[key]) is not bool:
            _reject()


def _question_text(value: object) -> None:
    if (
        type(value) is not str or not 1 <= len(value) <= 1_000
        or unicodedata.normalize("NFKC", value).strip() != value
    ):
        _reject()


def _delivery(value: object, verification: dict) -> None:
    _fields(value, _DELIVERY_FIELDS)
    if type(value["route"]) is not str or value["route"] not in {"ask", "generate"}:
        _reject()
    if (
        type(value["follow_up_rounds"]) is not int
        or not 0 <= value["follow_up_rounds"] <= 2
        or type(value["follow_up_limit"]) is not int
        or value["follow_up_limit"] != 2
    ):
        _reject()
    questions = value["follow_up_questions"]
    if type(questions) is not list:
        _reject()
    if value["route"] == "generate":
        if questions or verification["status"] == "SKIPPED_FOLLOW_UP":
            _reject()
        return
    if not 1 <= len(questions) <= 3 or verification["status"] != "SKIPPED_FOLLOW_UP":
        _reject()
    seen: set[str] = set()
    for item in questions:
        _fields(item, {"question", "options"})
        _question_text(item["question"])
        if item["question"] in seen:
            _reject()
        seen.add(item["question"])
        options = item["options"]
        if type(options) is not list or not 2 <= len(options) <= 3:
            _reject()
        for option in options:
            _question_text(option)
        if len(set(options)) != len(options):
            _reject()


def _policy_reasons(verification: dict, safety: dict, delivery: dict) -> list[str]:
    present: set[str] = set()
    for field, code in (
        ("risk_check_failed", "RISK_CHECK_FAILED"), ("llm_failed", "LLM_FAILED"),
        ("tool_loop_failed", "TOOL_LOOP_FAILED"), ("need_human", "NEED_HUMAN"),
    ):
        if safety[field]:
            present.add(code)
    if safety["risk_level"] == "高":
        present.add("HIGH_RISK")
    if safety["urgency"] == "高":
        present.add("HIGH_URGENCY")
    if verification["status"] == "SKIPPED_SAFETY":
        present.add("SAFETY_DRAFT")
    if delivery["route"] == "ask" and delivery["follow_up_rounds"] == 2:
        present.add("FOLLOW_UP_LIMIT_REACHED")
    if verification["status"] == "UNVERIFIED":
        reason = verification["failure_reason"]
        if reason in _SOURCE_FAILURES:
            present.add("CLAIM_SOURCE_FAILED")
        elif reason == "SOURCE_EMPTY":
            present.add("CLAIM_EVIDENCE_EMPTY")
        elif reason in _SEMANTIC_FAILURES:
            present.add("CLAIM_SEMANTIC_FAILED")
        else:
            # SEMANTIC_NOT_RUN is an intermediate snapshot, never a publication.
            _reject()
    if verification["checks"]["injection_ids"]:
        present.add("CLAIM_INJECTION_DETECTED")
    if verification["checks"]["conflicts"] or any(
        result["verdict"] == "CONTRADICTED" for result in verification["results"]
    ):
        present.add("CLAIM_CONTRADICTION_DETECTED")
    if verification["status"] == "VERIFIED":
        results = {result["claim_id"]: result for result in verification["results"]}
        if any(results[claim["claim_id"]]["verdict"] != "SUPPORTED"
               for claim in verification["claims"] if claim["kind"] == "RISK"):
            present.add("CLAIM_RISK_UNSUPPORTED")
    if verification["mode"] == "VERIFY" and safety["risk_level"] != verification["risk_level"]:
        present.add("RISK_CHANGED")
    return [code for code in POLICY_REASON_ORDER if code in present]


def _references(verification: dict) -> list[dict]:
    selected = {
        identifier for result in verification["results"] if result["verdict"] == "SUPPORTED"
        for identifier in result["supporting_ids"]
    }
    references: list[dict] = []
    for item in verification["sources"]["inventory"]:
        if item["evidence_id"] not in selected:
            continue
        locator = item["locator"]
        document = item["source_kind"] in {"MANUAL_CHUNK", "FAULT_CASE_CHUNK"}
        references.append({
            "n": len(references) + 1, "evidence_id": item["evidence_id"],
            "source_kind": item["source_kind"],
            "filename": locator["filename"] if document else locator["resource_id"],
            "chunk_index": locator["chunk_index"] if document else None,
            "page": None if document else (
                f"窗口 [{locator['start_ms']},{locator['end_ms']}) ms；"
                f"ts_ms={locator['record']['ts_ms']}；record_index={locator['record_index']}"
            ),
            "text": item["text"],
        })
    return references


def _follow_up_reply(questions: list[dict]) -> str:
    lines = ["【请补充信息】，请补充以下信息："]
    for index, question in enumerate(questions, 1):
        options = " / ".join(question["options"])
        lines.append(f"{index}. {question['question']}（可选：{options}）")
    return "\n".join(lines)


def _diagnostic_reply(diagnosis: dict, references: list[dict], annotations: dict) -> str:
    lines = [
        f"【设备型号】{diagnosis['device_model'] or '未知'}",
        f"【故障类型】{diagnosis['fault_category'] or '未知'}",
        f"【风险等级】{diagnosis['risk_level'] or '未评估'}{annotations.get('risk', '')}",
    ]
    if diagnosis["need_human"]:
        lines.insert(0, "系统安全政策：当前安全性需由专业人员确认，请停止操作并联系人工支持。")
    if diagnosis["causes"]:
        lines.append("【可能原因】\n" + "\n".join(
            f"- {text}{annotations.get(('cause', index), '')}"
            for index, text in enumerate(diagnosis["causes"])
        ))
    if diagnosis["steps"]:
        lines.append("【排查步骤】\n" + "\n".join(
            f"{index + 1}. {text}{annotations.get(('step', index), '')}"
            for index, text in enumerate(diagnosis["steps"])
        ))
    if references:
        lines.append("【引用来源】\n" + "\n".join(
            f"[{ref['n']}] 《{ref['filename']}》 · "
            + (ref["page"] if ref["page"] is not None else f"片段 {ref['chunk_index'] + 1}")
            for ref in references
        ))
    return "\n\n".join(lines)


def _projection(authority: dict, verification: dict, safety: dict, delivery: dict,
                reasons: list[str]) -> dict:
    diagnosis = {
        "device_model": authority["device_model"], "fault_category": authority["fault_category"],
        "risk_level": safety["risk_level"], "causes": [], "steps": [], "need_human": bool(reasons),
    }
    if not reasons and delivery["route"] == "ask":
        questions = delivery["follow_up_questions"]
        return {
            "status": "AWAITING_USER", "reply": _follow_up_reply(questions),
            "diagnosis": diagnosis, "references": [], "follow_up": questions,
        }
    annotations: dict = {}
    if reasons:
        guidance = human_guidance("高", check_failed=safety["risk_check_failed"])
        diagnosis.update(
            risk_level="高",
            causes=["系统安全政策：" + text for text in guidance["diagnosis"]],
            steps=["系统安全政策：" + text for text in guidance["solution_steps"]],
        )
        references = []
    else:
        if verification["status"] != "VERIFIED":
            _reject()
        references = _references(verification)
        numbers = {item["evidence_id"]: item["n"] for item in references}
        results = {result["claim_id"]: result for result in verification["results"]}
        for claim in verification["claims"]:
            result = results[claim["claim_id"]]
            suffix = "".join(f"[{numbers[identifier]}]" for identifier in result["supporting_ids"])
            if claim["kind"] == "RISK":
                annotations["risk"] = suffix
            elif claim["kind"] == "CAUSE":
                index = len(diagnosis["causes"])
                text = claim["text"] if result["verdict"] == "SUPPORTED" else (
                    f"原因{claim['ordinal']}：不确定（现有证据不足）"
                )
                diagnosis["causes"].append(text)
                annotations[("cause", index)] = suffix
            elif result["verdict"] == "SUPPORTED":
                index = len(diagnosis["steps"])
                diagnosis["steps"].append(claim["text"])
                annotations[("step", index)] = suffix
    return {
        "status": "COMPLETED", "reply": _diagnostic_reply(diagnosis, references, annotations),
        "diagnosis": diagnosis, "references": references, "follow_up": [],
    }


def finalize_claim_publication(*, authority: object, verification: object,
                               safety: object, delivery: object) -> dict:
    """Create one detached projection from validated audit and authority inputs."""
    try:
        inputs = _json_copy({
            "authority": authority, "verification": verification, "safety": safety, "delivery": delivery,
        }, maximum=MAX_PUBLICATION_BYTES)
        authority, safety, delivery = (inputs[key] for key in ("authority", "safety", "delivery"))
        context = _authority(authority)
        verified = validate_claim_verification(inputs["verification"], expected_context=context)
        _safety(safety)
        _delivery(delivery, verified)
        reasons = _policy_reasons(verified, safety, delivery)
        publication = {
            "schema_version": 1, "profile": PUBLICATION_PROFILE, "authority": authority,
            "verification": {"assessment": verified, "delivery": delivery},
            "safety": {**safety, "policy_reasons": reasons},
            "projection": _projection(authority, verified, safety, delivery, reasons),
        }
        publication["publication_sha256"] = _digest(publication)
        return _json_copy(publication, maximum=MAX_PUBLICATION_BYTES)
    except ClaimPublicationError:
        raise
    except Exception:
        raise ClaimPublicationError() from None


def validate_claim_publication(snapshot: object, *, expected_authority: object) -> dict:
    """Rebuild all derived content without I/O; callers authenticate authority."""
    try:
        raw = _json_copy(snapshot, maximum=MAX_PUBLICATION_BYTES)
        expected = _json_copy(expected_authority, maximum=MAX_PUBLICATION_BYTES)
        _fields(raw, _PUBLICATION_FIELDS)
        _authority(expected)
        if _canonical(raw["authority"]) != _canonical(expected):
            _reject()
        _fields(raw["verification"], {"assessment", "delivery"})
        _fields(raw["safety"], _SAFETY_FIELDS | {"policy_reasons"})
        rebuilt = finalize_claim_publication(
            authority=expected, verification=raw["verification"]["assessment"],
            safety={key: raw["safety"][key] for key in _SAFETY_FIELDS},
            delivery=raw["verification"]["delivery"],
        )
        if _canonical(raw) != _canonical(rebuilt):
            _reject()
        return rebuilt
    except ClaimPublicationError:
        raise
    except Exception:
        raise ClaimPublicationError() from None
