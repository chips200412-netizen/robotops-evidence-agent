"""MOD-021：服务端权威会话驱动的智能诊断页。"""

from __future__ import annotations

import html
import hashlib
import json
import math
import os
import re
import secrets
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import httpx
import streamlit as st

from app.agent.claim_publication import ClaimPublicationError, validate_claim_publication
from frontend.effect_v2 import parse_v2_effect_projection
from frontend.theme import load_theme, page_header


API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
APPROVAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32}$")
ACTIVE_RUN_STATUSES = {"QUEUED", "RUNNING", "INTERRUPTED", "CANCEL_REQUESTED"}
EFFECT_STATUSES = {
    "PREPARED",
    "EXECUTING",
    "APPLIED",
    "VERIFIED",
    "EXECUTION_FAILED",
    "VERIFICATION_FAILED",
    "UNKNOWN",
}
APPROVAL_STATUSES = {"PENDING", "CONSUMED", "CANCELLED", "EXPIRED"}
VERIFIER_CHECK_LABELS = {
    "plan": "计划完整",
    "trajectory": "轨迹连续",
    "authoritative_arguments": "参数权威",
    "normalized_results": "结果规范",
    "evidence_binding": "证据已绑定",
    "stop_preconditions": "停止条件成立",
}
NODE_LABELS = {
    "parse": "解析",
    "identify": "型号识别",
    "classify": "分类",
    "retrieve": "检索",
    "judge": "判断",
    "ask": "追问",
    "generate": "生成",
    "risk": "风险",
    "ticket": "草稿",
    "answer": "回答",
}
PIPELINE_NODES = [
    "parse",
    "identify",
    "classify",
    "retrieve",
    "judge",
    "generate",
    "risk",
    "ticket",
    "answer",
]
EXAMPLES = (
    "机器人上电后无法连接 Wi-Fi，状态灯持续闪烁",
    "单个关节没有动作，其他关节运行正常",
    "电池明显发热并出现鼓包",
)


class UiApiError(RuntimeError):
    def __init__(self, kind: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code


def _escape(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 20,
) -> tuple[int, dict[str, Any]]:
    try:
        if method == "GET":
            response = httpx.get(
                f"{API_BASE}{path}", params=params, timeout=timeout
            )
        else:
            response = httpx.post(
                f"{API_BASE}{path}", json=payload or {}, timeout=timeout
            )
        try:
            data = response.json()
        except Exception:
            data = {}
        return int(response.status_code), data if isinstance(data, dict) else {}
    except Exception as exc:
        raise UiApiError(
            "offline", "无法连接诊断服务；输入已保留，可在网络恢复后重试连接。"
        ) from exc


def _api_failure(status_code: int) -> UiApiError:
    if status_code in {401, 403}:
        return UiApiError(
            "permission",
            "当前账号无权执行此操作。系统不会自动重放写请求。",
            status_code,
        )
    if status_code == 404:
        return UiApiError("not_found", "诊断会话不存在或已不可访问。", status_code)
    if status_code == 409:
        return UiApiError(
            "conflict", "服务端状态已变化，请先刷新权威状态。", status_code
        )
    if status_code == 422:
        return UiApiError("invalid", "提交内容格式无效，请检查后重试。", status_code)
    return UiApiError(
        "degraded", "诊断服务暂时不可用，请稍后重试。", status_code
    )


def _clear_approval_flow() -> None:
    st.session_state.diagnostic_approval_confirm_scope = None
    st.session_state.diagnostic_approval_submitting = False
    st.session_state.diagnostic_approval_response_unknown = False
    st.session_state.diagnostic_pending_approval_decision = None
    st.session_state.diagnostic_pending_approval_scope = None
    st.session_state.diagnostic_approval_replay_available = False
    st.session_state.diagnostic_approval_write_locked = False
    st.session_state.diagnostic_approval_expiry_reconciled_scope = None
    st.session_state.diagnostic_approval_clock_scope = None
    st.session_state.diagnostic_approval_clock_server_ms = None
    st.session_state.diagnostic_approval_clock_monotonic = None
    st.session_state.diagnostic_approval_countdown_bucket = None
    st.session_state.diagnostic_approval_focus_key = None
    st.session_state.diagnostic_restore_approval_trigger = False


def _canonical_sha256(value: object) -> str | None:
    try:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _valid_synthetic_result(result: object, effect: dict[str, Any]) -> bool:
    expected_fields = {
        "schema_version",
        "effect_type",
        "source",
        "device_model",
        "effect_key",
        "binding_sha256",
        "boot_generation_before",
        "boot_generation_after",
        "applied",
    }
    if not isinstance(result, dict) or set(result) != expected_fields:
        return False
    before = result.get("boot_generation_before")
    after = result.get("boot_generation_after")
    return bool(
        type(result.get("schema_version")) is int
        and result["schema_version"] == 1
        and result.get("effect_type") == effect.get("effect_type")
        and result.get("source") == effect.get("source")
        and result.get("device_model")
        == (effect.get("arguments") or {}).get("device_model")
        and result.get("effect_key") == effect.get("effect_key")
        and result.get("binding_sha256") == effect.get("binding_sha256")
        and type(before) is int
        and before >= 0
        and type(after) is int
        and after == before + 1
        and result.get("applied") is True
        and isinstance(effect.get("result_sha256"), str)
        and DIGEST_PATTERN.fullmatch(effect["result_sha256"]) is not None
        and _canonical_sha256(result) == effect["result_sha256"]
    )


def _valid_synthetic_verification(
    verification: object,
    effect: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    expected_fields = {
        "schema_version",
        "status",
        "source",
        "device_model",
        "effect_key",
        "binding_sha256",
        "expected_boot_generation",
        "observed_boot_generation",
        "observed_effect_key",
        "observed_binding_sha256",
        "matched",
        "reason_code",
    }
    if not isinstance(verification, dict) or set(verification) != expected_fields:
        return False
    status = verification.get("status")
    observed_generation = verification.get("observed_boot_generation")
    observed_effect_key = verification.get("observed_effect_key")
    observed_binding = verification.get("observed_binding_sha256")
    if not (
        type(verification.get("schema_version")) is int
        and verification["schema_version"] == 1
        and status in {"VERIFIED", "VERIFICATION_FAILED", "UNKNOWN"}
        and verification.get("source") == effect.get("source")
        and verification.get("device_model")
        == (effect.get("arguments") or {}).get("device_model")
        and verification.get("effect_key") == effect.get("effect_key")
        and verification.get("binding_sha256") == effect.get("binding_sha256")
        and verification.get("expected_boot_generation")
        == result.get("boot_generation_after")
        and type(verification.get("matched")) is bool
        and isinstance(verification.get("reason_code"), str)
        and isinstance(effect.get("verification_sha256"), str)
        and DIGEST_PATTERN.fullmatch(effect["verification_sha256"]) is not None
        and _canonical_sha256(verification) == effect["verification_sha256"]
    ):
        return False
    if status == "UNKNOWN":
        return bool(
            observed_generation is None
            and observed_effect_key is None
            and observed_binding is None
            and verification["matched"] is False
            and verification["reason_code"] == "READBACK_UNAVAILABLE"
        )
    if not (
        type(observed_generation) is int
        and observed_generation >= 0
        and (
            observed_effect_key is None
            or isinstance(observed_effect_key, str)
            and DIGEST_PATTERN.fullmatch(observed_effect_key) is not None
        )
        and (
            observed_binding is None
            or isinstance(observed_binding, str)
            and DIGEST_PATTERN.fullmatch(observed_binding) is not None
        )
        and (observed_effect_key is None) == (observed_binding is None)
        and verification["reason_code"] == ""
    ):
        return False
    matched = (
        observed_generation == result["boot_generation_after"]
        and observed_effect_key == effect["effect_key"]
        and observed_binding == effect["binding_sha256"]
    )
    return bool(
        verification["matched"] is matched
        and (status == "VERIFIED") is matched
    )


def _valid_effect_lifecycle(effect: dict[str, Any]) -> bool:
    status = effect.get("status")
    attempt_count = effect.get("attempt_count")
    result = effect.get("result")
    result_sha256 = effect.get("result_sha256")
    verification = effect.get("verification")
    verification_sha256 = effect.get("verification_sha256")

    result_valid = _valid_synthetic_result(result, effect) if result is not None else False
    verification_valid = (
        _valid_synthetic_verification(verification, effect, result)
        if verification is not None and isinstance(result, dict) and result_valid
        else False
    )
    result_absent = result is None and result_sha256 is None
    verification_absent = verification is None and verification_sha256 is None

    if status == "PREPARED":
        return attempt_count == 0 and result_absent and verification_absent
    if status in {"EXECUTING", "EXECUTION_FAILED"}:
        return attempt_count == 1 and result_absent and verification_absent
    if status == "APPLIED":
        return attempt_count == 1 and result_valid and verification_absent
    if status == "VERIFIED":
        return bool(
            attempt_count == 1
            and result_valid
            and verification_valid
            and verification.get("status") == "VERIFIED"
            and verification.get("matched") is True
        )
    if status == "VERIFICATION_FAILED":
        return bool(
            attempt_count == 1
            and result_valid
            and verification_valid
            and verification.get("status") == "VERIFICATION_FAILED"
            and verification.get("matched") is False
        )
    if status == "UNKNOWN":
        return bool(
            attempt_count == 1
            and (
                result_absent and verification_absent
                or result_valid
                and (
                    verification_absent
                    or verification_valid
                    and verification.get("status") == "UNKNOWN"
                    and verification.get("matched") is False
                )
            )
        )
    return False


def _typed_effect_projection(run: dict[str, Any] | None) -> dict[str, Any] | None:
    """Parse the GET-safe typed effect projection and fail closed on mismatch."""
    run = run or {}
    effect = run.get("effect")
    if effect is None:
        return None
    if not isinstance(effect, dict):
        return {"valid": False, "effect": {}}
    if effect.get("approval_kind") == "EFFECT_V2":
        return parse_v2_effect_projection(run)
    if "approval_kind" in effect:
        return {"valid": False, "effect": {}}

    arguments = effect.get("arguments")
    risk = effect.get("risk")
    verifier = effect.get("verifier")
    checks = verifier.get("checks") if isinstance(verifier, dict) else None
    supporting = (
        verifier.get("supporting_evidence") if isinstance(verifier, dict) else None
    )
    missing = verifier.get("missing_evidence") if isinstance(verifier, dict) else None
    issued_at = effect.get("issued_at")
    expires_at = effect.get("expires_at")
    valid = (
        isinstance(run.get("session_id"), str)
        and SESSION_PATTERN.fullmatch(run["session_id"]) is not None
        and isinstance(run.get("run_id"), str)
        and SESSION_PATTERN.fullmatch(run["run_id"]) is not None
        and isinstance(effect.get("effect_key"), str)
        and DIGEST_PATTERN.fullmatch(effect["effect_key"]) is not None
        and isinstance(effect.get("proposal_sha256"), str)
        and DIGEST_PATTERN.fullmatch(effect["proposal_sha256"]) is not None
        and isinstance(effect.get("binding_sha256"), str)
        and DIGEST_PATTERN.fullmatch(effect["binding_sha256"]) is not None
        and effect.get("effect_type") == "restart_simulated_robot"
        and effect.get("source") == "SYNTHETIC_LOCAL"
        and isinstance(arguments, dict)
        and set(arguments) == {"device_model", "expected_boot_generation_delta"}
        and isinstance(arguments.get("device_model"), str)
        and bool(arguments["device_model"].strip())
        and arguments.get("expected_boot_generation_delta") == 1
        and type(arguments.get("expected_boot_generation_delta")) is int
        and isinstance(risk, dict)
        and set(risk)
        == {
            "risk_level",
            "urgency",
            "need_human",
            "risk_check_failed",
            "llm_failed",
            "tool_loop_failed",
        }
        and risk.get("risk_level") in {"低", "中"}
        and risk.get("urgency") in {"低", "中"}
        and risk.get("need_human") is False
        and risk.get("risk_check_failed") is False
        and risk.get("llm_failed") is False
        and risk.get("tool_loop_failed") is False
        and isinstance(verifier, dict)
        and set(verifier)
        == {
            "status",
            "checks",
            "reason_codes",
            "supporting_evidence",
            "missing_evidence",
            "verifier_result_sha256",
            "subject_sha256",
            "evidence_sha256",
            "evidence",
        }
        and verifier.get("status") == "PASSED"
        and isinstance(checks, dict)
        and set(checks) == set(VERIFIER_CHECK_LABELS)
        and all(type(value) is bool for value in checks.values())
        and all(checks.values())
        and isinstance(supporting, list)
        and supporting == sorted(checks)
        and isinstance(missing, list)
        and missing == []
        and isinstance(verifier.get("reason_codes"), list)
        and verifier["reason_codes"] == []
        and isinstance(verifier.get("evidence"), dict)
        and _canonical_sha256(verifier["evidence"])
        == verifier.get("evidence_sha256")
        and all(
            isinstance(verifier.get(name), str)
            and DIGEST_PATTERN.fullmatch(verifier[name]) is not None
            for name in (
                "verifier_result_sha256",
                "subject_sha256",
                "evidence_sha256",
            )
        )
        and type(issued_at) is int
        and type(expires_at) is int
        and 0 <= issued_at < expires_at
        and effect.get("approval_status") in APPROVAL_STATUSES
        and type(effect.get("can_decide")) is bool
        and effect.get("status") in EFFECT_STATUSES
        and type(effect.get("attempt_count")) is int
        and 0 <= effect["attempt_count"] <= 1
        and type(effect.get("updated_at")) is int
        and effect["updated_at"] >= 0
        and _valid_effect_lifecycle(effect)
    )

    approval_status = effect.get("approval_status")
    decision = effect.get("decision")
    effect_status = effect.get("status")
    approval_lifecycle_valid = (
        approval_status == "PENDING"
        and decision is None
        and effect_status == "PREPARED"
        and effect.get("attempt_count") == 0
        or approval_status in {"EXPIRED", "CANCELLED"}
        and decision is None
        and effect_status == "PREPARED"
        and effect.get("attempt_count") == 0
        and effect.get("can_decide") is False
        or approval_status == "CONSUMED"
        and decision == "REJECT"
        and effect_status == "PREPARED"
        and effect.get("attempt_count") == 0
        and effect.get("can_decide") is False
        or approval_status == "CONSUMED"
        and decision == "APPROVE"
        and effect_status in EFFECT_STATUSES
        and effect.get("can_decide") is False
    )
    valid = bool(valid and approval_lifecycle_valid)

    interrupt = run.get("interrupt")
    payload: dict[str, Any] = {}
    if run.get("status") == "INTERRUPTED" and effect.get("approval_status") in {
        "PENDING",
        "EXPIRED",
    }:
        if isinstance(interrupt, dict) and isinstance(interrupt.get("payload"), dict):
            payload = interrupt["payload"]
        proposal = payload.get("proposal") if isinstance(payload, dict) else None
        valid = bool(
            valid
            and isinstance(interrupt, dict)
            and isinstance(interrupt.get("interrupt_id"), str)
            and bool(interrupt["interrupt_id"])
            and isinstance(interrupt.get("approval_id"), str)
            and APPROVAL_ID_PATTERN.fullmatch(interrupt["approval_id"]) is not None
            and payload.get("approval_kind") == "EFFECT_V1"
            and payload.get("effect_key") == effect.get("effect_key")
            and payload.get("binding_sha256") == effect.get("binding_sha256")
            and payload.get("proposal_sha256") == effect.get("proposal_sha256")
            and payload.get("risk") == risk
            and payload.get("verifier") == verifier
            and payload.get("issued_at") == issued_at
            and payload.get("expires_at") == expires_at
            and payload.get("ttl_ms") == expires_at - issued_at
            and payload.get("status") == effect.get("approval_status")
            and payload.get("decision") == effect.get("decision")
            and payload.get("can_decide") == effect.get("can_decide")
            and type(payload.get("server_time")) is int
            and isinstance(proposal, dict)
            and set(proposal)
            == {"schema_version", "effect_type", "source", "arguments"}
            and proposal.get("schema_version") == 1
            and proposal.get("effect_type") == effect.get("effect_type")
            and proposal.get("source") == effect.get("source")
            and proposal.get("arguments") == arguments
        )
    elif interrupt is not None:
        valid = False

    return {
        "valid": bool(valid),
        "kind": "EFFECT_V1",
        "effect": effect,
        "interrupt": interrupt if isinstance(interrupt, dict) else None,
        "payload": payload,
    }


def _approval_scope(run: dict[str, Any] | None) -> str | None:
    typed = _typed_effect_projection(run)
    if not typed or not typed["valid"] or not typed.get("interrupt"):
        return None
    effect = typed["effect"]
    interrupt = typed["interrupt"]
    if typed.get("kind") == "EFFECT_V2":
        value = {
            "approval_kind": "EFFECT_V2",
            "execution_contract": effect["execution_contract"],
            "authority": effect["authority"],
            "interrupt_id": interrupt["interrupt_id"],
            "approval_id": interrupt["approval_id"],
            "effect_key": effect["effect_key"],
            "proof_sha256": effect["proof_sha256"],
            "proposal_sha256": effect["proposal_sha256"],
            "binding_sha256": effect["binding_sha256"],
            "proposal": effect["proposal"],
            "issued_at": effect["issued_at"],
            "expires_at": effect["expires_at"],
        }
    else:
        value = {
            "session_id": run["session_id"],
            "run_id": run["run_id"],
            "interrupt_id": interrupt["interrupt_id"],
            "approval_id": interrupt["approval_id"],
            "effect_key": effect["effect_key"],
            "proposal_sha256": effect["proposal_sha256"],
            "binding_sha256": effect["binding_sha256"],
            "effect_type": effect["effect_type"],
            "source": effect["source"],
            "arguments": effect["arguments"],
            "issued_at": effect["issued_at"],
            "expires_at": effect["expires_at"],
        }
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _init_state() -> None:
    defaults = {
        "diagnostic_session_id": None,
        "diagnostic_creation_request_id": None,
        "diagnostic_run_id": None,
        "diagnostic_pending_request_id": None,
        "diagnostic_pending_payload": None,
        "diagnostic_response_unknown": False,
        "diagnostic_request_missing": False,
        "diagnostic_cancel_confirm": False,
        "diagnostic_cancel_sent": False,
        "diagnostic_cancel_response_unknown": False,
        "diagnostic_poll_started_at": 0.0,
        "diagnostic_last_poll_at": 0.0,
        "diagnostic_ui_error_kind": None,
        "diagnostic_ui_error_message": None,
        "diagnostic_focus_key": None,
        "diagnostic_focus_input": False,
        "diagnostic_input": "",
        "diagnostic_approval_confirm_scope": None,
        "diagnostic_approval_submitting": False,
        "diagnostic_approval_response_unknown": False,
        "diagnostic_pending_approval_decision": None,
        "diagnostic_pending_approval_scope": None,
        "diagnostic_approval_replay_available": False,
        "diagnostic_approval_write_locked": False,
        "diagnostic_approval_countdown_bucket": None,
        "diagnostic_approval_focus_key": None,
        "diagnostic_restore_approval_trigger": False,
        "diagnostic_approval_expiry_reconciled_scope": None,
        "diagnostic_approval_clock_scope": None,
        "diagnostic_approval_clock_server_ms": None,
        "diagnostic_approval_clock_monotonic": None,
        "service_health": "checking",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.pop("diagnostic_clear_input", False):
        st.session_state["diagnostic_input"] = ""


def _resolve_session_id() -> tuple[str | None, str | None]:
    raw = st.query_params.get("session_id")
    if raw is not None:
        if isinstance(raw, str) and SESSION_PATTERN.fullmatch(raw):
            st.session_state.diagnostic_session_id = raw
            return raw, None
        return None, "URL 中的 session_id 无效；未创建或覆盖任何会话。"
    cached = st.session_state.diagnostic_session_id
    if isinstance(cached, str) and SESSION_PATTERN.fullmatch(cached):
        st.query_params["session_id"] = cached
        return cached, None
    return None, None


def _replace_session(session_id: str, *, parent_id: str | None = None) -> None:
    st.session_state.diagnostic_session_id = session_id
    st.session_state.diagnostic_creation_request_id = None
    st.session_state.diagnostic_run_id = None
    st.session_state.diagnostic_pending_request_id = None
    st.session_state.diagnostic_pending_payload = None
    st.session_state.diagnostic_response_unknown = False
    st.session_state.diagnostic_request_missing = False
    st.session_state.diagnostic_cancel_confirm = False
    st.session_state.diagnostic_cancel_sent = False
    st.session_state.diagnostic_cancel_response_unknown = False
    st.session_state.diagnostic_ui_error_kind = None
    st.session_state.diagnostic_ui_error_message = None
    st.session_state.diagnostic_parent_id = parent_id
    st.session_state.diagnostic_poll_started_at = 0.0
    st.session_state.diagnostic_last_poll_at = 0.0
    _clear_approval_flow()
    st.session_state.diagnostic_approval_countdown_bucket = None
    st.session_state.diagnostic_approval_focus_key = None
    st.query_params.clear()
    st.query_params["session_id"] = session_id


def _create_session() -> str:
    request_id = st.session_state.diagnostic_creation_request_id
    if not isinstance(request_id, str):
        request_id = secrets.token_urlsafe(24)
        st.session_state.diagnostic_creation_request_id = request_id
    status_code, data = _request(
        "POST", "/diagnostic-sessions", payload={"request_id": request_id}
    )
    if status_code != 201:
        raise _api_failure(status_code)
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not SESSION_PATTERN.fullmatch(session_id):
        raise UiApiError("degraded", "服务端返回了无效的会话标识。")
    _replace_session(session_id)
    return session_id


def _record_write_error(error: UiApiError) -> None:
    st.session_state.diagnostic_ui_error_kind = error.kind
    st.session_state.diagnostic_ui_error_message = error.message
    st.session_state.service_health = (
        "offline" if error.kind == "offline" else "degraded"
    )


def _accept_run_response(data: dict[str, Any]) -> None:
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not SESSION_PATTERN.fullmatch(run_id):
        raise UiApiError("degraded", "服务端返回了无效的运行标识。")
    st.session_state.diagnostic_run_id = run_id
    st.session_state.diagnostic_response_unknown = False
    st.session_state.diagnostic_request_missing = False
    st.session_state.diagnostic_ui_error_kind = None
    st.session_state.diagnostic_ui_error_message = None
    st.session_state.diagnostic_cancel_confirm = False
    st.session_state.diagnostic_cancel_sent = False
    st.session_state.diagnostic_cancel_response_unknown = False
    st.session_state.diagnostic_poll_started_at = time.monotonic()
    st.session_state.diagnostic_last_poll_at = 0.0
    st.session_state.service_health = "ready"
    _clear_approval_flow()


def _submit_message(text: str, *, reuse_pending: bool = False) -> None:
    message = text.strip()
    if not message:
        return
    try:
        session_id = st.session_state.diagnostic_session_id or _create_session()
        if reuse_pending:
            request_id = st.session_state.diagnostic_pending_request_id
            pending = st.session_state.diagnostic_pending_payload
            if not isinstance(request_id, str) or pending != message:
                raise UiApiError("conflict", "待确认请求已变化，无法安全重放。")
        else:
            request_id = secrets.token_urlsafe(24)
            st.session_state.diagnostic_pending_request_id = request_id
            st.session_state.diagnostic_pending_payload = message
        st.session_state.diagnostic_response_unknown = False
        st.session_state.diagnostic_request_missing = False
        status_code, data = _request(
            "POST",
            f"/diagnostic-sessions/{session_id}/runs",
            payload={"message": message, "request_id": request_id},
            timeout=65,
        )
        if status_code in {200, 202}:
            _accept_run_response(data)
            st.session_state.diagnostic_clear_input = True
            return
        if status_code >= 500:
            st.session_state.diagnostic_response_unknown = True
            st.session_state.service_health = "degraded"
            return
        raise _api_failure(status_code)
    except UiApiError as exc:
        # 只有已拿到权威 session_id 后，run POST 的响应丢失才进入 run
        # 对账态。首次会话创建响应未知时保留输入与 creation request_id，
        # 由人工重试同一创建请求并取回服务端原 session。
        if exc.kind == "offline" and st.session_state.diagnostic_session_id:
            st.session_state.diagnostic_response_unknown = True
        elif not st.session_state.diagnostic_session_id:
            st.session_state.diagnostic_response_unknown = False
            st.session_state.diagnostic_request_missing = False
        _record_write_error(exc)


def _resume_effect_decision(
    session_id: str,
    run_id: str,
    run: dict[str, Any],
    decision: str,
    *,
    replay: bool = False,
) -> None:
    """Submit one exact typed decision; an uncertain response is never replayed."""
    typed = _typed_effect_projection(run)
    scope = _approval_scope(run)
    effect = typed.get("effect") if typed else None
    interrupt = typed.get("interrupt") if typed else None
    pending_decision = st.session_state.diagnostic_pending_approval_decision
    pending_scope = st.session_state.diagnostic_pending_approval_scope
    valid_replay = (
        replay
        and st.session_state.diagnostic_approval_response_unknown
        and st.session_state.diagnostic_approval_replay_available
        and decision == pending_decision
        and scope == pending_scope
    )
    valid_fresh = not replay and not st.session_state.diagnostic_approval_write_locked
    if (
        not typed
        or not typed["valid"]
        or not isinstance(effect, dict)
        or not isinstance(interrupt, dict)
        or not isinstance(scope, str)
        or decision not in {"APPROVE", "REJECT"}
        or not (valid_replay or valid_fresh)
        or run.get("session_id") != session_id
        or run.get("run_id") != run_id
        or run.get("status") != "INTERRUPTED"
        or effect.get("approval_status") != "PENDING"
        or effect.get("status") != "PREPARED"
        or effect.get("can_decide") is not True
    ):
        st.session_state.diagnostic_approval_write_locked = True
        _record_write_error(
            UiApiError("conflict", "审批范围已变化；已禁止提交，请先读取权威状态。")
        )
        return

    st.session_state.diagnostic_pending_approval_decision = decision
    st.session_state.diagnostic_pending_approval_scope = scope
    st.session_state.diagnostic_approval_confirm_scope = None
    st.session_state.diagnostic_approval_submitting = True
    st.session_state.diagnostic_approval_write_locked = True
    st.session_state.diagnostic_approval_replay_available = False
    try:
        status_code, data = _request(
            "POST",
            f"/diagnostic-sessions/{session_id}/runs/{run_id}/resume",
            payload={
                "interrupt_id": interrupt["interrupt_id"],
                "approval_id": interrupt["approval_id"],
                "decision": decision,
            },
            timeout=65,
        )
        if status_code in {200, 202}:
            _accept_run_response(data)
            return
        if status_code >= 500:
            st.session_state.diagnostic_approval_response_unknown = True
            st.session_state.diagnostic_approval_submitting = False
            st.session_state.diagnostic_approval_write_locked = True
            st.session_state.service_health = "degraded"
            st.session_state.diagnostic_ui_error_kind = "degraded"
            st.session_state.diagnostic_ui_error_message = (
                "审批响应未能确认；页面将先用只读 GET 对账，绝不会自动重放。"
            )
            return
        raise _api_failure(status_code)
    except UiApiError as exc:
        if exc.kind == "offline":
            st.session_state.diagnostic_approval_response_unknown = True
            st.session_state.diagnostic_approval_write_locked = True
            _record_write_error(exc)
            st.session_state.diagnostic_ui_error_message = (
                "审批响应未能确认；网络恢复后先只读对账，绝不会自动重放。"
            )
        else:
            st.session_state.diagnostic_approval_response_unknown = False
            st.session_state.diagnostic_approval_write_locked = True
            _record_write_error(exc)
    finally:
        st.session_state.diagnostic_approval_submitting = False


def _retry_run(session_id: str, run_id: str) -> None:
    try:
        status_code, data = _request(
            "POST", f"/diagnostic-sessions/{session_id}/runs/{run_id}/retry"
        )
        if status_code not in {200, 202}:
            raise _api_failure(status_code)
        _accept_run_response(data)
    except UiApiError as exc:
        _record_write_error(exc)


def _cancel_run(session_id: str, run_id: str) -> None:
    if st.session_state.diagnostic_cancel_sent:
        return
    st.session_state.diagnostic_cancel_sent = True
    st.session_state.diagnostic_cancel_response_unknown = False
    try:
        status_code, data = _request(
            "POST", f"/diagnostic-sessions/{session_id}/runs/{run_id}/cancel"
        )
        if status_code not in {200, 202}:
            raise _api_failure(status_code)
        _accept_run_response(data)
        st.session_state.diagnostic_cancel_sent = True
    except UiApiError as exc:
        # POST 的响应可能在服务端已处理后丢失。保留确认态，并允许页面先用
        # 权威 GET 对账；若 run 仍可取消，用户可安全重试同一幂等 cancel。
        st.session_state.diagnostic_cancel_sent = False
        st.session_state.diagnostic_cancel_response_unknown = True
        _record_write_error(exc)


def _retry_draft(session_id: str, draft: dict[str, Any]) -> None:
    try:
        status_code, _data = _request(
            "POST",
            f"/diagnostic-sessions/{session_id}/draft-ticket/{draft['draft_id']}/retry",
            payload={"version": draft["version"]},
        )
        if status_code != 200:
            raise _api_failure(status_code)
        st.session_state.diagnostic_ui_error_kind = None
        st.session_state.diagnostic_ui_error_message = None
    except UiApiError as exc:
        _record_write_error(exc)


def _new_diagnosis(parent_session_id: str | None = None) -> None:
    try:
        if parent_session_id:
            status_code, data = _request(
                "POST", f"/diagnostic-sessions/{parent_session_id}/reopen"
            )
            if status_code not in {200, 201}:
                raise _api_failure(status_code)
            session_id = data.get("session_id")
            if not isinstance(session_id, str) or not SESSION_PATTERN.fullmatch(
                session_id
            ):
                raise UiApiError("degraded", "服务端返回了无效的关联会话。")
            _replace_session(session_id, parent_id=parent_session_id)
        else:
            _create_session()
        st.session_state.diagnostic_clear_input = True
        st.iframe(
            "<script>window.parent.scrollTo({top:0,left:0,behavior:'auto'});</script>",
            height=1,
            tab_index=-1,
        )
    except UiApiError as exc:
        _record_write_error(exc)


def _reconcile_approval_state(run: dict[str, Any]) -> None:
    typed = _typed_effect_projection(run)
    scope = _approval_scope(run)
    if typed and typed["valid"] and isinstance(scope, str):
        server_time = (typed.get("payload") or {}).get("server_time")
        if type(server_time) is int:
            st.session_state.diagnostic_approval_clock_scope = scope
            st.session_state.diagnostic_approval_clock_server_ms = server_time
            st.session_state.diagnostic_approval_clock_monotonic = time.monotonic()
            if server_time < typed["effect"]["expires_at"]:
                st.session_state.diagnostic_approval_expiry_reconciled_scope = None
    if not st.session_state.diagnostic_approval_response_unknown:
        if typed and typed["valid"]:
            confirm_scope = st.session_state.diagnostic_approval_confirm_scope
            if confirm_scope is not None and confirm_scope != scope:
                st.session_state.diagnostic_approval_confirm_scope = None
            effect = typed["effect"]
            if (
                run.get("status") == "INTERRUPTED"
                and effect.get("approval_status") == "PENDING"
                and effect.get("status") == "PREPARED"
                and effect.get("can_decide") is True
                and st.session_state.diagnostic_ui_error_kind != "permission"
            ):
                st.session_state.diagnostic_approval_write_locked = False
        return

    pending_scope = st.session_state.diagnostic_pending_approval_scope
    effect = typed.get("effect") if typed else None
    same_pending_scope = (
        typed
        and typed["valid"]
        and isinstance(effect, dict)
        and isinstance(scope, str)
        and scope == pending_scope
        and run.get("status") == "INTERRUPTED"
        and effect.get("approval_status") == "PENDING"
        and effect.get("status") == "PREPARED"
        and effect.get("can_decide") is True
    )
    if same_pending_scope:
        st.session_state.diagnostic_approval_replay_available = True
        st.session_state.diagnostic_approval_write_locked = True
        return

    _clear_approval_flow()
    if st.session_state.diagnostic_ui_error_kind in {"offline", "degraded"}:
        st.session_state.diagnostic_ui_error_kind = None
        st.session_state.diagnostic_ui_error_message = None


def _restore(session_id: str | None, url_error: str | None) -> dict[str, Any]:
    view: dict[str, Any] = {
        "session": None,
        "run": None,
        "messages": [],
        "draft": None,
        "history": [],
        "history_error": None,
        "error_kind": None,
        "error_message": None,
    }
    if url_error:
        view.update(error_kind="invalid_url", error_message=url_error)
        st.session_state.service_health = "checking"
        return view
    if not session_id:
        try:
            status_code, data = _request("GET", "/health", timeout=3)
            if status_code != 200 or data.get("status") != "ok":
                raise _api_failure(status_code)
            st.session_state.service_health = "ready"
            return view
        except UiApiError as exc:
            st.session_state.service_health = (
                "offline" if exc.kind == "offline" else "degraded"
            )
            view.update(error_kind=exc.kind, error_message=exc.message)
            return view
    try:
        status_code, session = _request(
            "GET", f"/diagnostic-sessions/{session_id}"
        )
        if status_code != 200:
            raise _api_failure(status_code)
        view["session"] = session

        reconciled_run: dict[str, Any] | None = None
        pending_request_id = st.session_state.diagnostic_pending_request_id
        if st.session_state.diagnostic_response_unknown and isinstance(
            pending_request_id, str
        ):
            request_status, request_run = _request(
                "GET",
                f"/diagnostic-sessions/{session_id}/requests/{pending_request_id}",
            )
            if request_status == 200:
                reconciled_run = request_run
                _accept_run_response(request_run)
            elif request_status == 404:
                st.session_state.diagnostic_request_missing = True
            else:
                raise _api_failure(request_status)

        run_id = (
            (reconciled_run or {}).get("run_id")
            or session.get("active_run_id")
            or session.get("latest_run_id")
        )
        if isinstance(run_id, str) and SESSION_PATTERN.fullmatch(run_id):
            run_status, run = _request(
                "GET", f"/diagnostic-sessions/{session_id}/runs/{run_id}"
            )
            if run_status != 200:
                raise _api_failure(run_status)
            view["run"] = run
            st.session_state.diagnostic_run_id = run_id
            _reconcile_approval_state(run)
            if run.get("status") in {"QUEUED", "RUNNING", "INTERRUPTED"}:
                st.session_state.diagnostic_cancel_sent = False
            else:
                if run.get("status") in {"CANCEL_REQUESTED", "CANCELLED"}:
                    st.session_state.diagnostic_cancel_sent = True
                st.session_state.diagnostic_cancel_confirm = False
                if st.session_state.diagnostic_cancel_response_unknown:
                    st.session_state.diagnostic_cancel_response_unknown = False
                    st.session_state.diagnostic_ui_error_kind = None
                    st.session_state.diagnostic_ui_error_message = None

        message_status, messages = _request(
            "GET", f"/diagnostic-sessions/{session_id}/messages"
        )
        if message_status != 200:
            raise _api_failure(message_status)
        view["messages"] = messages.get("items") or []

        draft_status, draft = _request(
            "GET", f"/diagnostic-sessions/{session_id}/draft-ticket"
        )
        if draft_status == 200:
            view["draft"] = draft
            history_status, history = _request(
                "GET",
                f"/diagnostic-sessions/{session_id}/similar-history",
                params={"limit": 3},
            )
            if history_status == 200:
                view["history"] = history.get("items") or []
            elif history_status == 503:
                view["history_error"] = (
                    "相似历史案例暂时不可用；本次诊断结果与本地草稿不受影响。"
                )
            elif history_status not in {404, 503}:
                raise _api_failure(history_status)
        elif draft_status != 404:
            raise _api_failure(draft_status)
        st.session_state.service_health = "ready"
        return view
    except UiApiError as exc:
        st.session_state.service_health = (
            "offline" if exc.kind == "offline" else "degraded"
        )
        view.update(error_kind=exc.kind, error_message=exc.message)
        return view


def _poll_interval(started_at: float, now: float) -> float:
    elapsed = max(0.0, now - started_at)
    if elapsed < 15:
        return 1.0
    if elapsed < 60:
        return 2.0
    return 5.0


def _effect_view_state(run: dict[str, Any]) -> str | None:
    typed = _typed_effect_projection(run)
    if typed is None:
        return None
    if not typed["valid"]:
        return "ERROR"
    if st.session_state.diagnostic_approval_response_unknown:
        return "APPROVAL_RESPONSE_UNKNOWN"
    if st.session_state.diagnostic_approval_submitting:
        return "APPROVAL_SUBMITTING"

    effect = typed["effect"]
    approval_status = effect["approval_status"]
    decision = effect.get("decision")
    effect_status = effect["status"]
    if run.get("status") == "CANCELLED" or approval_status == "CANCELLED":
        return "CANCELLED"
    if approval_status == "EXPIRED":
        return "APPROVAL_EXPIRED"
    if approval_status == "PENDING":
        payload = typed.get("payload") or {}
        server_time = payload.get("server_time")
        at_or_after_deadline = (
            type(server_time) is int and server_time >= effect["expires_at"]
        )
        if effect.get("can_decide") is not True or at_or_after_deadline:
            return "APPROVAL_EXPIRED"
        scope = _approval_scope(run)
        if (
            isinstance(scope, str)
            and st.session_state.diagnostic_approval_confirm_scope == scope
        ):
            return "APPROVAL_CONFIRM"
        return "AWAITING_APPROVAL"
    if decision == "REJECT":
        return "APPROVAL_REJECTED"
    if effect_status == "PREPARED" and decision == "APPROVE" and typed.get("kind") == "EFFECT_V2":
        return "EFFECT_WAITING_EXECUTION"
    if effect_status == "EXECUTING" or (effect_status == "PREPARED" and decision == "APPROVE"):
        return "EFFECT_EXECUTING"
    if effect_status == "APPLIED":
        return "VERIFYING_EFFECT"
    if effect_status == "VERIFIED":
        return "EFFECT_VERIFIED"
    if effect_status == "EXECUTION_FAILED":
        return "EFFECT_EXECUTION_FAILED"
    if effect_status == "VERIFICATION_FAILED":
        return "EFFECT_VERIFICATION_FAILED"
    if effect_status == "UNKNOWN":
        return "EFFECT_RESULT_UNKNOWN"
    return "ERROR"


def _view_state(view: dict[str, Any]) -> str:
    if view["error_kind"] == "invalid_url":
        return "INVALID_SESSION"
    if view["error_kind"] == "not_found":
        return "UNKNOWN_SESSION"
    if view["error_kind"] == "permission":
        return "PERMISSION_DENIED"
    if view["error_kind"] == "offline":
        return "OFFLINE"
    if view["error_kind"]:
        return "ERROR"
    if st.session_state.diagnostic_ui_error_kind == "permission":
        return "PERMISSION_DENIED"
    if st.session_state.diagnostic_response_unknown:
        return "RESPONSE_UNKNOWN"
    session = view.get("session")
    if session is None:
        return "INITIAL"
    run = view.get("run") or {}
    run_status = run.get("status")
    if st.session_state.diagnostic_cancel_confirm and run_status in {
        "QUEUED",
        "RUNNING",
        "INTERRUPTED",
    }:
        return "CANCEL_CONFIRM"
    effect_state = _effect_view_state(run)
    if effect_state is not None:
        return effect_state
    if run_status in {"QUEUED", "RUNNING", "INTERRUPTED", "CANCEL_REQUESTED"}:
        return str(run_status)
    if run_status == "FAILED":
        return (
            "FAILED_RETRYABLE"
            if run.get("failure_disposition") == "RETRYABLE"
            else "FAILED_FINAL"
        )
    if run_status == "CANCELLED" or session.get("status") == "CANCELLED":
        return "CANCELLED"
    if session.get("status") == "AWAITING_USER":
        return "AWAITING_USER"
    if session.get("status") == "COMPLETED":
        return "COMPLETED"
    if session.get("status") == "FAILED":
        return "FAILED_FINAL"
    return "OPEN"


STATE_COPY = {
    "INITIAL": ("开始一次可信诊断", "输入故障现象即可创建服务端会话。", "neutral"),
    "OPEN": ("会话已就绪", "服务端已建立空白会话，等待故障描述。", "ready"),
    "RESPONSE_UNKNOWN": (
        "正在确认请求结果",
        "写请求的响应未能确认。新写入已禁用，只会用原 request_id 对账。",
        "warning",
    ),
    "QUEUED": ("诊断已排队", "请求已持久登记，正在等待执行。", "running"),
    "RUNNING": ("诊断进行中", "服务端正在执行受限诊断流程。", "running"),
    "INTERRUPTED": (
        "诊断已安全中断",
        "当前 run 正等待后续审批能力处理；本页不会自行恢复。",
        "warning",
    ),
    "AWAITING_APPROVAL": (
        "等待一次性执行批准",
        "Verifier 已通过；请核对合成动作、目标、证据和有效期后决定。",
        "warning",
    ),
    "APPROVAL_CONFIRM": (
        "确认批准一次合成执行",
        "首次点击尚未消费 capability；只有再次确认才会提交决定。",
        "warning",
    ),
    "APPROVAL_SUBMITTING": (
        "正在提交审批决定",
        "决定按钮已锁定；页面不会并发或自动重复提交。",
        "running",
    ),
    "APPROVAL_RESPONSE_UNKNOWN": (
        "审批响应待确认",
        "全部新写入已冻结；页面只读对账后才允许人工重放原决定。",
        "warning",
    ),
    "APPROVAL_EXPIRED": (
        "审批能力已过期",
        "旧 capability 已禁用；页面不会延长、复活或执行它。",
        "danger",
    ),
    "APPROVAL_REJECTED": (
        "已拒绝并转人工",
        "该决定已被服务端记录，合成动作未执行。",
        "neutral",
    ),
    "EFFECT_EXECUTING": (
        "已批准 · 正在执行",
        "一次性决定已消费；正在执行本地合成动作，取消不能回滚已开始动作。",
        "running",
    ),
    "EFFECT_WAITING_EXECUTION": (
        "已批准 · 等待执行",
        "一次性决定已消费；合成动作尚未进入执行边界，页面不会提前显示成功。",
        "running",
    ),
    "VERIFYING_EFFECT": (
        "执行返回 · 正在独立读回",
        "动作返回不等于成功；服务端正在验证设备后置条件。",
        "running",
    ),
    "EFFECT_VERIFIED": (
        "独立读回已验证",
        "服务端读回与预期后置条件一致，动作成功完成。",
        "success",
    ),
    "EFFECT_EXECUTION_FAILED": (
        "动作执行失败 · 已安全关闭",
        "没有成功证据，系统不会自动或人工重执行该 effect。",
        "danger",
    ),
    "EFFECT_VERIFICATION_FAILED": (
        "独立读回验证失败",
        "执行结果与预期后置条件不一致，已停止并转人工复核。",
        "danger",
    ),
    "EFFECT_RESULT_UNKNOWN": (
        "执行结果未知 · 禁止重执行",
        "最多一次边界已生效；为避免重复副作用，只允许人工核查。",
        "danger",
    ),
    "CANCEL_CONFIRM": (
        "确认取消本次诊断",
        "首次点击尚未发送取消请求；可继续运行或确认取消。",
        "warning",
    ),
    "CANCEL_REQUESTED": (
        "正在安全取消",
        "服务端将在执行边界完成取消，请稍候。",
        "warning",
    ),
    "CANCELLED": ("诊断已取消", "该 run 不会继续执行，可新建诊断。", "neutral"),
    "AWAITING_USER": (
        "需要补充信息",
        "请回答服务端提出的问题；补充内容会进入同一会话。",
        "warning",
    ),
    "COMPLETED": ("诊断已完成", "结果、证据与唯一草稿已由服务端保存。", "success"),
    "FAILED_RETRYABLE": (
        "提交前调度失败",
        "图尚未开始，可安全重试原 run，不会追加消息或草稿。",
        "danger",
    ),
    "FAILED_FINAL": (
        "诊断已失败关闭",
        "为避免重放未知副作用，本次 run 不允许自动重试。",
        "danger",
    ),
    "OFFLINE": (
        "诊断服务离线",
        "未提交输入仍保留；自动恢复和手动重试都只执行只读 GET。",
        "danger",
    ),
    "PERMISSION_DENIED": (
        "权限不足",
        "写请求不会自动重放，请联系管理员确认权限。",
        "danger",
    ),
    "UNKNOWN_SESSION": (
        "找不到该诊断会话",
        "链接可能已失效。系统没有静默创建新会话。",
        "danger",
    ),
    "INVALID_SESSION": (
        "会话链接无效",
        "session_id 格式不正确，未向服务端发送恢复请求。",
        "danger",
    ),
    "ERROR": ("暂时无法恢复诊断", "服务端返回异常，请稍后重试连接。", "danger"),
}


def _focus_state_once(state_name: str, *, target_id: str) -> None:
    if state_name not in {
        "APPROVAL_CONFIRM",
        "APPROVAL_SUBMITTING",
        "APPROVAL_RESPONSE_UNKNOWN",
        "APPROVAL_EXPIRED",
        "APPROVAL_REJECTED",
        "EFFECT_EXECUTING",
        "VERIFYING_EFFECT",
        "EFFECT_VERIFIED",
        "EFFECT_EXECUTION_FAILED",
        "EFFECT_VERIFICATION_FAILED",
        "EFFECT_RESULT_UNKNOWN",
        "AWAITING_USER",
        "COMPLETED",
        "CANCELLED",
        "FAILED_RETRYABLE",
        "FAILED_FINAL",
        "OFFLINE",
        "PERMISSION_DENIED",
        "UNKNOWN_SESSION",
        "INVALID_SESSION",
        "ERROR",
    }:
        return
    focus_key = f"{st.session_state.diagnostic_session_id}:{state_name}:{target_id}"
    if st.session_state.diagnostic_focus_key == focus_key:
        return
    st.session_state.diagnostic_focus_key = focus_key
    focus_instance = secrets.token_hex(8)
    st.iframe(
        f"""
        <script>
        const focusInstance = '{focus_instance}';
        let focusAttempts = 0;
        const focusWhenReady = () => {{
          const target = window.parent.document.getElementById('{target_id}');
          if (target) {{
            target.focus({{preventScroll: false}});
            return;
          }}
          focusAttempts += 1;
          if (focusAttempts < 20) setTimeout(focusWhenReady, 50);
        }};
        setTimeout(focusWhenReady, 80);
        </script>
        """,
        height=1,
        tab_index=-1,
    )


def _render_session_header(view: dict[str, Any], state_name: str) -> None:
    title, description, tone = STATE_COPY[state_name]
    session = view.get("session") or {}
    session_id = session.get("session_id") or st.session_state.diagnostic_session_id
    parent_id = session.get("parent_session_id") or st.session_state.get(
        "diagnostic_parent_id"
    )
    metadata = "尚未建立服务端会话"
    if session_id:
        metadata = f"会话 {session_id[:8]}…{session_id[-4:]}"
    if parent_id:
        metadata += f" · 关联父会话 {parent_id[:8]}…{parent_id[-4:]}"
    st.markdown(
        f"""
        <section class="session-state state-{tone}" aria-live="polite">
          <div>
            <div id="diagnostic-state-title" class="diagnostic-state-heading"
              role="heading" aria-level="2" tabindex="-1">{_escape(title)}</div>
            <p>{_escape(description)}</p>
          </div>
          <span class="session-meta">{_escape(metadata)}</span>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_state_actions(view: dict[str, Any], state_name: str) -> None:
    session = view.get("session") or {}
    run = view.get("run") or {}
    session_id = session.get("session_id")
    run_id = run.get("run_id")

    write_error = st.session_state.diagnostic_ui_error_message
    if write_error:
        if st.session_state.diagnostic_ui_error_kind == "offline":
            st.warning(write_error)
        else:
            st.error(write_error)

    if state_name == "CANCEL_CONFIRM" and session_id and run_id:
        cols = st.columns(2)
        if cols[0].button(
            "确认取消",
            type="primary",
            key=f"confirm-cancel-{run_id}",
            use_container_width=True,
        ):
            _cancel_run(session_id, run_id)
            st.rerun(scope="app")
        if cols[1].button(
            "继续运行",
            key=f"keep-running-{run_id}",
            use_container_width=True,
        ):
            st.session_state.diagnostic_cancel_confirm = False
            st.rerun(scope="app")
        st.iframe(
            """
            <script>
            const doc = window.parent.document;
            if (!doc.documentElement.dataset.robotopsEscBound) {
              doc.documentElement.dataset.robotopsEscBound = '1';
              doc.addEventListener('keydown', (event) => {
                if (event.key !== 'Escape') return;
                const button = [...doc.querySelectorAll('button')]
                  .find((item) => item.textContent.trim() === '继续运行');
                if (button) button.click();
              });
            }
            </script>
            """,
            height=1,
            tab_index=-1,
        )
    elif state_name in {"QUEUED", "RUNNING", "INTERRUPTED"} and run_id:
        if st.button(
            "取消本次诊断",
            key=f"cancel-{run_id}",
        ):
            st.session_state.diagnostic_cancel_confirm = True
            st.rerun(scope="app")
    elif state_name == "FAILED_RETRYABLE" and session_id and run_id:
        if st.button("重试原请求", type="primary", key=f"retry-run-{run_id}"):
            _retry_run(session_id, run_id)
            st.rerun(scope="app")
    elif state_name == "RESPONSE_UNKNOWN":
        if st.session_state.diagnostic_request_missing:
            if st.button("继续原请求", type="primary", key="replay-pending"):
                _submit_message(
                    st.session_state.diagnostic_pending_payload or "",
                    reuse_pending=True,
                )
                st.rerun(scope="app")
        else:
            st.caption("页面正在使用原 request_id 只读对账，不会创建第二个 run。")
    elif state_name in {
        "OFFLINE",
        "ERROR",
        "PERMISSION_DENIED",
    }:
        if st.button("重试连接", key="retry-connection"):
            st.session_state.diagnostic_ui_error_kind = None
            st.session_state.diagnostic_ui_error_message = None
            st.rerun(scope="app")
    elif state_name in {"INITIAL", "UNKNOWN_SESSION", "INVALID_SESSION"}:
        if st.button("新建诊断", type="primary", key="new-diagnosis-empty"):
            _new_diagnosis()
            st.rerun(scope="app")
    elif state_name in {"COMPLETED", "CANCELLED", "FAILED_FINAL"}:
        first, second = st.columns(2)
        if first.button("新建诊断", key="new-diagnosis", use_container_width=True):
            _new_diagnosis()
            st.rerun(scope="app")
        if state_name == "COMPLETED" and session_id:
            if second.button(
                "关联新诊断", key="linked-diagnosis", use_container_width=True
            ):
                _new_diagnosis(session_id)
                st.rerun(scope="app")


def _render_pipeline(run: dict[str, Any] | None) -> None:
    run = run or {}
    result = run.get("result") or {}
    nodes = result.get("nodes") or []
    done = set(nodes)
    active = run.get("status") in ACTIVE_RUN_STATUSES
    steps: list[str] = []
    for index, name in enumerate(PIPELINE_NODES):
        css_class = "pstep done" if name in done else "pstep"
        if active and index == 0 and not done:
            css_class = "pstep run"
        steps.append(
            f'<div class="{css_class}">{_escape(NODE_LABELS.get(name, name))}</div>'
        )
    st.markdown(
        f'<div class="pipeline" aria-label="诊断流程">{"".join(steps)}</div>',
        unsafe_allow_html=True,
    )
    if active:
        st.caption("当前只展示服务端 run 生命周期；节点级真实 trace 将在升级阶段四接入。")
    elif nodes:
        st.caption("已完成节点为结果快照推断，不作为 Agent trace 证据。")


def _format_epoch_ms(value: object) -> str:
    if type(value) is not int or value < 0:
        return "—"
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def _ttl_copy(
    typed: dict[str, Any], scope: str | None
) -> tuple[str, str, str]:
    effect = typed["effect"]
    payload = typed.get("payload") or {}
    server_now = payload.get("server_time")
    if type(server_now) is not int:
        server_now = int(time.time() * 1000)
    anchor_server = st.session_state.diagnostic_approval_clock_server_ms
    anchor_monotonic = st.session_state.diagnostic_approval_clock_monotonic
    if (
        isinstance(scope, str)
        and st.session_state.diagnostic_approval_clock_scope == scope
        and type(anchor_server) is int
        and isinstance(anchor_monotonic, (int, float))
    ):
        server_now = max(
            server_now,
            anchor_server + int(max(0.0, time.monotonic() - anchor_monotonic) * 1000),
        )
    remaining_ms = max(0, int(effect["expires_at"]) - server_now)
    seconds = math.ceil(remaining_ms / 1000)
    visible = "已到期" if seconds <= 0 else f"{seconds // 60:02d}:{seconds % 60:02d}"
    bucket = "expired" if seconds <= 0 else (
        "under-one-minute" if seconds <= 60 else f"{math.ceil(seconds / 60)}-minutes"
    )
    if bucket == "expired":
        assistive = "审批能力已到期"
    elif bucket == "under-one-minute":
        assistive = "审批能力剩余不足一分钟"
    else:
        assistive = f"审批能力约剩余 {math.ceil(seconds / 60)} 分钟"
    announce = "polite" if (
        st.session_state.diagnostic_approval_countdown_bucket != bucket
    ) else "off"
    st.session_state.diagnostic_approval_countdown_bucket = bucket
    if (
        remaining_ms == 0
        and effect.get("approval_status") == "PENDING"
        and effect.get("can_decide") is True
        and isinstance(scope, str)
        and st.session_state.diagnostic_approval_expiry_reconciled_scope != scope
    ):
        st.session_state.diagnostic_approval_expiry_reconciled_scope = scope
        st.session_state.diagnostic_last_poll_at = 0.0
        st.rerun(scope="app")
    return visible, assistive, announce


def _view_refresh_fingerprint(view: dict[str, Any]) -> str | None:
    """Hash user-visible authority while excluding the moving server clock."""
    snapshot = deepcopy(view)
    run = snapshot.get("run")
    if isinstance(run, dict):
        interrupt = run.get("interrupt")
        if isinstance(interrupt, dict):
            payload = interrupt.get("payload")
            if isinstance(payload, dict):
                payload.pop("server_time", None)
    return _canonical_sha256(snapshot)


def _approval_countdown_bucket(run: dict[str, Any] | None) -> str | None:
    typed = _typed_effect_projection(run)
    if not typed or not typed["valid"]:
        return None
    effect = typed["effect"]
    if effect.get("approval_status") != "PENDING":
        return str(effect.get("approval_status") or "").lower() or None
    server_time = (typed.get("payload") or {}).get("server_time")
    if type(server_time) is not int:
        return None
    seconds = math.ceil(max(0, effect["expires_at"] - server_time) / 1000)
    if seconds <= 0:
        return "expired"
    if seconds <= 60:
        return "under-one-minute"
    return f"{math.ceil(seconds / 60)}-minutes"


def _latest_fault_description(view: dict[str, Any]) -> str:
    for message in reversed(view.get("messages") or []):
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"])
    return "服务端尚未返回可展示的故障描述"


def _effect_arguments(effect: dict[str, Any]) -> dict[str, Any]:
    if effect.get("approval_kind") == "EFFECT_V2":
        return effect["proposal"]["arguments"]
    return effect["arguments"]


def _effect_type(effect: dict[str, Any]) -> str:
    if effect.get("approval_kind") == "EFFECT_V2":
        return effect["proposal"]["effect_type"]
    return effect["effect_type"]


def _effect_source(effect: dict[str, Any]) -> str:
    if effect.get("approval_kind") == "EFFECT_V2":
        return effect["proposal"]["source"]
    return effect["source"]


def _render_effect_device_summary(view: dict[str, Any], typed: dict[str, Any]) -> None:
    effect = typed["effect"]
    device_model = _effect_arguments(effect)["device_model"]
    fault = (
        effect["authority"]["fault_description"]
        if typed.get("kind") == "EFFECT_V2"
        else _latest_fault_description(view)
    )
    st.markdown(
        f"""
        <section class="mod024-device-summary">
          <div class="mod024-section-heading">
            <span class="material-symbols-rounded" aria-hidden="true">precision_manufacturing</span>
            <h3>设备与故障</h3>
          </div>
          <dl>
            <div><dt>目标设备</dt><dd>{_escape(device_model)}</dd></div>
            <div><dt>故障描述</dt><dd>{_escape(fault)}</dd></div>
            <div><dt>执行环境</dt><dd>本地合成设备 · 不连接真实机器人</dd></div>
          </dl>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_approval_banner(
    view: dict[str, Any], typed: dict[str, Any], state_name: str
) -> None:
    effect = typed["effect"]
    session_id = str((view.get("session") or {}).get("session_id") or "")
    session_meta = (
        f"会话 {session_id[:8]}…{session_id[-4:]}" if len(session_id) >= 12 else ""
    )
    title, description, _tone = STATE_COPY[state_name]
    if typed.get("kind") == "EFFECT_V2" and state_name in {
        "AWAITING_APPROVAL", "APPROVAL_CONFIRM",
    }:
        description = (
            "服务端已核验结论并绑定本地合成遥测依据；当前仅公开编号摘要。"
            "请核对动作、目标、风险和有效期后决定。"
        )
    if typed.get("kind") == "EFFECT_V2" and state_name == "EFFECT_VERIFIED":
        title = "本地合成执行已验证"
        description = "服务端独立读回与本次动作的后置条件一致；不代表真实设备已执行。"
    if typed.get("kind") == "EFFECT_V2" and state_name == "VERIFYING_EFFECT":
        title = "执行已返回 · 待读回"
        description = "动作返回并非成功结论；正在等待服务端独立核对后置条件。"
    if typed.get("kind") == "EFFECT_V2" and state_name in {
        "EFFECT_EXECUTION_FAILED", "EFFECT_VERIFICATION_FAILED", "EFFECT_RESULT_UNKNOWN",
    }:
        description = f"{description} 请保留现场并转人工核查，不要重执行。"
    if state_name == "CANCELLED" and effect.get("attempt_count") == 1:
        description = (
            "run 已停止后续工作流提交；已进入执行边界的动作不会被回滚，"
            "下方保留服务端权威执行与读回结果。"
        )
    risk_level = effect["risk"].get("risk_level") or "—"
    icon = "verified_user" if state_name == "EFFECT_VERIFIED" else "gpp_maybe"
    st.markdown(
        f"""
        <section class="mod024-approval-banner state-{_escape(state_name.lower())}"
          aria-labelledby="mod024-approval-title" aria-live="polite">
          <span class="material-symbols-rounded" aria-hidden="true">{icon}</span>
          <div>
            <div id="mod024-approval-title" class="mod024-approval-heading"
              role="heading" aria-level="2">{_escape(title)}</div>
            <p>{_escape(description)}</p>
          </div>
          <div class="mod024-banner-meta">
            <span class="mod024-risk-badge">{_escape(risk_level)}风险</span>
            <span>{_escape(session_meta)}</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_verifier_summary(typed: dict[str, Any]) -> None:
    verifier = typed["effect"]["verifier"]
    checks = verifier["checks"]
    supporting_html = "".join(
        (
            '<span class="mod024-check-ok"><span class="material-symbols-rounded" '
            f'aria-hidden="true">check_circle</span>{_escape(VERIFIER_CHECK_LABELS[name])}</span>'
        )
        for name in verifier["supporting_evidence"]
    )
    missing = verifier["missing_evidence"]
    missing_copy = "无缺失项" if not missing else "、".join(
        VERIFIER_CHECK_LABELS.get(name, name) for name in missing
    )
    st.markdown(
        f"""
        <section class="mod024-verifier-summary">
          <div class="mod024-section-heading">
            <span class="material-symbols-rounded" aria-hidden="true">fact_check</span>
            <h3>Verifier 已通过</h3>
            <span class="mod024-status-chip status-passed">PASSED</span>
          </div>
          <p>确定性校验已覆盖计划、工具轨迹、服务端参数、规范结果、证据绑定与停止条件。</p>
          <div class="mod024-checks">{supporting_html}</div>
          <div class="mod024-missing"><strong>缺失证据</strong><span>{_escape(missing_copy)}</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("查看完整证据", expanded=False):
        st.caption(
            "以下为服务端校验并绑定的合成遥测/ROS 证据；内容按不可信数据展示，不作为指令执行。"
        )
        st.json(verifier["evidence"], expanded=1)
        st.code(verifier["evidence_sha256"], language=None)


def _render_claim_summary(typed: dict[str, Any], *, completed: bool) -> None:
    """Only display the V2 public summary; private proof is not a UI source."""
    effect = typed["effect"]
    summary = effect["claim_summary"]
    basis = effect["action_basis"]
    claim_ids = summary["claim_ids"]
    telemetry_ids = basis["telemetry_evidence_ids"]
    boundary_copy = (
        "此卡仍为审批时公开的编号摘要；下方仅在完成结果通过服务端核验后展示详细结论与引用。"
        if completed else "审批前仅公开结论与依据编号；详细结论与引用在完成后查看。"
    )
    st.markdown(
        f"""
        <section class="mod026-claim-summary" aria-label="已核验结论摘要">
          <div class="mod024-section-heading">
            <span class="material-symbols-rounded" aria-hidden="true">fact_check</span>
            <h3>已核验结论摘要</h3>
            <span class="mod024-status-chip status-passed">{_escape(summary['status'])}</span>
          </div>
          <p>服务端已核验 {_escape(len(claim_ids))} 条结论；
          合成演示政策记录 {_escape(len(telemetry_ids))} 项遥测依据。</p>
          <p class="mod026-evidence-limit">{_escape(boundary_copy)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("查看结论与合成遥测编号", expanded=False):
        st.caption("编号来自服务端公开审批摘要，不是证据正文或相似度评分；资料内容不能改变审批权限。")
        st.markdown("**已核验结论编号**")
        for claim_id in claim_ids:
            st.code(claim_id, language=None)
        st.markdown("**合成遥测依据编号**")
        for evidence_id in telemetry_ids:
            st.code(evidence_id, language=None)
        st.caption(
            f"演示政策：{basis['policy']}；详细引用仅由下方核验通过的完成结果展示。"
            if completed else
            f"演示政策：{basis['policy']}；完整逐条引用在诊断完成后展示。"
        )


def _render_effect_proposal(typed: dict[str, Any]) -> None:
    effect = typed["effect"]
    arguments = _effect_arguments(effect)
    device_model = arguments["device_model"]
    st.markdown(
        f"""
        <section class="mod024-effect-proposal">
          <div class="mod024-section-heading">
            <span class="material-symbols-rounded" aria-hidden="true">restart_alt</span>
            <h3>拟执行动作</h3>
          </div>
          <strong>重启合成机器人</strong>
          <code>{_escape(_effect_type(effect))}</code>
          <dl>
            <div><dt>来源</dt><dd>{_escape(_effect_source(effect))}</dd></div>
            <div><dt>目标</dt><dd>{_escape(device_model)}</dd></div>
            <div><dt>风险</dt><dd>{_escape(effect['risk']['risk_level'])}风险 · 仅本地合成 effect</dd></div>
          </dl>
          <div class="mod024-verification-plan">
            <span class="material-symbols-rounded" aria-hidden="true">manage_search</span>
            <span><b>验证计划</b>执行返回后独立读回 boot generation 与 binding，只有后置条件一致才标记成功。</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("查看 canonical arguments", expanded=False):
        st.caption("参数来自服务端权威提案；展开、折叠或刷新都不会改变审批绑定。")
        st.json(arguments, expanded=1)


def _render_capability_summary(
    typed: dict[str, Any], scope: str | None
) -> None:
    effect = typed["effect"]
    approval_status = str(effect.get("approval_status") or "")
    if approval_status == "PENDING":
        ttl_label = "剩余有效期"
        visible_ttl, assistive_ttl, announce = _ttl_copy(typed, scope)
    else:
        ttl_label = "Capability 状态"
        state_copy = {
            "CONSUMED": ("已消费", "审批能力已消费，不能再次决定"),
            "EXPIRED": ("已过期", "审批能力已过期，不能再次决定"),
            "CANCELLED": ("已取消", "审批能力已取消，不能再次决定"),
        }
        visible_ttl, assistive_ttl = state_copy.get(
            approval_status, ("不可决定", "审批能力当前不可决定")
        )
        state_bucket = f"approval-{approval_status.lower() or 'unavailable'}"
        announce = (
            "polite"
            if st.session_state.diagnostic_approval_countdown_bucket != state_bucket
            else "off"
        )
        st.session_state.diagnostic_approval_countdown_bucket = state_bucket
    st.markdown(
        f"""
        <section class="mod024-capability-summary">
          <div class="mod024-section-heading">
            <span class="material-symbols-rounded" aria-hidden="true">key</span>
            <h3>一次性 capability</h3>
          </div>
          <div class="mod024-ttl status-{_escape(approval_status.lower())}">
            <span>{_escape(ttl_label)}</span>
            <strong aria-hidden="true">{_escape(visible_ttl)}</strong>
            <span class="sr-only" aria-live="{announce}">{_escape(assistive_ttl)}</span>
          </div>
          <dl>
            <div><dt>有效截止</dt><dd>{_escape(_format_epoch_ms(effect['expires_at']))}</dd></div>
            <div><dt>审批范围</dt><dd>仅此会话、run、interrupt 与提案的一次决定</dd></div>
          </dl>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_capability_bindings(typed: dict[str, Any]) -> None:
    effect = typed["effect"]
    st.markdown(
        f"""
        <section class="mod024-binding-summary" aria-label="审批绑定标识">
          <div class="mod024-section-heading">
            <span class="material-symbols-rounded" aria-hidden="true">fingerprint</span>
            <h3>审批绑定标识</h3>
          </div>
          <dl>
            <div><dt>Effect key</dt><dd class="mod024-binding-value">{_escape(effect['effect_key'])}</dd></div>
            <div><dt>Binding SHA-256</dt><dd class="mod024-binding-value">{_escape(effect['binding_sha256'])}</dd></div>
            <div><dt>Proposal SHA-256</dt><dd class="mod024-binding-value">{_escape(effect['proposal_sha256'])}</dd></div>
            {('<div><dt>Proof SHA-256</dt><dd class="mod024-binding-value">' + _escape(effect['proof_sha256']) + '</dd></div>') if typed.get('kind') == 'EFFECT_V2' else ''}
          </dl>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _restore_approval_trigger_focus() -> None:
    if not st.session_state.diagnostic_restore_approval_trigger:
        return
    st.session_state.diagnostic_restore_approval_trigger = False
    focus_instance = secrets.token_hex(8)
    st.iframe(
        f"""
        <script>
        const focusInstance = '{focus_instance}';
        let focusAttempts = 0;
        const focusApprovalTrigger = () => {{
          const target = [...window.parent.document.querySelectorAll('button')]
            .find((item) => item.textContent.trim() === '批准一次执行');
          if (target) {{
            target.focus({{preventScroll: false}});
            return;
          }}
          focusAttempts += 1;
          if (focusAttempts < 20) setTimeout(focusApprovalTrigger, 50);
        }};
        setTimeout(focusApprovalTrigger, 80);
        </script>
        """,
        height=1,
        tab_index=-1,
    )


def _install_approval_escape(scope: str) -> None:
    should_focus = st.session_state.diagnostic_approval_focus_key != scope
    if should_focus:
        st.session_state.diagnostic_approval_focus_key = scope
    focus_instance = secrets.token_hex(8) if should_focus else "stable"
    focus_script = (
        """
        setTimeout(() => {
          const title = doc.getElementById('mod024-confirm-title');
          if (title) title.focus({preventScroll: false});
        }, 80);
        """
        if should_focus
        else ""
    )
    st.iframe(
        f"""
        <script>
        const focusInstance = '{focus_instance}';
        const doc = window.parent.document;
        if (!doc.documentElement.dataset.robotopsApprovalEscBoundV2) {{
          doc.documentElement.dataset.robotopsApprovalEscBoundV2 = '1';
          doc.addEventListener('keydown', (event) => {{
            if (event.key !== 'Escape') return;
            const button = [...doc.querySelectorAll('button')]
              .find((item) => item.textContent.trim() === '放弃批准');
            if (button) {{
              event.preventDefault();
              button.click();
            }}
          }}, true);
        }}
        {focus_script}
        </script>
        """,
        height=1,
        tab_index=-1,
    )


def _render_effect_decision_bar(
    view: dict[str, Any], typed: dict[str, Any], state_name: str
) -> None:
    session = view.get("session") or {}
    run = view.get("run") or {}
    effect = typed["effect"]
    session_id = session.get("session_id")
    run_id = run.get("run_id")
    scope = _approval_scope(run)
    can_decide = (
        typed["valid"]
        and isinstance(scope, str)
        and effect.get("approval_status") == "PENDING"
        and effect.get("status") == "PREPARED"
        and effect.get("can_decide") is True
        and not st.session_state.diagnostic_approval_write_locked
        and state_name == "AWAITING_APPROVAL"
    )

    with st.container(border=True):
        st.markdown(
            '<span class="mod024-decision-bar" aria-hidden="true"></span>'
            '<div class="mod024-section-heading"><span class="material-symbols-rounded" '
            'aria-hidden="true">rule</span><h3>决定与恢复</h3></div>',
            unsafe_allow_html=True,
        )

        if state_name == "APPROVAL_CONFIRM" and can_decide is False and scope:
            confirm_matches = (
                st.session_state.diagnostic_approval_confirm_scope == scope
                and effect.get("approval_status") == "PENDING"
                and effect.get("can_decide") is True
                and not st.session_state.diagnostic_approval_write_locked
            )
            visible_ttl, assistive_ttl, announce = _ttl_copy(typed, scope)
            st.markdown(
                f"""
                <section class="mod024-approval-confirm">
                  <div id="mod024-confirm-title" class="mod024-confirm-heading"
                    role="heading" aria-level="3" tabindex="-1">确认批准一次合成执行</div>
                  <p>你将批准 <b>{_escape(_effect_type(effect))}</b>，目标为
                  <b>{_escape(_effect_arguments(effect)['device_model'])}</b>，风险为
                  <b>{_escape(effect['risk']['risk_level'])}风险</b>。</p>
                  <p class="mod024-confirm-ttl">剩余 TTL
                    <strong aria-hidden="true">{_escape(visible_ttl)}</strong>
                    <span class="sr-only" aria-live="{announce}">{_escape(assistive_ttl)}</span>
                  </p>
                </section>
                """,
                unsafe_allow_html=True,
            )
            confirm_col, abandon_col = st.columns(2)
            if confirm_col.button(
                "确认批准并执行",
                type="primary",
                key=f"confirm-effect-{run_id}",
                use_container_width=True,
                disabled=not confirm_matches,
            ):
                with st.spinner("正在提交原审批绑定…"):
                    _resume_effect_decision(
                        session_id, run_id, run, "APPROVE"
                    )
                st.rerun(scope="app")
            if abandon_col.button(
                "放弃批准",
                key=f"abandon-effect-{run_id}",
                use_container_width=True,
            ):
                st.session_state.diagnostic_approval_confirm_scope = None
                st.session_state.diagnostic_approval_focus_key = None
                st.session_state.diagnostic_focus_key = None
                st.session_state.diagnostic_restore_approval_trigger = True
                st.rerun(scope="app")
            _install_approval_escape(scope)
            return

        if state_name == "APPROVAL_RESPONSE_UNKNOWN":
            st.warning("先只读 GET 对账；在结果明确前，禁止发起任何新决定或取消请求。")
            frozen = st.columns(3)
            frozen[0].button(
                "批准一次执行", key=f"frozen-approve-{run_id}", disabled=True,
                use_container_width=True,
            )
            frozen[1].button(
                "拒绝并转人工", key=f"frozen-reject-{run_id}", disabled=True,
                use_container_width=True,
            )
            frozen[2].button(
                "取消本次诊断", key=f"frozen-cancel-{run_id}", disabled=True,
                use_container_width=True,
            )
            decision = st.session_state.diagnostic_pending_approval_decision
            replay_label = (
                "重放原批准决定" if decision == "APPROVE" else "重放原拒绝决定"
            )
            if st.button(
                replay_label,
                type="primary",
                key=f"replay-effect-{run_id}",
                use_container_width=True,
                disabled=not st.session_state.diagnostic_approval_replay_available,
            ):
                with st.spinner("正在重放同一决定与同一绑定…"):
                    _resume_effect_decision(
                        session_id, run_id, run, str(decision), replay=True
                    )
                st.rerun(scope="app")
            return

        if typed.get("kind") == "EFFECT_V2" and effect.get("approval_status") == "CONSUMED":
            st.caption("一次性审批已消费；本页面不会再次提供批准或拒绝入口。")
            if run.get("status") in {"QUEUED", "RUNNING", "INTERRUPTED"}:
                if st.button(
                    "取消本次诊断",
                    key=f"cancel-effect-{run_id}",
                    use_container_width=True,
                ):
                    st.session_state.diagnostic_cancel_confirm = True
                    st.rerun(scope="app")
            if effect.get("attempt_count") == 1:
                st.caption("动作已经进入最多一次执行边界；取消 run 不代表回滚设备动作。")
            return

        decision_columns = st.columns(3)
        if decision_columns[0].button(
            "批准一次执行",
            type="primary",
            key=f"approve-effect-{run_id}",
            use_container_width=True,
            disabled=not can_decide,
        ):
            st.session_state.diagnostic_approval_confirm_scope = scope
            st.rerun(scope="app")
        if decision_columns[1].button(
            "拒绝并转人工",
            key=f"reject-effect-{run_id}",
            use_container_width=True,
            disabled=not can_decide,
        ):
            with st.spinner("正在提交拒绝决定…"):
                _resume_effect_decision(session_id, run_id, run, "REJECT")
            st.rerun(scope="app")

        active_run = run.get("status") in {"QUEUED", "RUNNING", "INTERRUPTED"}
        cancel_locked = (
            not active_run
            or state_name in {
                "APPROVAL_SUBMITTING",
                "APPROVAL_RESPONSE_UNKNOWN",
            }
        )
        if decision_columns[2].button(
            "取消本次诊断",
            key=f"cancel-effect-{run_id}",
            use_container_width=True,
            disabled=cancel_locked,
        ):
            st.session_state.diagnostic_cancel_confirm = True
            st.rerun(scope="app")
        if effect.get("attempt_count") == 1:
            st.caption("effect 已进入最多一次执行边界；取消 run 不代表回滚已开始或已完成的动作。")
        _restore_approval_trigger_focus()


def _effect_status_state(effect: dict[str, Any]) -> str | None:
    effect_status = effect.get("status")
    if (effect_status == "PREPARED" and effect.get("decision") == "APPROVE"
            and effect.get("approval_kind") == "EFFECT_V2"):
        return "EFFECT_WAITING_EXECUTION"
    if effect_status == "EXECUTING" or (
        effect_status == "PREPARED" and effect.get("decision") == "APPROVE"
    ):
        return "EFFECT_EXECUTING"
    if effect_status == "APPLIED":
        return "VERIFYING_EFFECT"
    if effect_status == "VERIFIED":
        return "EFFECT_VERIFIED"
    if effect_status == "EXECUTION_FAILED":
        return "EFFECT_EXECUTION_FAILED"
    if effect_status == "VERIFICATION_FAILED":
        return "EFFECT_VERIFICATION_FAILED"
    if effect_status == "UNKNOWN":
        return "EFFECT_RESULT_UNKNOWN"
    return None


def _phase_classes(state_name: str) -> list[str]:
    if state_name in {"AWAITING_APPROVAL", "APPROVAL_CONFIRM", "APPROVAL_EXPIRED"}:
        return ["active", "pending", "pending", "pending"]
    if state_name == "APPROVAL_REJECTED":
        return ["stopped", "pending", "pending", "pending"]
    if state_name in {"APPROVAL_SUBMITTING", "APPROVAL_RESPONSE_UNKNOWN"}:
        return ["active", "pending", "pending", "pending"]
    if state_name == "EFFECT_EXECUTING":
        return ["done", "active", "pending", "pending"]
    if state_name == "EFFECT_WAITING_EXECUTION":
        return ["done", "pending", "pending", "pending"]
    if state_name == "VERIFYING_EFFECT":
        return ["done", "done", "active", "pending"]
    if state_name == "EFFECT_VERIFIED":
        return ["done", "done", "done", "done"]
    if state_name == "EFFECT_EXECUTION_FAILED":
        return ["done", "error", "stopped", "error"]
    if state_name == "EFFECT_VERIFICATION_FAILED":
        return ["done", "done", "error", "error"]
    if state_name == "EFFECT_RESULT_UNKNOWN":
        return ["done", "error", "error", "error"]
    return ["pending", "pending", "pending", "pending"]


def _render_effect_progress(typed: dict[str, Any], state_name: str) -> None:
    labels = ("审批决定", "合成执行", "独立读回", "最终结论")
    effect_state = _effect_status_state(typed["effect"])
    progress_state = effect_state or state_name
    classes = _phase_classes(progress_state)
    items = "".join(
        f'<li class="{css}"><span aria-hidden="true"></span>{_escape(label)}</li>'
        for label, css in zip(labels, classes, strict=True)
    )
    current = STATE_COPY[state_name][0]
    if state_name == "CANCELLED" and effect_state is not None:
        current = f"{current} · {STATE_COPY[effect_state][0]}"
    st.markdown(
        f"""
        <section class="mod024-effect-progress" aria-labelledby="mod024-stage-title">
          <div class="mod024-section-heading">
            <span class="material-symbols-rounded" aria-hidden="true">account_tree</span>
            <div id="mod024-stage-title" class="mod024-stage-heading"
              role="heading" aria-level="3" tabindex="-1">当前阶段：{_escape(current)}</div>
          </div>
          <ol>{items}</ol>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_readback_result(typed: dict[str, Any], state_name: str) -> None:
    effect = typed["effect"]
    verification = effect.get("verification")
    result = effect.get("result")
    effect_status = effect.get("status")
    if effect_status == "VERIFIED":
        title = "读回与预期一致"
        copy = "独立 read-after-write 已证明 boot generation 与本次 binding 一致。"
        tone = "success"
    elif effect_status == "VERIFICATION_FAILED":
        title = "读回与预期不一致"
        copy = "没有成功结论；请保留现场并转人工核查。"
        tone = "danger"
    elif effect_status == "UNKNOWN":
        title = "读回结果未知"
        copy = "禁止重新执行；请使用线下或独立通道确认设备状态。"
        tone = "danger"
    elif effect_status == "EXECUTION_FAILED":
        title = "动作没有成功返回"
        copy = "未形成成功证据，服务端已失败关闭。"
        tone = "danger"
    elif effect_status == "PREPARED" and effect.get("decision") == "REJECT":
        title = "动作未执行"
        copy = "拒绝决定已记录；一次性 capability 未进入执行边界。"
        tone = "neutral"
    elif (
        effect_status == "PREPARED"
        and effect.get("decision") == "APPROVE"
        and typed.get("kind") == "EFFECT_V2"
    ):
        title = "等待合成执行"
        copy = "批准决定已记录，但动作尚未进入执行边界；请等待服务端状态。"
        tone = "neutral"
    elif effect_status == "PREPARED" and state_name == "CANCELLED":
        title = "动作未开始"
        copy = "run 已取消，未进入合成执行边界。"
        tone = "neutral"
    elif effect_status == "PREPARED" and typed.get("kind") == "EFFECT_V2":
        if state_name == "APPROVAL_EXPIRED":
            title = "审批已过期 · 动作未执行"
            copy = "一次性 capability 已禁用；没有进入合成执行或独立读回。"
        elif state_name in {"APPROVAL_SUBMITTING", "APPROVAL_RESPONSE_UNKNOWN"}:
            title = "审批结果待确认 · 禁止推断执行成功"
            copy = "请先读取服务端权威状态；页面不会自动重发或改换决定。"
        else:
            title = "尚未执行 · 等待审批"
            copy = "只有批准后才可能进入合成执行；执行返回后仍须独立读回。"
        tone = "neutral"
    else:
        title = "等待独立读回"
        copy = "只有服务端独立读回通过，页面才会显示动作成功。"
        tone = "neutral"
    if state_name == "CANCELLED" and effect.get("attempt_count") == 1:
        copy = f"{copy} 取消 run 不会回滚已经开始或完成的动作。"
    st.markdown(
        f"""
        <section class="mod024-readback-result state-{tone}">
          <div class="mod024-section-heading">
            <span class="material-symbols-rounded" aria-hidden="true">{('task_alt' if tone == 'success' else 'policy')}</span>
            <h3>{_escape(title)}</h3>
          </div>
          <p>{_escape(copy)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if result is not None or verification is not None:
        with st.expander("查看执行与读回记录", expanded=False):
            st.json({"result": result, "verification": verification}, expanded=1)


def _render_typed_effect_workspace(
    view: dict[str, Any], typed: dict[str, Any], state_name: str
) -> None:
    st.markdown(
        '<span class="mod024-workspace-anchor" aria-hidden="true"></span>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.36, 0.64], gap="large")
    with left:
        st.markdown(
            '<span class="mod024-primary-column" aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
        if typed["valid"]:
            _render_effect_device_summary(view, typed)
        else:
            st.error("服务端审批投影不完整；页面已禁止所有决定。")
        with st.container(border=True):
            st.markdown('<h3 class="panel-title">诊断进度</h3>', unsafe_allow_html=True)
            _render_pipeline(view.get("run"))
    with right:
        st.markdown(
            '<span class="mod024-primary-column" aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
        if not typed["valid"]:
            st.error("无法验证 effect、Verifier 与审批 binding 的一致性。请刷新或转人工。")
            return
        _render_approval_banner(view, typed, state_name)
        if typed.get("kind") == "EFFECT_V2":
            _render_claim_summary(
                typed, completed=(view.get("run") or {}).get("status") == "COMPLETED"
            )
        else:
            _render_verifier_summary(typed)
        st.markdown(
            '<span class="mod024-detail-grid-anchor" aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
        proposal_col, decision_col = st.columns([0.52, 0.48], gap="medium")
        with proposal_col:
            st.markdown(
                '<span class="mod024-detail-column" aria-hidden="true"></span>',
                unsafe_allow_html=True,
            )
            _render_effect_proposal(typed)
        with decision_col:
            st.markdown(
                '<span class="mod024-detail-column" aria-hidden="true"></span>',
                unsafe_allow_html=True,
            )
            _render_capability_summary(typed, _approval_scope(view.get("run")))
            _render_effect_decision_bar(view, typed, state_name)
            _render_effect_progress(typed, state_name)
            _render_readback_result(typed, state_name)
            _render_capability_bindings(typed)
    _render_timeline(view, state_name)


def _fill_example(value: str) -> None:
    st.session_state.diagnostic_input = value
    st.session_state.diagnostic_focus_input = True


def _install_composer_behavior(*, focus_input: bool) -> None:
    """Bind Enter-to-submit and restore focus after an example fill."""
    focus_script = "area.focus({preventScroll: true});" if focus_input else ""
    st.iframe(
        f"""
        <script>
        const doc = window.parent.document;
        const bindComposer = () => {{
          const area = [...doc.querySelectorAll('textarea')].find((item) =>
            ['故障现象', '补充信息'].includes(item.getAttribute('aria-label'))
          );
          if (!area) return;
          if (!area.dataset.robotopsEnterBound) {{
            area.dataset.robotopsEnterBound = '1';
            area.addEventListener('keydown', (event) => {{
              if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
              const form = area.closest('form, [data-testid="stForm"]');
              const submit = form && form.querySelector('[data-testid="stBaseButton-primaryFormSubmit"]');
              if (!submit || submit.disabled || !area.value.trim()) return;
              event.preventDefault();
              submit.click();
            }});
          }}
          {focus_script}
        }};
        setTimeout(bindComposer, 80);
        </script>
        """,
        height=1,
        tab_index=-1,
    )


def _render_composer(view: dict[str, Any], state_name: str) -> None:
    writable = state_name in {"INITIAL", "OPEN", "AWAITING_USER"}
    label = "补充信息" if state_name == "AWAITING_USER" else "故障现象"
    with st.container(border=True):
        st.markdown('<h3 class="panel-title">故障描述</h3>', unsafe_allow_html=True)
        with st.form("diagnostic-composer", clear_on_submit=False):
            text = st.text_area(
                label,
                key="diagnostic_input",
                placeholder="例如：机器人上电后无法连接 Wi-Fi，状态灯持续闪烁",
                height=126,
                disabled=not writable,
                max_chars=8000,
            )
            submitted = st.form_submit_button(
                "提交补充" if state_name == "AWAITING_USER" else "开始诊断",
                type="primary",
                use_container_width=True,
                disabled=not writable,
            )
        st.markdown('<p class="field-hint">示例仅填充，不会自动提交</p>', unsafe_allow_html=True)
        example_columns = st.columns(3)
        for index, example in enumerate(EXAMPLES):
            example_columns[index].button(
                ("网络异常", "关节异常", "电池风险")[index],
                key=f"example-{index}",
                on_click=_fill_example,
                args=(example,),
                disabled=not writable,
                use_container_width=True,
            )
        _install_composer_behavior(
            focus_input=bool(
                st.session_state.pop("diagnostic_focus_input", False)
            )
        )
        if submitted:
            with st.spinner("正在登记诊断请求…"):
                _submit_message(text)
            st.rerun(scope="app")

    with st.container(border=True):
        st.markdown('<h3 class="panel-title">运行进度</h3>', unsafe_allow_html=True)
        _render_pipeline(view.get("run"))


def _render_result_card(diagnosis: dict[str, Any]) -> None:
    risk = str(diagnosis.get("risk_level") or "—")
    risk_class = {"低": "low", "中": "mid", "高": "high"}.get(risk, "gray")
    causes = diagnosis.get("causes") or []
    steps = diagnosis.get("steps") or []
    causes_html = "".join(f"<li>{_escape(item)}</li>" for item in causes)
    steps_html = "".join(f"<li>{_escape(item)}</li>" for item in steps)
    risk_alert = ""
    if risk == "高":
        risk_alert = (
            '<div class="risk-notice"><strong>高风险提示</strong>'
            "请停止带电操作，不要自行拆机，并联系专业技术人员。</div>"
        )
    st.markdown(
        f"""
        <section class="diag-result">
          <div class="head">
            <div class="kpi"><span class="k">设备型号</span><span class="v">{_escape(diagnosis.get('device_model') or '—')}</span></div>
            <div class="kpi"><span class="k">故障类型</span><span class="v">{_escape(diagnosis.get('fault_category') or '—')}</span></div>
            <div class="kpi"><span class="k">风险等级</span><span class="tag {risk_class}">{_escape(risk)}</span></div>
          </div>
          {risk_alert}
          <div class="result-body">
            <section><h3>可能原因</h3><ul>{causes_html or '<li>暂无结构化原因</li>'}</ul></section>
            <section><h3>排查步骤</h3><ol>{steps_html or '<li>请按回答中的安全建议处理</li>'}</ol></section>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_references(references: list[dict[str, Any]]) -> None:
    if not references:
        return
    st.markdown("### 引用来源")
    for index, reference in enumerate(references, 1):
        number = reference.get("n") or index
        filename = reference.get("filename") or "未命名资料"
        with st.expander(f"[{number}] {filename}"):
            location = reference.get("page")
            if location is None:
                chunk_index = reference.get("chunk_index")
                location = (
                    f"片段 {int(chunk_index) + 1}"
                    if isinstance(chunk_index, int)
                    else "位置未标注"
                )
            similarity = reference.get("similarity")
            similarity_text = (
                f" · 相似度 {float(similarity):.2f}"
                if isinstance(similarity, (int, float))
                else ""
            )
            st.caption(f"{filename} · {location}{similarity_text}")
            st.write(reference.get("text") or reference.get("snippet") or "暂无原文片段")


def _render_history(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    st.markdown("### 相似历史案例")
    st.caption("已在服务端排除当前草稿及其关联外部工单。")
    for item in items:
        st.markdown(
            f"""
            <div class="history-match">
              <span>{_escape(item.get('source_id'))}</span>
              <p>{_escape(item.get('summary'))}</p>
              <b>{float(item.get('similarity') or 0):.2f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_draft(
    session_id: str | None,
    draft: dict[str, Any] | None,
    *,
    allow_retry: bool,
) -> None:
    if not session_id or not draft:
        return
    status = str(draft.get("status") or "")
    labels = {
        "COLLECTING": "信息收集中",
        "READY": "本地草稿已就绪",
        "PROJECTION_FAILED": "工单投影失败",
        "PROJECTED": "已生成外部工单",
    }
    ticket_id = draft.get("external_ticket_id")
    session_suffix = f"?session_id={session_id}"
    st.markdown(
        f"""
        <section class="draft-strip draft-{_escape(status.lower())}">
          <div>
            <span class="draft-label">唯一草稿 · v{int(draft.get('version') or 1)}</span>
            <strong>{_escape(labels.get(status, status))}</strong>
            <small>{_escape(ticket_id or draft.get('draft_id'))}</small>
          </div>
          <a href="./tickets{session_suffix}">查看工单页</a>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if draft.get("projection_note"):
        st.caption(str(draft["projection_note"]))
    if status == "PROJECTION_FAILED" and allow_retry:
        if st.button(
            "重试草稿投影",
            key=f"retry-draft-{draft.get('draft_id')}-{draft.get('version')}",
        ):
            _retry_draft(session_id, draft)
            st.rerun(scope="app")


def _validated_v2_publication(
    run: dict[str, Any], typed: dict[str, Any]
) -> dict[str, Any] | None:
    """Read completed public claims only after the server projection binds them."""
    if run.get("status") != "COMPLETED" or typed.get("kind") != "EFFECT_V2":
        return None
    try:
        result = run.get("result")
        if type(result) is not dict or type(result.get("diagnosis")) is not dict:
            return None
        raw = result["diagnosis"].get("claim_evidence")
        if type(raw) is not dict:
            return None
        publication = validate_claim_publication(
            raw, expected_authority=typed["effect"]["authority"]
        )
        projection = publication["projection"]
        if any(
            result.get(field) != projection[field]
            for field in ("status", "reply", "references", "follow_up")
        ) or {
            key: value for key, value in result["diagnosis"].items()
            if key != "claim_evidence"
        } != projection["diagnosis"]:
            return None
        assessment = publication["verification"]["assessment"]
        summary = typed["effect"]["claim_summary"]
        if summary != {
            "status": "VERIFIED",
            "verification_sha256": assessment["verification_sha256"],
            "source_snapshot_sha256": assessment["sources"]["snapshot_sha256"],
            "verifier_result_sha256": assessment["sources"]["tool_snapshot"]["verifier_result_sha256"],
            "claim_ids": [item["claim_id"] for item in assessment["claims"]],
        }:
            return None
        return publication
    except (ClaimPublicationError, KeyError, TypeError, ValueError):
        return None


def _render_v2_claim_evidence(publication: dict[str, Any] | None) -> None:
    if publication is None:
        st.warning("逐条证据与审批摘要不一致或暂不可用；页面不展示未经核验的引用。")
        return
    assessment = publication["verification"]["assessment"]
    results = {item["claim_id"]: item for item in assessment["results"]}
    sources = {item["evidence_id"]: item for item in assessment["sources"]["inventory"]}
    st.markdown("### 逐条结论与证据")
    st.caption("以下仅来自已完成 run 的服务端公开核验结果；合成遥测不是生产设备状态。")
    for claim in assessment["claims"]:
        result = results[claim["claim_id"]]
        with st.expander(f"{claim['kind']} · {claim['claim_id']} · {result['verdict']}"):
            st.text(claim["text"])
            supporting = result["supporting_ids"]
            contradicting = result["contradicting_ids"]
            if not supporting and not contradicting:
                st.caption("现有来源不足以支持此结论；不能将其当成已确认事实。")
            for label, identifiers in (("支持", supporting), ("反证", contradicting)):
                for identifier in identifiers:
                    source = sources[identifier]
                    st.markdown(f"**{label}来源 · {_escape(source['source_kind'])}**")
                    st.code(identifier, language=None)
                    st.text(source["text"])
                    st.json(source["locator"], expanded=False)


def _render_timeline(view: dict[str, Any], state_name: str) -> None:
    messages = view.get("messages") or []
    session = view.get("session") or {}
    session_id = session.get("session_id")
    run = view.get("run") or {}
    typed = _typed_effect_projection(run)
    is_v2 = typed is not None and typed.get("kind") == "EFFECT_V2"
    v2_completed = is_v2 and run.get("status") == "COMPLETED"
    publication = _validated_v2_publication(run, typed) if v2_completed else None
    latest_assistant = next(
        (item for item in reversed(messages) if item.get("role") == "assistant"), None
    )
    with st.container(border=True):
        st.markdown('<h3 class="panel-title">权威消息</h3>', unsafe_allow_html=True)
        if not messages:
            st.markdown(
                """
                <div class="diagnostic-empty">
                  <strong>尚无诊断消息</strong>
                  <span>提交故障描述后，服务端消息会按顺序显示在这里。</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for message in messages:
            role = "user" if message.get("role") == "user" else "assistant"
            with st.chat_message(role):
                if v2_completed and message is latest_assistant:
                    if publication is not None:
                        st.markdown(publication["projection"]["reply"])
                    else:
                        st.warning("本次完成结果无法核验，暂不展示诊断正文。")
                else:
                    st.markdown(str(message.get("content") or ""))

        payload = (latest_assistant or {}).get("payload") or {}
        if is_v2:
            payload = publication["projection"] if publication is not None else {}
        follow_up = payload.get("follow_up") or []
        if state_name == "AWAITING_USER" and follow_up:
            st.markdown("### 快速补充")
            for question_index, question in enumerate(follow_up):
                st.write(question.get("question") or "请补充信息")
                options = question.get("options") or []
                option_columns = st.columns(max(1, min(3, len(options))))
                for option_index, option in enumerate(options[:3]):
                    if option_columns[option_index % len(option_columns)].button(
                        str(option),
                        key=(
                            f"follow-up-{latest_assistant.get('message_id')}-"
                            f"{question_index}-{option_index}"
                        ),
                        use_container_width=True,
                    ):
                        _submit_message(str(option))
                        st.rerun(scope="app")

        diagnosis = payload.get("diagnosis") or {}
        if diagnosis and diagnosis.get("risk_level"):
            _render_result_card(diagnosis)
        _render_references(payload.get("references") or [])
        if v2_completed:
            _render_v2_claim_evidence(publication)
        _render_history(view.get("history") or [])
        if view.get("history_error"):
            st.warning(str(view["history_error"]))
        _render_draft(
            session_id,
            view.get("draft"),
            allow_retry=state_name == "COMPLETED",
        )


def _render_workspace(view: dict[str, Any]) -> None:
    state_name = _view_state(view)
    typed = _typed_effect_projection(view.get("run"))
    if typed is None or typed.get("valid") is not True:
        _render_session_header(view, state_name)
    _render_state_actions(view, state_name)
    if typed is not None:
        _render_typed_effect_workspace(view, typed, state_name)
    else:
        st.markdown(
            '<span class="diagnosis-grid-anchor" aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([0.38, 0.62], gap="large")
        with left:
            st.markdown(
                '<span class="diagnosis-column-marker" aria-hidden="true"></span>',
                unsafe_allow_html=True,
            )
            _render_composer(view, state_name)
        with right:
            st.markdown(
                '<span class="diagnosis-column-marker" aria-hidden="true"></span>',
                unsafe_allow_html=True,
            )
            _render_timeline(view, state_name)
    if state_name == "APPROVAL_CONFIRM" and typed is not None and typed.get("valid") is True:
        focus_target = "mod024-confirm-title"
    elif typed is not None and typed.get("valid") is True:
        focus_target = "mod024-stage-title"
    else:
        focus_target = "diagnostic-state-title"
    _focus_state_once(state_name, target_id=focus_target)


def main() -> None:
    load_theme()
    _init_state()
    session_id, url_error = _resolve_session_id()
    view = _restore(session_id, url_error)
    page_header(
        "智能诊断",
        "服务端权威会话、可恢复运行、证据与单一草稿",
        route="/diagnosis",
        service_status=st.session_state.service_health,
    )

    run = view.get("run") or {}
    state_name = _view_state(view)
    should_tick = (
        run.get("status") in ACTIVE_RUN_STATUSES
        or st.session_state.diagnostic_approval_response_unknown
        or (
            st.session_state.diagnostic_response_unknown
            and not st.session_state.diagnostic_request_missing
        )
        or state_name == "OFFLINE"
    )
    if should_tick and not st.session_state.diagnostic_poll_started_at:
        st.session_state.diagnostic_poll_started_at = time.monotonic()

    _render_workspace(view)

    @st.fragment(run_every=1.0 if should_tick else None)
    def live_poll() -> None:
        nonlocal view
        before = _view_state(view)
        now = time.monotonic()
        should_refresh = False
        if before == "OFFLINE":
            should_refresh = now - st.session_state.diagnostic_last_poll_at >= 5.0
        elif before == "RESPONSE_UNKNOWN":
            should_refresh = now - st.session_state.diagnostic_last_poll_at >= 1.0
        elif before == "APPROVAL_RESPONSE_UNKNOWN":
            should_refresh = now - st.session_state.diagnostic_last_poll_at >= 1.0
        elif (view.get("run") or {}).get("status") in ACTIVE_RUN_STATUSES:
            started = st.session_state.diagnostic_poll_started_at or now
            should_refresh = now - st.session_state.diagnostic_last_poll_at >= (
                _poll_interval(started, now)
            )
        if should_refresh:
            st.session_state.diagnostic_last_poll_at = now
            before_fingerprint = _view_refresh_fingerprint(view)
            before_bucket = st.session_state.diagnostic_approval_countdown_bucket
            refreshed = _restore(session_id, url_error)
            after_fingerprint = _view_refresh_fingerprint(refreshed)
            after_bucket = _approval_countdown_bucket(refreshed.get("run"))
            view = refreshed
            if (
                before_fingerprint != after_fingerprint
                or before_bucket != after_bucket
            ):
                st.rerun(scope="app")

    live_poll()


main()
