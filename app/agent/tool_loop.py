"""MOD-022/023：服务端权威只读工具注册表与严格 Agent 合同。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent import synthetic_evidence
from app.agent import tools as agent_tools
from app.agent.catalog import DEVICE_INFO
from app.agent.state import MAX_TOOL_ITERATIONS


PLANNER_TERMINATION_CONDITION = "ENOUGH_EVIDENCE_OR_NO_SAFE_ACTION"
CORE_TOOL_PROFILE = "core_v1"
SYNTHETIC_EVIDENCE_TOOL_PROFILE = "synthetic_evidence_v1"
TOOL_TIMEOUT_SECONDS = 5.0
TOOL_OUTPUT_MAX_BYTES = 64 * 1024
TOOL_RESULT_TEXT_MAX_CHARS = 8_000
TOOL_SUMMARY_TEXT_MAX_CHARS = 600
TOOL_QUERY_MAX_CHARS = 2_000
TOOL_RESULT_EVIDENCE_MAX_BYTES = 192 * 1024
VERIFIER_RESULT_MAX_BYTES = 8 * 1024
VERIFICATION_FINAL_STATUSES = frozenset({"PASSED", "FAILED", "ERROR"})
EFFECT_POLICY_STATUSES = frozenset(
    {"NOT_STARTED", "NOT_APPLICABLE", "PROPOSED", "DENIED", "ERROR"}
)
EFFECT_TYPE_RESTART_SIMULATED_ROBOT = "restart_simulated_robot"
EFFECT_SOURCE_SYNTHETIC_LOCAL = "SYNTHETIC_LOCAL"
VERIFIER_CHECK_NAMES = (
    "plan",
    "trajectory",
    "authoritative_arguments",
    "normalized_results",
    "evidence_binding",
    "stop_preconditions",
)
STOP_REASONS = frozenset({"ENOUGH_EVIDENCE", "NO_SAFE_ACTION", "PLAN_COMPLETE"})
TRAJECTORY_ERROR_CODES = frozenset(
    {
        "AGENT_DECISION_INVALID",
        "AGENT_DECISION_SCHEMA_INVALID",
        "AGENT_HANDOFF_INVALID",
        "AGENT_SAFETY_SIGNAL",
        "AGENT_SAFETY_STATE_INVALID",
        "AGENT_STOP_PRECONDITION_INVALID",
        "AGENT_STOP_REASON_INVALID",
        "AGENT_TOOL_LOOP_DISABLED",
        "MAX_TOOL_ITERATIONS",
        "PLANNER_ARGUMENT_SOURCE_INVALID",
        "PLANNER_DUPLICATE_STEP",
        "PLANNER_DUPLICATE_TOOL",
        "PLANNER_MAX_ITERATIONS_INVALID",
        "PLANNER_PRIVILEGE_INTENT",
        "PLANNER_SCHEMA_INVALID",
        "PLANNER_TERMINATION_INVALID",
        "PLANNER_TOOL_NOT_ALLOWED",
        "TOOL_ARGUMENT_INVALID",
        "TOOL_EXECUTION_ERROR",
        "TOOL_JSON_INVALID",
        "TOOL_OUTPUT_INVALID",
        "TOOL_PENDING_ACTION_INVALID",
        "TOOL_PROFILE_INVALID",
        "TOOL_PATH_CONTAINMENT_INVALID",
        "TOOL_DATASET_INVALID",
        "TOOL_SCOPE_INVALID",
        "TOOL_TIME_WINDOW_INVALID",
        "TOOL_PATH_INVALID",
        "TOOL_INPUT_TOO_LARGE",
        "TOOL_INPUT_ENCODING_INVALID",
        "TOOL_INPUT_FORMAT_INVALID",
        "TOOL_SOURCE_CHANGED",
        "TOOL_STATE_INVALID",
        "TOOL_STEP_NOT_PLANNED",
        "TOOL_STEP_REUSED",
        "TOOL_TIMEOUT",
        "TOOL_TRAJECTORY_INVALID",
        "TOOL_UNKNOWN",
    }
)
TOOL_LOOP_FAILURE_NOTE = (
    "自动诊断工具未能安全完成，当前结论需要人工核对，请停止自行操作并联系人工支持。"
)


class _CodedContractError(ValueError):
    """只公开稳定错误码，不携带模型、工具或底层异常内容。"""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class PlanContractError(_CodedContractError):
    """Planner 输出违反服务端合同。"""


class DecisionContractError(_CodedContractError):
    """下一动作输出违反服务端合同。"""


class ToolLoopError(_CodedContractError):
    """工具解析、执行或 observation 违反服务端合同。"""


@dataclass(frozen=True)
class ReadonlyToolSpec:
    """不可变的服务端工具权限与参数来源定义。"""

    name: str
    description: str
    argument_sources: Mapping[str, tuple[str, ...]]
    executor_name: str
    output_kind: str
    permission: str = "READ_ONLY"
    read_only: bool = True
    timeout_seconds: float = TOOL_TIMEOUT_SECONDS


def _sources(**items: tuple[str, ...]) -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType(dict(items))


READONLY_TOOL_REGISTRY: Mapping[str, ReadonlyToolSpec] = MappingProxyType(
    {
        "search_manual": ReadonlyToolSpec(
            name="search_manual",
            description="检索机器人维修手册中的只读证据",
            argument_sources=_sources(
                query=("device_model", "fault_category", "fault_description")
            ),
            executor_name="search_manual_strict",
            output_kind="documents",
        ),
        "search_fault_cases": ReadonlyToolSpec(
            name="search_fault_cases",
            description="检索历史故障案例中的只读证据",
            argument_sources=_sources(
                query=("device_model", "fault_category", "fault_description")
            ),
            executor_name="search_fault_cases_strict",
            output_kind="documents",
        ),
        "query_device_information": ReadonlyToolSpec(
            name="query_device_information",
            description="查询内置机器人型号与参数档案",
            argument_sources=_sources(device_model=("device_model",)),
            executor_name="query_device_information_strict",
            output_kind="device",
        ),
    }
)

SYNTHETIC_EVIDENCE_TOOL_REGISTRY: Mapping[str, ReadonlyToolSpec] = MappingProxyType(
    {
        "query_simulated_telemetry": ReadonlyToolSpec(
            name="query_simulated_telemetry",
            description="查询授权合成设备遥测中的只读证据",
            argument_sources=_sources(
                device_model=("device_model",),
                resource_id=("device_model", "server.synthetic_manifest"),
                start_ms=("device_model", "server.synthetic_manifest"),
                end_ms=("device_model", "server.synthetic_manifest"),
                max_points=("server.policy",),
                max_bytes=("server.policy",),
            ),
            executor_name="query_simulated_telemetry_strict",
            output_kind="telemetry",
        ),
        "analyze_synthetic_ros_log": ReadonlyToolSpec(
            name="analyze_synthetic_ros_log",
            description="分析授权合成 ROS 日志中的只读证据",
            argument_sources=_sources(
                device_model=("device_model",),
                resource_id=("device_model", "server.synthetic_manifest"),
                start_ms=("device_model", "server.synthetic_manifest"),
                end_ms=("device_model", "server.synthetic_manifest"),
                max_entries=("server.policy",),
                max_bytes=("server.policy",),
            ),
            executor_name="analyze_synthetic_ros_log_strict",
            output_kind="ros_log",
        ),
    }
)

_SYNTHETIC_EVIDENCE_REGISTRY: Mapping[str, ReadonlyToolSpec] = MappingProxyType(
    {**READONLY_TOOL_REGISTRY, **SYNTHETIC_EVIDENCE_TOOL_REGISTRY}
)


def registry_for_profile(profile: object) -> Mapping[str, ReadonlyToolSpec]:
    """按服务端私有能力档案返回不可变 allowlist。"""
    if profile == CORE_TOOL_PROFILE:
        return READONLY_TOOL_REGISTRY
    if profile == SYNTHETIC_EVIDENCE_TOOL_PROFILE:
        return _SYNTHETIC_EVIDENCE_REGISTRY
    raise ToolLoopError("TOOL_PROFILE_INVALID")


def _profile_from_state(state: Mapping[str, Any]) -> str:
    profile = state.get("tool_capability_profile", CORE_TOOL_PROFILE)
    registry_for_profile(profile)
    return profile

_PRIVILEGED_INTENT_MARKERS = (
    "create_support_ticket",
    "get_ticket_history",
    "check_solution_risk",
    "创建工单",
    "更新工单",
    "删除工单",
    "写入业务库",
    "修改配置",
    "重启设备",
    "停止设备",
    "关闭设备",
    "外部通知",
    "发送通知",
    "提权",
    "绕过",
    "跳过审批",
    "关闭风险检查",
    "发送邮件",
    "写数据库",
    "删除记录",
    "bypass",
    "disable safety",
    "restart device",
    "shutdown device",
    "notify external",
    "send email",
    "update configuration",
    "write database",
    "delete ticket",
)
_TRAJECTORY_KEYS = {
    "sequence",
    "iteration",
    "event",
    "step_id",
    "tool_name",
    "arguments",
    "result_summary",
    "result_sha256",
    "error_code",
    "stop_reason",
}
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_PATTERNS = (
    re.compile(
        r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?"
        r"(?:-----END(?: [A-Z0-9]+)? PRIVATE KEY-----|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?i)\b(?:proxy-)?authorization\s*:\s*"
        r"(?:basic|bearer|token)\s+[a-z0-9+/._~=-]{4,}"
    ),
    re.compile(
        r"(?i)\baws_(?:secret_access_key|access_key_id|session_token)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bgithub_pat_[a-z0-9_]{20,}"),
    re.compile(r"(?i)\bgh[pousr]_[a-z0-9]{20,}"),
    re.compile(r"(?i)\bxox[baprs]-[a-z0-9-]{20,}"),
    re.compile(r"(?i)\bglpat-[a-z0-9_-]{20,}"),
    re.compile(r"(?i)\bsk-[a-z0-9_-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|passwd|secret)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(
        r"(?i)\b(?:[a-z][a-z0-9_]*_)?(?:api_key|access_token|password|passwd|"
        r"secret|token)\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@"),
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class _PlanStep(_StrictModel):
    step_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    tool_name: str = Field(min_length=1, max_length=80)
    argument_sources: dict[str, list[str]]
    purpose: str = Field(min_length=1, max_length=300)


class _DiagnosticPlan(_StrictModel):
    objective: str = Field(min_length=1, max_length=500)
    steps: list[_PlanStep] = Field(min_length=1, max_length=3)
    termination_condition: str = Field(min_length=1, max_length=80)
    max_iterations: int


class _ToolDecision(_StrictModel):
    action: str = Field(min_length=1, max_length=20)
    step_id: str = Field(max_length=64)
    stop_reason: str = Field(max_length=80)


class _VerifierChecks(_StrictModel):
    plan: bool
    trajectory: bool
    authoritative_arguments: bool
    normalized_results: bool
    evidence_binding: bool
    stop_preconditions: bool


class _VerifierResult(_StrictModel):
    schema_version: int = Field(ge=1, le=1)
    status: str = Field(min_length=5, max_length=6)
    checks: _VerifierChecks
    reason_codes: list[str]
    subject_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    """生成无 NaN、稳定键序的 JSON；供摘要、轨迹和测试复现。"""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ToolLoopError("TOOL_JSON_INVALID") from exc


def canonical_sha256(value: Any) -> str:
    """对 strict JSON-safe 值生成稳定的小写 SHA-256。"""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _effect_proposal_from_validated_progress(
    state: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the only MOD-024 proposal from already-validated authority data."""
    if (
        state.get("agent_tool_loop_enabled") is not True
        or progress.get("tool_capability_profile")
        != SYNTHETIC_EVIDENCE_TOOL_PROFILE
        or progress.get("agent_route") != "done"
        or progress.get("verification_status") != "PASSED"
        or state.get("risk_level") not in {"低", "中"}
        or state.get("urgency") not in {"低", "中"}
        or any(
            state.get(field) is not False
            for field in (
                "need_human",
                "risk_check_failed",
                "llm_failed",
                "tool_loop_failed",
            )
        )
    ):
        raise ToolLoopError("EFFECT_POLICY_DENIED")
    device_model = state.get("device_model")
    if not isinstance(device_model, str) or not device_model or len(device_model) > 200:
        raise ToolLoopError("EFFECT_POLICY_DENIED")

    has_bound_bad_telemetry = False
    evidence = progress.get("tool_result_evidence")
    if not isinstance(evidence, dict):
        raise ToolLoopError("EFFECT_POLICY_DENIED")
    for item in evidence.values():
        if not isinstance(item, dict) or item.get("tool_name") != "query_simulated_telemetry":
            continue
        result = item.get("result")
        if (
            not isinstance(result, dict)
            or result.get("source") != EFFECT_SOURCE_SYNTHETIC_LOCAL
            or result.get("device_model") != device_model
            or not isinstance(result.get("samples"), list)
        ):
            continue
        if any(
            isinstance(sample, dict) and sample.get("quality") == "BAD"
            for sample in result["samples"]
        ):
            has_bound_bad_telemetry = True
            break
    if not has_bound_bad_telemetry:
        raise ToolLoopError("EFFECT_POLICY_DENIED")
    return {
        "schema_version": 1,
        "effect_type": EFFECT_TYPE_RESTART_SIMULATED_ROBOT,
        "source": EFFECT_SOURCE_SYNTHETIC_LOCAL,
        "arguments": {
            "device_model": device_model,
            "expected_boot_generation_delta": 1,
        },
    }


def validate_effect_proposal(
    proposal: object,
    *,
    state: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a proposal by re-deriving it; no model/client field is trusted."""
    if not isinstance(proposal, dict):
        raise ToolLoopError("EFFECT_PROPOSAL_INVALID")
    expected = _effect_proposal_from_validated_progress(state, progress)
    if canonical_json(proposal) != canonical_json(expected):
        raise ToolLoopError("EFFECT_PROPOSAL_INVALID")
    return expected


def build_effect_proposal(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic proposal from a final, verified, read-only checkpoint."""
    if (
        not isinstance(state, Mapping)
        or state.get("effect_policy_status", "NOT_STARTED") != "NOT_STARTED"
        or state.get("effect_proposal", {}) != {}
        or state.get("effect_proposal_sha256", "") != ""
    ):
        raise ToolLoopError("EFFECT_POLICY_DENIED")
    progress = validate_checkpoint_progress(state)
    return _effect_proposal_from_validated_progress(state, progress)


def validate_verifier_result(
    value: object,
    *,
    subject_sha256: str,
    evidence: object,
) -> dict[str, Any]:
    """严格验证 MOD-024 Verifier 结果及其完整对象/证据摘要绑定。"""
    try:
        parsed = _VerifierResult.model_validate(value)
    except ValidationError as exc:
        raise ToolLoopError("TOOL_STATE_INVALID") from exc
    result = parsed.model_dump(mode="json")
    status = result["status"]
    reasons = result["reason_codes"]
    checks = result["checks"]
    if (
        status not in VERIFICATION_FINAL_STATUSES
        or len(reasons) > 16
        or reasons != sorted(set(reasons))
        or any(
            not isinstance(code, str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", code) is None
            for code in reasons
        )
        or set(checks) != set(VERIFIER_CHECK_NAMES)
        or _HEX_64.fullmatch(subject_sha256) is None
        or result["subject_sha256"] != subject_sha256
        or result["evidence_sha256"] != canonical_sha256(evidence)
        or (status == "PASSED" and (not all(checks.values()) or reasons))
        or (status != "PASSED" and (all(checks.values()) or not reasons))
    ):
        raise ToolLoopError("TOOL_STATE_INVALID")
    if len(canonical_json(result).encode("utf-8")) > VERIFIER_RESULT_MAX_BYTES:
        raise ToolLoopError("TOOL_STATE_INVALID")
    return result


def build_verifier_result(
    *,
    status: str,
    checks: Mapping[str, bool],
    reason_codes: list[str],
    subject_sha256: str,
    evidence: object,
) -> dict[str, Any]:
    """构造无时钟、无随机量且可逐字复现的 Verifier 输出。"""
    value = {
        "schema_version": 1,
        "status": status,
        "checks": dict(checks),
        "reason_codes": sorted(set(reason_codes)),
        "subject_sha256": subject_sha256,
        "evidence_sha256": canonical_sha256(evidence),
    }
    return validate_verifier_result(
        value,
        subject_sha256=subject_sha256,
        evidence=evidence,
    )


def allowed_tool_catalog(*, profile: str = CORE_TOOL_PROFILE) -> list[dict[str, Any]]:
    """返回给 Planner 的最小公开目录，不暴露 Python callable。"""
    registry = registry_for_profile(profile)
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "permission": spec.permission,
            "argument_sources": {
                name: list(sources)
                for name, sources in sorted(spec.argument_sources.items())
            },
        }
        for spec in sorted(registry.values(), key=lambda item: item.name)
    ]


def validate_diagnostic_plan(
    payload: object, *, profile: str = CORE_TOOL_PROFILE
) -> dict[str, Any]:
    """严格验证 Planner 计划及服务端权限/来源，不执行任何工具。"""
    try:
        registry = registry_for_profile(profile)
    except ToolLoopError as exc:
        raise PlanContractError(exc.code) from exc
    try:
        parsed = _DiagnosticPlan.model_validate(payload)
    except ValidationError as exc:
        raise PlanContractError("PLANNER_SCHEMA_INVALID") from exc

    if parsed.termination_condition != PLANNER_TERMINATION_CONDITION:
        raise PlanContractError("PLANNER_TERMINATION_INVALID")
    if parsed.max_iterations != MAX_TOOL_ITERATIONS:
        raise PlanContractError("PLANNER_MAX_ITERATIONS_INVALID")

    objective = parsed.objective.casefold()
    if any(marker.casefold() in objective for marker in _PRIVILEGED_INTENT_MARKERS):
        raise PlanContractError("PLANNER_PRIVILEGE_INTENT")

    seen_steps: set[str] = set()
    seen_tools: set[str] = set()
    for step in parsed.steps:
        if step.step_id in seen_steps:
            raise PlanContractError("PLANNER_DUPLICATE_STEP")
        seen_steps.add(step.step_id)
        if step.tool_name in seen_tools:
            raise PlanContractError("PLANNER_DUPLICATE_TOOL")
        seen_tools.add(step.tool_name)
        spec = registry.get(step.tool_name)
        if spec is None:
            raise PlanContractError("PLANNER_TOOL_NOT_ALLOWED")
        expected_sources = {
            name: list(sources) for name, sources in spec.argument_sources.items()
        }
        if step.argument_sources != expected_sources:
            raise PlanContractError("PLANNER_ARGUMENT_SOURCE_INVALID")
        purpose = step.purpose.casefold()
        if (
            step.purpose != spec.description
            or any(
                marker.casefold() in purpose
                for marker in _PRIVILEGED_INTENT_MARKERS
            )
        ):
            raise PlanContractError("PLANNER_PRIVILEGE_INTENT")

    return parsed.model_dump(mode="json")


def validate_tool_decision(payload: object) -> dict[str, str]:
    """验证模型只能选择计划 step 或停止，不能直接提交参数。"""
    try:
        parsed = _ToolDecision.model_validate(payload)
    except ValidationError as exc:
        raise DecisionContractError("AGENT_DECISION_SCHEMA_INVALID") from exc
    decision = parsed.model_dump(mode="json")
    if parsed.action == "CALL_TOOL":
        if not parsed.step_id or parsed.stop_reason:
            raise DecisionContractError("AGENT_DECISION_INVALID")
        return decision
    if parsed.action == "STOP":
        if parsed.step_id:
            raise DecisionContractError("AGENT_DECISION_INVALID")
        if parsed.stop_reason not in STOP_REASONS:
            raise DecisionContractError("AGENT_STOP_REASON_INVALID")
        return decision
    raise DecisionContractError("AGENT_DECISION_INVALID")


def _normalized_text(value: object, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ToolLoopError("TOOL_ARGUMENT_INVALID")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > maximum:
        raise ToolLoopError("TOOL_ARGUMENT_INVALID")
    return normalized


def materialize_tool_call(
    plan: object,
    decision: object,
    state: Mapping[str, Any],
    *,
    completed_steps: list[str],
    resource_metadata: object | None = None,
) -> dict[str, Any]:
    """从已验证状态派生参数；任何模型参数字段已在 decision schema 被拒绝。"""
    resource_metadata = _validated_resource_metadata(resource_metadata)
    profile = _profile_from_state(state)
    registry = registry_for_profile(profile)
    validated_plan = validate_diagnostic_plan(plan, profile=profile)
    validated_decision = validate_tool_decision(decision)
    if validated_decision["action"] != "CALL_TOOL":
        raise DecisionContractError("AGENT_DECISION_INVALID")
    if (
        not isinstance(completed_steps, list)
        or any(not isinstance(item, str) for item in completed_steps)
        or len(set(completed_steps)) != len(completed_steps)
    ):
        raise ToolLoopError("TOOL_STATE_INVALID")
    step_id = validated_decision["step_id"]
    if step_id in completed_steps:
        raise ToolLoopError("TOOL_STEP_REUSED")
    step = next(
        (item for item in validated_plan["steps"] if item["step_id"] == step_id),
        None,
    )
    if step is None:
        raise ToolLoopError("TOOL_STEP_NOT_PLANNED")
    spec = registry.get(step["tool_name"])
    if spec is None:
        raise ToolLoopError("TOOL_UNKNOWN")

    if spec.output_kind == "documents":
        query_parts = []
        for source in spec.argument_sources["query"]:
            value = state.get(source, "")
            if not isinstance(value, str):
                raise ToolLoopError("TOOL_ARGUMENT_INVALID")
            normalized = " ".join(value.split())
            if normalized:
                query_parts.append(normalized)
        query = _normalized_text(" ".join(query_parts), maximum=TOOL_QUERY_MAX_CHARS)
        arguments = {"query": query, "top_k": 5}
    elif spec.output_kind == "device":
        model = _normalized_text(state.get("device_model", ""), maximum=200)
        arguments = {"device_model": model}
    elif spec.output_kind in {"telemetry", "ros_log"}:
        model = _normalized_text(state.get("device_model", ""), maximum=200)
        try:
            arguments = synthetic_evidence.authoritative_tool_arguments(
                spec.name,
                model,
                resource_metadata=resource_metadata,
            )
        except synthetic_evidence.EvidenceToolError as exc:
            raise ToolLoopError(exc.code) from exc
    else:
        raise ToolLoopError("TOOL_STATE_INVALID")
    return {
        "step_id": step_id,
        "tool_name": spec.name,
        "arguments": arguments,
    }


def _validated_resource_metadata(value: object | None) -> list[dict[str, Any]] | None:
    """Preserve the legacy default, but never fall back for explicit metadata."""
    if value is None:
        return None
    try:
        return synthetic_evidence.validate_resource_metadata(value)
    except synthetic_evidence.EvidenceToolError as exc:
        raise ToolLoopError(exc.code) from exc


def _validate_arguments(
    spec: ReadonlyToolSpec,
    arguments: object,
    *,
    resource_metadata: object | None = None,
) -> dict[str, Any]:
    resource_metadata = _validated_resource_metadata(resource_metadata)
    if not isinstance(arguments, dict):
        raise ToolLoopError("TOOL_ARGUMENT_INVALID")
    if spec.output_kind == "documents":
        if set(arguments) != {"query", "top_k"}:
            raise ToolLoopError("TOOL_ARGUMENT_INVALID")
        top_k = arguments.get("top_k")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k != 5:
            raise ToolLoopError("TOOL_ARGUMENT_INVALID")
        return {
            "query": _normalized_text(
                arguments.get("query"), maximum=TOOL_QUERY_MAX_CHARS
            ),
            "top_k": 5,
        }
    if spec.output_kind == "device":
        if set(arguments) != {"device_model"}:
            raise ToolLoopError("TOOL_ARGUMENT_INVALID")
        return {
            "device_model": _normalized_text(
                arguments.get("device_model"), maximum=200
            )
        }
    if spec.output_kind in {"telemetry", "ros_log"}:
        try:
            return synthetic_evidence.validate_tool_arguments(
                spec.name, arguments, resource_metadata=resource_metadata
            )
        except synthetic_evidence.EvidenceToolError as exc:
            raise ToolLoopError(exc.code) from exc
    raise ToolLoopError("TOOL_ARGUMENT_INVALID")


def _run_with_timeout(
    function: Callable[..., Any],
    arguments: Mapping[str, Any],
    timeout_seconds: float,
) -> Any:
    """只读同步工具的有界等待；超时 worker 不复用，不能冒充进程级强杀。"""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robotops-readonly-tool")
    future = executor.submit(function, **dict(arguments))
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _normalize_documents(
    output: object, *, max_items: int = 5
) -> list[dict[str, Any]]:
    if not isinstance(output, list) or len(output) > max_items:
        raise ToolLoopError("TOOL_OUTPUT_INVALID")
    result: list[dict[str, Any]] = []
    expected = {"id", "filename", "chunk_index", "text", "similarity"}
    for item in output:
        if not isinstance(item, dict) or set(item) != expected:
            raise ToolLoopError("TOOL_OUTPUT_INVALID")
        identifier = item.get("id")
        filename = item.get("filename")
        text = item.get("text")
        chunk_index = item.get("chunk_index")
        similarity = item.get("similarity")
        if (
            not isinstance(identifier, str)
            or not identifier
            or len(identifier) > 200
            or not isinstance(filename, str)
            or not filename
            or len(filename) > 500
            or not isinstance(text, str)
            or not text
            or len(text) > TOOL_RESULT_TEXT_MAX_CHARS
            or not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or chunk_index < 0
            or not isinstance(similarity, (int, float))
            or isinstance(similarity, bool)
            or not math.isfinite(float(similarity))
            or not 0 <= float(similarity) <= 1
        ):
            raise ToolLoopError("TOOL_OUTPUT_INVALID")
        result.append(
            {
                "id": identifier,
                "filename": filename,
                "chunk_index": chunk_index,
                "text": text,
                "similarity": float(similarity),
            }
        )
    return result


def _bounded_string(value: object, *, maximum: int = 1_000, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not allow_empty and not value):
        raise ToolLoopError("TOOL_OUTPUT_INVALID")
    return value


def _normalize_device(output: object) -> dict[str, Any]:
    if not isinstance(output, dict) or not isinstance(output.get("found"), bool):
        raise ToolLoopError("TOOL_OUTPUT_INVALID")
    if output["found"]:
        if set(output) != {"found", "model", "name", "params"}:
            raise ToolLoopError("TOOL_OUTPUT_INVALID")
        return {
            "found": True,
            "model": _bounded_string(output.get("model"), maximum=200),
            "name": _bounded_string(output.get("name"), maximum=500),
            "params": _bounded_string(output.get("params"), maximum=2_000),
        }
    if set(output) != {"found", "model", "message"}:
        raise ToolLoopError("TOOL_OUTPUT_INVALID")
    return {
        "found": False,
        "model": _bounded_string(output.get("model"), maximum=200),
        "message": _bounded_string(output.get("message"), maximum=1_000),
    }


def _normalize_result(
    spec: ReadonlyToolSpec,
    output: object,
    *,
    arguments: Mapping[str, Any],
    resource_metadata: object | None = None,
) -> object:
    resource_metadata = _validated_resource_metadata(resource_metadata)
    if spec.output_kind == "documents":
        return _normalize_documents(output)
    if spec.output_kind == "device":
        result = _normalize_device(output)
        if result["model"] != arguments["device_model"]:
            raise ToolLoopError("TOOL_OUTPUT_INVALID")
        return result
    if spec.output_kind in {"telemetry", "ros_log"}:
        try:
            return synthetic_evidence.normalize_tool_result(
                spec.name,
                output,
                arguments,
                resource_metadata=resource_metadata,
            )
        except synthetic_evidence.EvidenceToolError as exc:
            raise ToolLoopError(exc.code) from exc
    raise ToolLoopError("TOOL_OUTPUT_INVALID")


def _result_summary(kind: str, result: object) -> dict[str, Any]:
    if kind == "documents":
        documents = result
        return {
            "count": len(documents),
            "items": [
                {
                    "id": item["id"],
                    "filename": item["filename"],
                    "chunk_index": item["chunk_index"],
                    "similarity": item["similarity"],
                    "text_excerpt": _redact_summary_text(
                        item["text"][:TOOL_SUMMARY_TEXT_MAX_CHARS]
                    ),
                }
                for item in documents
            ],
        }
    if kind == "device":
        return _redact_trajectory_value(dict(result))
    if kind == "telemetry":
        metrics: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for sample in result["samples"]:
            metrics.setdefault((sample["metric"], sample["unit"]), []).append(sample)
        aggregates = []
        for (metric, unit), samples in sorted(metrics.items()):
            values = [float(item["value"]) for item in samples]
            latest = max(samples, key=lambda item: (item["ts_ms"], item["metric"]))
            aggregates.append(
                {
                    "metric": metric,
                    "unit": unit,
                    "count": len(samples),
                    "min": min(values),
                    "max": max(values),
                    "latest": float(latest["value"]),
                    "bad_count": sum(
                        1 for item in samples if item["quality"] != "GOOD"
                    ),
                }
            )
        return {
            "schema_version": result["schema_version"],
            "source": result["source"],
            "trust_boundary": result["trust_boundary"],
            "device_model": result["device_model"],
            "resource_id": result["resource_id"],
            "start_ms": result["start_ms"],
            "end_ms": result["end_ms"],
            "bytes_read": result["bytes_read"],
            "resource_sha256": result["resource_sha256"],
            "count": result["count"],
            "truncated": result["truncated"],
            "metrics": aggregates,
        }
    if kind == "ros_log":
        level_counts = {
            level: sum(1 for item in result["entries"] if item["level"] == level)
            for level in ("DEBUG", "INFO", "WARN", "ERROR", "FATAL")
        }
        excerpts = [
            {
                "ts_ms": item["ts_ms"],
                "level": item["level"],
                "node": item["node"],
                "message_excerpt": _redact_summary_text(
                    item["message"][:TOOL_SUMMARY_TEXT_MAX_CHARS]
                ),
            }
            for item in result["entries"][:5]
        ]
        return {
            "schema_version": result["schema_version"],
            "source": result["source"],
            "trust_boundary": result["trust_boundary"],
            "device_model": result["device_model"],
            "resource_id": result["resource_id"],
            "start_ms": result["start_ms"],
            "end_ms": result["end_ms"],
            "bytes_read": result["bytes_read"],
            "resource_sha256": result["resource_sha256"],
            "count": result["count"],
            "truncated": result["truncated"],
            "level_counts": level_counts,
            "entries": excerpts,
        }
    raise ToolLoopError("TOOL_OUTPUT_INVALID")


def _validate_evidence_summary(kind: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
    common = {
        "schema_version",
        "source",
        "trust_boundary",
        "device_model",
        "resource_id",
        "start_ms",
        "end_ms",
        "bytes_read",
        "resource_sha256",
        "count",
        "truncated",
    }
    expected = common | ({"metrics"} if kind == "telemetry" else {"level_counts", "entries"})
    if set(value) != expected:
        raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
    if (
        value.get("schema_version") != 1
        or value.get("source") != "SYNTHETIC_LOCAL"
        or value.get("trust_boundary") != "UNTRUSTED_DATA"
        or not isinstance(value.get("device_model"), str)
        or not value["device_model"]
        or not isinstance(value.get("resource_id"), str)
        or not value["resource_id"]
        or any(
            type(value.get(key)) is not int
            for key in ("start_ms", "end_ms", "bytes_read", "count")
        )
        or not 0 <= value["start_ms"] < value["end_ms"]
        or not 0 <= value["bytes_read"] <= synthetic_evidence.MAX_EVIDENCE_BYTES
        or value["count"] < 0
        or type(value.get("truncated")) is not bool
        or not isinstance(value.get("resource_sha256"), str)
        or _HEX_64.fullmatch(value["resource_sha256"]) is None
    ):
        raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
    if kind == "telemetry":
        metrics = value.get("metrics")
        if not isinstance(metrics, list) or len(metrics) > value["count"]:
            raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
        expected_metric_keys = {
            "metric",
            "unit",
            "count",
            "min",
            "max",
            "latest",
            "bad_count",
        }
        for item in metrics:
            if (
                not isinstance(item, dict)
                or set(item) != expected_metric_keys
                or not isinstance(item.get("metric"), str)
                or not isinstance(item.get("unit"), str)
                or type(item.get("count")) is not int
                or type(item.get("bad_count")) is not int
                or not 0 <= item["bad_count"] <= item["count"]
                or any(
                    isinstance(item.get(key), bool)
                    or not isinstance(item.get(key), (int, float))
                    or not math.isfinite(float(item[key]))
                    for key in ("min", "max", "latest")
                )
            ):
                raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
    else:
        counts = value.get("level_counts")
        entries = value.get("entries")
        if (
            not isinstance(counts, dict)
            or set(counts) != {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}
            or any(type(item) is not int or item < 0 for item in counts.values())
            or sum(counts.values()) != value["count"]
            or not isinstance(entries, list)
            or len(entries) > min(5, value["count"])
        ):
            raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
        for item in entries:
            if (
                not isinstance(item, dict)
                or set(item) != {"ts_ms", "level", "node", "message_excerpt"}
                or type(item.get("ts_ms")) is not int
                or item.get("level") not in counts
                or not isinstance(item.get("node"), str)
                or not isinstance(item.get("message_excerpt"), str)
                or len(item["message_excerpt"]) > TOOL_SUMMARY_TEXT_MAX_CHARS
            ):
                raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
    canonical_json(value)
    return dict(value)


def _redact_summary_text(value: str) -> str:
    """仅对进入 trajectory/下一轮模型的摘要做高置信凭证脱敏。"""
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_CREDENTIAL]", redacted)
    return redacted


def _redact_trajectory_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_summary_text(value)
    if isinstance(value, list):
        return [_redact_trajectory_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _redact_trajectory_value(item)
            for key, item in value.items()
        }
    return value


def _trajectory_safe_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """生成可持久化参数投影；校验端必须以同一投影绑定权威原始参数。"""
    redacted = _redact_trajectory_value(dict(arguments))
    if not isinstance(redacted, dict):
        raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
    return redacted


def validate_stop_preconditions(
    plan: object,
    stop_reason: str,
    completed_steps: object,
    trajectory: object,
    *,
    profile: str = CORE_TOOL_PROFILE,
    resource_metadata: object | None = None,
) -> None:
    """模型停止必须与已执行 step 和正向 observation 一致。"""
    validated_plan = validate_diagnostic_plan(plan, profile=profile)
    events = validate_trajectory(
        trajectory, profile=profile, resource_metadata=resource_metadata
    )
    if (
        not isinstance(completed_steps, list)
        or any(not isinstance(item, str) for item in completed_steps)
        or len(set(completed_steps)) != len(completed_steps)
    ):
        raise ToolLoopError("TOOL_STATE_INVALID")
    planned_steps = {item["step_id"] for item in validated_plan["steps"]}
    if not set(completed_steps).issubset(planned_steps):
        raise ToolLoopError("TOOL_STATE_INVALID")
    positive = False
    for item in events:
        if item["event"] != "TOOL":
            continue
        count = item["result_summary"].get("count")
        if (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
        ) or item["result_summary"].get("found") is True:
            positive = True
            break
    if stop_reason == "ENOUGH_EVIDENCE" and not positive:
        raise ToolLoopError("AGENT_STOP_PRECONDITION_INVALID")
    if stop_reason == "PLAN_COMPLETE" and (
        set(completed_steps) != planned_steps or not positive
    ):
        raise ToolLoopError("AGENT_STOP_PRECONDITION_INVALID")


def execute_readonly_tool(
    tool_name: object,
    arguments: object,
    *,
    profile: str = CORE_TOOL_PROFILE,
) -> dict[str, Any]:
    """执行单个 allowlisted 只读工具并返回有界、可摘要的 observation。"""
    if not isinstance(tool_name, str):
        raise ToolLoopError("TOOL_UNKNOWN")
    spec = registry_for_profile(profile).get(tool_name)
    if spec is None:
        raise ToolLoopError("TOOL_UNKNOWN")
    normalized_arguments = _validate_arguments(spec, arguments)
    function = getattr(agent_tools, spec.executor_name, None)
    if not callable(function):
        raise ToolLoopError("TOOL_EXECUTION_ERROR")
    try:
        output = _run_with_timeout(function, normalized_arguments, spec.timeout_seconds)
    except TimeoutError as exc:
        raise ToolLoopError("TOOL_TIMEOUT") from exc
    except synthetic_evidence.EvidenceToolError as exc:
        raise ToolLoopError(exc.code) from exc
    except ToolLoopError:
        raise
    except Exception as exc:
        raise ToolLoopError("TOOL_EXECUTION_ERROR") from exc

    result = _normalize_result(spec, output, arguments=normalized_arguments)
    serialized = canonical_json(result)
    if len(serialized.encode("utf-8")) > TOOL_OUTPUT_MAX_BYTES:
        raise ToolLoopError("TOOL_OUTPUT_INVALID")
    return {
        "tool_name": spec.name,
        "arguments": normalized_arguments,
        "result": result,
        "result_summary": _result_summary(spec.output_kind, result),
        "result_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def validate_tool_observation(
    observation: object,
    *,
    expected_tool_name: str,
    expected_arguments: Mapping[str, Any],
    profile: str = CORE_TOOL_PROFILE,
    resource_metadata: object | None = None,
) -> dict[str, Any]:
    """防止执行适配器返回错配/伪造 observation。"""
    expected_keys = {
        "tool_name",
        "arguments",
        "result",
        "result_summary",
        "result_sha256",
    }
    if not isinstance(observation, dict) or set(observation) != expected_keys:
        raise ToolLoopError("TOOL_OUTPUT_INVALID")
    spec = registry_for_profile(profile).get(expected_tool_name)
    if spec is None or observation.get("tool_name") != expected_tool_name:
        raise ToolLoopError("TOOL_OUTPUT_INVALID")
    normalized_arguments = _validate_arguments(
        spec, expected_arguments, resource_metadata=resource_metadata
    )
    if observation.get("arguments") != normalized_arguments:
        raise ToolLoopError("TOOL_OUTPUT_INVALID")
    result = _normalize_result(
        spec,
        observation.get("result"),
        arguments=normalized_arguments,
        resource_metadata=resource_metadata,
    )
    serialized = canonical_json(result)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    summary = _result_summary(spec.output_kind, result)
    if (
        observation.get("result_summary") != summary
        or observation.get("result_sha256") != digest
    ):
        raise ToolLoopError("TOOL_OUTPUT_INVALID")
    return {
        "tool_name": expected_tool_name,
        "arguments": normalized_arguments,
        "result": result,
        "result_summary": summary,
        "result_sha256": digest,
    }


def append_trajectory_event(
    trajectory: object,
    *,
    profile: str = CORE_TOOL_PROFILE,
    iteration: int,
    event: str,
    step_id: str = "",
    tool_name: str = "",
    arguments: Mapping[str, Any] | None = None,
    result_summary: Mapping[str, Any] | None = None,
    result_sha256: str = "",
    error_code: str = "",
    stop_reason: str = "",
    resource_metadata: object | None = None,
) -> list[dict[str, Any]]:
    """追加无随机量/时间戳的确定性轨迹事件。"""
    existing = validate_trajectory(
        trajectory, profile=profile, resource_metadata=resource_metadata
    )
    if (
        not isinstance(iteration, int)
        or isinstance(iteration, bool)
        or not 0 <= iteration <= MAX_TOOL_ITERATIONS
        or event not in {"TOOL", "STOP", "ERROR"}
        or not all(
            isinstance(value, str)
            for value in (step_id, tool_name, result_sha256, error_code, stop_reason)
        )
    ):
        raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
    item = {
        "sequence": len(existing) + 1,
        "iteration": iteration,
        "event": event,
        "step_id": step_id,
        "tool_name": tool_name,
        "arguments": _trajectory_safe_arguments(arguments or {}),
        "result_summary": _redact_trajectory_value(dict(result_summary or {})),
        "result_sha256": result_sha256,
        "error_code": error_code,
        "stop_reason": stop_reason,
    }
    canonical_json(item)
    return validate_trajectory(
        [*existing, item], profile=profile, resource_metadata=resource_metadata
    )


def validate_trajectory(
    value: object,
    *,
    profile: str = CORE_TOOL_PROFILE,
    resource_metadata: object | None = None,
) -> list[dict[str, Any]]:
    """验证 checkpoint 中的 trajectory；拒绝部分/篡改事件。"""
    resource_metadata = _validated_resource_metadata(resource_metadata)
    registry = registry_for_profile(profile)
    if not isinstance(value, list) or len(value) > 12:
        raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
    validated: list[dict[str, Any]] = []
    terminal_seen = False
    previous_iteration = 0
    for expected_sequence, item in enumerate(value, 1):
        if terminal_seen:
            raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
        if not isinstance(item, dict) or set(item) != _TRAJECTORY_KEYS:
            raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
        if (
            item.get("sequence") != expected_sequence
            or (resource_metadata is not None and type(item.get("sequence")) is not int)
            or not isinstance(item.get("iteration"), int)
            or isinstance(item.get("iteration"), bool)
            or not 0 <= item["iteration"] <= MAX_TOOL_ITERATIONS
            or item["iteration"] < previous_iteration
            or item.get("event") not in {"TOOL", "STOP", "ERROR"}
            or not isinstance(item.get("arguments"), dict)
            or not isinstance(item.get("result_summary"), dict)
            or not all(
                isinstance(item.get(key), str)
                for key in (
                    "step_id",
                    "tool_name",
                    "result_sha256",
                    "error_code",
                    "stop_reason",
                )
            )
            or (
                item.get("result_sha256")
                and _HEX_64.fullmatch(item["result_sha256"]) is None
            )
        ):
            raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
        event = item["event"]
        if event == "TOOL" and (
            not item["step_id"]
            or not item["tool_name"]
            or not item["arguments"]
            or not item["result_summary"]
            or not item["result_sha256"]
            or item["error_code"]
            or item["stop_reason"]
        ):
            raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
        if event == "TOOL":
            spec = registry.get(item["tool_name"])
            if spec is None:
                raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
            if _validate_arguments(
                spec, item["arguments"], resource_metadata=resource_metadata
            ) != item["arguments"]:
                raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
            summary = item["result_summary"]
            if spec.output_kind == "documents":
                count = summary.get("count")
                items = summary.get("items")
                if (
                    set(summary) != {"count", "items"}
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or not 0 <= count <= 5
                    or not isinstance(items, list)
                    or len(items) != count
                ):
                    raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
                expected_item_keys = {
                    "id",
                    "filename",
                    "chunk_index",
                    "similarity",
                    "text_excerpt",
                }
                for summary_item in items:
                    similarity = summary_item.get("similarity") if isinstance(summary_item, dict) else None
                    if (
                        not isinstance(summary_item, dict)
                        or set(summary_item) != expected_item_keys
                        or not isinstance(summary_item.get("id"), str)
                        or not summary_item["id"]
                        or not isinstance(summary_item.get("filename"), str)
                        or not summary_item["filename"]
                        or not isinstance(summary_item.get("chunk_index"), int)
                        or isinstance(summary_item.get("chunk_index"), bool)
                        or summary_item["chunk_index"] < 0
                        or not isinstance(similarity, (int, float))
                        or isinstance(similarity, bool)
                        or not math.isfinite(float(similarity))
                        or not 0 <= float(similarity) <= 1
                        or not isinstance(summary_item.get("text_excerpt"), str)
                        or len(summary_item["text_excerpt"]) > TOOL_SUMMARY_TEXT_MAX_CHARS
                    ):
                        raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
            elif spec.output_kind == "device":
                try:
                    if _normalize_device(summary) != summary:
                        raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
                except ToolLoopError as exc:
                    raise ToolLoopError("TOOL_TRAJECTORY_INVALID") from exc
            elif spec.output_kind in {"telemetry", "ros_log"}:
                _validate_evidence_summary(spec.output_kind, summary)
            else:
                raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
        if event == "STOP" and (
            item["step_id"]
            or item["tool_name"]
            or item["arguments"]
            or item["result_summary"]
            or item["result_sha256"]
            or item["error_code"]
            or item["stop_reason"] not in STOP_REASONS
        ):
            raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
        if event == "ERROR" and (
            item["result_summary"]
            or item["result_sha256"]
            or not item["error_code"]
            or item["error_code"] not in TRAJECTORY_ERROR_CODES
            or item["stop_reason"] != item["error_code"]
        ):
            raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
        if event == "ERROR":
            if item["tool_name"]:
                spec = registry.get(item["tool_name"])
                if (
                    spec is None
                    or not item["step_id"]
                    or not item["arguments"]
                ):
                    raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
                try:
                    if _validate_arguments(
                        spec, item["arguments"], resource_metadata=resource_metadata
                    ) != item["arguments"]:
                        raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
                except ToolLoopError as exc:
                    raise ToolLoopError("TOOL_TRAJECTORY_INVALID") from exc
            elif item["arguments"]:
                raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
            if not item["step_id"] and (item["tool_name"] or item["arguments"]):
                raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
        if event in {"STOP", "ERROR"}:
            terminal_seen = True
        previous_iteration = item["iteration"]
        try:
            if _redact_trajectory_value(item) != item:
                raise ToolLoopError("TOOL_TRAJECTORY_INVALID")
            canonical_json(item)
        except ToolLoopError as exc:
            raise ToolLoopError("TOOL_TRAJECTORY_INVALID") from exc
        validated.append(dict(item))
    return validated


def validate_agent_progress(
    plan: object,
    completed_steps: object,
    trajectory: object,
    *,
    tool_iterations: object | None = None,
    profile: str = CORE_TOOL_PROFILE,
    resource_metadata: object | None = None,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """交叉验证计划、成功事件与 completed step，防 checkpoint 轨迹伪造。"""
    validated_plan = validate_diagnostic_plan(plan, profile=profile)
    events = validate_trajectory(
        trajectory, profile=profile, resource_metadata=resource_metadata
    )
    if (
        not isinstance(completed_steps, list)
        or any(not isinstance(item, str) for item in completed_steps)
        or len(set(completed_steps)) != len(completed_steps)
    ):
        raise ToolLoopError("TOOL_STATE_INVALID")
    steps_by_id = {item["step_id"]: item for item in validated_plan["steps"]}
    tool_events = [item for item in events if item["event"] == "TOOL"]
    event_steps = [item["step_id"] for item in tool_events]
    if event_steps != completed_steps:
        raise ToolLoopError("TOOL_STATE_INVALID")
    if tool_iterations is not None:
        expected_iterations = max(
            (item["iteration"] for item in events),
            default=0,
        )
        if (
            not isinstance(tool_iterations, int)
            or isinstance(tool_iterations, bool)
            or not 0 <= tool_iterations <= MAX_TOOL_ITERATIONS
            or tool_iterations != expected_iterations
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        if (
            events
            and events[-1]["event"] in {"STOP", "ERROR"}
            and events[-1]["iteration"] != tool_iterations
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        if (
            events
            and events[-1]["event"] == "STOP"
            and events[-1]["iteration"] != len(tool_events)
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
    for expected_iteration, item in enumerate(tool_events, 1):
        step = steps_by_id.get(item["step_id"])
        if (
            step is None
            or step["tool_name"] != item["tool_name"]
            or item["iteration"] != expected_iteration
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
    return validated_plan, list(completed_steps), events


def _validate_result_binding(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    authoritative_arguments: Mapping[str, Any],
    resource_metadata: object | None = None,
) -> object:
    """把 trajectory 摘要/摘要哈希绑定到持久化的规范 observation。"""
    profile = _profile_from_state(state)
    spec = registry_for_profile(profile)[event["tool_name"]]
    try:
        normalized_arguments = _validate_arguments(
            spec, authoritative_arguments, resource_metadata=resource_metadata
        )
    except ToolLoopError as exc:
        raise ToolLoopError("TOOL_STATE_INVALID") from exc
    if spec.output_kind == "documents":
        persisted = _normalize_documents(
            state.get("retrieved_documents"),
            max_items=10,
        )
        by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for item in persisted:
            key = (item["id"], item["filename"], item["chunk_index"])
            if key in by_key:
                raise ToolLoopError("TOOL_STATE_INVALID")
            by_key[key] = item
        result: list[dict[str, Any]] = []
        for summary_item in event["result_summary"]["items"]:
            key = (
                summary_item["id"],
                summary_item["filename"],
                summary_item["chunk_index"],
            )
            item = by_key.get(key)
            if item is None:
                raise ToolLoopError("TOOL_STATE_INVALID")
            result.append(item)
    elif spec.output_kind == "device":
        model = normalized_arguments["device_model"]
        info = DEVICE_INFO.get(model)
        expected_device = (
            {"found": True, "model": model, **info}
            if info is not None
            else {
                "found": False,
                "model": model,
                "message": f"设备档案中未找到「{model}」的型号信息",
            }
        )
        result = _normalize_device(expected_device)
    elif spec.output_kind in {"telemetry", "ros_log"}:
        evidence = state.get("tool_result_evidence", {})
        item = evidence.get(event["step_id"]) if isinstance(evidence, dict) else None
        expected_item_keys = {
            "sequence",
            "tool_name",
            "arguments",
            "result",
            "result_sha256",
        }
        if (
            not isinstance(item, dict)
            or set(item) != expected_item_keys
            or item.get("sequence") != event["sequence"]
            or (resource_metadata is not None and type(item.get("sequence")) is not int)
            or item.get("tool_name") != event["tool_name"]
            or item.get("arguments")
            != _trajectory_safe_arguments(normalized_arguments)
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        try:
            result = _normalize_result(
                spec,
                item.get("result"),
                arguments=normalized_arguments,
                resource_metadata=resource_metadata,
            )
        except ToolLoopError as exc:
            raise ToolLoopError("TOOL_STATE_INVALID") from exc
        evidence_digest = hashlib.sha256(
            canonical_json(result).encode("utf-8")
        ).hexdigest()
        if item.get("result_sha256") != evidence_digest:
            raise ToolLoopError("TOOL_STATE_INVALID")
        if (
            resource_metadata is not None
            and canonical_json(item["result"]) != canonical_json(result)
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
    else:
        raise ToolLoopError("TOOL_STATE_INVALID")
    expected_summary = _result_summary(spec.output_kind, result)
    expected_digest = hashlib.sha256(
        canonical_json(result).encode("utf-8")
    ).hexdigest()
    if (
        event["result_summary"] != expected_summary
        or event["result_sha256"] != expected_digest
    ):
        raise ToolLoopError("TOOL_STATE_INVALID")
    if (
        resource_metadata is not None
        and canonical_json(event["result_summary"]) != canonical_json(expected_summary)
    ):
        raise ToolLoopError("TOOL_STATE_INVALID")
    return result


def _normalized_verifier_source(value: object, *, maximum: int) -> str:
    """规范化允许为空的服务端事实，避免无意义空白改变 subject 摘要。"""
    if not isinstance(value, str):
        raise ToolLoopError("TOOL_STATE_INVALID")
    normalized = " ".join(value.split())
    if len(normalized) > maximum:
        raise ToolLoopError("TOOL_STATE_INVALID")
    return normalized


def verifier_subject_sha256(
    state: Mapping[str, Any],
    *,
    progress: Mapping[str, Any],
) -> str:
    """绑定 Verifier 核验对象；不包含会在 risk/policy 节点变化的字段。"""
    if not isinstance(state, Mapping) or not isinstance(progress, Mapping):
        raise ToolLoopError("TOOL_STATE_INVALID")
    profile = progress.get("tool_capability_profile")
    registry_for_profile(profile)
    plan = progress.get("diagnostic_plan")
    pending = progress.get("pending_tool_action")
    completed = progress.get("completed_plan_steps")
    trajectory = progress.get("tool_trajectory")
    route = progress.get("agent_route")
    failed = progress.get("tool_loop_failed")
    evidence = progress.get("tool_result_evidence")
    iterations = state.get("tool_iterations")
    stop_reason = state.get("execution_stop_reason")
    if (
        not isinstance(plan, dict)
        or not isinstance(pending, dict)
        or not isinstance(completed, list)
        or any(not isinstance(item, str) for item in completed)
        or not isinstance(trajectory, list)
        or route not in {"", "decide", "execute", "handoff", "done"}
        or not isinstance(failed, bool)
        or not isinstance(evidence, dict)
        or not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or not 0 <= iterations <= MAX_TOOL_ITERATIONS
        or not isinstance(stop_reason, str)
        or len(stop_reason) > 80
    ):
        raise ToolLoopError("TOOL_STATE_INVALID")
    try:
        documents = _normalize_documents(
            state.get("retrieved_documents"),
            max_items=10,
        )
    except ToolLoopError as exc:
        raise ToolLoopError("TOOL_STATE_INVALID") from exc
    subject = {
        "schema_version": "robotops_verifier_subject_v1",
        "tool_capability_profile": profile,
        "diagnostic_plan": plan,
        "pending_tool_action": pending,
        "completed_plan_steps": completed,
        "tool_trajectory": trajectory,
        "tool_iterations": iterations,
        "agent_route": route,
        "execution_stop_reason": stop_reason,
        "tool_loop_failed": failed,
        "retrieved_documents": documents,
        "tool_result_evidence": evidence,
        "authoritative_sources": {
            "device_model": _normalized_verifier_source(
                state.get("device_model", ""), maximum=200
            ),
            "fault_category": _normalized_verifier_source(
                state.get("fault_category", ""), maximum=200
            ),
            "fault_description": _normalized_verifier_source(
                state.get("fault_description", ""), maximum=10_000
            ),
        },
    }
    return hashlib.sha256(canonical_json(subject).encode("utf-8")).hexdigest()


def validate_checkpoint_progress(
    state: Mapping[str, Any], *, resource_metadata: object | None = None
) -> dict[str, Any]:
    """校验可恢复 Agent 字段的形状、终止语义与跨字段一致性。"""
    resource_metadata = _validated_resource_metadata(resource_metadata)
    if not isinstance(state, Mapping):
        raise ToolLoopError("TOOL_STATE_INVALID")
    profile = _profile_from_state(state)
    registry = registry_for_profile(profile)
    evidence = state.get("tool_result_evidence", {})
    plan = state.get("diagnostic_plan")
    pending = state.get("pending_tool_action")
    completed = state.get("completed_plan_steps")
    trajectory = state.get("tool_trajectory")
    route = state.get("agent_route")
    failed = state.get("tool_loop_failed")
    verification = state.get("verification_status", "NOT_STARTED")
    verifier_result = state.get("verifier_result", {})
    verifier_result_sha256 = state.get("verifier_result_sha256", "")
    effect_policy_status = state.get("effect_policy_status", "NOT_STARTED")
    effect_proposal = state.get("effect_proposal", {})
    effect_proposal_sha256 = state.get("effect_proposal_sha256", "")
    iterations = state.get("tool_iterations")
    stop_reason = state.get("execution_stop_reason")
    if (
        not isinstance(plan, dict)
        or not isinstance(pending, dict)
        or not isinstance(completed, list)
        or not isinstance(trajectory, list)
        or not isinstance(evidence, dict)
        or route not in {"", "decide", "execute", "handoff", "done"}
        or not isinstance(failed, bool)
        or verification
        not in {"NOT_STARTED", "PENDING", *VERIFICATION_FINAL_STATUSES}
        or not isinstance(verifier_result, dict)
        or not isinstance(verifier_result_sha256, str)
        or effect_policy_status not in EFFECT_POLICY_STATUSES
        or not isinstance(effect_proposal, dict)
        or not isinstance(effect_proposal_sha256, str)
        or not isinstance(stop_reason, str)
    ):
        raise ToolLoopError("TOOL_STATE_INVALID")
    if pending:
        if (
            set(pending) != {"step_id"}
            or not isinstance(pending.get("step_id"), str)
            or not pending["step_id"]
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
    events = validate_trajectory(
        trajectory, profile=profile, resource_metadata=resource_metadata
    )
    terminal = events[-1] if events and events[-1]["event"] in {"STOP", "ERROR"} else None

    if plan:
        try:
            validated_plan, validated_completed, events = validate_agent_progress(
                plan,
                completed,
                events,
                tool_iterations=iterations,
                profile=profile,
                resource_metadata=resource_metadata,
            )
        except (PlanContractError, ToolLoopError) as exc:
            raise ToolLoopError("TOOL_STATE_INVALID") from exc
        planned_ids = {item["step_id"] for item in validated_plan["steps"]}
        if pending and (
            pending["step_id"] not in planned_ids
            or pending["step_id"] in validated_completed
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        tool_events = [item for item in events if item["event"] == "TOOL"]
        produced_documents: list[dict[str, Any]] = []
        produced_keys: set[tuple[str, str, int]] = set()
        produced_evidence_steps: set[str] = set()
        for index, event in enumerate(tool_events):
            try:
                expected_call = materialize_tool_call(
                    validated_plan,
                    {
                        "action": "CALL_TOOL",
                        "step_id": event["step_id"],
                        "stop_reason": "",
                    },
                    state,
                    completed_steps=validated_completed[:index],
                    resource_metadata=resource_metadata,
                )
            except (PlanContractError, DecisionContractError, ToolLoopError) as exc:
                raise ToolLoopError("TOOL_STATE_INVALID") from exc
            if event["arguments"] != _trajectory_safe_arguments(
                expected_call["arguments"]
            ):
                raise ToolLoopError("TOOL_STATE_INVALID")
            bound_result = _validate_result_binding(
                state,
                event,
                authoritative_arguments=expected_call["arguments"],
                resource_metadata=resource_metadata,
            )
            output_kind = registry[event["tool_name"]].output_kind
            if output_kind == "documents":
                for item in bound_result:
                    key = (item["id"], item["filename"], item["chunk_index"])
                    if key not in produced_keys:
                        produced_documents.append(item)
                        produced_keys.add(key)
            elif output_kind in {"telemetry", "ros_log"}:
                produced_evidence_steps.add(event["step_id"])
        if terminal is not None and terminal["event"] == "ERROR":
            error_step_id = terminal["step_id"]
            if error_step_id:
                step = next(
                    (
                        item
                        for item in validated_plan["steps"]
                        if item["step_id"] == error_step_id
                    ),
                    None,
                )
                if step is None or error_step_id in validated_completed:
                    raise ToolLoopError("TOOL_STATE_INVALID")
                if terminal["tool_name"]:
                    try:
                        expected_call = materialize_tool_call(
                            validated_plan,
                            {
                                "action": "CALL_TOOL",
                                "step_id": error_step_id,
                                "stop_reason": "",
                            },
                            state,
                            completed_steps=validated_completed,
                            resource_metadata=resource_metadata,
                        )
                    except (
                        PlanContractError,
                        DecisionContractError,
                        ToolLoopError,
                    ) as exc:
                        raise ToolLoopError("TOOL_STATE_INVALID") from exc
                    if (
                        terminal["tool_name"] != step["tool_name"]
                        or terminal["tool_name"] != expected_call["tool_name"]
                        or terminal["arguments"]
                        != _trajectory_safe_arguments(expected_call["arguments"])
                    ):
                        raise ToolLoopError("TOOL_STATE_INVALID")
                elif (
                    terminal["arguments"]
                    or terminal["error_code"] != "MAX_TOOL_ITERATIONS"
                ):
                    raise ToolLoopError("TOOL_STATE_INVALID")
        persisted_documents = _normalize_documents(
            state.get("retrieved_documents"),
            max_items=10,
        )
        if persisted_documents != produced_documents:
            raise ToolLoopError("TOOL_STATE_INVALID")
        if (
            resource_metadata is not None
            and canonical_json(state["retrieved_documents"]) != canonical_json(persisted_documents)
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        if set(evidence) != produced_evidence_steps:
            raise ToolLoopError("TOOL_STATE_INVALID")
        try:
            if len(canonical_json(evidence).encode("utf-8")) > TOOL_RESULT_EVIDENCE_MAX_BYTES:
                raise ToolLoopError("TOOL_STATE_INVALID")
        except ToolLoopError as exc:
            raise ToolLoopError("TOOL_STATE_INVALID") from exc
    else:
        validated_plan = {}
        validated_completed = []
        if completed or pending or evidence:
            raise ToolLoopError("TOOL_STATE_INVALID")
        if events and not (len(events) == 1 and events[0]["event"] == "ERROR"):
            raise ToolLoopError("TOOL_STATE_INVALID")
        if events and (
            events[0]["step_id"]
            or events[0]["tool_name"]
            or events[0]["arguments"]
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        if events and iterations != events[0]["iteration"]:
            raise ToolLoopError("TOOL_STATE_INVALID")

    if route == "execute":
        if (
            not plan
            or not pending
            or terminal is not None
            or failed
            or verification != "NOT_STARTED"
            or stop_reason
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
    elif route == "decide":
        if (
            not plan
            or pending
            or terminal is not None
            or failed
            or verification != "NOT_STARTED"
            or stop_reason
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
    elif route in {"handoff", "done"}:
        if pending or terminal is None:
            raise ToolLoopError("TOOL_STATE_INVALID")
        if route == "done" and verification not in {
            "PENDING",
            *VERIFICATION_FINAL_STATUSES,
        }:
            raise ToolLoopError("TOOL_STATE_INVALID")
        if route == "handoff" and verification != "NOT_STARTED":
            raise ToolLoopError("TOOL_STATE_INVALID")
        if failed != (terminal["event"] == "ERROR"):
            raise ToolLoopError("TOOL_STATE_INVALID")
        expected_reason = (
            terminal["error_code"]
            if terminal["event"] == "ERROR"
            else f"AGENT_STOP_{terminal['stop_reason']}"
        )
        if stop_reason != expected_reason:
            raise ToolLoopError("TOOL_STATE_INVALID")
        if terminal["event"] == "STOP":
            validate_stop_preconditions(
                validated_plan,
                terminal["stop_reason"],
                validated_completed,
                events,
                profile=profile,
                resource_metadata=resource_metadata,
            )
        if terminal["event"] == "ERROR" or terminal["stop_reason"] == "NO_SAFE_ACTION":
            if (
                state.get("need_human") is not True
                or state.get("risk_level") != "高"
                or state.get("urgency") != "高"
            ):
                raise ToolLoopError("TOOL_STATE_INVALID")
    elif route == "":
        if (
            plan
            or pending
            or completed
            or events
            or failed
            or verification != "NOT_STARTED"
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")

    subject_progress = {
        "diagnostic_plan": validated_plan,
        "pending_tool_action": dict(pending),
        "completed_plan_steps": validated_completed,
        "tool_trajectory": events,
        "agent_route": route,
        "tool_loop_failed": failed,
        "tool_capability_profile": profile,
        "tool_result_evidence": dict(evidence),
    }
    expected_subject_sha256 = verifier_subject_sha256(
        state,
        progress=subject_progress,
    )

    if verification in {"NOT_STARTED", "PENDING"}:
        if (
            verifier_result
            or verifier_result_sha256
            or effect_policy_status != "NOT_STARTED"
            or effect_proposal
            or effect_proposal_sha256
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        validated_verifier_result: dict[str, Any] = {}
    else:
        if route != "done" or not verifier_result or not verifier_result_sha256:
            raise ToolLoopError("TOOL_STATE_INVALID")
        validated_verifier_result = validate_verifier_result(
            verifier_result,
            subject_sha256=expected_subject_sha256,
            evidence=evidence,
        )
        if (
            validated_verifier_result["status"] != verification
            or verifier_result_sha256
            != canonical_sha256(validated_verifier_result)
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        if verification in {"FAILED", "ERROR"} and (
            state.get("risk_level") != "高"
            or state.get("urgency") != "高"
            or state.get("need_human") is not True
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        proposal_progress = {
            **subject_progress,
            "verification_status": verification,
            "verifier_result": validated_verifier_result,
            "verifier_result_sha256": verifier_result_sha256,
        }
        if effect_proposal:
            if verification != "PASSED" or effect_policy_status != "PROPOSED":
                raise ToolLoopError("TOOL_STATE_INVALID")
            try:
                validated_proposal = validate_effect_proposal(
                    effect_proposal,
                    state=state,
                    progress=proposal_progress,
                )
            except ToolLoopError as exc:
                raise ToolLoopError("TOOL_STATE_INVALID") from exc
            if effect_proposal_sha256 != canonical_sha256(validated_proposal):
                raise ToolLoopError("TOOL_STATE_INVALID")
        else:
            validated_proposal = {}
            if effect_proposal_sha256 or effect_policy_status == "PROPOSED":
                raise ToolLoopError("TOOL_STATE_INVALID")
        if effect_policy_status in {"DENIED", "ERROR"} and (
            state.get("risk_level") != "高"
            or state.get("urgency") != "高"
            or state.get("need_human") is not True
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")

    return {
        "diagnostic_plan": validated_plan,
        "pending_tool_action": dict(pending),
        "completed_plan_steps": validated_completed,
        "tool_trajectory": events,
        "agent_route": route,
        "tool_loop_failed": failed,
        "tool_capability_profile": profile,
        "tool_result_evidence": dict(evidence),
        "verification_status": verification,
        "verifier_result": validated_verifier_result,
        "verifier_result_sha256": verifier_result_sha256,
        "effect_policy_status": effect_policy_status,
        "effect_proposal": validated_proposal if verification not in {"NOT_STARTED", "PENDING"} else {},
        "effect_proposal_sha256": effect_proposal_sha256,
    }


def merge_retrieved_documents(existing: object, observation: Mapping[str, Any]) -> list[dict]:
    """稳定合并检索 observation，防止重复工具结果扩张状态。"""
    current = _normalize_documents(existing, max_items=10)
    result = observation.get("result")
    additions = _normalize_documents(result)
    seen = {(item["id"], item["filename"], item["chunk_index"]) for item in current}
    merged = list(current)
    for item in additions:
        key = (item["id"], item["filename"], item["chunk_index"])
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged[:10]


def merge_tool_result_evidence(
    existing: object,
    observation: Mapping[str, Any],
    *,
    step_id: str,
    sequence: int,
    profile: str,
    resource_metadata: object | None = None,
) -> dict[str, dict[str, Any]]:
    """持久化一个 MOD-023 规范结果；最终完整关系由 checkpoint 校验。"""
    if (
        not isinstance(existing, dict)
        or not isinstance(step_id, str)
        or re.fullmatch(r"^[a-z][a-z0-9_]{0,63}$", step_id) is None
        or step_id in existing
        or type(sequence) is not int
        or not 1 <= sequence <= MAX_TOOL_ITERATIONS
    ):
        raise ToolLoopError("TOOL_STATE_INVALID")
    tool_name = observation.get("tool_name") if isinstance(observation, Mapping) else None
    spec = registry_for_profile(profile).get(tool_name)
    if spec is None or spec.output_kind not in {"telemetry", "ros_log"}:
        raise ToolLoopError("TOOL_STATE_INVALID")
    arguments = _validate_arguments(
        spec, observation.get("arguments"), resource_metadata=resource_metadata
    )
    result = _normalize_result(
        spec,
        observation.get("result"),
        arguments=arguments,
        resource_metadata=resource_metadata,
    )
    digest = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
    if observation.get("result_sha256") != digest:
        raise ToolLoopError("TOOL_STATE_INVALID")
    merged = {
        **existing,
        step_id: {
            "sequence": sequence,
            "tool_name": spec.name,
            "arguments": _trajectory_safe_arguments(arguments),
            "result": result,
            "result_sha256": digest,
        },
    }
    if len(canonical_json(merged).encode("utf-8")) > TOOL_RESULT_EVIDENCE_MAX_BYTES:
        raise ToolLoopError("TOOL_STATE_INVALID")
    return merged


def prompt_tool_evidence_summary(
    evidence: object,
    *,
    profile: str,
    resource_metadata: object | None = None,
) -> dict[str, dict[str, Any]]:
    """将已绑定的完整结果投影为可送入模型的有界不可信摘要。"""
    resource_metadata = _validated_resource_metadata(resource_metadata)
    if not isinstance(evidence, dict) or len(evidence) > MAX_TOOL_ITERATIONS:
        raise ToolLoopError("TOOL_STATE_INVALID")
    registry = registry_for_profile(profile)
    projected: dict[str, dict[str, Any]] = {}
    seen_sequences: set[int] = set()
    expected_keys = {
        "sequence",
        "tool_name",
        "arguments",
        "result",
        "result_sha256",
    }
    for step_id, item in sorted(evidence.items()):
        if (
            not isinstance(step_id, str)
            or re.fullmatch(r"^[a-z][a-z0-9_]{0,63}$", step_id) is None
            or not isinstance(item, dict)
            or set(item) != expected_keys
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        sequence = item.get("sequence")
        tool_name = item.get("tool_name")
        spec = registry.get(tool_name) if isinstance(tool_name, str) else None
        if (
            type(sequence) is not int
            or not 1 <= sequence <= MAX_TOOL_ITERATIONS
            or sequence in seen_sequences
            or spec is None
            or spec.output_kind not in {"telemetry", "ros_log"}
        ):
            raise ToolLoopError("TOOL_STATE_INVALID")
        arguments = _validate_arguments(
            spec, item.get("arguments"), resource_metadata=resource_metadata
        )
        if item.get("arguments") != _trajectory_safe_arguments(arguments):
            raise ToolLoopError("TOOL_STATE_INVALID")
        result = _normalize_result(
            spec,
            item.get("result"),
            arguments=arguments,
            resource_metadata=resource_metadata,
        )
        digest = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
        if item.get("result_sha256") != digest:
            raise ToolLoopError("TOOL_STATE_INVALID")
        projected[step_id] = {
            "sequence": sequence,
            "tool_name": spec.name,
            "result_summary": _result_summary(spec.output_kind, result),
        }
        seen_sequences.add(sequence)
    if len(canonical_json(projected).encode("utf-8")) > TOOL_OUTPUT_MAX_BYTES:
        raise ToolLoopError("TOOL_OUTPUT_INVALID")
    return projected
