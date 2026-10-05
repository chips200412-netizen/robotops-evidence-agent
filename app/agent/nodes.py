"""LangGraph 节点实现（阶段 3/4 + MOD-022 受限诊断子图）。

节点清单对齐 spec/03 §2 节点流转：
parse → identify → classify → retrieve → judge →（ask | generate）
→ risk → ticket → answer。

切片 3.1：注册全部节点与骨架（占位）。
切片 3.2–3.4：全部节点真实实现（LLM + 规则兜底）。
切片 3.5：ticket_node 接入 create_support_ticket 工具。
阶段 4 切片 3：工单真实落库 MySQL，失败经 ticket_note 提示（03 §5.4）。
"""
import hashlib
import json
import re
from collections.abc import Mapping

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.agent.catalog import (
    DEVICE_MODELS,
    FAULT_CATEGORIES,
    detect_raw_input_hazards,
    normalize_model,
    rule_classify_fault,
    rule_identify_device,
    rule_risk_level,
)
from app.agent.prompts import (
    build_ask_prompt,
    build_classify_prompt,
    build_generate_prompt,
    build_identify_prompt,
    build_parse_prompt,
    build_planner_prompt,
    build_tool_decision_prompt,
)
from app.agent.state import AgentState, MAX_FOLLOW_UP_ROUNDS, MAX_TOOL_ITERATIONS
from app.agent.runtime_context import (
    ASYNC_AUTHORITY_SURFACE,
    SYNTHETIC_RESTART_EFFECT_MODE,
    AgentRuntimeContext,
    EffectAuthority,
)
from app.agent.safety import (
    RISK_FAILURE_NOTE,
    LLM_FAILURE_NOTE,
    fail_closed_state,
    is_risk_level,
    llm_failure_state,
    merge_risk_levels,
    needs_human,
    safety_handoff,
)
from app.agent import tool_loop
from app.agent.execution import claim_tool_iteration
from app.observability.tracing import (
    safe_span, set_safe_handoff_required, set_safe_outcome,
    set_safe_process_verification_status, set_safe_tool_name,
)
from app.agent.tools import (
    check_solution_risk,
    create_support_ticket,
    search_fault_cases,
    search_manual,
)
from app.services.llm import generate_json
from app.services import effect_capabilities


_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EFFECT_RUNTIME_FIELDS = frozenset(
    {
        "schema_version",
        "effect_key",
        "binding_sha256",
        "status",
        "attempt_count",
        "result_sha256",
        "verification_sha256",
    }
)
_EXECUTION_RETURN_STATUSES = frozenset(
    {
        "APPLIED",
        "EXECUTION_FAILED",
        "VERIFIED",
        "VERIFICATION_FAILED",
        "UNKNOWN",
    }
)
_VERIFICATION_RETURN_STATUSES = frozenset(
    {"VERIFIED", "VERIFICATION_FAILED", "UNKNOWN"}
)
EFFECT_FAILURE_NOTE = (
    "合成处置未获得独立读回确认，已失败关闭并转人工处理。"
)


def _object(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _string(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError("Expected string field")
    return value.strip()


def _strings(value: object, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError("Expected string array")
    if not allow_empty and not value:
        raise ValueError("Expected nonempty array")
    return [v.strip() for v in value]


def _raw_hazards(query: str) -> list[str]:
    """保留 1A 解析器，只校验其调用边界，不让异常/非法结果静默通过。"""
    if not isinstance(query, str):
        raise ValueError("Invalid raw query")
    hazards = detect_raw_input_hazards(query)
    if not isinstance(hazards, list) or not all(isinstance(h, str) and h for h in hazards):
        raise ValueError("Invalid raw hazard result")
    return hazards


def parse_node(state: AgentState) -> dict:
    """[解析节点] 原始危险预检先于模型；失败保留原文并进入安全人工出口。"""
    query = state.get("user_query", "")
    # R1-ACC-001：必须先检查未经模型改写的原始输入，避免 LLM 遗漏危险词。
    try:
        hazards = _raw_hazards(query)
    except Exception:
        return {"fault_description": query, "missing_information": [], **fail_closed_state()}
    safety_update = (
        {"risk_level": "高", "urgency": "高", "need_human": True}
        if hazards
        else {}
    )
    if state.get("llm_failed") is True:
        return {"fault_description": query, "missing_information": [], **safety_handoff(state), **safety_update}
    try:
        prompt = build_parse_prompt(query)
        with safe_span("robotops.llm.parse") as span:
            raw_payload = generate_json(prompt)
            set_safe_outcome(span, "COMPLETED")
        data = _object(raw_payload)
        fd = _string(data, "fault_description") or query
        missing = _strings(data.get("missing_information", []))
        result = {
            "fault_description": fd,
            "missing_information": missing,
        }
    except Exception:
        result = {"fault_description": query, "missing_information": [], **llm_failure_state("parse")}
    return {**result, **safety_update}


def identify_node(state: AgentState) -> dict:
    """[识别节点] 设备型号识别（LLM → 规范型号校验 → 规则兜底）。"""
    query = state.get("user_query", "")
    fd = state.get("fault_description", "")
    text = f"{query} {fd}"
    model = ""
    if state.get("llm_failed") is True:
        return {"device_model": rule_identify_device(text), **safety_handoff(state)}
    try:
        prompt = build_identify_prompt(query, fd)
        with safe_span("robotops.llm.identify") as span:
            raw_payload = generate_json(prompt)
            set_safe_outcome(span, "COMPLETED")
        data = _object(raw_payload)
        candidate = _string(data, "device_model")
        model = _canonical_model(candidate) or rule_identify_device(text)
    except Exception:
        return {"device_model": rule_identify_device(text), **llm_failure_state("identify")}
    if not model:
        model = rule_identify_device(text)
    return {"device_model": model}


def classify_node(state: AgentState) -> dict:
    """[分类节点] 故障分类（10 类）+ 紧急程度 + 是否需人工（LLM → 类别校验 → 规则兜底）。"""
    fd = state.get("fault_description", "")
    model = state.get("device_model", "")
    category, urgency, need_human = "", "", False
    model_risk, invalid_risk = "", False
    if state.get("llm_failed") is True:
        return {"fault_category": rule_classify_fault(fd), **safety_handoff(state)}
    try:
        prompt = build_classify_prompt(fd, model)
        with safe_span("robotops.llm.classify") as span:
            raw_payload = generate_json(prompt)
            set_safe_outcome(span, "COMPLETED")
        data = _object(raw_payload)
        # 风险字段独立检查，其他坏字段不得遮蔽已出现的风险合同失败。
        if "risk_level" in data:
            model_risk = data["risk_level"]
            invalid_risk = not is_risk_level(model_risk)
        category = _string(data, "fault_category")
        urgency = _string(data, "urgency")
        if not is_risk_level(urgency) or not isinstance(data.get("need_human"), bool):
            raise ValueError("Invalid classification contract")
        need_human = data["need_human"]
    except Exception:
        failure = llm_failure_state("classify")
        if invalid_risk:
            failure = {**fail_closed_state(), **failure}
        return {"fault_category": rule_classify_fault(fd), **failure}
    if category not in FAULT_CATEGORIES:
        category = rule_classify_fault(fd)
    if invalid_risk:
        return {"fault_category": category, **fail_closed_state(), **llm_failure_state("classify")}
    if state.get("risk_check_failed") is True:
        return {"fault_category": category, **fail_closed_state()}
    try:
        risk = merge_risk_levels(state.get("risk_level", ""), model_risk)
    except ValueError:
        return {"fault_category": category, **fail_closed_state()}
    previous_urgency = state.get("urgency", "")
    urgency = merge_risk_levels(urgency, previous_urgency if is_risk_level(previous_urgency) else "")
    if risk == "高":
        urgency = "高"
    need_human = need_human or state.get("need_human") is True or urgency == "高"
    return {
        "fault_category": category, "risk_level": risk, "urgency": urgency,
        "need_human": need_human,
    }


def _canonical_model(candidate: str) -> str:
    """将模型输出映射为规范型号：表内精确/归一化匹配；否则返回空串。"""
    if not candidate:
        return ""
    norm = normalize_model(candidate)
    for model in DEVICE_MODELS:
        if normalize_model(model) == norm:
            return model
    return ""


def _safe_iteration(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return 0
    return min(max(value, 0), MAX_TOOL_ITERATIONS)


def _tool_profile(state: AgentState | dict) -> str:
    """只采用服务端已识别档案；损坏状态的失败事件回落到 core 校验。"""
    candidate = state.get("tool_capability_profile", tool_loop.CORE_TOOL_PROFILE)
    try:
        tool_loop.registry_for_profile(candidate)
    except tool_loop.ToolLoopError:
        return tool_loop.CORE_TOOL_PROFILE
    return candidate


def _agent_failure(
    state: AgentState | dict,
    code: str,
    *,
    llm_node: str = "",
    step_id: str = "",
    tool_name: str = "",
    arguments: dict | None = None,
    base_update: dict | None = None,
) -> dict:
    """将 Planner/decision/tool 异常转换为确定性高风险人工出口。"""
    update = dict(base_update or {})
    profile = _tool_profile(state)
    failure_code = (
        code
        if isinstance(code, str) and code in tool_loop.TRAJECTORY_ERROR_CODES
        else "TOOL_STATE_INVALID"
    )
    iteration = _safe_iteration(update.get("tool_iterations", state.get("tool_iterations", 0)))
    try:
        trajectory = tool_loop.append_trajectory_event(
            state.get("tool_trajectory", []),
            profile=profile,
            iteration=iteration,
            event="ERROR",
            step_id=step_id,
            tool_name=tool_name,
            arguments=arguments,
            error_code=failure_code,
            stop_reason=failure_code,
        )
    except tool_loop.ToolLoopError:
        try:
            salvage = tool_loop.validate_trajectory(
                state.get("tool_trajectory", []), profile=profile
            )
            if salvage and salvage[-1]["event"] in {"STOP", "ERROR"}:
                salvage = salvage[:-1]
        except tool_loop.ToolLoopError:
            salvage = []
        trajectory = tool_loop.append_trajectory_event(
            salvage,
            profile=profile,
            iteration=iteration,
            event="ERROR",
            error_code=failure_code,
            stop_reason=failure_code,
        )
    completed_steps = [item["step_id"] for item in trajectory if item["event"] == "TOOL"]
    existing_evidence = state.get("tool_result_evidence", {})
    retained_evidence = (
        {
            step: existing_evidence[step]
            for step in completed_steps
            if step in existing_evidence
        }
        if isinstance(existing_evidence, dict)
        else {}
    )
    update.update(
        {
            "risk_level": "高",
            "urgency": "高",
            "need_human": True,
            "pending_tool_action": {},
            "completed_plan_steps": completed_steps,
            "tool_trajectory": trajectory,
            "tool_result_evidence": retained_evidence,
            "tool_capability_profile": profile,
            "agent_route": "handoff",
            "tool_loop_failed": True,
            "execution_stop_reason": failure_code,
        }
    )
    try:
        tool_loop.validate_checkpoint_progress({**state, **update})
    except (tool_loop.PlanContractError, tool_loop.ToolLoopError):
        trajectory = tool_loop.append_trajectory_event(
            [],
            profile=profile,
            iteration=iteration,
            event="ERROR",
            error_code=failure_code,
            stop_reason=failure_code,
        )
        update.update(
            {
                "max_tool_iterations": MAX_TOOL_ITERATIONS,
                "tool_iterations": iteration,
                "diagnostic_plan": {},
                "pending_tool_action": {},
                "completed_plan_steps": [],
                "tool_trajectory": trajectory,
                "tool_result_evidence": {},
                "retrieved_documents": [],
                "tool_capability_profile": profile,
                "verification_status": "NOT_STARTED",
                "verifier_result": {},
                "verifier_result_sha256": "",
                "effect_policy_status": "NOT_STARTED",
                "effect_proposal": {},
                "effect_proposal_sha256": "",
                "effect_runtime": {},
                "agent_route": "handoff",
                "tool_loop_failed": True,
                "execution_stop_reason": failure_code,
            }
        )
    if llm_node:
        update.update(llm_failure_state(llm_node))
        update["execution_stop_reason"] = failure_code
        update["tool_trajectory"] = trajectory
        update["agent_route"] = "handoff"
        update["tool_loop_failed"] = True
        update["pending_tool_action"] = {}
    return update


def _agent_boundary_failure(state: AgentState | dict) -> dict | None:
    """每个可恢复节点都重新核对服务端启用位与粘性安全信号。"""
    if state.get("agent_tool_loop_enabled") is not True:
        return _agent_failure(state, "AGENT_TOOL_LOOP_DISABLED")
    if (
        not isinstance(state.get("need_human"), bool)
        or not isinstance(state.get("risk_check_failed"), bool)
        or not isinstance(state.get("llm_failed"), bool)
        or state.get("risk_level") not in {"", "低", "中", "高"}
        or state.get("urgency") not in {"", "低", "中", "高"}
        or (
            (
                state.get("risk_check_failed") is True
                or state.get("llm_failed") is True
                or state.get("risk_level") == "高"
            )
            and (
                state.get("need_human") is not True
                or state.get("risk_level") != "高"
                or state.get("urgency") != "高"
            )
        )
        or (state.get("urgency") == "高" and state.get("need_human") is not True)
    ):
        return _agent_failure(state, "AGENT_SAFETY_STATE_INVALID")
    if needs_human(state):
        return _agent_failure(state, "AGENT_SAFETY_SIGNAL")
    return None


def planner_node(state: AgentState) -> dict:
    """[Planner] 生成并严格验证只读计划；任何合同错误均零工具调用关闭。"""
    boundary_failure = _agent_boundary_failure(state)
    if boundary_failure is not None:
        return boundary_failure
    try:
        profile = state.get(
            "tool_capability_profile", tool_loop.CORE_TOOL_PROFILE
        )
        tool_loop.registry_for_profile(profile)
        progress = tool_loop.validate_checkpoint_progress(state)
        if progress["agent_route"] != "":
            raise tool_loop.ToolLoopError("TOOL_STATE_INVALID")
    except (tool_loop.PlanContractError, tool_loop.ToolLoopError) as exc:
        return _agent_failure(state, exc.code)
    count = state.get("tool_iterations", 0)
    limit = state.get("max_tool_iterations", MAX_TOOL_ITERATIONS)
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count != 0
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit != MAX_TOOL_ITERATIONS
    ):
        return _agent_failure(state, "MAX_TOOL_ITERATIONS")
    try:
        prompt = build_planner_prompt(
            state.get("device_model", ""),
            state.get("fault_category", ""),
            state.get("fault_description", ""),
            tool_loop.allowed_tool_catalog(profile=profile),
            termination_condition=tool_loop.PLANNER_TERMINATION_CONDITION,
            max_iterations=MAX_TOOL_ITERATIONS,
        )
        with safe_span("robotops.llm.planner") as span:
            raw_payload = generate_json(prompt)
            set_safe_outcome(span, "COMPLETED")
        payload = _object(raw_payload)
        plan = tool_loop.validate_diagnostic_plan(payload, profile=profile)
    except tool_loop.PlanContractError as exc:
        return _agent_failure(state, exc.code, llm_node="planner")
    except Exception:
        return _agent_failure(state, "PLANNER_SCHEMA_INVALID", llm_node="planner")
    return {
        "diagnostic_plan": plan,
        "pending_tool_action": {},
        "completed_plan_steps": [],
        "tool_trajectory": [],
        "agent_route": "decide",
        "tool_loop_failed": False,
        "tool_result_evidence": {},
        "verification_status": "NOT_STARTED",
        "verifier_result": {},
        "verifier_result_sha256": "",
        "effect_policy_status": "NOT_STARTED",
        "effect_proposal": {},
        "effect_proposal_sha256": "",
        "effect_runtime": {},
        "execution_stop_reason": "",
    }


def tool_decision_node(state: AgentState) -> dict:
    """[Action] 消费先前 observation，仅选择计划 step 或固定原因停止。"""
    boundary_failure = _agent_boundary_failure(state)
    if boundary_failure is not None:
        return boundary_failure
    try:
        profile = state.get("tool_capability_profile", tool_loop.CORE_TOOL_PROFILE)
        progress = tool_loop.validate_checkpoint_progress(state)
        if progress["agent_route"] != "decide":
            raise tool_loop.ToolLoopError("TOOL_STATE_INVALID")
        plan = progress["diagnostic_plan"]
        completed = progress["completed_plan_steps"]
        trajectory = progress["tool_trajectory"]
        prompt = build_tool_decision_prompt(
            state.get("device_model", ""),
            state.get("fault_category", ""),
            state.get("fault_description", ""),
            plan,
            trajectory,
            completed,
            stop_reasons=sorted(tool_loop.STOP_REASONS),
        )
        with safe_span("robotops.llm.tool_decision") as span:
            raw_payload = generate_json(prompt)
            set_safe_outcome(span, "COMPLETED")
        payload = _object(raw_payload)
        decision = tool_loop.validate_tool_decision(payload)
        if decision["action"] == "STOP":
            reason = decision["stop_reason"]
            tool_loop.validate_stop_preconditions(
                plan,
                reason,
                completed,
                trajectory,
                profile=profile,
            )
            trajectory = tool_loop.append_trajectory_event(
                trajectory,
                profile=profile,
                iteration=_safe_iteration(state.get("tool_iterations", 0)),
                event="STOP",
                stop_reason=reason,
            )
            update = {
                "pending_tool_action": {},
                "tool_trajectory": trajectory,
                "agent_route": "handoff",
                "execution_stop_reason": f"AGENT_STOP_{reason}",
            }
            if reason == "NO_SAFE_ACTION":
                update.update(
                    {
                        "risk_level": "高",
                        "urgency": "高",
                        "need_human": True,
                    }
                )
            return update
        call = tool_loop.materialize_tool_call(
            plan,
            decision,
            state,
            completed_steps=completed,
        )
    except tool_loop.PlanContractError as exc:
        return _agent_failure(state, exc.code, llm_node="tool_decision")
    except tool_loop.DecisionContractError as exc:
        return _agent_failure(state, exc.code, llm_node="tool_decision")
    except tool_loop.ToolLoopError as exc:
        return _agent_failure(state, exc.code)
    except Exception:
        return _agent_failure(
            state, "AGENT_DECISION_SCHEMA_INVALID", llm_node="tool_decision"
        )
    return {
        "pending_tool_action": {"step_id": call["step_id"]},
        "agent_route": "execute",
    }


def tool_execution_node(state: AgentState) -> dict:
    """[Observation] 执行服务端派生的只读调用并记录有界、可复现 observation。"""
    boundary_failure = _agent_boundary_failure(state)
    if boundary_failure is not None:
        return boundary_failure
    pending = state.get("pending_tool_action", {})
    profile = state.get("tool_capability_profile", tool_loop.CORE_TOOL_PROFILE)
    if (
        not isinstance(pending, dict)
        or set(pending) != {"step_id"}
        or not isinstance(pending.get("step_id"), str)
        or not pending["step_id"]
    ):
        return _agent_failure(state, "TOOL_PENDING_ACTION_INVALID")
    claim = claim_tool_iteration(state)
    if not claim.allowed:
        return _agent_failure(
            state,
            "MAX_TOOL_ITERATIONS",
            step_id=pending["step_id"],
            base_update=claim.update,
        )
    decision = {"action": "CALL_TOOL", "step_id": pending["step_id"], "stop_reason": ""}
    try:
        progress = tool_loop.validate_checkpoint_progress(state)
        if progress["agent_route"] != "execute":
            raise tool_loop.ToolLoopError("TOOL_STATE_INVALID")
        plan = progress["diagnostic_plan"]
        completed = progress["completed_plan_steps"]
        call = tool_loop.materialize_tool_call(
            plan,
            decision,
            state,
            completed_steps=completed,
        )
    except (tool_loop.PlanContractError, tool_loop.DecisionContractError) as exc:
        return _agent_failure(state, exc.code)
    except tool_loop.ToolLoopError as exc:
        return _agent_failure(state, exc.code)

    claimed_state = {**state, **claim.update}
    try:
        with safe_span("robotops.tool.execute") as span:
            set_safe_tool_name(span, call["tool_name"])
            observation = tool_loop.execute_readonly_tool(
                call["tool_name"], call["arguments"], profile=profile
            )
            observation = tool_loop.validate_tool_observation(
                observation,
                expected_tool_name=call["tool_name"],
                expected_arguments=call["arguments"],
                profile=profile,
            )
            set_safe_outcome(span, "COMPLETED")
        trajectory = tool_loop.append_trajectory_event(
            state.get("tool_trajectory", []),
            profile=profile,
            iteration=claim.update["tool_iterations"],
            event="TOOL",
            step_id=call["step_id"],
            tool_name=call["tool_name"],
            arguments=observation["arguments"],
            result_summary=observation["result_summary"],
            result_sha256=observation["result_sha256"],
        )
        retrieved = state.get("retrieved_documents", [])
        if call["tool_name"] in {"search_manual", "search_fault_cases"}:
            retrieved = tool_loop.merge_retrieved_documents(retrieved, observation)
        evidence = state.get("tool_result_evidence", {})
        if call["tool_name"] in tool_loop.SYNTHETIC_EVIDENCE_TOOL_REGISTRY:
            evidence = tool_loop.merge_tool_result_evidence(
                evidence,
                observation,
                step_id=call["step_id"],
                sequence=len(trajectory),
                profile=profile,
            )
    except tool_loop.ToolLoopError as exc:
        return _agent_failure(
            claimed_state,
            exc.code,
            step_id=call["step_id"],
            tool_name=call["tool_name"],
            arguments=call["arguments"],
            base_update=claim.update,
        )
    except Exception:
        return _agent_failure(
            claimed_state,
            "TOOL_EXECUTION_ERROR",
            step_id=call["step_id"],
            tool_name=call["tool_name"],
            arguments=call["arguments"],
            base_update=claim.update,
        )

    completed = [*completed, call["step_id"]]
    return {
        **claim.update,
        "pending_tool_action": {},
        "completed_plan_steps": completed,
        "tool_trajectory": trajectory,
        "retrieved_documents": retrieved,
        "tool_result_evidence": evidence,
        "agent_route": "decide",
    }


def agent_handoff_node(state: AgentState) -> dict:
    """MOD-024 Verifier 前的单一出口；当前只验证轨迹完整性并标记 PENDING。"""
    try:
        profile = state.get("tool_capability_profile", tool_loop.CORE_TOOL_PROFILE)
        trajectory = tool_loop.validate_trajectory(
            state.get("tool_trajectory", []), profile=profile
        )
        if state.get("agent_route") != "handoff" or state.get("pending_tool_action") != {}:
            raise tool_loop.ToolLoopError("AGENT_HANDOFF_INVALID")
        failed = state.get("tool_loop_failed") is True
        if not trajectory:
            raise tool_loop.ToolLoopError("AGENT_HANDOFF_INVALID")
        if failed:
            if trajectory[-1]["event"] != "ERROR":
                raise tool_loop.ToolLoopError("AGENT_HANDOFF_INVALID")
            if state.get("diagnostic_plan"):
                _, _, trajectory = tool_loop.validate_agent_progress(
                    state.get("diagnostic_plan", {}),
                    state.get("completed_plan_steps", []),
                    trajectory,
                    tool_iterations=state.get("tool_iterations"),
                    profile=profile,
                )
        else:
            plan, completed, trajectory = tool_loop.validate_agent_progress(
                state.get("diagnostic_plan", {}),
                state.get("completed_plan_steps", []),
                trajectory,
                tool_iterations=state.get("tool_iterations"),
                profile=profile,
            )
            if trajectory[-1]["event"] != "STOP":
                raise tool_loop.ToolLoopError("AGENT_HANDOFF_INVALID")
            tool_loop.validate_stop_preconditions(
                plan,
                trajectory[-1]["stop_reason"],
                completed,
                trajectory,
                profile=profile,
            )
        terminal = trajectory[-1]
        if terminal["event"] == "ERROR" or terminal["stop_reason"] == "NO_SAFE_ACTION":
            if (
                state.get("need_human") is not True
                or state.get("risk_level") != "高"
                or state.get("urgency") != "高"
            ):
                raise tool_loop.ToolLoopError("AGENT_SAFETY_STATE_INVALID")
        tool_loop.validate_checkpoint_progress(state)
    except (tool_loop.PlanContractError, tool_loop.ToolLoopError) as exc:
        return {
            **_agent_failure(state, exc.code),
            "agent_route": "done",
            "verification_status": "PENDING",
            "verifier_result": {},
            "verifier_result_sha256": "",
            "effect_policy_status": "NOT_STARTED",
            "effect_proposal": {},
            "effect_proposal_sha256": "",
            "effect_runtime": {},
        }
    return {
        "tool_trajectory": trajectory,
        "agent_route": "done",
        "verification_status": "PENDING",
        "verifier_result": {},
        "verifier_result_sha256": "",
        "effect_policy_status": "NOT_STARTED",
        "effect_proposal": {},
        "effect_proposal_sha256": "",
        "effect_runtime": {},
    }


def _verifier_checks(value: bool) -> dict[str, bool]:
    return {name: value for name in tool_loop.VERIFIER_CHECK_NAMES}


def _verifier_evidence(state: AgentState | dict) -> dict:
    """只为摘要选择 JSON-safe evidence；不回显任何不可信正文。"""
    evidence = state.get("tool_result_evidence", {})
    if not isinstance(evidence, dict):
        return {}
    try:
        tool_loop.canonical_json(evidence)
    except tool_loop.ToolLoopError:
        return {}
    return evidence


def _local_canonical_sha256(value: object) -> str:
    """Verifier 最后一道本地兜底；不依赖可注入的工具层摘要函数。"""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provisional_verifier_subject_sha256(state: AgentState | dict) -> str:
    """异常前预绑定原始 checkpoint；合法状态与规范化 subject 完全一致。"""
    progress = {
        "diagnostic_plan": state.get("diagnostic_plan", {}),
        "pending_tool_action": state.get("pending_tool_action", {}),
        "completed_plan_steps": state.get("completed_plan_steps", []),
        "tool_trajectory": state.get("tool_trajectory", []),
        "agent_route": state.get("agent_route", ""),
        "tool_loop_failed": state.get("tool_loop_failed", False),
        "tool_capability_profile": state.get(
            "tool_capability_profile", tool_loop.CORE_TOOL_PROFILE
        ),
        "tool_result_evidence": state.get("tool_result_evidence", {}),
    }
    return tool_loop.verifier_subject_sha256(state, progress=progress)


def _verifier_error_update(
    *,
    evidence: object,
    subject_sha256: object,
) -> dict:
    """无条件可构造的脱敏 ERROR 终态；不会再调用工具层 helper。"""
    try:
        evidence_sha256 = _local_canonical_sha256(evidence)
    except (TypeError, ValueError):
        evidence_sha256 = _local_canonical_sha256({})
    if (
        not isinstance(subject_sha256, str)
        or len(subject_sha256) != 64
        or any(character not in "0123456789abcdef" for character in subject_sha256)
    ):
        subject_sha256 = _local_canonical_sha256(
            {"schema_version": "robotops_verifier_unbound_error_v1"}
        )
    result = {
        "schema_version": 1,
        "status": "ERROR",
        "checks": _verifier_checks(False),
        "reason_codes": ["VERIFIER_INTERNAL_ERROR"],
        "subject_sha256": subject_sha256,
        "evidence_sha256": evidence_sha256,
    }
    return {
        "risk_level": "高",
        "urgency": "高",
        "need_human": True,
        "verification_status": "ERROR",
        "verifier_result": result,
        "verifier_result_sha256": _local_canonical_sha256(result),
        "effect_policy_status": "NOT_STARTED",
        "effect_proposal": {},
        "effect_proposal_sha256": "",
        "effect_runtime": {},
    }


def verifier_node(state: AgentState) -> dict:
    """Observe one real deterministic process-verification node execution."""
    with safe_span("robotops.node.process_verifier") as span:
        update = _verifier_node_impl(state)
        set_safe_process_verification_status(span, update.get("verification_status"))
        set_safe_outcome(span, "COMPLETED")
        return update


def _verifier_node_impl(state: AgentState) -> dict:
    """MOD-024 确定性过程/证据 Verifier；不调用模型、IO 或副作用。"""
    evidence: object = {}
    subject_sha256: object = _local_canonical_sha256(
        {"schema_version": "robotops_verifier_unbound_error_v1"}
    )
    try:
        evidence = _verifier_evidence(state)
        try:
            subject_sha256 = _provisional_verifier_subject_sha256(state)
        except Exception:
            pass
        try:
            progress = tool_loop.validate_checkpoint_progress(state)
            if (
                progress["agent_route"] != "done"
                or progress["verification_status"] != "PENDING"
            ):
                raise tool_loop.ToolLoopError("TOOL_STATE_INVALID")
        except (tool_loop.PlanContractError, tool_loop.ToolLoopError) as exc:
            status = "FAILED"
            checks = _verifier_checks(False)
            reason_codes = [exc.code]
            safety_update: dict = {
                "risk_level": "高",
                "urgency": "高",
                "need_human": True,
            }
        else:
            evidence = progress["tool_result_evidence"]
            subject_sha256 = tool_loop.verifier_subject_sha256(
                state,
                progress=progress,
            )
            status = "PASSED"
            checks = _verifier_checks(True)
            reason_codes = []
            safety_update = {}

        result = tool_loop.build_verifier_result(
            status=status,
            checks=checks,
            reason_codes=reason_codes,
            subject_sha256=subject_sha256,
            evidence=evidence,
        )
        return {
            **safety_update,
            "verification_status": status,
            "verifier_result": result,
            "verifier_result_sha256": tool_loop.canonical_sha256(result),
            "effect_policy_status": "NOT_STARTED",
            "effect_proposal": {},
            "effect_proposal_sha256": "",
            "effect_runtime": {},
        }
    except Exception:
        return _verifier_error_update(
            evidence=evidence,
            subject_sha256=subject_sha256,
        )


def retrieve_node(state: AgentState) -> dict:
    """[检索节点] 知识库检索：device_model + fault_category 组合查询，双库检索（03 §6）。

    单库失败不影响整体（03 §6 工具兜底）；无查询词返回空。
    """
    parts = [
        state.get("device_model", ""),
        state.get("fault_category", ""),
        state.get("fault_description", ""),
    ]
    query = " ".join(p for p in parts if p)
    if not query:
        return {"retrieved_documents": []}
    docs: list[dict] = []
    for tool_name, fn in (
        ("search_manual", search_manual),
        ("search_fault_cases", search_fault_cases),
    ):
        try:
            with safe_span("robotops.retrieval.lookup") as span:
                set_safe_tool_name(span, tool_name)
                found = fn(query, top_k=5)
                set_safe_outcome(span, "COMPLETED")
            docs.extend(found)
        except Exception:
            continue  # 单库失败兜底为空，不中断工作流
    return {"retrieved_documents": docs}


def judge_node(state: AgentState) -> dict:
    """[判断节点] 信息充足性判断（03 §3.1）→ 写入路由决策 route。

    信息不足条件：设备型号未识别 / 故障类别为空或「其他」（歧义）/ 检索无结果（低于阈值）。
    追问条件：信息不足 且 不需人工 且 追问未满 2 轮；否则直接生成（03 §3.3/§八）。
    """
    missing: list[str] = []
    if not state.get("device_model"):
        missing.append("设备型号")
    category = state.get("fault_category", "")
    if not category or category == "其他":
        missing.append("具体故障类型")
    if not state.get("retrieved_documents"):
        missing.append("知识库匹配资料")

    # 一旦进入 MOD-022 工具子图，所有 STOP/ERROR 必须继续经过 generate/risk，
    # 不允许 ask → END 绕过确定性风险门禁。缺失信息仍进入最终初步建议。
    if (
        state.get("agent_route") == "done"
        and state.get("verification_status")
        in {"PENDING", *tool_loop.VERIFICATION_FINAL_STATUSES}
    ):
        return {"missing_information": missing, "route": "generate"}

    rounds = int(state.get("follow_up_rounds", 0))
    if missing and not needs_human(state) and rounds < MAX_FOLLOW_UP_ROUNDS:
        return {"missing_information": missing, "route": "ask"}
    return {"missing_information": missing, "route": "generate"}


def ask_node(state: AgentState) -> dict:
    """[追问节点] 正常 ≤3 问后结束；失败清空追问并在同一图转安全交付。

    追问轮次由调用方（API 层）在下一轮注入 follow_up_rounds，此处不递增。
    """
    missing = state.get("missing_information") or []
    questions: list[dict] = []
    if needs_human(state):
        return {"follow_up_questions": [], "route": "generate", **safety_handoff(state)}
    try:
        prompt = build_ask_prompt(
            missing, state.get("fault_description", ""), state.get("device_model", ""),
        )
        with safe_span("robotops.llm.ask") as span:
            raw_payload = generate_json(prompt)
            set_safe_outcome(span, "COMPLETED")
        data = _object(raw_payload)
        raw = data.get("questions")
        if not isinstance(raw, list) or not raw:
            raise ValueError("Expected questions")
        for q in raw:
            question = _string(_object(q), "question")
            options = _strings(q.get("options"), allow_empty=False)
            if not question or len(options) < 2:
                raise ValueError("Invalid question contract")
            questions.append({"question": question, "options": options[:3]})
    except Exception:
        return {"follow_up_questions": [], "route": "generate", **llm_failure_state("ask")}
    return {"follow_up_questions": questions[:3]}


def generate_node(state: AgentState) -> dict:
    """[生成节点] 基于资料生成方案；已需人工或模型失败时只交付确定性安全说明。

    检索空 → 资料段要求输出「知识库未匹配」（I6）；信息不足（强制生成）→ 标注初步建议（03 §3.3）。
    """
    if needs_human(state):
        return safety_handoff(state)
    try:
        evidence_summary = tool_loop.prompt_tool_evidence_summary(
            state.get("tool_result_evidence", {}),
            profile=state.get(
                "tool_capability_profile", tool_loop.CORE_TOOL_PROFILE
            ),
        )
        prompt = build_generate_prompt(
            state.get("device_model", ""),
            state.get("fault_category", ""),
            state.get("fault_description", ""),
            state.get("retrieved_documents", []),
            state.get("missing_information", []),
            evidence_summary,
        )
        with safe_span("robotops.llm.generate") as span:
            raw_payload = generate_json(prompt)
            set_safe_outcome(span, "COMPLETED")
        data = _object(raw_payload)
        diagnosis = _strings(data.get("diagnosis"), allow_empty=False)
        steps = _strings(data.get("solution_steps"), allow_empty=False)
    except Exception:
        return llm_failure_state("generate")
    return {"diagnosis": diagnosis, "solution_steps": steps}


def risk_node(state: AgentState) -> dict:
    """[风险节点] 合并全部来源，严格验证工具合同，失败关闭（R1-ACC-002/004）。"""
    with safe_span("robotops.node.risk") as span:
        result = _risk_node_impl(state)
        set_safe_handoff_required(span, result.get("need_human"))
        set_safe_outcome(span, "COMPLETED")
        return result


def _risk_node_impl(state: AgentState) -> dict:
    """Keep the pre-observability gate and its fail-closed result unchanged."""
    try:
        steps = state.get("solution_steps", [])
        description = state.get("fault_description", "")
        category = state.get("fault_category", "")
        if (
            not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
            or not isinstance(description, str)
            or not isinstance(category, str)
        ):
            raise ValueError("Invalid risk input")
        raw_level = "高" if _raw_hazards(state.get("user_query", "")) else ""
        local_level = rule_risk_level(" ".join([description, category, *steps]))
        if not is_risk_level(local_level):
            raise ValueError("Invalid local risk result")
        # 即使已有高风险，也执行检查；失败标记不能被既有等级掩盖。
        result = check_solution_risk(list(steps), description)
        if (
            not isinstance(result, dict)
            or result.get("ok") is not True
            or not is_risk_level(result.get("risk_level"))
            or ("need_human" in result and not isinstance(result["need_human"], bool))
        ):
            raise ValueError("Invalid risk tool result")
        risk = merge_risk_levels(state.get("risk_level", ""), raw_level, local_level, result["risk_level"])
    except Exception:
        return safety_handoff({**state, **fail_closed_state()})
    urgency = state.get("urgency", "")
    if risk == "高":
        urgency = "高"
    need_human = needs_human(state) or result.get("need_human") is True or risk == "高"
    update = {
        "risk_level": risk, "urgency": urgency, "need_human": need_human,
        "risk_check_failed": False, "risk_note": "",
    }
    # 不用成功检查清除先前失败；LLM 和风险失败可同时保留。
    merged = {**state, **update, "risk_check_failed": state.get("risk_check_failed") is True}
    return {**update, **safety_handoff(merged)}


def _effect_fail_closed_update(
    state: AgentState | dict,
    *,
    policy_status: str,
    effect_runtime: dict | None = None,
) -> dict:
    """Close the effect path without exposing authority or device details."""
    safety_state = {
        **state,
        "risk_level": "高",
        "urgency": "高",
        "need_human": True,
    }
    return {
        "risk_level": "高",
        "urgency": "高",
        "need_human": True,
        **safety_handoff(safety_state),
        "effect_policy_status": policy_status,
        "effect_proposal": {},
        "effect_proposal_sha256": "",
        "effect_runtime": effect_runtime or {},
    }


def _valid_runtime_token(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and re.fullmatch(r"[A-Za-z0-9_-]{32}", value) is not None
    )


def _runtime_context(
    runtime: Runtime[AgentRuntimeContext] | None,
) -> AgentRuntimeContext | None:
    context = getattr(runtime, "context", None)
    return context if isinstance(context, AgentRuntimeContext) else None


def _valid_async_context(
    context: AgentRuntimeContext,
    *,
    require_approval: bool | None,
) -> bool:
    if require_approval is True:
        approval_ok = _valid_runtime_token(context.approval_id)
    elif require_approval is False:
        approval_ok = context.approval_id is None
    else:
        approval_ok = context.approval_id is None or _valid_runtime_token(
            context.approval_id
        )
    return (
        context.surface == ASYNC_AUTHORITY_SURFACE
        and context.effect_mode == SYNTHETIC_RESTART_EFFECT_MODE
        and _valid_runtime_token(context.session_id)
        and _valid_runtime_token(context.run_id)
        and approval_ok
        and isinstance(context.effect_authority, EffectAuthority)
    )


def effect_policy_node(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    """Derive the sole synthetic proposal from verified server-owned evidence."""
    context = _runtime_context(runtime)
    if context is None or context.surface != ASYNC_AUTHORITY_SURFACE:
        # `/chat`, synchronous turns, absent runtime context and forged checkpoint
        # flags all stay on the compatibility path with no capability material.
        return {
            "effect_policy_status": "NOT_APPLICABLE",
            "effect_proposal": {},
            "effect_proposal_sha256": "",
            "effect_runtime": {},
        }
    if not _valid_async_context(context, require_approval=False):
        return _effect_fail_closed_update(state, policy_status="ERROR")

    try:
        progress = tool_loop.validate_checkpoint_progress(state)
    except (tool_loop.PlanContractError, tool_loop.ToolLoopError):
        return _effect_fail_closed_update(state, policy_status="ERROR")
    if progress.get("verification_status") != "PASSED":
        status = (
            "ERROR"
            if progress.get("verification_status") == "ERROR"
            else "DENIED"
        )
        return _effect_fail_closed_update(state, policy_status=status)
    if (
        state.get("agent_tool_loop_enabled") is not True
        or progress.get("tool_capability_profile")
        != tool_loop.SYNTHETIC_EVIDENCE_TOOL_PROFILE
    ):
        return _effect_fail_closed_update(state, policy_status="DENIED")
    if (
        state.get("risk_level") not in {"低", "中"}
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
        return _effect_fail_closed_update(state, policy_status="DENIED")

    device_model = state.get("device_model")
    has_bad_telemetry = any(
        isinstance(item, dict)
        and item.get("tool_name") == "query_simulated_telemetry"
        and isinstance(item.get("result"), dict)
        and item["result"].get("source") == tool_loop.EFFECT_SOURCE_SYNTHETIC_LOCAL
        and item["result"].get("device_model") == device_model
        and isinstance(item["result"].get("samples"), list)
        and any(
            isinstance(sample, dict) and sample.get("quality") == "BAD"
            for sample in item["result"]["samples"]
        )
        for item in progress.get("tool_result_evidence", {}).values()
    )
    if not has_bad_telemetry:
        return {
            "effect_policy_status": "NOT_APPLICABLE",
            "effect_proposal": {},
            "effect_proposal_sha256": "",
            "effect_runtime": {},
        }

    try:
        proposal = tool_loop.build_effect_proposal(state)
        proposal_sha256 = tool_loop.canonical_sha256(proposal)
    except tool_loop.ToolLoopError:
        return _effect_fail_closed_update(state, policy_status="ERROR")
    except Exception:
        return _effect_fail_closed_update(state, policy_status="ERROR")
    return {
        "effect_policy_status": "PROPOSED",
        "effect_proposal": proposal,
        "effect_proposal_sha256": proposal_sha256,
        "effect_runtime": {},
    }


def approval_interrupt_node(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    """Pause for one strict typed decision; only the service may resume it."""
    context = _runtime_context(runtime)
    if context is None or not _valid_async_context(context, require_approval=None):
        return _effect_fail_closed_update(state, policy_status="ERROR")
    try:
        progress = tool_loop.validate_checkpoint_progress(state)
        proposal = tool_loop.validate_effect_proposal(
            state.get("effect_proposal"),
            state=state,
            progress=progress,
        )
        if (
            state.get("effect_policy_status") != "PROPOSED"
            or state.get("effect_proposal_sha256")
            != tool_loop.canonical_sha256(proposal)
            or state.get("effect_runtime", {}) != {}
        ):
            raise tool_loop.ToolLoopError("EFFECT_PROPOSAL_INVALID")
        request = effect_capabilities.build_effect_interrupt_request(proposal)
    except Exception:
        return _effect_fail_closed_update(state, policy_status="ERROR")

    decision = interrupt(request)
    resumed_context = _runtime_context(runtime)
    if (
        resumed_context is None
        or not _valid_async_context(resumed_context, require_approval=True)
    ):
        return _effect_fail_closed_update(state, policy_status="ERROR")
    if type(decision) is not str or decision not in {"APPROVE", "REJECT"}:
        return _effect_fail_closed_update(state, policy_status="ERROR")
    if decision == "REJECT":
        return _effect_fail_closed_update(state, policy_status="DENIED")
    return {}


def _validated_effect_input(state: AgentState | dict) -> tuple[dict, str]:
    progress = tool_loop.validate_checkpoint_progress(state)
    proposal = tool_loop.validate_effect_proposal(
        state.get("effect_proposal"),
        state=state,
        progress=progress,
    )
    proposal_sha256 = tool_loop.canonical_sha256(proposal)
    if (
        state.get("effect_policy_status") != "PROPOSED"
        or state.get("effect_proposal_sha256") != proposal_sha256
    ):
        raise tool_loop.ToolLoopError("EFFECT_PROPOSAL_INVALID")
    return proposal, proposal_sha256


def _validated_effect_runtime(
    value: object,
    *,
    allowed_statuses: frozenset[str],
    previous: dict | None = None,
) -> dict:
    if not isinstance(value, Mapping) or set(value) != _EFFECT_RUNTIME_FIELDS:
        raise ValueError("invalid effect authority response")
    normalized = dict(value)
    result_sha256 = normalized["result_sha256"]
    verification_sha256 = normalized["verification_sha256"]
    if (
        type(normalized["schema_version"]) is not int
        or normalized["schema_version"] != 1
        or not isinstance(normalized["effect_key"], str)
        or _DIGEST_PATTERN.fullmatch(normalized["effect_key"]) is None
        or not isinstance(normalized["binding_sha256"], str)
        or _DIGEST_PATTERN.fullmatch(normalized["binding_sha256"]) is None
        or normalized["status"] not in allowed_statuses
        or type(normalized["attempt_count"]) is not int
        or normalized["attempt_count"] != 1
        or (
            result_sha256 is not None
            and (
                not isinstance(result_sha256, str)
                or _DIGEST_PATTERN.fullmatch(result_sha256) is None
            )
        )
        or (
            verification_sha256 is not None
            and (
                not isinstance(verification_sha256, str)
                or _DIGEST_PATTERN.fullmatch(verification_sha256) is None
            )
        )
    ):
        raise ValueError("invalid effect authority response")
    status = normalized["status"]
    if status == "APPLIED" and (
        result_sha256 is None or verification_sha256 is not None
    ):
        raise ValueError("invalid effect authority response")
    if status == "EXECUTION_FAILED" and verification_sha256 is not None:
        raise ValueError("invalid effect authority response")
    if status in {"VERIFIED", "VERIFICATION_FAILED"} and (
        result_sha256 is None or verification_sha256 is None
    ):
        raise ValueError("invalid effect authority response")
    if previous is not None and (
        normalized["effect_key"] != previous.get("effect_key")
        or normalized["binding_sha256"] != previous.get("binding_sha256")
        or normalized["attempt_count"] != previous.get("attempt_count")
        or (
            previous.get("result_sha256") is not None
            and result_sha256 != previous.get("result_sha256")
        )
    ):
        raise ValueError("invalid effect authority response")
    return normalized


def effect_execution_node(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    """Delegate durable CAS/execution to the private service authority."""
    context = _runtime_context(runtime)
    if context is None or not _valid_async_context(context, require_approval=True):
        return _effect_fail_closed_update(state, policy_status="ERROR")
    try:
        proposal, proposal_sha256 = _validated_effect_input(state)
        outcome = context.effect_authority.execute_effect(
            session_id=context.session_id,
            run_id=context.run_id,
            approval_id=context.approval_id,
            proposal=proposal,
            proposal_sha256=proposal_sha256,
        )
        effect_runtime = _validated_effect_runtime(
            outcome,
            allowed_statuses=_EXECUTION_RETURN_STATUSES,
        )
    except Exception:
        return _effect_fail_closed_update(state, policy_status="ERROR")
    if effect_runtime["status"] in {
        "EXECUTION_FAILED",
        "VERIFICATION_FAILED",
        "UNKNOWN",
    }:
        return _effect_fail_closed_update(
            state,
            policy_status="ERROR",
            effect_runtime=effect_runtime,
        )
    return {"effect_runtime": effect_runtime}


def effect_verification_node(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    """Require an independent service-owned read-after-write verification."""
    context = _runtime_context(runtime)
    if context is None or not _valid_async_context(context, require_approval=True):
        return _effect_fail_closed_update(state, policy_status="ERROR")
    current = state.get("effect_runtime")
    try:
        current = _validated_effect_runtime(
            current,
            allowed_statuses=_EXECUTION_RETURN_STATUSES,
        )
    except Exception:
        return _effect_fail_closed_update(state, policy_status="ERROR")
    if current["status"] in {
        "EXECUTION_FAILED",
        "VERIFICATION_FAILED",
        "UNKNOWN",
    }:
        return _effect_fail_closed_update(
            state,
            policy_status="ERROR",
            effect_runtime=current,
        )
    try:
        proposal, proposal_sha256 = _validated_effect_input(state)
        if current["status"] == "VERIFIED":
            return {"effect_runtime": current}
        if current["status"] != "APPLIED":
            return _effect_fail_closed_update(
                state,
                policy_status="ERROR",
                effect_runtime=current,
            )
        outcome = context.effect_authority.verify_effect(
            session_id=context.session_id,
            run_id=context.run_id,
            approval_id=context.approval_id,
            proposal=proposal,
            proposal_sha256=proposal_sha256,
        )
        effect_runtime = _validated_effect_runtime(
            outcome,
            allowed_statuses=_VERIFICATION_RETURN_STATUSES,
            previous=current,
        )
    except Exception:
        return _effect_fail_closed_update(state, policy_status="ERROR")
    if effect_runtime["status"] != "VERIFIED":
        return _effect_fail_closed_update(
            state,
            policy_status="ERROR",
            effect_runtime=effect_runtime,
        )
    return {"effect_runtime": effect_runtime}


def _effect_publication_guard(
    state: AgentState | dict,
    runtime: Runtime[AgentRuntimeContext] | None,
) -> dict:
    """Prevent an unfinished/tampered typed path from publishing success."""
    context = _runtime_context(runtime)
    if (
        context is None
        or context.surface != ASYNC_AUTHORITY_SURFACE
        or context.effect_mode != SYNTHETIC_RESTART_EFFECT_MODE
    ):
        return {}
    if not _valid_async_context(context, require_approval=None):
        return _effect_fail_closed_update(state, policy_status="ERROR")
    try:
        progress = tool_loop.validate_checkpoint_progress(state)
        policy_status = progress["effect_policy_status"]
        if policy_status == "PROPOSED":
            if not _valid_async_context(context, require_approval=True):
                raise ValueError("approved runtime required")
            _validated_effect_input(state)
            effect_runtime = _validated_effect_runtime(
                state.get("effect_runtime"),
                allowed_statuses=frozenset({"VERIFIED"}),
            )
            if effect_runtime["status"] != "VERIFIED":
                raise ValueError("effect not verified")
        elif policy_status in {"NOT_APPLICABLE", "DENIED", "ERROR"}:
            if state.get("effect_runtime", {}) != {}:
                if not _valid_async_context(context, require_approval=True):
                    raise ValueError("approved runtime required")
                # Failed effect terminal records are safe to publish only with
                # the already fail-closed ERROR state.
                effect_runtime = _validated_effect_runtime(
                    state.get("effect_runtime"),
                    allowed_statuses=frozenset(
                        {
                            "EXECUTION_FAILED",
                            "VERIFICATION_FAILED",
                            "UNKNOWN",
                        }
                    ),
                )
                if policy_status != "ERROR" or effect_runtime["status"] not in {
                    "EXECUTION_FAILED",
                    "VERIFICATION_FAILED",
                    "UNKNOWN",
                }:
                    raise ValueError("invalid effect terminal")
        else:
            raise ValueError("effect policy incomplete")
    except Exception:
        return _effect_fail_closed_update(state, policy_status="ERROR")
    return {}


def answer_node(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    """[回答节点] 组装最终回答：风险横幅 + 知识库未匹配提示 + 结构化诊断 + 引用来源 [n]（A2-4/I6）。"""
    publication_update = _effect_publication_guard(state, runtime)
    state = {**state, **publication_update}
    state = {**state, **safety_handoff(state)}
    refs = [
        {
            "n": i,
            "filename": h.get("filename", ""),
            "chunk_index": h.get("chunk_index", 0),
            "similarity": h.get("similarity", 0),
        }
        for i, h in enumerate(state.get("retrieved_documents", []), 1)
    ]
    lines: list[str] = []
    if state.get("risk_level") == "高":
        lines.append(
            "⚠️ **高风险提示**：请停止操作、远离危险源，并联系专业技术人员或人工支持"
            "（非专业勿操作）。"
        )
    if needs_human(state):
        lines.append("【人工介入】当前问题需要专业人员进一步处理，请联系人工支持，暂勿自行维修。")
    if state.get("risk_check_failed") is True:
        lines.append(RISK_FAILURE_NOTE)
    if state.get("llm_failed") is True:
        lines.append(LLM_FAILURE_NOTE)
    if state.get("tool_loop_failed") is True:
        lines.append(tool_loop.TOOL_LOOP_FAILURE_NOTE)
    if not state.get("retrieved_documents"):
        lines.append("知识库未匹配到相关资料（或相似度低于阈值），以下为通用建议；如需精准方案，请补充资料后重试。")
    missing = state.get("missing_information") or []
    if missing:
        lines.append(
            "（初步建议）以下诊断基于不完整信息，缺少：" + "、".join(missing)
            + "；建议补充后再确认，必要时联系人工支持。"
        )
    lines.append(f"【设备型号】{state.get('device_model') or '未知'}")
    lines.append(f"【故障类型】{state.get('fault_category') or '未知'}")
    lines.append(f"【风险等级】{state.get('risk_level') or '低'}")
    diagnosis = state.get("diagnosis") or []
    if diagnosis:
        lines.append("【可能原因】\n" + "\n".join(f"- {d}" for d in diagnosis))
    steps = state.get("solution_steps") or []
    if steps:
        lines.append("【排查步骤】\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
    if refs:
        lines.append(
            "**引用来源**\n"
            + "\n".join(
                f"[{r['n']}] 《{r['filename']}》第 {r['chunk_index'] + 1} 段（相似度 {r['similarity']:.2f}）"
                for r in refs
            )
        )
    ticket = state.get("ticket_id")
    if ticket:
        lines.append(f"【工单】已生成工单：{ticket}")
    elif state.get("ticket_note"):
        # 03 §5.4：工单保存失败时回答仍正常输出，但明确提示失败原因（I8）
        lines.append(f"【工单】{state.get('ticket_note')}（回答不受影响，可稍后重试）")
    return {
        **publication_update,
        "final_answer": "\n\n".join(lines),
        "references": refs,
    }


def ticket_node(state: AgentState) -> dict:
    """[工单节点] 保存工单：调用 create_support_ticket 真实落库（阶段 4 切片 3）。

    失败（saved=False）时 ticket_id 为空、ticket_note 记录原因，
    回答侧给出失败提示（03 §5.4：不静默吞错，不中断回答）。
    """
    safety_update = safety_handoff(state)
    state = {**state, **safety_update}
    if (
        state.get("ticket_mode") != "persist"
        or state.get("agent_tool_loop_enabled") is True
    ):
        # 只有旧 `/chat` 显式注入的 persist 才允许触发 MySQL；缺失、拼写
        # 错误、Agent 路径或恢复 checkpoint 中的未知值都失败关闭为本地草稿。
        return {**safety_update, "ticket_id": "", "ticket_note": ""}
    result = create_support_ticket(
        device_model=state.get("device_model", ""),
        fault_category=state.get("fault_category", ""),
        fault_description=state.get("fault_description", ""),
        diagnosis=state.get("diagnosis", []),
        solution_steps=state.get("solution_steps", []),
        risk_level=state.get("risk_level", "低"),
        need_human=needs_human(state),
    )
    return {
        **safety_update,
        "ticket_id": result.get("ticket_id", ""),
        "ticket_note": result.get("note", ""),
    }
