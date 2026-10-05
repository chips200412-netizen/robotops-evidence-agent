"""AgentState：LangGraph 智能体状态定义（阶段 3 切片 1）。

规范字段严格对齐 spec/01 §5（A3-1）；另附图运行所需的内部控制字段
（追问轮次/对话上下文/紧急程度/工单编号，对应 spec/03 §2–§4 节点映射），
内部字段以注释标注，不影响 A3-1 的字段验收。
"""
from typing import TypedDict

# 追问最多循环 2 轮（03 §3.3）
MAX_FOLLOW_UP_ROUNDS = 2
# MOD-020 服务端权威工具轮次上限；客户端不得覆盖。
MAX_TOOL_ITERATIONS = 4


class AgentState(TypedDict):
    """智能体全链路共享状态。

    —— 01 §5 规范字段 ——
    user_query:        用户原始输入
    device_model:      识别出的设备型号
    fault_category:    故障类型（10 类之一）
    fault_description: 故障现象描述（解析节点抽取）
    retrieved_documents: 检索到的文档 [{id, filename, chunk_index, text, similarity}]
    missing_information: 缺失信息（追问依据）
    diagnosis:         可能原因列表
    solution_steps:    解决步骤列表
    risk_level:        风险等级（低/中/高）
    references:        引用来源 [{n, filename, chunk_index, similarity}]
    final_answer:      最终回答
    —— 内部控制字段（03 §2/§3/§4）——
    urgency:           紧急程度（低/中/高，分类节点输出；高则跳过追问，03 §3.4）
    follow_up_rounds:  已追问轮数（最多 2 轮，03 §3.3）
    follow_up_questions: 本轮追问问题 [{question, options}]（03 §3.2 每问 2–3 选项）
    conversation:      OpenAI 格式对话历史（多轮上下文，03 §4）
    ticket_id:         工单编号（工单节点输出，03 §2）
    ticket_note:       工单保存结果说明（成功/失败原因，03 §5.4 失败可感知）
    ticket_mode:       服务端私有工单模式（persist / draft）；客户端不得覆盖
    route:             判断节点路由决策（ask / generate，供条件边读取）
    need_human:        是否需要人工介入（分类节点输出，01 §4 功能 2）
    risk_check_failed: 本次图内风险检查是否失败（只升不降，03 §八）
    risk_note:         风险检查失败的固定安全说明，不包含底层异常
    llm_failed:        本次图内 LLM 是否最终失败（与风险工具失败独立、只升不降）
    llm_failure_node:  首个 LLM 失败节点名，不含异常文本
    tool_iterations:   已完成的 action-observation 工具轮次（跨 checkpoint 累计）
    max_tool_iterations: 服务端固定业务上限，当前为 4
    execution_stop_reason: 确定性安全收尾原因；空字符串表示未触发
    agent_tool_loop_enabled: 服务端私有启用位；客户端不得覆盖
    tool_capability_profile: 服务端私有只读工具能力档案；客户端不得覆盖
    diagnostic_plan:   经服务端严格验证的 Planner 计划
    pending_tool_action: 待执行的计划 step（参数在执行节点由服务端重新派生）
    completed_plan_steps: 已成功执行的唯一 step_id
    tool_trajectory:   无随机量/时间戳的 action-observation 轨迹
    agent_route:       诊断子图内部路由（decide/execute/handoff/done）
    tool_loop_failed:  工具子图是否失败关闭
    tool_result_evidence: MOD-023 typed tools 的规范化 checkpoint 结果证据
    verification_status: MOD-024 Verifier 状态（NOT_STARTED/PENDING/PASSED/FAILED/ERROR）
    verifier_result:   确定性过程/证据完整性核验结果（strict schema）
    verifier_result_sha256: verifier_result 的 canonical SHA-256
    effect_policy_status: 确定性 effect policy 状态
    effect_proposal:   服务端确定性生成的 typed proposal
    effect_proposal_sha256: effect_proposal 的 canonical SHA-256
    effect_runtime:    服务端 effect ledger 的脱敏权威投影（strict schema）
    claim_effect_intent: V2 冻结审批意图（无动作时空，不是执行权限）
    claim_effect_runtime: V2 服务权威执行投影，与旧 effect_runtime 独立
    claim_effect_decision: V2 恢复路由决定（空/APPROVE/REJECT，不是授权）
    """

    user_query: str
    device_model: str
    fault_category: str
    fault_description: str
    retrieved_documents: list[dict]
    missing_information: list[str]
    diagnosis: list[str]
    solution_steps: list[str]
    risk_level: str
    references: list[dict]
    final_answer: str
    urgency: str
    follow_up_rounds: int
    follow_up_questions: list[dict]
    conversation: list[dict]
    ticket_id: str
    ticket_note: str
    ticket_mode: str
    route: str
    need_human: bool
    risk_check_failed: bool
    risk_note: str
    llm_failed: bool
    llm_failure_node: str
    tool_iterations: int
    max_tool_iterations: int
    execution_stop_reason: str
    agent_tool_loop_enabled: bool
    tool_capability_profile: str
    diagnostic_plan: dict
    pending_tool_action: dict
    completed_plan_steps: list[str]
    tool_trajectory: list[dict]
    agent_route: str
    tool_loop_failed: bool
    tool_result_evidence: dict
    verification_status: str
    verifier_result: dict
    verifier_result_sha256: str
    effect_policy_status: str
    effect_proposal: dict
    effect_proposal_sha256: str
    effect_runtime: dict
    claim_evidence_profile: str
    claim_draft: dict
    claim_verification: dict
    claim_publication: dict
    claim_context_seed: dict
    claim_effect_intent: dict
    claim_effect_runtime: dict
    claim_effect_decision: str


def initial_state(user_query: str, conversation: list[dict] | None = None,
                  follow_up_rounds: int = 0) -> AgentState:
    """构造图运行的初始状态：公开字段置空/默认，输入字段由调用方注入。"""
    return {
        "user_query": user_query,
        "device_model": "",
        "fault_category": "",
        "fault_description": "",
        "retrieved_documents": [],
        "missing_information": [],
        "diagnosis": [],
        "solution_steps": [],
        "risk_level": "",
        "references": [],
        "final_answer": "",
        "urgency": "",
        "follow_up_rounds": follow_up_rounds,
        "follow_up_questions": [],
        "conversation": conversation or [],
        "ticket_id": "",
        "ticket_note": "",
        "ticket_mode": "persist",
        "route": "",
        "need_human": False,
        "risk_check_failed": False,
        "risk_note": "",
        "llm_failed": False,
        "llm_failure_node": "",
        "tool_iterations": 0,
        "max_tool_iterations": MAX_TOOL_ITERATIONS,
        "execution_stop_reason": "",
        "agent_tool_loop_enabled": False,
        "tool_capability_profile": "core_v1",
        "diagnostic_plan": {},
        "pending_tool_action": {},
        "completed_plan_steps": [],
        "tool_trajectory": [],
        "agent_route": "",
        "tool_loop_failed": False,
        "tool_result_evidence": {},
        "verification_status": "NOT_STARTED",
        "verifier_result": {},
        "verifier_result_sha256": "",
        "effect_policy_status": "NOT_STARTED",
        "effect_proposal": {},
        "effect_proposal_sha256": "",
        "effect_runtime": {},
        "claim_evidence_profile": "disabled_v0",
        "claim_draft": {},
        "claim_verification": {},
        "claim_publication": {},
        "claim_context_seed": {},
        "claim_effect_intent": {},
        "claim_effect_runtime": {},
        "claim_effect_decision": "",
    }
