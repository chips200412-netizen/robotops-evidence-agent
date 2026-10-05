"""LangGraph 图编排：确定性安全外层 + MOD-022 受限诊断子图。

服务端权威会话启用 MOD-022 时：
  START → parse → identify → classify → planner
  → tool_decision ↔ tool_execution → agent_handoff → verifier → judge
  → generate → risk → effect_policy
  → (ticket | approval_interrupt → effect_execution → effect_verification)
  → ticket → answer → END

旧 `/chat` 未启用服务端私有位时仍走 classify → retrieve → judge，
并由 judge 按既有合同选择 ask 或 generate，保持兼容。
Planner/工具异常只能经 agent_handoff 进入既有 judge/generate/risk 安全路径。
独立 V2 权威从 claim_verify 进入 claim_effect_policy 与专用审批/执行链，
然后直接 END；冻结 claim 证据与旧 effect 域保持独立。
"""
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime, get_runtime
from langchain_core.runnables import RunnableLambda

from app.agent import nodes
from app.agent import claim_graph
from app.agent import claim_effect_graph
from app.agent.runtime_context import (
    ASYNC_AUTHORITY_SURFACE,
    SYNTHETIC_RESTART_EFFECT_MODE,
    AgentRuntimeContext,
)
from app.agent.state import AgentState
from app.agent.safety import needs_human

# 节点注册表：名称 → 函数（03 §2 全部节点，A3-2 判定依据）
NODE_NAMES = (
    "parse",
    "identify",
    "classify",
    "planner",
    "tool_decision",
    "tool_execution",
    "agent_handoff",
    "verifier",
    "retrieve",
    "judge",
    "ask",
    "generate",
    "risk",
    "claim_verify",
    "claim_effect_policy",
    "claim_approval_interrupt",
    "claim_effect_execution",
    "claim_effect_verification",
    "effect_policy",
    "approval_interrupt",
    "effect_execution",
    "effect_verification",
    "answer",
    "ticket",
)


def route_after_classify(state: AgentState) -> str:
    """高风险先收尾；基础信息不足仍保留既有追问；其余进入 Agent。"""
    if claim_graph.is_claim_active(state):
        claim_graph.validate_claim_context(state)
        claim_graph.validate_claim_context_seed(state)
    if state.get("agent_tool_loop_enabled") is not True:
        return "retrieve"
    if needs_human(state):
        return "judge"
    if not state.get("device_model") or state.get("fault_category") in {"", "其他"}:
        return "retrieve"
    return "planner"


def route_after_planner(state: AgentState) -> str:
    return "tool_decision" if state.get("agent_route") == "decide" else "agent_handoff"


def route_after_tool_decision(state: AgentState) -> str:
    return (
        "tool_execution"
        if state.get("agent_route") == "execute"
        else "agent_handoff"
    )


def route_after_tool_execution(state: AgentState) -> str:
    return (
        "tool_decision"
        if state.get("agent_route") == "decide"
        else "agent_handoff"
    )


def route_after_judge(state: AgentState) -> str:
    """读取 judge 决策：返回目标节点名（ask / generate）。"""
    return state.get("route", "generate")


def route_after_risk(
    _state: AgentState,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> str:
    """Keep compatibility surfaces out of policy using private runtime context."""
    if claim_graph.is_claim_active(_state):
        return "claim_verify"
    context = getattr(runtime, "context", None)
    if (isinstance(context, AgentRuntimeContext)
            and context.execution_contract == claim_effect_graph.EXECUTION_CONTRACT):
        # An authoritative V2 session must keep its active claim profile.
        raise claim_effect_graph.ClaimEffectGraphError()
    if (
        isinstance(context, AgentRuntimeContext)
        and context.surface == ASYNC_AUTHORITY_SURFACE
        and context.effect_mode == SYNTHETIC_RESTART_EFFECT_MODE
    ):
        return "effect_policy"
    return "ticket"


def route_after_effect_policy(state: AgentState) -> str:
    """Only one deterministic proposal may enter the typed approval node."""
    return (
        "approval_interrupt"
        if state.get("effect_policy_status") == "PROPOSED"
        else "ticket"
    )


def route_after_approval_interrupt(state: AgentState) -> str:
    """An approved resume preserves PROPOSED; reject/error closes to ticket."""
    return (
        "effect_execution"
        if state.get("effect_policy_status") == "PROPOSED"
        else "ticket"
    )


def route_after_claim_verify(state: AgentState) -> str:
    """Use server-supplied runtime when present; sync checkpoint edits have none."""
    return claim_effect_graph.route_after_claim_verify(state, get_runtime())


def build_graph(*, checkpointer=None):
    """注册全部节点与边，可注入持久 checkpointer 后编译返回 LangGraph 图。"""
    builder = StateGraph(AgentState, context_schema=AgentRuntimeContext)
    overrides = {
        "parse": claim_graph.parse_claim_node,
        "classify": claim_graph.classify_claim_node,
        "generate": claim_graph.generate_claim_node,
        "claim_verify": RunnableLambda(claim_graph.claim_verify_node,
                                       afunc=claim_graph.aclaim_verify_node),
        **{name: getattr(claim_effect_graph, f"{name}_node") for name in (
            "claim_effect_policy", "claim_approval_interrupt",
            "claim_effect_execution", "claim_effect_verification",
        )},
    }
    for name in NODE_NAMES:
        builder.add_node(name, overrides[name] if name in overrides
                         else getattr(nodes, f"{name}_node"))

    builder.add_edge(START, "parse")
    builder.add_edge("parse", "identify")
    builder.add_edge("identify", "classify")
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {"planner": "planner", "retrieve": "retrieve", "judge": "judge"},
    )
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tool_decision": "tool_decision", "agent_handoff": "agent_handoff"},
    )
    builder.add_conditional_edges(
        "tool_decision",
        route_after_tool_decision,
        {"tool_execution": "tool_execution", "agent_handoff": "agent_handoff"},
    )
    builder.add_conditional_edges(
        "tool_execution",
        route_after_tool_execution,
        {"tool_decision": "tool_decision", "agent_handoff": "agent_handoff"},
    )
    builder.add_edge("agent_handoff", "verifier")
    builder.add_edge("verifier", "judge")
    builder.add_edge("retrieve", "judge")
    # judge → ask（追问）或 generate（生成），由 route 决定
    builder.add_conditional_edges(
        "judge", route_after_judge, {"ask": "ask", "generate": "generate"}
    )
    builder.add_conditional_edges(
        "ask", route_after_ask,
        {"generate": "generate", "claim_verify": "claim_verify", "end": END}
    )
    builder.add_edge("generate", "risk")
    builder.add_conditional_edges(
        "risk", route_after_risk,
        {"effect_policy": "effect_policy", "ticket": "ticket", "claim_verify": "claim_verify"}
    )
    builder.add_conditional_edges(
        "claim_verify", route_after_claim_verify,
        {"claim_effect_policy": "claim_effect_policy", "end": END},
    )
    builder.add_conditional_edges(
        "claim_effect_policy", claim_effect_graph.route_after_claim_effect_policy,
        {"claim_approval_interrupt": "claim_approval_interrupt", "end": END},
    )
    builder.add_conditional_edges(
        "claim_approval_interrupt", claim_effect_graph.route_after_claim_approval_interrupt,
        {"claim_effect_execution": "claim_effect_execution", "end": END},
    )
    builder.add_edge("claim_effect_execution", "claim_effect_verification")
    builder.add_edge("claim_effect_verification", END)
    builder.add_conditional_edges(
        "effect_policy",
        route_after_effect_policy,
        {"approval_interrupt": "approval_interrupt", "ticket": "ticket"},
    )
    builder.add_conditional_edges(
        "approval_interrupt",
        route_after_approval_interrupt,
        {"effect_execution": "effect_execution", "ticket": "ticket"},
    )
    builder.add_edge("effect_execution", "effect_verification")
    builder.add_edge("effect_verification", "ticket")
    builder.add_edge("ticket", "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)


def route_after_ask(state: AgentState) -> str:
    """模型追问失败必须继续安全交付，不能在本轮 END 丢失人工结论。"""
    active = claim_graph.is_claim_active(state)
    if needs_human(state) or (active and state.get("tool_loop_failed") is True):
        return "generate"
    return "claim_verify" if active else "end"


graph = build_graph()
