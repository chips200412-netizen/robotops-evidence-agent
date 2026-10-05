"""智能体对话接口（阶段 3 切片 5 + 阶段 4 切片 3）：/chat 调用 LangGraph 工作流。

链路：问题解析 → 型号识别 → 故障分类 → 检索 → 信息判断 →（追问/生成）→ 风险 → 工单 → 回答。
- 追问轮：返回「【请补充信息】」格式的追问问题（带选项），前端展示后用户补充，
  下一轮请求携带历史，API 按历史中追问前缀条数推算 follow_up_rounds（外部多轮驱动，03 §3.3）；
- 正常轮：返回 final_answer（含引用来源 [n]）与 references；生成轮把 user/assistant
  消息按 ticket_id 落库 conversation_messages（A4-3），落库失败不阻塞回答；
- 新会话首轮自动查同型号+同故障历史工单，命中先提示「有相似历史案例」（03 §4/I5，阶段 4 切片 4）。
"""
from fastapi import APIRouter, HTTPException

from app.agent.graph import graph as agent_graph
from app.agent.state import initial_state
from app.agent import tools as agent_tools
from app.core.config import settings
from app.database import repositories
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

# 追问回复固定前缀：用于 API 层统计已追问轮数（_count_follow_up_rounds）
FOLLOW_UP_PREFIX = "【请补充信息】"


def _count_follow_up_rounds(messages: list[dict] | None) -> int:
    """统计历史中已发生的追问轮数（assistant 消息以追问前缀开头）。"""
    if not messages:
        return 0
    return sum(
        1
        for m in messages
        if m.get("role") == "assistant"
        and str(m.get("content", "")).startswith(FOLLOW_UP_PREFIX)
    )


def _build_follow_up_reply(questions: list[dict]) -> str:
    """将追问问题列表组装为面向用户的回复（≤3 问，每问带选项）。"""
    lines = [FOLLOW_UP_PREFIX + "，请补充以下信息："]
    for i, q in enumerate(questions, 1):
        opts = " / ".join(q.get("options") or [])
        suffix = f"（可选：{opts}）" if opts else ""
        lines.append(f"{i}. {q.get('question', '')}{suffix}")
    return "\n".join(lines)


def _history_hint(device_model: str, fault_category: str) -> str | None:
    """新会话首轮：同型号+同故障历史工单提示（03 §4 / I5）。

    命中返回提示文案；无历史/查询失败返回 None（不影响正常流程）。
    """
    if not device_model or not fault_category:
        return None
    try:
        hits = agent_tools.get_ticket_history(device_model, fault_category, limit=3)
    except Exception:
        return None
    if not hits:
        return None
    ids = "、".join(h["ticket_id"] for h in hits[:3])
    return f"📋 检测到 {len(hits)} 条相同型号与故障类型的历史工单（{ids}），可供参考。"


def _infer_nodes(result: dict) -> list[str]:
    """根据执行结果推断本次实际走过的节点序列（快照回放，切片 2 步骤条）。

    主链节点（parse→identify→classify→retrieve→judge）在正常工作流中总是执行
    （即使识别/分类输出为空），分支按结果判定：
    - 追问轮（route=ask）：止于 ask；
    - 生成轮：generate →（risk 有输出）risk →（ticket 有输出）ticket → answer。
    """
    nodes = ["parse", "identify", "classify", "retrieve", "judge"]
    if result.get("route") == "ask":
        nodes.append("ask")
    else:
        if result.get("llm_failure_node") == "ask":
            nodes.append("ask")
        nodes.append("generate")
        if result.get("risk_level"):
            nodes.append("risk")
        if result.get("ticket_id"):
            nodes.append("ticket")
        nodes.append("answer")
    return nodes


def _build_diagnosis(result: dict) -> dict:
    """组装结果卡片三要素所需的结构化诊断数据（A5-3）。"""
    return {
        "device_model": result.get("device_model") or "",
        "fault_category": result.get("fault_category") or "",
        "risk_level": result.get("risk_level") or "",
        "causes": result.get("diagnosis") or [],
        "steps": result.get("solution_steps") or [],
        "need_human": result.get("need_human") is True,
    }


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """调用 LangGraph 智能体工作流，返回最终回答或追问问题。"""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="未配置 OPENAI_API_KEY：请复制 .env.example 为 .env 并填入 DeepSeek API Key",
        )

    history = req.messages or []
    st = initial_state(
        req.message,
        conversation=[*history, {"role": "user", "content": req.message}],
        follow_up_rounds=_count_follow_up_rounds(history),
    )
    try:
        result = agent_graph.invoke(st)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"智能体工作流执行失败：{e}")

    questions = result.get("follow_up_questions") or []
    if questions and result.get("route") == "ask":
        return ChatResponse(
            reply=_build_follow_up_reply(questions),
            session_id=req.session_id,
            references=[],
            nodes=_infer_nodes(result),
            follow_up=questions,
            diagnosis=_build_diagnosis(result),
        )
    reply = result.get("final_answer") or "（工作流未产出回答，请重试）"
    # 03 §4（I5）：首轮提示历史；03 §八：安全/人工通知优先于历史提示。
    if not history and not questions:
        hint = _history_hint(result.get("device_model", ""), result.get("fault_category", ""))
        if hint:
            safety_first = result.get("risk_level") == "高" or result.get("need_human") is True
            reply = reply + "\n\n" + hint if safety_first else hint + "\n\n" + reply
    # A4-3：生成轮落库会话消息（关联到新工单；失败不影响回答，03 §5.4 精神）
    ticket_id = result.get("ticket_id") or ""
    if ticket_id:
        try:
            repositories.add_message(ticket_id, "user", req.message)
            repositories.add_message(ticket_id, "assistant", reply)
        except Exception:
            pass  # 消息落库失败不阻塞回答（工单本身已保存）
    return ChatResponse(
        reply=reply,
        session_id=req.session_id,
        references=result.get("references") or [],
        nodes=_infer_nodes(result),
        diagnosis=_build_diagnosis(result),
        ticket_id=result.get("ticket_id") or None,
        ticket_note=result.get("ticket_note") or None,
    )
