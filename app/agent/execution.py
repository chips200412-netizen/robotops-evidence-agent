"""MOD-020：服务端权威的业务工具轮次门禁。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.agent.state import MAX_TOOL_ITERATIONS


MAX_TOOL_ITERATIONS_REPLY = (
    "已达到最大工具执行轮次，系统已停止自动操作，请联系人工支持。"
)


@dataclass(frozen=True)
class ToolIterationClaim:
    """一次工具调度前的确定性决定；调用方只有 allowed 时才可执行工具。"""

    allowed: bool
    update: dict[str, Any]


def claim_tool_iteration(state: Mapping[str, Any]) -> ToolIterationClaim:
    """在工具执行前原子式计算下一轮；异常/篡改状态一律失败关闭。"""
    count = state.get("tool_iterations", 0)
    limit = state.get("max_tool_iterations", MAX_TOOL_ITERATIONS)
    valid = (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        and isinstance(limit, int)
        and not isinstance(limit, bool)
        and limit == MAX_TOOL_ITERATIONS
    )
    if not valid or count >= limit:
        safe_count = count if isinstance(count, int) and not isinstance(count, bool) else 0
        safe_count = min(max(safe_count, 0), MAX_TOOL_ITERATIONS)
        return ToolIterationClaim(
            allowed=False,
            update={
                "tool_iterations": safe_count,
                "max_tool_iterations": MAX_TOOL_ITERATIONS,
                "execution_stop_reason": "MAX_TOOL_ITERATIONS",
                "need_human": True,
                "route": "generate",
                "final_answer": MAX_TOOL_ITERATIONS_REPLY,
            },
        )
    return ToolIterationClaim(
        allowed=True,
        update={
            "tool_iterations": count + 1,
            "max_tool_iterations": MAX_TOOL_ITERATIONS,
            "execution_stop_reason": "",
        },
    )
