"""工单接口（阶段 5 切片 4，A4-2/I7/I10 前端对接）。

- GET  /tickets：工单列表（状态/型号/故障类型筛选 + 分页），附消息/追问统计
- GET  /tickets/{ticket_id}：工单详情 + 会话消息（A4-3，含追问记录）
- PATCH /tickets/{ticket_id}/status：状态流转（仅向前，03 §5.3 / I10）

数据库不可用（MySQL 未启动）返回 503 并给明确提示（A4-4 精神）；
状态回退/非法值返回 400（由仓储层校验）。
"""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.database import repositories
from app.database.connection import DatabaseError

router = APIRouter(tags=["tickets"])

FOLLOW_UP_PREFIX = "【请补充信息】"


class StatusUpdateRequest(BaseModel):
    """状态流转请求（仅允许四个合法状态，向前校验在仓储层）。"""

    status: Literal["待处理", "处理中", "已解决", "已关闭"] = Field(
        ..., description="目标状态（仅允许向前流转，03 §5.3）"
    )


def _is_db_unavailable(e: DatabaseError) -> bool:
    """区分「连接失败」（503）与「业务校验失败」（400）。"""
    return "数据库连接失败" in str(e)


def _stats(ticket_id: str) -> dict:
    """统计某工单的会话消息数 / 追问轮数 / 回答数（演示规模，单工单查询）。"""
    msgs = repositories.list_messages(ticket_id)
    follow_ups = sum(
        1
        for m in msgs
        if m.get("role") == "assistant"
        and str(m.get("content", "")).startswith(FOLLOW_UP_PREFIX)
    )
    answers = sum(1 for m in msgs if m.get("role") == "assistant")
    return {
        "message_count": len(msgs),
        "follow_up_count": follow_ups,
        "answer_count": answers,
    }


@router.get("/tickets")
def list_tickets(
    status: str | None = Query(default=None, description="按处理状态筛选"),
    device_model: str | None = Query(default=None, description="按设备型号筛选"),
    fault_category: str | None = Query(default=None, description="按故障类型筛选"),
    limit: int = Query(default=100, ge=1, le=200, description="返回条数上限"),
    offset: int = Query(default=0, ge=0, description="分页偏移"),
) -> dict:
    """工单列表（最新在前，附消息/追问统计）；count 为本页返回条数（分页简化为单页，limit 上限 200）。"""
    try:
        tickets = repositories.list_tickets(
            limit=limit,
            offset=offset,
            status=status,
            device_model=device_model,
            fault_category=fault_category,
        )
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e))
    rows = [{**t, **_stats(t["ticket_id"])} for t in tickets]
    return {"tickets": rows, "count": len(rows)}


@router.get("/tickets/{ticket_id}")
def get_ticket_detail(ticket_id: str) -> dict:
    """工单详情 + 会话消息（A4-3，含追问记录与统计）。"""
    try:
        ticket = repositories.get_ticket(ticket_id)
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"工单不存在：{ticket_id}")
    try:
        msgs = repositories.list_messages(ticket_id)
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {**ticket, "messages": msgs, **_stats(ticket_id)}


@router.patch("/tickets/{ticket_id}/status")
def update_status(ticket_id: str, req: StatusUpdateRequest) -> dict:
    """状态流转（仅向前；回退/非法 → 400，连接失败 → 503，工单不存在 → 404）。"""
    try:
        updated = repositories.update_ticket_status(ticket_id, req.status)
    except DatabaseError as e:
        status_code = 503 if _is_db_unavailable(e) else 400
        raise HTTPException(status_code=status_code, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail=f"工单不存在：{ticket_id}")
    return updated
