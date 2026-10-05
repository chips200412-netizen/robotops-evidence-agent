from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """基础问答请求。"""

    message: str = Field(..., min_length=1, description="用户输入的故障现象/问题")
    session_id: str | None = Field(default=None, description="会话标识（多轮对话用）")
    messages: list[dict] | None = Field(
        default=None,
        description="多轮对话历史（OpenAI 格式 role/content 列表，可选；无则仅当前消息）",
    )


class ChatResponse(BaseModel):
    """基础问答响应。"""

    reply: str = Field(..., description="助手回答（末尾附引用来源 [n]）")
    session_id: str | None = Field(default=None, description="会话标识（原样回传）")
    references: list[dict] | None = Field(
        default=None,
        description="引用来源列表：[{n, filename, chunk_index, similarity}]（A2-4）",
    )
    # ---- 阶段 5 切片 2 新增（A5-3，前端智能诊断页）----
    nodes: list[str] | None = Field(
        default=None,
        description="本次工作流实际走过的节点序列（快照回放，前端步骤条渲染）",
    )
    follow_up: list[dict] | None = Field(
        default=None,
        description="追问问题列表：[{question, options}]（仅追问轮非空，切片 2 选项按钮）",
    )
    diagnosis: dict | None = Field(
        default=None,
        description="结构化诊断结果：{device_model, fault_category, risk_level, causes, steps, need_human}",
    )
    ticket_id: str | None = Field(
        default=None,
        description="本次诊断生成/关联的工单编号（工单提示行；追问轮为空）",
    )
    ticket_note: str | None = Field(
        default=None,
        description="工单保存结果说明（成功/失败原因，I8 失败可感知）",
    )
