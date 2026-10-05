"""历史记录（阶段 5 切片 4，A4-3 前端对接）。

- 筛选（设备型号/故障类型）→ GET /tickets → 会话历史列表（问题/追问轮数/风险/工单/回答数）
- 选中会话 → GET /tickets/{id} → 展开完整对话（含追问记录）
- 三态：空态（无会话）/ 错误态（后端不可达 503）/ 加载态（spinner）
"""
import os

from html import escape

import httpx
import streamlit as st

from frontend.theme import empty_card, load_theme, page_header

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
RISK_CLS = {"低": "tag low", "中": "tag mid", "高": "tag high"}


def _init_state() -> None:
    st.session_state.setdefault("sessions", None)
    st.session_state.setdefault("filter_key", None)
    st.session_state.setdefault("list_error", None)
    st.session_state.setdefault("detail", None)
    st.session_state.setdefault("detail_error", None)


def _api_error(resp: httpx.Response) -> str:
    try:
        return resp.json().get("detail", f"后端返回异常（HTTP {resp.status_code}）")
    except Exception:  # noqa: BLE001
        return f"后端返回异常（HTTP {resp.status_code}）"


def load_sessions(device: str | None, fault: str | None) -> list[dict] | None:
    """加载会话历史（复用 /tickets 列表 + 消息统计）；失败返回 None。"""
    params = {}
    if device:
        params["device_model"] = device
    if fault:
        params["fault_category"] = fault
    try:
        resp = httpx.get(f"{API_BASE}/tickets", params=params, timeout=20)
    except Exception as e:  # noqa: BLE001
        st.session_state.list_error = f"无法连接后端服务（{API_BASE}）：{e}"
        return None
    if resp.status_code != 200:
        st.session_state.list_error = _api_error(resp)
        return None
    st.session_state.list_error = None
    return resp.json().get("tickets", [])


def load_detail(ticket_id: str) -> dict | None:
    try:
        resp = httpx.get(f"{API_BASE}/tickets/{ticket_id}", timeout=20)
    except Exception as e:  # noqa: BLE001
        st.session_state.detail_error = f"无法连接后端服务（{API_BASE}）：{e}"
        return None
    if resp.status_code != 200:
        st.session_state.detail_error = _api_error(resp)
        return None
    st.session_state.detail_error = None
    return resp.json()


def render_session_list(sessions: list[dict]) -> None:
    """会话历史列表（方案 C hist-item：时间/问题/追问轮数/风险/工单/回答数）。"""
    items = []
    for s in sessions:
        risk = s.get("risk_level", "") or "—"
        risk_cls = RISK_CLS.get(risk, "tag gray")
        follow = s.get("follow_up_count", 0)
        answers = s.get("answer_count", 0)
        ticket = s.get("ticket_id", "")
        follow_text = f"追问 {follow} 轮" if follow else "未追问"
        ticket_text = f"关联工单 {ticket}" if ticket else "未生成工单"
        sub = f"{follow_text} · 风险 {risk} · {ticket_text} · {answers} 条回答"
        items.append(
            '<div class="hist-item">'
            f'<span class="time">{escape(str(s.get("created_at", "")))}</span>'
            f'<div class="q"><b>{escape(str(s.get("fault_description", "")))}</b>'
            f'<div class="sub">{escape(sub)}</div></div>'
            f'<span class="{risk_cls}">{escape(risk)}</span>'
            "</div>"
        )
    st.markdown(
        f'<div class="card"><div class="card-head">会话历史 <span class="hint">共 {len(sessions)} 条</span></div>'
        f"{''.join(items)}</div>",
        unsafe_allow_html=True,
    )


def render_conversation(detail: dict) -> None:
    """展开某会话的完整对话（user/assistant 消息均 escape 纯文本，单个卡片容器）。"""
    msgs = detail.get("messages") or []
    html = (
        '<div class="card"><div class="card-head">对话详情 · '
        f'<span class="ticket-no">{escape(str(detail.get("ticket_id", "")))}</span>'
        f'<span class="hint">{escape(str(detail.get("fault_category", "")))}</span></div>'
    )
    if not msgs:
        html += '<div class="empty"><b>暂无对话记录</b><span>该工单尚未关联会话消息</span></div>'
    for m in msgs:
        role = m.get("role")
        cls = "msg-user" if role == "user" else "msg-assistant"
        html += f'<div class="{cls}">{escape(str(m.get("content", "")))}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    load_theme()
    page_header("历史记录", "会话历史与追问记录，按设备 / 故障类型筛选", route="/history")
    _init_state()

    c1, c2 = st.columns(2)
    device_filter = c1.text_input("设备型号", placeholder="全部")
    fault_filter = c2.text_input("故障类型", placeholder="全部")

    filter_key = (device_filter.strip(), fault_filter.strip())
    if st.session_state.filter_key != filter_key or st.session_state.sessions is None:
        with st.spinner("加载会话历史…"):
            st.session_state.sessions = load_sessions(
                device_filter.strip() or None, fault_filter.strip() or None
            )
        st.session_state.filter_key = filter_key

    if st.session_state.list_error:
        st.error(st.session_state.list_error)

    sessions = st.session_state.sessions
    if sessions is None:
        pass  # 加载失败：list_error 已提示
    elif not sessions:
        empty_card("↻", "暂无历史会话", "诊断完成后，会话与追问记录会沉淀在这里")
    else:
        render_session_list(sessions)
        labels = [f'{s.get("ticket_id", "")} · {s.get("fault_description", "")}' for s in sessions]
        choice = st.selectbox("查看会话对话", range(len(sessions)), format_func=lambda i: labels[i])
        selected = sessions[choice]
        with st.spinner("加载对话详情…"):
            st.session_state.detail = load_detail(selected["ticket_id"])
        if st.session_state.detail_error:
            st.error(st.session_state.detail_error)
        if st.session_state.detail:
            render_conversation(st.session_state.detail)


main()
