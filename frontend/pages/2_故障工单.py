"""故障工单（阶段 5 切片 4，A4-2/I7/I10 前端对接）。

- 筛选（状态/设备型号/故障类型）→ GET /tickets → 工单列表表格（语义色标签）
- 选中工单 → GET /tickets/{id} → 详情卡片（三要素/状态流转图/描述/诊断/方案）
- 状态流转：PATCH /tickets/{id}/status（仅向前，I10），回退由后端拒绝并提示
- 三态：空态（无工单）/ 错误态（后端不可达 503）/ 加载态（spinner）
"""
import os

from html import escape

import httpx
import streamlit as st

from frontend.theme import empty_card, load_theme, page_header

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

TICKET_STATUSES = ("待处理", "处理中", "已解决", "已关闭")
RISK_CLS = {"低": "tag low", "中": "tag mid", "高": "tag high"}


def _init_state() -> None:
    st.session_state.setdefault("tickets", None)
    st.session_state.setdefault("filter_key", None)
    st.session_state.setdefault("list_error", None)
    st.session_state.setdefault("detail", None)
    st.session_state.setdefault("detail_error", None)


def _api_error(resp: httpx.Response) -> str:
    try:
        return resp.json().get("detail", f"后端返回异常（HTTP {resp.status_code}）")
    except Exception:  # noqa: BLE001
        return f"后端返回异常（HTTP {resp.status_code}）"


def load_tickets(status: str | None, device: str | None, fault: str | None) -> list[dict] | None:
    """加载工单列表；失败写 list_error 并返回 None。"""
    params = {}
    if status:
        params["status"] = status
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
    """加载工单详情（含会话消息）；失败写 detail_error 并返回 None。"""
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


def update_status(ticket_id: str, new_status: str) -> bool:
    """状态流转（I10）；成功写流转结果提示，失败写 detail_error。"""
    try:
        resp = httpx.patch(
            f"{API_BASE}/tickets/{ticket_id}/status",
            json={"status": new_status},
            timeout=20,
        )
    except Exception as e:  # noqa: BLE001
        st.session_state.detail_error = f"无法连接后端服务（{API_BASE}）：{e}"
        return False
    if resp.status_code != 200:
        st.session_state.detail_error = _api_error(resp)
        return False
    st.session_state.detail_error = None
    st.session_state.detail = resp.json()
    return True


def render_table(tickets: list[dict]) -> None:
    """工单列表表格（方案 C：语义色风险/状态标签）。"""
    rows = []
    for t in tickets:
        risk = t.get("risk_level", "") or "—"
        risk_cls = RISK_CLS.get(risk, "tag gray")
        status = t.get("status", "") or "—"
        rows.append(
            f'<tr><td><span class="ticket-no">{escape(str(t.get("ticket_id", "")))}</span></td>'
            f'<td>{escape(str(t.get("device_model", "")))}</td>'
            f'<td>{escape(str(t.get("fault_category", "")))}</td>'
            f'<td><span class="{risk_cls}">{escape(risk)}</span></td>'
            f'<td><span class="tag gray">{escape(status)}</span></td>'
            f'<td>{escape(str(t.get("created_at", "")))}</td></tr>'
        )
    st.markdown(
        f'<div class="card"><div class="card-head">工单列表 <span class="hint">共 {len(tickets)} 条</span></div>'
        '<table class="kb-table"><tr><th>工单编号</th><th>设备型号</th><th>故障类型</th>'
        "<th>风险</th><th>处理状态</th><th>创建时间</th></tr>"
        f"{''.join(rows)}</table></div>",
        unsafe_allow_html=True,
    )


def render_status_flow(current: str) -> str:
    """状态流转图：已走到的状态点亮，未到的半透明（03 §5.3）。"""
    idx = TICKET_STATUSES.index(current) if current in TICKET_STATUSES else 0
    parts = []
    for i, s in enumerate(TICKET_STATUSES):
        opacity = "" if i <= idx else ' style="opacity:.5"'
        parts.append(f'<span class="tag gray"{opacity}>{s}</span>')
        if i < len(TICKET_STATUSES) - 1:
            parts.append('<span class="arrow">→</span>')
    return f'<span class="status-flow">{"".join(parts)}</span>'


def render_detail(d: dict) -> None:
    """工单详情卡片：三要素 + 状态流转图 + 问题描述 + 诊断 + 方案（I7 全字段）。"""
    risk = d.get("risk_level", "") or "—"
    risk_cls = RISK_CLS.get(risk, "tag gray")
    diagnosis = d.get("diagnosis") or []
    steps = d.get("solution_steps") or []
    html = (
        '<div class="card"><div class="card-head">工单详情 · '
        f'<span class="ticket-no">{escape(str(d.get("ticket_id", "")))}</span></div>'
        '<div class="ticket-detail">'
        f'<div><div class="label">设备型号 / 故障类型</div><div class="v">'
        f'{escape(str(d.get("device_model", "")))} · {escape(str(d.get("fault_category", "")))} '
        f'<span class="{risk_cls}">{escape(risk)}</span></div></div>'
        f'<div><div class="label">处理状态</div>{render_status_flow(d.get("status", ""))}</div>'
        f'<div class="full"><div class="label">问题描述</div><div class="v">'
        f'{escape(str(d.get("fault_description", "")))}</div></div>'
        f'<div class="full"><div class="label">诊断结果</div><ul class="steps">'
        + "".join(f"<li>{escape(str(c))}</li>" for c in diagnosis)
        + f'</ul></div><div class="full"><div class="label">解决方案</div><ol class="steps">'
        + "".join(f"<li>{escape(str(s))}</li>" for s in steps)
        + "</ol></div></div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_flow_buttons(ticket_id: str, current: str) -> None:
    """状态流转按钮：每个可达的「更前」状态一个按钮（I10 仅向前）。"""
    idx = TICKET_STATUSES.index(current) if current in TICKET_STATUSES else 0
    targets = TICKET_STATUSES[idx + 1:]
    if not targets:
        st.caption("工单已处于终态「已关闭」，无后续流转。")
        return
    cols = st.columns(len(targets))
    for i, target in enumerate(targets):
        if cols[i].button(f"标记为{target}", key=f"flow-{ticket_id}-{target}"):
            update_status(ticket_id, target)
            st.session_state.tickets = None  # 触发列表刷新（状态已变）
            st.rerun()


def main() -> None:
    load_theme()
    page_header("故障工单", "全部诊断工单，支持筛选与状态流转（待处理 → 处理中 → 已解决 → 已关闭）", route="/tickets")
    _init_state()

    c1, c2, c3 = st.columns(3)
    status_filter = c1.selectbox("状态", ["全部", *TICKET_STATUSES])
    device_filter = c2.text_input("设备型号", placeholder="全部")
    fault_filter = c3.text_input("故障类型", placeholder="全部")

    status_val = None if status_filter == "全部" else status_filter
    filter_key = (status_val, device_filter.strip(), fault_filter.strip())
    if st.session_state.filter_key != filter_key or st.session_state.tickets is None:
        with st.spinner("加载工单列表…"):
            st.session_state.tickets = load_tickets(
                status_val, device_filter.strip() or None, fault_filter.strip() or None
            )
        st.session_state.filter_key = filter_key

    if st.session_state.list_error:
        st.error(st.session_state.list_error)

    tickets = st.session_state.tickets
    if tickets is None:
        pass  # 加载失败：list_error 已提示
    elif not tickets:
        empty_card("▤", "暂无工单", "诊断完成后，工单会出现在这里，可筛选与状态流转")
    else:
        render_table(tickets)
        ids = [t["ticket_id"] for t in tickets]
        selected = st.selectbox("查看工单详情", ids)
        if selected:
            with st.spinner("加载工单详情…"):
                st.session_state.detail = load_detail(selected)
            if st.session_state.detail_error:
                st.error(st.session_state.detail_error)
            detail = st.session_state.detail
            if detail:
                render_detail(detail)
                render_flow_buttons(detail["ticket_id"], detail.get("status", ""))


main()
