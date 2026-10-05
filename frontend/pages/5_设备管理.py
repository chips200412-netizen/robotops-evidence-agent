"""设备管理（阶段 5 切片 5，A5-4 前端对接）。

- 设备列表：GET /devices → 型号/名称/参数/创建时间表格
- 新增/编辑：POST /devices（按型号幂等 upsert）；编辑态型号锁定，名称/参数可改
- 删除：DELETE /devices/{model}（二次确认，A4-2 设备 CRUD）
- 三态：空态（无设备）/ 错误态（后端不可达 503）/ 加载态（spinner）
- 响应式（A5-4）：窄屏表单单列、表格横向滚动（style.css @media）
"""
import os

from html import escape

import httpx
import streamlit as st

from frontend.theme import empty_card, load_theme, page_header

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

NEW_OPTION = "＋ 新增设备"


def _init_state() -> None:
    st.session_state.setdefault("devices", None)          # 设备列表缓存（None=未加载）
    st.session_state.setdefault("list_error", None)
    st.session_state.setdefault("device_select", NEW_OPTION)  # 编辑目标选择
    st.session_state.setdefault("edit_model", "")          # 当前编辑型号（""=新增）
    st.session_state.setdefault("form_model", "")
    st.session_state.setdefault("form_name", "")
    st.session_state.setdefault("form_params", "")
    st.session_state.setdefault("save_result", None)       # 保存成功提示文案
    st.session_state.setdefault("save_error", None)
    st.session_state.setdefault("delete_result", None)
    st.session_state.setdefault("delete_error", None)
    st.session_state.setdefault("confirm_delete", None)    # 待确认删除的型号


def _api_error(resp: httpx.Response) -> str:
    try:
        return resp.json().get("detail", f"后端返回异常（HTTP {resp.status_code}）")
    except Exception:  # noqa: BLE001
        return f"后端返回异常（HTTP {resp.status_code}）"


def _find_device(devices: list[dict], model: str) -> dict | None:
    return next((d for d in devices if d.get("model") == model), None)


def _on_select() -> None:
    """编辑目标变化时同步表单字段（新增清空；编辑预填）。"""
    model = st.session_state.device_select
    if model == NEW_OPTION:
        st.session_state.edit_model = ""
        st.session_state.form_model = ""
        st.session_state.form_name = ""
        st.session_state.form_params = ""
        return
    dev = _find_device(st.session_state.devices or [], model) or {}
    st.session_state.edit_model = model
    st.session_state.form_model = dev.get("model", "")
    st.session_state.form_name = dev.get("name", "")
    st.session_state.form_params = dev.get("params", "")


def _reset_to_new() -> None:
    """保存/删除成功后回到「新增」态并清空表单。"""
    st.session_state.device_select = NEW_OPTION
    st.session_state.edit_model = ""
    st.session_state.form_model = ""
    st.session_state.form_name = ""
    st.session_state.form_params = ""


def load_devices() -> list[dict] | None:
    """加载设备列表；失败写 list_error 并返回 None（不缓存误导性空列表）。"""
    try:
        resp = httpx.get(f"{API_BASE}/devices", timeout=20)
    except Exception as e:  # noqa: BLE001
        st.session_state.list_error = f"无法连接后端服务（{API_BASE}）：{e}"
        return None
    if resp.status_code != 200:
        st.session_state.list_error = _api_error(resp)
        return None
    st.session_state.list_error = None
    return resp.json().get("devices", [])


def save_device(model: str, name: str, params: str) -> bool:
    """新增/编辑设备（upsert）；成功写 save_result，失败写 save_error。"""
    try:
        resp = httpx.post(
            f"{API_BASE}/devices",
            json={"model": model, "name": name, "params": params},
            timeout=20,
        )
    except Exception as e:  # noqa: BLE001
        st.session_state.save_result = None
        st.session_state.save_error = f"无法连接后端服务（{API_BASE}）：{e}"
        return False
    if resp.status_code != 200:
        st.session_state.save_result = None
        st.session_state.save_error = _api_error(resp)
        return False
    st.session_state.save_result = f"设备「{model}」已保存"
    st.session_state.save_error = None
    return True


def delete_device(model: str) -> bool:
    """删除设备；成功写 delete_result，失败写 delete_error。"""
    try:
        resp = httpx.delete(f"{API_BASE}/devices/{model}", timeout=20)
    except Exception as e:  # noqa: BLE001
        st.session_state.delete_result = None
        st.session_state.delete_error = f"无法连接后端服务（{API_BASE}）：{e}"
        return False
    if resp.status_code != 200:
        st.session_state.delete_result = None
        st.session_state.delete_error = _api_error(resp)
        return False
    st.session_state.delete_result = f"设备「{model}」已删除"
    st.session_state.delete_error = None
    return True


def render_table(devices: list[dict]) -> None:
    """设备档案表格（方案 C table：型号/名称/参数/创建时间）。"""
    rows = []
    for d in devices:
        rows.append(
            f'<tr><td><b>{escape(str(d.get("model", "")))}</b></td>'
            f'<td>{escape(str(d.get("name", "")))}</td>'
            f'<td>{escape(str(d.get("params", "")))}</td>'
            f'<td>{escape(str(d.get("created_at", "")))}</td></tr>'
        )
    st.markdown(
        f'<div class="card"><div class="card-head">设备档案 <span class="hint">共 {len(devices)} 个型号</span></div>'
        '<div class="table-scroll"><table class="kb-table"><tr><th>型号</th><th>名称</th><th>参数</th><th>创建时间</th></tr>'
        f"{''.join(rows)}</table></div></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    load_theme()
    page_header("设备管理", "设备型号与参数档案，支持新增 / 编辑 / 删除", route="/devices")
    _init_state()

    # ---------- 加载设备列表（首次或显式刷新后） ----------
    if st.session_state.devices is None:
        with st.spinner("加载设备列表…"):
            st.session_state.devices = load_devices()

    if st.session_state.list_error:
        st.error(st.session_state.list_error)

    devices = st.session_state.devices or []
    left, right = st.columns([1, 1.25], gap="large")

    # ---------- 左列：新增/编辑表单 ----------
    with left:
        options = [NEW_OPTION] + [d.get("model", "") for d in devices]
        current = st.session_state.device_select
        if current not in options:  # 目标被删除等边界：回退到新增
            st.session_state.device_select = NEW_OPTION
            current = NEW_OPTION
        st.markdown(
            '<div class="card"><div class="card-head">'
            f'{"编辑设备" if current != NEW_OPTION else "新增设备"}</div>',
            unsafe_allow_html=True,
        )
        st.selectbox(
            "操作对象",
            options,
            key="device_select",
            on_change=_on_select,
        )
        is_edit = st.session_state.device_select != NEW_OPTION
        st.text_input(
            "型号 *",
            key="form_model",
            disabled=is_edit,
            placeholder="如 Tonybot",
            help="型号为设备唯一标识，编辑时不可修改",
        )
        st.text_input("名称", key="form_name", placeholder="如 教育人形机器人")
        st.text_area("参数", key="form_params", height=90, placeholder="主控 / 舵机 / 电池 / 通信方式等")

        model = st.session_state.form_model.strip()
        if st.button("保存设备", disabled=not model):
            if save_device(model, st.session_state.form_name.strip(), st.session_state.form_params.strip()):
                st.session_state.devices = None  # 触发列表刷新
                _reset_to_new()
                st.rerun()

        if is_edit:
            if st.button("删除设备"):
                st.session_state.confirm_delete = st.session_state.edit_model
        st.markdown("</div>", unsafe_allow_html=True)

        # 保存/删除/确认提示
        if st.session_state.save_result:
            st.success(st.session_state.save_result)
        if st.session_state.save_error:
            st.error(st.session_state.save_error)
        if st.session_state.delete_result:
            st.success(st.session_state.delete_result)
        if st.session_state.delete_error:
            st.error(st.session_state.delete_error)

        if st.session_state.confirm_delete:
            target = st.session_state.confirm_delete
            st.warning(f"确认删除设备「{target}」？此操作不可撤销。")
            c1, c2 = st.columns(2)
            if c1.button("确认删除", key="confirm_yes"):
                if delete_device(target):
                    st.session_state.confirm_delete = None
                    st.session_state.devices = None
                    _reset_to_new()
                    st.rerun()
            if c2.button("取消", key="confirm_no"):
                st.session_state.confirm_delete = None
                st.rerun()

    # ---------- 右列：设备档案列表 ----------
    with right:
        if st.session_state.devices is not None:
            if not devices:
                empty_card("◍", "暂无设备档案", "新增设备后，型号与参数会显示在这里")
            else:
                render_table(devices)


main()
