"""前端主题与公共组件（阶段 5 切片 1，方案 C：Console 企业 SaaS）。

- load_theme(): 注入 frontend/static/style.css（方案 C token + Streamlit 覆盖，按文件版本刷新）
- page_header(): 页面标题 + 路由提示 + 状态 chips
- brand_html() / sidebar_foot_html(): 侧边栏品牌区与底部状态
- empty_card(): 空态占位卡片（三态之一，V6）
"""
from pathlib import Path

import streamlit as st

_STATIC = Path(__file__).parent / "static"
_STYLE_CACHE: tuple[int, str] | None = None


def load_theme() -> None:
    """注入方案 C 样式；CSS 修改后在下一次 Streamlit rerun 自动刷新。"""
    global _STYLE_CACHE
    style_path = _STATIC / "style.css"
    style_version = style_path.stat().st_mtime_ns
    if _STYLE_CACHE is None or _STYLE_CACHE[0] != style_version:
        _STYLE_CACHE = (style_version, style_path.read_text(encoding="utf-8"))
    st.markdown(f"<style>{_STYLE_CACHE[1]}</style>", unsafe_allow_html=True)


def brand_html() -> str:
    """侧边栏品牌区（logo + 名称 + 副标题）。"""
    return (
        '<div class="brand-block">'
        '<div class="logo">支</div>'
        '<div><div class="name">机器人技术支持</div><div class="sub">Robot Support Console</div></div>'
        "</div>"
    )


def _service_status(status: str | None = None) -> tuple[str, str]:
    current = status or st.session_state.get("service_health", "checking")
    return {
        "ready": ("ready", "服务可用"),
        "degraded": ("degraded", "服务降级"),
        "offline": ("offline", "服务离线"),
    }.get(current, ("checking", "正在检查服务"))


def sidebar_foot_html(status: str | None = None) -> str:
    """侧边栏底部只展示真实/检查中的服务状态，不伪造数据库连接。"""
    css_class, label = _service_status(status)
    return (
        f'<div class="side-foot service-{css_class}"><span class="dot"></span>'
        f"{label} · v0.6-session</div>"
    )


def page_header(
    title: str, desc: str, route: str, *, service_status: str | None = None
) -> None:
    """方案 C 页面头：标题、描述、路由与当前服务探测状态。"""
    css_class, label = _service_status(service_status)
    st.markdown(
        f"""
        <div class="page-head">
          <div>
            <h1>{title}</h1>
            <div class="desc">{desc}</div>
          </div>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <span class="route-tip">页面路由：{route}</span>
            <span class="status-chip service-{css_class}"><span class="dot"></span>{label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_card(icon: str, title: str, hint: str) -> None:
    """空态占位卡片（V6 空态：图标 + 标题 + 说明，无空白页）。"""
    st.markdown(
        f"""
        <div class="card">
          <div class="empty">
            <div class="ic">{icon}</div>
            <b>{title}</b>
            <span>{hint}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
