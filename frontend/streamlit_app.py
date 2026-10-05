"""机器人技术支持智能体 · Streamlit 入口（阶段 5 切片 1：多页面导航）。

运行：streamlit run frontend/streamlit_app.py
多页面应用（方案 C · Console 企业 SaaS）：
- st.navigation(position="hidden") 负责路由执行；侧边栏用 st.page_link 自定义导航
  （品牌区在顶、工作台/配置分组、状态在底，顺序可控）
- 5 个独立页面 URL：/diagnosis /tickets /history /knowledge /devices（独立跳转）
页面实现位于 frontend/pages/，公共样式见 frontend/static/style.css。
"""
from pathlib import Path
import json
import re
import runpy
from urllib.parse import quote

import streamlit as st

from frontend.theme import brand_html, load_theme, sidebar_foot_html

PAGES = Path(__file__).parent / "pages"

st.set_page_config(page_title="机器人技术支持智能体", page_icon="🤖", layout="wide")

# 在路由组件创建前注入方案 C 样式，避免页面切换时先绘制 Streamlit 默认主题。
load_theme()

# Streamlit 会强制把 default=True 页的 url_path 改为根路径。用一个隐藏的
# 同页执行入口兼容 `/`，把真正的智能诊断页保留在规范地址 `/diagnosis`。
def _render_root_diagnosis() -> None:
    runpy.run_path(str(PAGES / "1_智能诊断.py"), run_name="__main__")


# 页面对象（st.navigation 路由 + st.page_link 导航共用）
p_root = st.Page(
    _render_root_diagnosis,
    title="智能诊断",
    icon=":material/smart_toy:",
    default=True,
    visibility="hidden",
)
p_diag = st.Page(
    str(PAGES / "1_智能诊断.py"),
    title="智能诊断",
    icon=":material/smart_toy:",
    url_path="diagnosis",
)
p_tickets = st.Page(str(PAGES / "2_故障工单.py"), title="故障工单", icon=":material/assignment:", url_path="tickets")
p_history = st.Page(str(PAGES / "3_历史记录.py"), title="历史记录", icon=":material/history:", url_path="history")
p_kb = st.Page(str(PAGES / "4_知识库管理.py"), title="知识库管理", icon=":material/library_books:", url_path="knowledge")
p_devices = st.Page(str(PAGES / "5_设备管理.py"), title="设备管理", icon=":material/memory:", url_path="devices")

# 路由执行（隐藏自动导航，侧栏导航由 st.page_link 自定义控制）
pg = st.navigation(
    [p_root, p_diag, p_tickets, p_history, p_kb, p_devices], position="hidden"
)
active_route = pg.url_path or "diagnosis"


def _install_known_route_not_found_guard(route: str) -> None:
    """Dismiss Streamlit's cold-load false 404 only on registered routes.

    Streamlit can briefly render the requested non-default page and still raise
    its framework-level "Page not found" dialog on the first browser load.  The
    guard is deliberately narrow: the browser path must exactly equal the
    configured ``baseUrlPath`` plus the route that ``st.navigation`` resolved,
    the framework dialog copy must match, and observation is bounded to ten
    seconds. Unknown paths therefore retain Streamlit's real not-found feedback.
    """
    base_url_path = str(st.get_option("server.baseUrlPath") or "").strip("/")
    expected_path = "/" + "/".join(
        part for part in (base_url_path, route) if part
    )
    frame_html = """
        <script>
        (() => {
          const parentWindow = window.parent;
          const doc = parentWindow.document;
          const expectedPath = __EXPECTED_PATH__;
          const routeMatches = () =>
            (parentWindow.location.pathname.replace(/\\/+$/g, '') || '/')
              === expectedPath;
          if (!routeMatches()) return;

          const dismissFalseNotFound = () => {
            if (!routeMatches()) return;
            for (const dialog of doc.querySelectorAll('[role="dialog"]')) {
              const text = (dialog.textContent || '')
                .replace(/\\s+/g, ' ')
                .trim();
              if (!text.includes('Page not found')) continue;
              if (!text.includes("Running the app's main page")) continue;
              const close = dialog.querySelector(
                'button[aria-label="Close"], button[data-testid="stModalCloseButton"]'
              );
              if (close) close.click();
            }
          };

          dismissFalseNotFound();
          const marker = 'robotopsKnownRouteGuardBound';
          if (doc.documentElement.dataset[marker]) return;
          doc.documentElement.dataset[marker] = '1';
          const observer = new MutationObserver(dismissFalseNotFound);
          observer.observe(doc.body, {childList: true, subtree: true});
          parentWindow.setTimeout(() => {
            observer.disconnect();
            delete doc.documentElement.dataset[marker];
          }, 10000);
        })();
        </script>
        """.replace("__EXPECTED_PATH__", json.dumps(expected_path))
    st.iframe(
        frame_html,
        height=1,
        tab_index=-1,
    )


_install_known_route_not_found_guard(active_route)

# 小屏使用真实顶部导航；只携带公开 session_id，不传播 run/request/thread 等内部标识。
raw_session_id = st.query_params.get("session_id")
session_suffix = ""
nav_query_params: dict[str, str] | None = None
if isinstance(raw_session_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{32}", raw_session_id):
    session_suffix = f"?session_id={quote(raw_session_id, safe='')}"
    nav_query_params = {"session_id": raw_session_id}
mobile_links = "".join(
    (
        f'<a class="{"on" if active_route == route else ""}" '
        f'href="./{route}{session_suffix}">{label}</a>'
    )
    for route, label in (
        ("diagnosis", "智能诊断"),
        ("tickets", "故障工单"),
        ("history", "历史记录"),
        ("knowledge", "知识库"),
        ("devices", "设备"),
    )
)
st.markdown(
    f'<nav class="mobile-nav" aria-label="主导航">{mobile_links}</nav>',
    unsafe_allow_html=True,
)

# 自定义侧边栏：品牌 → 分组导航 → 底部状态
with st.sidebar:
    st.markdown(
        f'<span class="nav-active-route" data-route="{active_route}" aria-hidden="true"></span>', unsafe_allow_html=True
    )
    st.markdown(brand_html(), unsafe_allow_html=True)
    st.markdown('<div class="side-nav-grp">工作台</div>', unsafe_allow_html=True)
    st.page_link(
        p_diag,
        label="智能诊断",
        icon=":material/smart_toy:",
        query_params=nav_query_params,
    )
    st.page_link(
        p_tickets,
        label="故障工单",
        icon=":material/assignment:",
        query_params=nav_query_params,
    )
    st.page_link(
        p_history,
        label="历史记录",
        icon=":material/history:",
        query_params=nav_query_params,
    )
    st.markdown('<div class="side-nav-grp">配置</div>', unsafe_allow_html=True)
    st.page_link(
        p_kb,
        label="知识库管理",
        icon=":material/library_books:",
        query_params=nav_query_params,
    )
    st.page_link(
        p_devices,
        label="设备管理",
        icon=":material/memory:",
        query_params=nav_query_params,
    )
    st.markdown(sidebar_foot_html(), unsafe_allow_html=True)

pg.run()
