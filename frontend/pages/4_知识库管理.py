"""知识库管理（阶段 5 切片 3，A2-1/A2-2/A2-3 前端对接）。

- 上传区：文件选择 + 文档类型 → POST /documents/upload → 解析状态卡片（A2-1/A2-2）
- 文档/切片列表：GET /documents（文档 + 切片数）→ 选中后 GET /documents/{id}/chunks（切片表格）
- 检索测试：关键词 + 类型过滤 → POST /documents/search → 命中卡片（含相似度，A2-3）
- 三态：空态（无文档/未检索）、错误态（后端不可达/失败）、加载态（spinner）
"""
from html import escape
import os

import httpx
import streamlit as st

from frontend.theme import empty_card, load_theme, page_header

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

DOC_TYPE_LABELS = {"manual": "说明书", "fault_case": "故障案例"}
DOC_TYPE_ICON = {"manual": "MD", "fault_case": "FC"}


def _init_state() -> None:
    st.session_state.setdefault("kb_docs", None)          # 文档列表缓存（None=未加载）
    st.session_state.setdefault("chunks_doc_id", None)    # 已加载切片的 doc_id
    st.session_state.setdefault("chunks", None)           # 当前文档切片
    st.session_state.setdefault("load_error", None)
    st.session_state.setdefault("chunk_error", None)
    st.session_state.setdefault("upload_result", None)
    st.session_state.setdefault("upload_error", None)
    st.session_state.setdefault("search_result", None)
    st.session_state.setdefault("search_error", None)


def _api_error(resp: httpx.Response) -> str:
    """从非 200 响应提取 detail 文案。"""
    try:
        return resp.json().get("detail", f"后端返回异常（HTTP {resp.status_code}）")
    except Exception:  # noqa: BLE001 - 非 JSON 响应统一提示
        return f"后端返回异常（HTTP {resp.status_code}）"


def upload_document(file, doc_type: str) -> bool:
    """上传并入库（A2-1/A2-2）；成功写 upload_result，失败写 upload_error。"""
    try:
        resp = httpx.post(
            f"{API_BASE}/documents/upload",
            files={"file": (file.name, file.getvalue(), file.type or "application/octet-stream")},
            data={"doc_type": doc_type},
            timeout=120,
        )
    except Exception as e:  # noqa: BLE001 - 连接类异常统一提示
        st.session_state.upload_result = None
        st.session_state.upload_error = f"无法连接后端服务（{API_BASE}）：{e}"
        return False
    if resp.status_code != 200:
        st.session_state.upload_result = None
        st.session_state.upload_error = _api_error(resp)
        return False
    st.session_state.upload_result = resp.json()
    st.session_state.upload_error = None
    return True


def load_documents() -> list[dict] | None:
    """加载知识库文档列表；失败写 load_error 并返回 None（不缓存误导性空列表）。"""
    try:
        resp = httpx.get(f"{API_BASE}/documents", timeout=20)
    except Exception as e:  # noqa: BLE001
        st.session_state.load_error = f"无法连接后端服务（{API_BASE}）：{e}"
        return None
    if resp.status_code != 200:
        st.session_state.load_error = _api_error(resp)
        return None
    st.session_state.load_error = None
    return resp.json().get("documents", [])


def load_chunks(doc_id: str) -> list[dict] | None:
    """加载某文档切片；失败写 chunk_error 并返回 None（不缓存误导性空列表）。"""
    try:
        resp = httpx.get(f"{API_BASE}/documents/{doc_id}/chunks", timeout=20)
    except Exception as e:  # noqa: BLE001
        st.session_state.chunk_error = f"无法连接后端服务（{API_BASE}）：{e}"
        return None
    if resp.status_code != 200:
        st.session_state.chunk_error = _api_error(resp)
        return None
    st.session_state.chunk_error = None
    return resp.json().get("chunks", [])


def search_kb(query: str, doc_type: str | None, top_k: int) -> bool:
    """检索测试（A2-3）；成功写 search_result，失败写 search_error 并清空旧结果。"""
    try:
        resp = httpx.post(
            f"{API_BASE}/documents/search",
            json={"query": query, "doc_type": doc_type, "top_k": top_k},
            timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        st.session_state.search_result = None
        st.session_state.search_error = f"无法连接后端服务（{API_BASE}）：{e}"
        return False
    if resp.status_code != 200:
        st.session_state.search_result = None
        st.session_state.search_error = _api_error(resp)
        return False
    st.session_state.search_result = resp.json()
    st.session_state.search_error = None
    return True


def render_upload_result(result: dict) -> None:
    """上传解析状态卡片：文件名/类型/字符数/切片数 + 首片段预览。"""
    doc_type = result.get("doc_type", "")
    tag = escape(DOC_TYPE_LABELS.get(doc_type, doc_type or "—"))
    filename = escape(str(result.get("filename", "")))
    html = (
        '<div class="card">'
        '<div class="card-head">解析完成 <span class="hint">已写入 Chroma</span></div>'
        '<div class="kb-meta">'
        f'<div class="kpi"><span class="k">文件名</span><span class="v">{filename}</span></div>'
        f'<div class="kpi"><span class="k">类型</span><span class="tag gray">{tag}</span></div>'
        f'<div class="kpi"><span class="k">字符数</span><span class="v">{escape(str(result.get("char_count", 0)))}</span></div>'
        f'<div class="kpi"><span class="k">切片数</span><span class="v">{escape(str(result.get("chunk_count", 0)))}</span></div>'
        "</div>"
    )
    if result.get("preview"):
        html += f'<div class="kb-preview">{escape(str(result["preview"]))}…</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def render_doc_list(docs: list[dict]) -> None:
    """文档列表卡片（方案 C doc-item：类型图标 + 文件名 + 切片数）。"""
    items = "".join(
        '<div class="doc-item">'
        f'<span class="ic">{escape(DOC_TYPE_ICON.get(d.get("doc_type", ""), "?"))}</span>'
        f'<span class="name">{escape(str(d.get("filename", "")))}</span>'
        f'<span class="chunk-count">{escape(str(d.get("chunk_count", 0)))}</span>'
        "</div>"
        for d in docs
    )
    st.markdown(
        f'<div class="card"><div class="card-head">文档列表 <span class="hint">共 {len(docs)} 篇</span></div>{items}</div>',
        unsafe_allow_html=True,
    )


def render_chunk_table(chunks: list[dict]) -> None:
    """切片列表表格（# / 切片序号 / 原文片段，方案 C table）。"""
    rows = "".join(
        f"<tr><td>{i + 1}</td><td>{escape(str(c.get('chunk_index', 0)))}</td>"
        f"<td>{escape(str(c.get('text', ''))[:120])}</td></tr>"
        for i, c in enumerate(chunks)
    )
    st.markdown(
        f'<div class="card"><div class="card-head">切片列表 <span class="hint">共 {len(chunks)} 片</span></div>'
        '<table class="kb-table"><tr><th>#</th><th>切片</th><th>原文片段</th></tr>'
        f"{rows}</table></div>",
        unsafe_allow_html=True,
    )


def render_search_result(result: dict) -> None:
    """检索命中卡片：文件名 / 相似度标签 / 片段；无命中提示「知识库未匹配」。"""
    hits = result.get("results") or []
    if not hits:
        st.markdown(
            '<div class="err-box">知识库未匹配：未检索到相似度达标的片段，请换关键词或补充资料。</div>',
            unsafe_allow_html=True,
        )
        return
    html = f'<div class="card"><div class="card-head">检索结果 <span class="hint">{escape(str(result.get("count", len(hits))))} 条命中</span></div>'
    for h in hits:
        sim = h.get("similarity", 0)
        html += (
            '<div class="hit"><div class="top">'
            f'<span>{escape(str(h.get("filename", "")))} · 切片 {escape(str(h.get("chunk_index", 0)))}</span>'
            f'<span class="sim">{escape(str(sim))}</span>'
            "</div>"
            f'<div class="snippet">{escape(str(h.get("text", ""))[:200])}</div></div>'
        )
    st.markdown(html + "</div>", unsafe_allow_html=True)


def main() -> None:
    load_theme()
    page_header("知识库管理", "文档上传、解析状态、切片列表与检索测试", route="/knowledge")
    _init_state()

    left, right = st.columns([1, 1.35], gap="large")

    # ---------- 左列：上传区 + 文档列表 ----------
    with left:
        st.markdown('<div class="card"><div class="card-head">上传文档</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "选择文档（PDF / DOCX / MD / TXT）",
            type=["pdf", "docx", "md", "markdown", "txt"],
        )
        doc_type = st.selectbox(
            "文档类型",
            options=["manual", "fault_case"],
            format_func=lambda v: DOC_TYPE_LABELS.get(v, v),
        )
        if st.button("上传并入库", disabled=uploaded is None):
            if upload_document(uploaded, doc_type):
                st.session_state.kb_docs = None  # 触发文档列表刷新
                st.session_state.chunks = None
                st.session_state.chunks_doc_id = None
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.upload_error:
            st.error(st.session_state.upload_error)
        if st.session_state.upload_result:
            render_upload_result(st.session_state.upload_result)

        # 文档列表（首次或上传后加载；失败保持 None，不缓存误导性空列表）
        if st.session_state.kb_docs is None:
            with st.spinner("加载文档列表…"):
                loaded = load_documents()
                if loaded is not None:
                    st.session_state.kb_docs = loaded
        if st.session_state.load_error:
            st.error(st.session_state.load_error)
        if st.session_state.kb_docs is None:
            pass  # 加载失败：load_error 已提示，不显示空态
        elif not st.session_state.kb_docs:
            empty_card("▦", "知识库为空", "上传文档后，这里会列出文档与切片")
        else:
            render_doc_list(st.session_state.kb_docs)

    # ---------- 右列：切片列表 + 检索测试 ----------
    with right:
        docs = st.session_state.kb_docs or []
        if docs:
            labels = [f'{d["filename"]}（{d["chunk_count"]} 切片）' for d in docs]
            current_id = st.session_state.get("chunks_doc_id")
            idx = 0
            for i, d in enumerate(docs):
                if d["doc_id"] == current_id:
                    idx = i
                    break
            choice = st.selectbox(
                "选择文档查看切片",
                range(len(docs)),
                format_func=lambda i: labels[i],
                index=idx,
            )
            selected = docs[choice]
            if st.session_state.chunks_doc_id != selected["doc_id"] or st.session_state.chunks is None:
                st.session_state.chunks = load_chunks(selected["doc_id"])  # None=失败
                st.session_state.chunks_doc_id = selected["doc_id"]
            if st.session_state.chunk_error:
                st.error(st.session_state.chunk_error)
            if st.session_state.chunks is None:
                pass  # 加载失败：chunk_error 已提示，不显示空态/旧文档切片
            elif not st.session_state.chunks:
                empty_card("▦", "该文档暂无切片", "切片可能尚未完成解析")
            else:
                render_chunk_table(st.session_state.chunks)

        st.markdown('<div class="card"><div class="card-head">检索测试</div>', unsafe_allow_html=True)
        query = st.text_input("检索关键词", placeholder="例如：Tonybot 无法连接 Wi-Fi")
        search_type = st.selectbox(
            "范围",
            options=[None, "manual", "fault_case"],
            format_func=lambda v: "全部" if v is None else DOC_TYPE_LABELS.get(v, v),
        )
        top_k = st.slider("返回条数", 1, 10, 5)
        if st.button("检索", disabled=not (query and query.strip())):
            search_kb(query.strip(), search_type, top_k)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.search_error:
            st.error(st.session_state.search_error)
        if st.session_state.search_result:
            render_search_result(st.session_state.search_result)


main()
