"""文档上传接口（阶段 2）+ 知识库列表/检索测试（阶段 5 切片 3）。

- POST /documents/upload：multipart 上传 → 保存 → 解析 → 切分 → 向量化 → Chroma 入库（A2-1/A2-2）
- GET  /documents：列出知识库文档列表（doc_id/filename/doc_type/chunk_count），供前端切片列表
- POST /documents/search：检索测试（关键词 → 命中列表，含相似度，A2-3 前端对接）
"""
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from app.rag import evidence_claims, evidence_reranker, hybrid_retriever, retriever
from app.rag.embeddings import embed_texts
from app.rag.loader import SUPPORTED_EXTENSIONS, load_document
from app.rag.splitter import split_text
from app.rag.vector_store import add_documents, delete_by_doc, get_chunks, list_documents
from app.schemas.evidence_search import EvidenceSearchRequest, EvidenceSearchResponse
from app.schemas.evidence_rerank import EvidenceRerankRequest, EvidenceRerankResponse
from app.schemas.evidence_claims import EvidenceClaimsRequest, EvidenceClaimsResponse

router = APIRouter(tags=["documents"])

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
MANUALS_DIR = DATA_ROOT / "manuals"
FAULT_CASES_DIR = DATA_ROOT / "fault_cases"
DOC_TYPE_DIRS = {"manual": MANUALS_DIR, "fault_case": FAULT_CASES_DIR}


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("manual"),
) -> dict:
    """上传并入库文档（PDF/DOCX/MD/TXT）：解析 → 切分 → Embedding → Chroma（A2-2）。"""
    if doc_type not in DOC_TYPE_DIRS:
        raise HTTPException(
            status_code=400,
            detail=f"doc_type 只能是 manual / fault_case，收到：{doc_type}",
        )
    original_name = file.filename or ""
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{suffix or '（无扩展名）'}（支持 PDF/DOCX/MD/TXT）",
        )

    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")

    # 保存到对应目录（uuid 前缀防文件名冲突）。目录按请求创建，避免模块导入触碰持久化路径。
    target_dir = DOC_TYPE_DIRS[doc_type]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid.uuid4().hex[:8]}_{original_name}"
    target.write_bytes(content)

    try:
        text = load_document(target)
    except Exception as e:  # 解析失败：清理已保存文件
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"文档解析失败：{e}")
    if not text.strip():
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="文档无可解析文本内容")

    # 切分 → 向量化 → 入库；任何一步失败回滚（删 chunks + 删文件）
    doc_id = uuid.uuid4().hex
    chunks = split_text(text)
    try:
        embeddings = embed_texts(chunks)
        add_documents(
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            texts=chunks,
            metadatas=[
                {
                    "doc_id": doc_id,
                    "filename": original_name,
                    "doc_type": doc_type,
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ],
            embeddings=embeddings,
        )
    except Exception as e:  # 向量化/入库失败：回滚
        delete_by_doc(doc_id)
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"文档入库失败：{e}")

    return {
        "filename": original_name,
        "saved_path": str(target),
        "doc_type": doc_type,
        "doc_id": doc_id,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "preview": chunks[0][:200] if chunks else "",
    }


class SearchRequest(BaseModel):
    """检索测试请求（阶段 5 切片 3）。"""

    query: str = Field(..., min_length=1, description="检索关键词")
    doc_type: Literal["manual", "fault_case"] | None = Field(
        default=None, description="过滤文档类型：manual / fault_case；空表示全部"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="返回命中数上限")


@router.get("/documents")
def list_kb_documents() -> dict:
    """列出知识库文档列表（阶段 5 切片 3）：{documents, total}。"""
    docs = list_documents()
    return {"documents": docs, "total": len(docs)}


@router.get("/documents/{doc_id}/chunks")
def get_document_chunks(doc_id: str) -> dict:
    """返回某文档的全部切片（阶段 5 切片 3，前端展开切片列表）。"""
    chunks = get_chunks(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"文档不存在或已无切片：{doc_id}")
    return {"doc_id": doc_id, "chunks": chunks, "count": len(chunks)}


@router.post("/documents/search")
def search_kb(req: SearchRequest) -> dict:
    """检索测试端点（阶段 5 切片 3，A2-3）：关键词 → 命中列表（含相似度）。"""
    doc_type = req.doc_type or None
    hits = retriever.retrieve(query=req.query, top_k=req.top_k, doc_type=doc_type)
    return {"query": req.query, "doc_type": doc_type, "count": len(hits), "results": hits}


@router.post(
    "/documents/evidence-search",
    response_model=EvidenceSearchResponse,
    responses={503: {"model": EvidenceSearchResponse}},
)
def evidence_search(req: EvidenceSearchRequest, response: Response) -> EvidenceSearchResponse:
    """运行可追溯双路召回；失败也保留同一 typed 顶层结构。"""
    try:
        raw_result = hybrid_retriever.search(
            query=req.query,
            doc_type=req.doc_type,
            top_k=req.top_k,
        )
    except Exception:
        raw_result = hybrid_retriever.failure_response(
            query=req.query,
            doc_type=req.doc_type,
            top_k=req.top_k,
            invalid_result=False,
        )
    try:
        result = EvidenceSearchResponse.model_validate(raw_result)
        if (
            result.query != req.query
            or result.doc_type != req.doc_type
            or result.top_k != req.top_k
        ):
            raise ValueError("response is not bound to the request")
    except Exception:
        result = EvidenceSearchResponse.model_validate(
            hybrid_retriever.failure_response(
                query=req.query,
                doc_type=req.doc_type,
                top_k=req.top_k,
                invalid_result=True,
            )
        )
    response.headers["Cache-Control"] = "no-store"
    if result.status == "FAILED":
        response.status_code = 503
    return result


@router.post(
    "/documents/evidence-rerank",
    response_model=EvidenceRerankResponse,
    responses={503: {"model": EvidenceRerankResponse}},
)
async def evidence_rerank(
    req: EvidenceRerankRequest,
    response: Response,
) -> EvidenceRerankResponse:
    """Rerank a fixed hybrid candidate window with typed, visible fallback."""
    try:
        raw_result = await evidence_reranker.rerank_evidence(
            query=req.query,
            doc_type=req.doc_type,
            top_k=req.top_k,
        )
    except evidence_reranker.InvalidRetrievalResultError:
        raw_result = evidence_reranker.failure_response(
            query=req.query,
            doc_type=req.doc_type,
            top_k=req.top_k,
            invalid_result=True,
        )
    except Exception:
        raw_result = evidence_reranker.failure_response(
            query=req.query,
            doc_type=req.doc_type,
            top_k=req.top_k,
            invalid_result=False,
        )
    try:
        result = EvidenceRerankResponse.model_validate(raw_result)
        if (
            result.query != req.query
            or result.doc_type != req.doc_type
            or result.requested_top_k != req.top_k
        ):
            raise ValueError("response is not bound to the request")
    except Exception:
        result = EvidenceRerankResponse.model_validate(
            evidence_reranker.failure_response(
                query=req.query,
                doc_type=req.doc_type,
                top_k=req.top_k,
                invalid_result=True,
            )
        )
    response.headers["Cache-Control"] = "no-store"
    if result.status == "FAILED":
        response.status_code = 503
    return result


@router.post(
    "/documents/evidence-claims",
    response_model=EvidenceClaimsResponse,
    responses={503: {"model": EvidenceClaimsResponse}},
)
async def evidence_claim_binding(
    req: EvidenceClaimsRequest,
    response: Response,
) -> EvidenceClaimsResponse:
    """Bind normalized draft claims to a server-owned strict rerank snapshot."""
    try:
        raw_result = await evidence_claims.verify_claims(
            query=req.query,
            doc_type=req.doc_type,
            draft=req.draft,
        )
    except Exception:
        raw_result = evidence_claims.failure_response(
            query=req.query,
            doc_type=req.doc_type,
            draft=req.draft,
            error_code="CLAIM_RESPONSE_INVALID",
            discarded_response=True,
        )
    try:
        raw_payload = (
            raw_result.model_dump(mode="python")
            if isinstance(raw_result, BaseModel)
            else raw_result
        )
        result = EvidenceClaimsResponse.model_validate(raw_payload)
        if (
            result.query != req.query
            or result.doc_type != req.doc_type
            or result.draft.model_dump(mode="python")
            != req.draft.model_dump(mode="python")
        ):
            raise ValueError("claim response is not bound to the request")
    except Exception:
        result = EvidenceClaimsResponse.model_validate(
            evidence_claims.failure_response(
                query=req.query,
                doc_type=req.doc_type,
                draft=req.draft,
                error_code="CLAIM_RESPONSE_INVALID",
                discarded_response=True,
            )
        )
    response.headers["Cache-Control"] = "no-store"
    if result.status == "FAILED":
        response.status_code = 503
    return result
