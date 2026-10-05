"""Chroma 向量库封装（阶段 2 切片 2）：持久化到 data/chroma。

集合按距离检索（默认 L2）；写入/删除均按文档级 doc_id 管理，便于回滚。
"""
from pathlib import Path

import chromadb
from chromadb.config import Settings

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
CHROMA_DIR = DATA_ROOT / "chroma"
COLLECTION_NAME = "robot_kb"

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


class _ExplicitVectorCollection(chromadb.Collection):
    """Refuse implicit embedding, including defaults persisted by older clients.

    Chroma can recover an embedding function from stored configuration/schema
    even when the Python constructor receives None. Keep the stored collection
    unchanged and reject that operation before configuration is interpreted.
    """

    def _embed(self, input, is_query=False):
        raise ValueError("automatic embedding function disabled; explicit vectors required")

    def _sparse_embed(self, input, sparse_embedding_function):
        raise ValueError("automatic embedding function disabled; explicit vectors required")

    def _get_sparse_embedding_targets(self):
        return {}


def get_collection() -> chromadb.Collection:
    """获取（或惰性创建）持久化集合。"""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )
        stored = _client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=None,
            configuration={"embedding_function": None},
        )
        _collection = _ExplicitVectorCollection(
            client=stored._client, model=stored.get_model(), embedding_function=None
        )
    return _collection


def add_documents(
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
) -> None:
    """按 chunk 写入（id 需全局唯一，metadata 含 doc_id 便于按文档删除）。"""
    get_collection().add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def query(
    embedding: list[float],
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """按向量检索，返回 [{id, text, metadata, distance}]；空库返回 []。"""
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[embedding],
        n_results=min(top_k, col.count()),
        where=where,
    )
    items = []
    for i, doc_id in enumerate(res["ids"][0]):
        items.append(
            {
                "id": doc_id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            }
        )
    return items


def delete_by_doc(doc_id: str) -> None:
    """删除某文档的全部 chunk（用于上传失败回滚）。"""
    col = get_collection()
    col.delete(where={"doc_id": doc_id})


def list_documents() -> list[dict]:
    """聚合知识库中文档元信息（阶段 5 切片 3）。

    按 metadata 的 doc_id 分组统计，返回 [{doc_id, filename, doc_type, chunk_count}]，
    按 filename 排序；空库返回 []。
    """
    col = get_collection()
    if col.count() == 0:
        return []
    data = col.get(include=["metadatas"])
    metas = data.get("metadatas") or []
    docs: dict[str, dict] = {}
    for m in metas:
        doc_id = m.get("doc_id") or ""
        if not doc_id:
            continue
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": m.get("filename", ""),
                "doc_type": m.get("doc_type", ""),
                "chunk_count": 0,
            }
        docs[doc_id]["chunk_count"] += 1
    result = list(docs.values())
    result.sort(key=lambda d: d["filename"])
    return result


def get_chunks(doc_id: str) -> list[dict]:
    """获取某文档的全部切片（阶段 5 切片 3），按 chunk_index 升序。

    返回 [{id, chunk_index, text}]；文档不存在返回 []。
    """
    col = get_collection()
    data = col.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metas = data.get("metadatas") or []
    chunks = []
    for i, cid in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        chunks.append(
            {
                "id": cid,
                "chunk_index": meta.get("chunk_index", i),
                "text": documents[i] if i < len(documents) else "",
            }
        )
    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks


def list_chunks(doc_type: str | None = None) -> list[dict]:
    """枚举原始知识切片，供 MOD-025A 词法 lane 使用。

    doc_type 过滤直接交给 Chroma，在 BM25 建语料与排名之前完成。返回值不为
    缺失 metadata 补默认身份；新 evidence 接缝会对来源字段严格失败关闭。
    """
    col = get_collection()
    if col.count() == 0:
        return []
    kwargs: dict = {"include": ["documents", "metadatas"]}
    if doc_type is not None:
        kwargs["where"] = {"doc_type": doc_type}
    data = col.get(**kwargs)
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    items = [
        {
            "id": chunk_id,
            "text": documents[index] if index < len(documents) else None,
            "metadata": metadatas[index] if index < len(metadatas) else None,
        }
        for index, chunk_id in enumerate(ids)
    ]
    items.sort(key=lambda item: str(item["id"]))
    return items
