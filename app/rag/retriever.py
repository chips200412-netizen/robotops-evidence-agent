"""检索器（阶段 2 切片 3）：向量检索 + 相似度阈值过滤。

相似度：Chroma 返回 L2 距离，归一化向量下余弦相似度 = 1 - d²/2，
结果夹取到 [0, 1]。低于阈值（默认 0.6，RETRIEVAL_SIMILARITY_THRESHOLD
可配）的命中被过滤，触发「知识库未匹配」提示（A2-5）。
"""
from app.core.config import settings
from app.rag import vector_store
from app.rag.embeddings import embed_query


def _distance_to_similarity(distance: float) -> float:
    """Chroma 的 L2 距离为平方欧氏距离（Σ(x-y)²），归一化向量下余弦相似度 = 1 - d/2。"""
    return max(0.0, 1.0 - distance / 2.0)


def retrieve(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> list[dict]:
    """检索知识库，返回按相似度降序的命中列表（低于阈值被过滤）。

    返回项：{id, filename, chunk_index, text, similarity}
    """
    embedding = embed_query(query)
    where = {"doc_type": doc_type} if doc_type else None
    hits = vector_store.query(embedding=embedding, top_k=top_k, where=where)

    threshold = settings.retrieval_similarity_threshold
    results = []
    for h in hits:
        sim = _distance_to_similarity(h["distance"])
        if sim < threshold:
            continue
        meta = h["metadata"]
        results.append(
            {
                "id": h["id"],
                "filename": meta.get("filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "text": h["text"],
                "similarity": round(sim, 4),
            }
        )
    return results
