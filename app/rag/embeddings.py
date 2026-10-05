"""Embedding 封装（阶段 2 切片 2）：本地 sentence-transformers 模型。

用户选定方案：本地模型，离线可用、无需 API Key。
模型通过 .env 的 EMBEDDING_MODEL 配置，默认中文专用模型
BAAI/bge-small-zh-v1.5（中文检索效果优于多语言 MiniLM，512 维）。
模型懒加载，仅使用已准备的本地安全权重；诊断请求不隐式下载。
"""
from functools import lru_cache

from app.core.config import settings

EMBEDDING_MODEL = settings.embedding_model
BGE_REVISION = "7999e1d3359715c523056ef9478215996d62a620"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBEDDING_MODEL,
        trust_remote_code=False,
        local_files_only=True,
        token=False,
        revision=BGE_REVISION if EMBEDDING_MODEL == "BAAI/bge-small-zh-v1.5" else None,
        model_kwargs={"use_safetensors": True},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量文本 → 归一化向量列表。"""
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """单条查询文本 → 向量（与库内向量同分布）。"""
    return embed_texts([text])[0]
