"""文本切分（阶段 2 切片 2）：LangChain RecursiveCharacterTextSplitter。"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# 分隔符按优先级：段落 → 换行 → 中文句末 → 空格
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """将长文本切分为带重叠的 chunk 列表。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
    )
    return splitter.split_text(text)
