"""文档解析器：PDF / Word / Markdown / TXT → 纯文本（阶段 2 切片 1，A2-1）。"""
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def load_pdf(path: Path) -> str:
    """解析 PDF：逐页提取文本，页间以空行分隔。"""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_docx(path: Path) -> str:
    """解析 Word：拼接段落文本（含表格单元格，docx 常规段落为主）。"""
    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs)


def load_text(path: Path) -> str:
    """解析 Markdown / TXT：按 UTF-8 读取原文。"""
    return path.read_text(encoding="utf-8")


def load_document(path: str | Path) -> str:
    """按扩展名分发解析，返回纯文本；不支持的类型抛 ValueError。"""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf(p)
    if ext == ".docx":
        return load_docx(p)
    if ext in {".md", ".markdown", ".txt"}:
        return load_text(p)
    raise ValueError(f"不支持的文档类型：{ext or '（无扩展名）'}（支持 PDF/DOCX/MD/TXT）")
