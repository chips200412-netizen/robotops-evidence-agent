"""LLM 调用封装（OpenAI 兼容接口，当前默认 DeepSeek）。

阶段 1 切片 2：/chat 接入真实大模型回答。
阶段 3 切片 2：新增 generate_json —— 结构化（JSON）输出，供解析/识别/分类节点使用。
密钥/模型/地址全部从 .env 读取（A1-4），可通过 OPENAI_BASE_URL /
OPENAI_MODEL 切换任意 OpenAI 兼容服务。
调用失败重试 1 次，仍失败抛 LLMError（03 §7：不静默吞错）。
"""
import json
import math
import re
import time
from typing import List

from openai import OpenAI

from app.core.config import settings
from app.observability.tracing import (
    safe_span, set_safe_llm_attempt, set_safe_outcome, set_safe_provider_usage,
)


class LLMError(RuntimeError):
    """LLM 调用失败（未配置密钥 / 网络 / 超时 / 鉴权 / JSON 解析失败等）。"""


def build_client() -> OpenAI:
    """构造 OpenAI 兼容客户端。"""
    try:
        timeout = float(settings.openai_timeout_seconds)
    except (TypeError, ValueError):
        raise LLMError("OPENAI_TIMEOUT_SECONDS 必须为有限正数") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise LLMError("OPENAI_TIMEOUT_SECONDS 必须为有限正数")
    return OpenAI(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url,
        timeout=timeout, max_retries=0,
    )


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise LLMError("模型 JSON 含重复字段")
        result[key] = value
    return result


def _invalid_constant(value: str):
    raise LLMError("模型 JSON 含非法数值")


def _load_json(text: str):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _is_explanatory_wrapper(text: str) -> bool:
    """只放行说明文字，不能把另一 JSON 值或分隔符当作说明丢弃。"""
    text = text.strip()
    if not text:
        return True
    if any(ch in text for ch in '[]{}"`'):
        return False
    # 可在说明句前接句号等自然语言标点，但不能以 JSON 分隔符/数值起始。
    prose = text.lstrip("。.!！?？ \t\r\n")
    if not prose or not prose[0].isalpha():
        return False
    return re.match(
        r"(?:true\b|false\b|null\b|NaN\b|[+-]?Infinity\b|[+-]?(?:\d|\.\d))",
        prose, re.I,
    ) is None


def _extract_json(text: str) -> str:
    """接受单个完整对象及说明包装；不从数组/残缺容器里截取内层对象。"""
    if not isinstance(text, str) or not text.strip():
        raise LLMError("模型未返回合法 JSON 对象")
    candidate = text.strip()
    try:
        data = _load_json(candidate)
    except json.JSONDecodeError:
        if "```" in candidate:
            fence = re.fullmatch(r"([^`]*)```(?:json)?\s*\n(.*?)\n?```([^`]*)", candidate, re.S)
            if not fence:
                raise LLMError("模型未返回合法 JSON 对象") from None
            prefix, candidate, suffix = fence.groups()
        else:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise LLMError("模型未返回合法 JSON 对象") from None
            prefix, suffix = candidate[:start], candidate[end + 1:]
            candidate = candidate[start:end + 1]
        # 两侧独立按同一规则检查，尾部的值/残缺分隔符也不能被剥除。
        if not all(_is_explanatory_wrapper(part) for part in (prefix, suffix)):
            raise LLMError("模型未返回合法 JSON 对象") from None
        try:
            data = _load_json(candidate)
        except json.JSONDecodeError:
            raise LLMError("模型未返回合法 JSON 对象") from None
    if not isinstance(data, dict):
        raise LLMError("模型未返回合法 JSON 对象")
    return candidate


def _request_completion(client: OpenAI, *, attempt: int, **kwargs):
    """Observe one actual SDK attempt without recording request or response text."""
    with safe_span("robotops.llm.request") as span:
        set_safe_llm_attempt(span, attempt)
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception:
            set_safe_provider_usage(span, None)
            set_safe_outcome(span, "ERROR")
            raise
        set_safe_provider_usage(span, response)
        set_safe_outcome(span, "COMPLETED")
        return response


def generate_json(
    messages: List[dict],
    max_retries: int = 1,
    temperature: float = 0.2,
    *,
    model: str | None = None,
) -> dict:
    """调用模型并要求 JSON 对象输出；解析失败或非对象时抛 LLMError。

    DeepSeek 的 json_object 模式要求 prompt 中含「json」字样（prompts.py 已保证）。
    """
    if not settings.openai_api_key:
        raise LLMError(
            "未配置 OPENAI_API_KEY：请复制 .env.example 为 .env 并填入 DeepSeek API Key"
        )

    try:
        client = build_client()
    except Exception as e:
        raise LLMError("LLM 调用失败：客户端初始化失败") from e
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = _request_completion(
                client, attempt=attempt + 1,
                model=settings.openai_model if model is None else model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if content is None or not content.strip():
                raise LLMError("模型返回空内容")
            data = json.loads(_extract_json(content))
            if not isinstance(data, dict):
                raise LLMError("模型返回的 JSON 不是对象")
            return data
        except Exception as e:  # 网络/超时/鉴权/JSON 解析失败等：均可重试 1 次
            last_err = e
            if attempt < max_retries:
                time.sleep(0.5)
    raise LLMError("LLM 调用失败：无法获取有效的 JSON 对象") from last_err


def generate_reply(messages: List[dict], max_retries: int = 1) -> str:
    """调用 OpenAI 兼容模型返回回答；失败重试 max_retries 次，仍失败抛 LLMError。

    Args:
        messages: OpenAI 格式消息列表，如 [{"role": "user", "content": "..."}]
        max_retries: 失败重试次数（默认 1，对应 03 §7「重试 1 次」）
    """
    if not settings.openai_api_key:
        raise LLMError(
            "未配置 OPENAI_API_KEY：请复制 .env.example 为 .env 并填入 DeepSeek API Key"
        )

    client = build_client()
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = _request_completion(
                client, attempt=attempt + 1,
                model=settings.openai_model,
                messages=messages,
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            if content is None or not content.strip():
                raise LLMError("模型返回空内容")
            return content.strip()
        except LLMError:
            raise
        except Exception as e:  # 网络 / 超时 / 鉴权等
            last_err = e
            if attempt < max_retries:
                time.sleep(0.5)
    raise LLMError(f"LLM 调用失败：{last_err}")
