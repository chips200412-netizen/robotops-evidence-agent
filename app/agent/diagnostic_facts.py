"""Shared authoritative fact merging for the graph and session boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def merge_authoritative_facts(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    cumulative_query: str,
) -> dict:
    """保留已确认事实，并以服务端累计原文兜住跨轮故障描述。"""
    if not isinstance(cumulative_query, str) or not cumulative_query:
        raise ValueError("invalid cumulative query")
    merged = dict(current)
    for field in ("device_model", "fault_category"):
        old_value = previous.get(field, "")
        new_value = current.get(field, "")
        if not isinstance(old_value, str) or not isinstance(new_value, str):
            raise ValueError("invalid persisted diagnostic fact")
        if old_value and not new_value:
            merged[field] = old_value

    old_description = previous.get("fault_description", "")
    new_description = current.get("fault_description", "")
    if not isinstance(old_description, str) or not isinstance(new_description, str):
        raise ValueError("invalid persisted fault description")
    if old_description and new_description:
        if old_description in new_description:
            merged["fault_description"] = new_description
        elif new_description in old_description:
            merged["fault_description"] = old_description
        else:
            merged["fault_description"] = cumulative_query
    elif old_description:
        merged["fault_description"] = old_description
    else:
        merged["fault_description"] = new_description or cumulative_query
    return merged
