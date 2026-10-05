"""离线合成演示：调用实际项目门禁，不连接模型、设备或外部服务。"""

from __future__ import annotations

import json

from app.agent.execution import claim_tool_iteration
from app.agent.safety import merge_risk_levels, needs_human, safety_handoff


def run_scenarios() -> list[dict]:
    """运行五组内存场景；结果来自候选中的项目代码。"""
    risk_inputs = ["低", "高", "中", ""]
    risk_result = merge_risk_levels(*risk_inputs)

    manual_input = {"risk_level": "中", "need_human": True}
    manual_result = {**manual_input, **safety_handoff(manual_input)}

    failures = []
    for name, failure_input in (
        ("风险检查失败", {"risk_level": "低", "risk_check_failed": True}),
        (
            "模型失败标记（未调用模型）",
            {"risk_level": "低", "llm_failed": True,
             "llm_failure_node": "synthetic_generate"},
        ),
    ):
        failures.append({"case": name, "result": safety_handoff(failure_input)})

    invalid_input = "未定义风险"
    try:
        merge_risk_levels("低", invalid_input)
    except ValueError as exc:
        rejection = {"rejected": True, "error_type": type(exc).__name__}
    else:
        raise RuntimeError("合成非法输入未被项目门禁拒绝")

    budget_state = {"tool_iterations": 0, "max_tool_iterations": 4}
    budget_decisions = []
    for attempt in range(1, 6):
        claim = claim_tool_iteration(budget_state)
        budget_state.update(claim.update)
        budget_decisions.append({
            "attempt": attempt,
            "allowed": claim.allowed,
            "tool_iterations": budget_state["tool_iterations"],
        })

    return [
        {"id": "risk_monotonic", "case": "风险只升不降",
         "input": risk_inputs, "result": risk_result},
        {"id": "human_handoff", "case": "人工接管信号优先",
         "result": manual_result, "needs_human": needs_human(manual_result)},
        {"id": "failure_closed", "case": "风险检查与模型失败均安全收尾",
         "variants": failures},
        {"id": "invalid_risk", "case": "非法风险枚举被拒绝",
         "input": invalid_input, "result": rejection},
        {"id": "tool_budget", "case": "四轮预算用尽后拒绝第五轮",
         "decisions": budget_decisions, "final_state": budget_state,
         "actual_tool_calls": 0},
    ]


def main() -> None:
    print(json.dumps({
        "mode": "SYNTHETIC_OFFLINE_DEMO",
        "notice": "全新合成输入；只运行确定性门禁，无模型、设备、数据库或网络调用。",
        "metric_status": "12_FORMAL_METRICS_PENDING",
        "scenarios": run_scenarios(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
