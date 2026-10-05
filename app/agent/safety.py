"""R1-1B：风险等级合同与确定性安全交接，不依赖 LLM 或外部服务。"""
from collections.abc import Mapping


RISK_LEVELS = ("低", "中", "高")
RISK_FAILURE_NOTE = "风险检查未能完成，当前安全性尚未确认，请停止操作并联系人工支持。"
LLM_FAILURE_NOTE = "智能诊断未能完成，当前安全性尚未确认，请停止操作并联系人工支持。"


def is_risk_level(value: object) -> bool:
    """外部风险结论必须是精确枚举；空字符串只允许作内部未评估值。"""
    return isinstance(value, str) and value in RISK_LEVELS


def merge_risk_levels(*levels: str) -> str:
    """只升不降；内部未评估值不覆盖已有结论，其他未知值拒绝合并。"""
    result = ""
    for level in levels:
        if isinstance(level, str) and level == "":
            continue
        if not is_risk_level(level):
            raise ValueError("Invalid risk level")
        if not result or RISK_LEVELS.index(level) > RISK_LEVELS.index(result):
            result = level
    return result


def needs_human(state: Mapping) -> bool:
    """任一安全信号均不可被普通追问或方案生成阻塞。"""
    return (
        state.get("need_human") is True
        or state.get("risk_check_failed") is True
        or state.get("llm_failed") is True
        or state.get("risk_level") == "高"
        or state.get("urgency") == "高"
    )


def human_guidance(risk_level: str, *, check_failed: bool = False) -> dict:
    """安全文案不含会触发维修关键词规则的操作描述，每次返回独立列表。"""
    if check_failed or risk_level == "高":
        diagnosis = (
            "风险检查未能完成，安全性尚未确认，需由专业人员评估。"
            if check_failed
            else "当前情况涉及安全风险，需由专业人员现场评估。"
        )
        steps = [
            "停止操作，暂停使用设备，不再尝试自行排查。",
            "远离危险源，提醒周围人员保持安全距离。",
            "联系专业技术人员或人工支持，等待现场安全评估。",
        ]
    else:
        diagnosis = "当前问题需要人工进一步确认，暂不提供自行维修方案。"
        steps = [
            "暂停自行排查，保留当前故障现象的文字记录。",
            "联系人工支持，提供设备型号和已知故障信息。",
            "等待专业人员确认后再进行后续处理。",
        ]
    return {"diagnosis": [diagnosis], "solution_steps": steps}


def fail_closed_state() -> dict:
    """检查失败视为无法确认安全，固定说明不包含底层异常或敏感参数。"""
    return {
        "risk_level": "高",
        "urgency": "高",
        "need_human": True,
        "risk_check_failed": True,
        "risk_note": RISK_FAILURE_NOTE,
        **human_guidance("高", check_failed=True),
    }


def llm_failure_state(node: str) -> dict:
    """模型失败与风险工具失败分开记录；不保留底层异常或模型内容。"""
    return {
        "risk_level": "高", "urgency": "高", "need_human": True,
        "llm_failed": True, "llm_failure_node": node,
        **human_guidance("高"),
        "diagnosis": [LLM_FAILURE_NOTE],
    }


def safety_handoff(state: Mapping) -> dict:
    """统一生成、风险、工单和回答的最终安全内容，保留并存的失败标记。"""
    update = {}
    if state.get("risk_check_failed") is True:
        update.update(fail_closed_state())
    if state.get("llm_failed") is True:
        update.update(llm_failure_state(state.get("llm_failure_node", "")))
    if not update and needs_human(state):
        update = {"need_human": True, **human_guidance(state.get("risk_level", ""))}
    return update
