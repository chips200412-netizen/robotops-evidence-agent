"""智能体工具包（阶段 3 切片 5 + 阶段 4 切片 3）：6 个真实工具全部实现 + 兜底（A3-3）。

工具签名与 01 §6 / 03 §6 对齐：
search_manual / search_fault_cases / query_device_information /
create_support_ticket / get_ticket_history / check_solution_risk。

每个工具独立 try/except 兜底：单工具失败返回空结果/预设文案，不中断工作流（03 §6）。
数据库类工具（工单/历史）已接入 MySQL（阶段 4 切片 3）。
"""
from app.agent.catalog import DEVICE_INFO, rule_risk_level
from app.agent.safety import RISK_FAILURE_NOTE, is_risk_level
from app.agent.synthetic_evidence import (
    analyze_synthetic_ros_log_strict,
    query_simulated_telemetry_strict,
)
from app.database import repositories
from app.rag.retriever import retrieve


def search_manual_strict(query: str, top_k: int = 5) -> list[dict]:
    """MOD-022 只读执行器：异常上抛，由 Agent registry 统一脱敏和失败关闭。"""
    return retrieve(query, top_k=top_k, doc_type="manual")


def search_fault_cases_strict(query: str, top_k: int = 5) -> list[dict]:
    """MOD-022 只读执行器：异常上抛，由 Agent registry 统一脱敏和失败关闭。"""
    return retrieve(query, top_k=top_k, doc_type="fault_case")


def query_device_information_strict(device_model: str) -> dict:
    """MOD-022 只读执行器：返回严格设备档案合同，不吞掉编程错误。"""
    info = DEVICE_INFO.get(device_model)
    if not info:
        return {
            "found": False,
            "model": device_model,
            "message": f"设备档案中未找到「{device_model}」的型号信息",
        }
    return {"found": True, "model": device_model, **info}


def search_manual(query: str, top_k: int = 5) -> list[dict]:
    """检索设备说明书（manual 类文档）。"""
    try:
        return search_manual_strict(query, top_k=top_k)
    except Exception:
        return []


def search_fault_cases(query: str, top_k: int = 5) -> list[dict]:
    """检索历史故障案例（fault_case 类文档）。"""
    try:
        return search_fault_cases_strict(query, top_k=top_k)
    except Exception:
        return []


def query_device_information(device_model: str) -> dict:
    """查询机器人型号与参数；未收录型号返回未找到说明。"""
    try:
        return query_device_information_strict(device_model)
    except Exception:
        return {"found": False, "model": device_model, "message": "设备信息查询失败"}


def create_support_ticket(
    device_model: str,
    fault_category: str,
    fault_description: str,
    diagnosis: list[str],
    solution_steps: list[str],
    risk_level: str,
    *,
    need_human: bool | None = None,
) -> dict:
    """保存技术支持工单（真实落库 MySQL，阶段 4 切片 3）。

    03 §5.4：保存失败不静默——返回 saved=False 与原因，回答侧给出失败提示。
    成功返回：{ticket_id, saved: True, note}; 失败返回：{ticket_id: "", saved: False, note}。
    """
    try:
        row = repositories.create_ticket(
            device_model=device_model,
            fault_category=fault_category,
            fault_description=fault_description,
            diagnosis=diagnosis,
            solution_steps=solution_steps,
            risk_level=risk_level,
            need_human=need_human,
        )
        return {
            "ticket_id": row["ticket_id"],
            "saved": True,
            "note": "工单已记录（MySQL）",
        }
    except Exception as e:
        return {
            "ticket_id": "",
            "saved": False,
            "note": f"工单保存失败：{e}",
        }


def get_ticket_history(device_model: str, fault_category: str, limit: int = 5) -> list[dict]:
    """查询同型号+同故障类型的历史工单（阶段 4 切片 4 接 MySQL）。"""
    try:
        return repositories.list_tickets(
            device_model=device_model,
            fault_category=fault_category,
            limit=limit,
        )
    except Exception:
        return []


def check_solution_risk(solution_steps: list[str], fault_description: str = "") -> dict:
    """检查输入及规则结果；失败关闭，不把未知风险降为低风险（R1-ACC-002）。"""
    try:
        if (
            not isinstance(solution_steps, list)
            or not all(isinstance(step, str) for step in solution_steps)
            or not isinstance(fault_description, str)
        ):
            raise ValueError("Invalid risk check input")
        text = " ".join(solution_steps) + " " + fault_description
        level = rule_risk_level(text)
        if not is_risk_level(level):
            raise ValueError("Invalid risk check output")
        return {"risk_level": level, "ok": True}
    except Exception:
        return {"risk_level": "高", "ok": False, "need_human": True, "note": RISK_FAILURE_NOTE}
