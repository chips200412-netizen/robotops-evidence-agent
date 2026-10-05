"""仓储层（阶段 4 切片 2，A4-2/A4-3）：工单/会话消息/设备/案例 CRUD。

- 工单编号：TK + YYYYMMDD + 4 位当日序号（03 §5.2），UNIQUE 约束兜底并发；
- diagnosis/solution_steps 以 JSON 文本存取（ensure_ascii=False 保留中文）；
- 状态流转只允许向前（待处理→处理中→已解决→已关闭，03 §5.3）；
- 数据库异常统一抛 DatabaseError（connection.py）。
"""
import json
from datetime import datetime

from app.database.connection import DatabaseError, get_connection
from app.database.models import TICKET_STATUSES


def _loads(text: str | None) -> list:
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _next_ticket_seq(conn, date_str: str) -> int:
    """当日最大序号 + 1（TK{date}-XXXX）。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ticket_id FROM support_tickets "
            "WHERE ticket_id LIKE %s ORDER BY ticket_id DESC LIMIT 1",
            (f"TK{date_str}-%",),
        )
        row = cur.fetchone()
    if not row:
        return 1
    try:
        return int(row["ticket_id"].split("-")[1]) + 1
    except (IndexError, ValueError):
        return 1


# ---------- 工单 ----------

def create_ticket(
    device_model: str,
    fault_category: str,
    fault_description: str,
    diagnosis: list[str],
    solution_steps: list[str],
    risk_level: str = "低",
    status: str = "待处理",
    *,
    need_human: bool | None = None,
) -> dict:
    """创建工单（编号自动生成），返回完整工单记录。"""
    if need_human is not None and not isinstance(need_human, bool):
        raise DatabaseError("人工结论必须为布尔或未记录（None）")
    conn = get_connection()
    try:
        date_str = datetime.now().strftime("%Y%m%d")
        seq = _next_ticket_seq(conn, date_str)
        ticket_id = f"TK{date_str}-{seq:04d}"
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO support_tickets
                   (ticket_id, device_model, fault_category, fault_description,
                    diagnosis, solution_steps, risk_level, status, need_human)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    ticket_id,
                    device_model,
                    fault_category,
                    fault_description,
                    json.dumps(diagnosis, ensure_ascii=False),
                    json.dumps(solution_steps, ensure_ascii=False),
                    risk_level,
                    status,
                    need_human,
                ),
            )
        conn.commit()
        return get_ticket(ticket_id)
    except Exception as e:
        conn.rollback()
        raise DatabaseError(f"创建工单失败：{e}") from e
    finally:
        conn.close()


def _decode_ticket(row: dict) -> dict:
    """SQL 三态转换；历史缺列/NULL 未记录，不用 truthiness 猜测人工结论。"""
    row["diagnosis"] = _loads(row.get("diagnosis"))
    row["solution_steps"] = _loads(row.get("solution_steps"))
    value = row.get("need_human")
    if value is not None and (not isinstance(value, (bool, int)) or value not in (0, 1)):
        raise DatabaseError("工单人工结论数据不合法")
    row["need_human"] = None if value is None else bool(value)
    return row


def get_ticket(ticket_id: str) -> dict | None:
    """按编号查工单（诊断/步骤反序列化为列表）；不存在返回 None。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM support_tickets WHERE ticket_id=%s", (ticket_id,))
            row = cur.fetchone()
        if not row:
            return None
        return _decode_ticket(row)
    finally:
        conn.close()


def list_tickets(
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    device_model: str | None = None,
    fault_category: str | None = None,
) -> list[dict]:
    """工单列表（最新在前，可按状态/型号/故障类型过滤）。"""
    conn = get_connection()
    try:
        sql = "SELECT * FROM support_tickets WHERE 1=1"
        params: list = []
        if status:
            sql += " AND status=%s"
            params.append(status)
        if device_model:
            sql += " AND device_model=%s"
            params.append(device_model)
        if fault_category:
            sql += " AND fault_category=%s"
            params.append(fault_category)
        sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
        params += [limit, offset]
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [_decode_ticket(row) for row in rows]
    finally:
        conn.close()


def update_ticket_status(ticket_id: str, new_status: str) -> dict | None:
    """工单状态流转：只允许向前（03 §5.3）；非法/回退抛 DatabaseError。"""
    if new_status not in TICKET_STATUSES:
        raise DatabaseError(f"非法工单状态：{new_status}（可选：{'/'.join(TICKET_STATUSES)}）")
    conn = get_connection()
    try:
        ticket = get_ticket(ticket_id)
        if ticket is None:
            return None
        old_idx = TICKET_STATUSES.index(ticket["status"])
        new_idx = TICKET_STATUSES.index(new_status)
        if new_idx < old_idx:
            raise DatabaseError(
                f"状态不允许回退：{ticket['status']} → {new_status}"
            )
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE support_tickets SET status=%s WHERE ticket_id=%s",
                (new_status, ticket_id),
            )
        conn.commit()
        return get_ticket(ticket_id)
    except DatabaseError:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise DatabaseError(f"更新工单状态失败：{e}") from e
    finally:
        conn.close()


# ---------- 会话消息（A4-3） ----------

def add_message(ticket_id: str, role: str, content: str) -> None:
    """按工单编号追加一条会话消息（role: user|assistant）。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_messages (ticket_id, role, content) VALUES (%s,%s,%s)",
                (ticket_id, role, content),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise DatabaseError(f"保存会话消息失败：{e}") from e
    finally:
        conn.close()


def list_messages(ticket_id: str) -> list[dict]:
    """按工单编号查会话消息（时间正序）。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, created_at FROM conversation_messages "
                "WHERE ticket_id=%s ORDER BY id ASC",
                (ticket_id,),
            )
            return list(cur.fetchall())
    finally:
        conn.close()


# ---------- 设备 ----------

def upsert_device(model: str, name: str = "", params: str = "") -> dict:
    """按型号幂等写入/更新设备档案（model UNIQUE）。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO devices (model, name, params) VALUES (%s,%s,%s)
                   ON DUPLICATE KEY UPDATE name=VALUES(name), params=VALUES(params)""",
                (model, name, params),
            )
        conn.commit()
        return get_device(model)
    except Exception as e:
        conn.rollback()
        raise DatabaseError(f"保存设备失败：{e}") from e
    finally:
        conn.close()


def get_device(model: str) -> dict | None:
    """按型号查设备；不存在返回 None。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM devices WHERE model=%s", (model,))
            return cur.fetchone()
    finally:
        conn.close()


def list_devices(limit: int = 200) -> list[dict]:
    """设备列表（按 id 倒序，最新插入在前；编辑不改顺序）。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM devices ORDER BY id DESC LIMIT %s", (limit,))
            return list(cur.fetchall())
    finally:
        conn.close()


def delete_device(model: str) -> bool:
    """按型号删除设备；返回是否删除了记录（不存在返回 False）。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM devices WHERE model=%s", (model,))
            affected = cur.rowcount
        conn.commit()
        return affected > 0
    except Exception as e:
        conn.rollback()
        raise DatabaseError(f"删除设备失败：{e}") from e
    finally:
        conn.close()


# ---------- 案例 ----------

def create_fault_case(
    device_model: str,
    fault_type: str,
    symptom: str = "",
    cause: str = "",
    solution: str = "",
    risk_level: str = "低",
) -> dict:
    """新增一条故障案例。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO fault_cases
                   (device_model, fault_type, symptom, cause, solution, risk_level)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (device_model, fault_type, symptom, cause, solution, risk_level),
            )
            case_id = cur.lastrowid
        conn.commit()
        return get_fault_case(case_id)
    except Exception as e:
        conn.rollback()
        raise DatabaseError(f"保存故障案例失败：{e}") from e
    finally:
        conn.close()


def get_fault_case(case_id: int) -> dict | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM fault_cases WHERE id=%s", (case_id,))
            return cur.fetchone()
    finally:
        conn.close()


def list_fault_cases(
    device_model: str | None = None,
    fault_type: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """案例列表（可按型号/故障类型过滤）。"""
    conn = get_connection()
    try:
        sql = "SELECT * FROM fault_cases WHERE 1=1"
        params: list = []
        if device_model:
            sql += " AND device_model=%s"
            params.append(device_model)
        if fault_type:
            sql += " AND fault_type=%s"
            params.append(fault_type)
        sql += " ORDER BY id DESC LIMIT %s"
        params.append(limit)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()
