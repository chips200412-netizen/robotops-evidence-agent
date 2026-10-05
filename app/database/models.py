"""5 张表模型与建表迁移（阶段 4 切片 1，A4-1，依据 spec/01 §9）。

表：devices / documents / support_tickets / conversation_messages / fault_cases。
建表幂等（CREATE TABLE IF NOT EXISTS），utf8mb4；列表字段（diagnosis/solution_steps）
以 JSON 文本存储，由仓储层序列化/反序列化。
"""
from app.database.connection import DatabaseError, get_connection

# 工单处理状态机（03 §5.3）：待处理 → 处理中 → 已解决 → 已关闭
TICKET_STATUSES = ("待处理", "处理中", "已解决", "已关闭")

TABLES: dict[str, str] = {
    "devices": """
        CREATE TABLE IF NOT EXISTS devices (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model VARCHAR(64) NOT NULL UNIQUE COMMENT '设备型号（规范名）',
            name VARCHAR(128) NOT NULL DEFAULT '' COMMENT '设备名称',
            params TEXT COMMENT '设备参数',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "documents": """
        CREATE TABLE IF NOT EXISTS documents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL COMMENT '原文件名',
            type VARCHAR(32) NOT NULL DEFAULT 'manual' COMMENT 'manual | fault_case',
            source VARCHAR(512) NOT NULL DEFAULT '' COMMENT '来源路径',
            chunk_count INT NOT NULL DEFAULT 0,
            uploaded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "support_tickets": """
        CREATE TABLE IF NOT EXISTS support_tickets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ticket_id VARCHAR(32) NOT NULL UNIQUE COMMENT '工单编号 TK+YYYYMMDD-XXXX（03 §5.2）',
            device_model VARCHAR(64) NOT NULL DEFAULT '',
            fault_category VARCHAR(64) NOT NULL DEFAULT '',
            fault_description TEXT,
            diagnosis TEXT COMMENT 'JSON 数组：可能原因',
            solution_steps TEXT COMMENT 'JSON 数组：解决步骤',
            risk_level VARCHAR(8) NOT NULL DEFAULT '低',
            need_human TINYINT(1) NULL DEFAULT NULL COMMENT '人工结论；NULL=未记录',
            status VARCHAR(16) NOT NULL DEFAULT '待处理',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "conversation_messages": """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ticket_id VARCHAR(32) NOT NULL COMMENT '关联工单编号（A4-3）',
            role VARCHAR(16) NOT NULL COMMENT 'user | assistant',
            content TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_ticket_id (ticket_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "fault_cases": """
        CREATE TABLE IF NOT EXISTS fault_cases (
            id INT AUTO_INCREMENT PRIMARY KEY,
            device_model VARCHAR(64) NOT NULL DEFAULT '',
            fault_type VARCHAR(64) NOT NULL DEFAULT '',
            symptom TEXT,
            cause TEXT,
            solution TEXT,
            risk_level VARCHAR(8) NOT NULL DEFAULT '低',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
}


def init_db() -> None:
    """执行全部建表语句（幂等）；任一步失败抛 DatabaseError。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for name, ddl in TABLES.items():
                cur.execute(ddl)
        conn.commit()
    except Exception as e:
        raise DatabaseError(f"建表失败：{e}") from e
    finally:
        conn.close()


def list_tables() -> list[str]:
    """返回当前库中表名列表（测试/诊断用）。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            return [list(row.values())[0] for row in cur.fetchall()]
    finally:
        conn.close()
