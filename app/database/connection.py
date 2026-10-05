"""数据库连接层（阶段 4 切片 1，A4-1/A4-4）。

PyMySQL 直连（纯 Python，兼容 MySQL 8 / 5.5）；配置全部来自 .env（E4）。
连接失败抛 DatabaseError，给出明确修复指引（A4-4：不静默）。
"""
import pymysql

from app.core.config import settings


class DatabaseError(RuntimeError):
    """数据库连接/操作失败（未启动 MySQL / 配置错误等）。"""


def get_connection() -> pymysql.Connection:
    """新建一个连接（dict cursor）。调用方负责 close/with 管理。"""
    try:
        return pymysql.connect(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            database=settings.db_name,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
    except pymysql.MySQLError as e:
        raise DatabaseError(
            f"数据库连接失败（{settings.db_host}:{settings.db_port}/{settings.db_name}）：{e}。"
            "请确认 MySQL 已启动：docker compose up -d mysql，且 .env 中 DB_* 配置正确。"
        ) from e
