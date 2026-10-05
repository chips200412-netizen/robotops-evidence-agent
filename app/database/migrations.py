"""R1-1C 显式增量迁移。默认只检查；--apply 只添加缺失的 need_human 列。

不在 API/import 时执行，不回填历史结论、不重建表、不回退删除字段。
MySQL DDL 隐式提交，失败不可当作事务回滚；须在授权目标上先备份后运行。
"""
import argparse

from app.database.connection import DatabaseError, get_connection


def _column_ready(cur) -> bool:
    cur.execute("SHOW COLUMNS FROM support_tickets WHERE Field='need_human'")
    row = cur.fetchone()
    if row is None:
        return False
    if (
        not {"Field", "Type", "Null", "Default", "Extra"} <= row.keys()
        or row["Field"] != "need_human"
        or row.get("Type", "").lower() not in ("tinyint", "tinyint(1)")
        or row.get("Null") != "YES"
        or row.get("Default") is not None
        or row.get("Extra", "") != ""
    ):
        raise DatabaseError("need_human 已存在但列定义不兼容；请人工核对，不自动改型")
    return True


def migrate_need_human(*, apply: bool = False) -> str:
    """返回 pending/ready/applied；只处理本次增量，重复执行无写操作。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if _column_ready(cur):
                return "ready"
            if not apply:
                return "pending"
            cur.execute(
                "ALTER TABLE support_tickets ADD COLUMN need_human "
                "TINYINT(1) NULL DEFAULT NULL COMMENT '人工结论；NULL=未记录', "
                "ALGORITHM=INSTANT"
            )
            if not _column_ready(cur):
                raise DatabaseError("need_human 迁移后校验失败")
        return "applied"
    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError("人工结论迁移失败；请检查目标库、表和 ALTER 权限，不会自动重建表") from e
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="在已授权目标库追加缺失列")
    args = parser.parse_args()
    try:
        result = migrate_need_human(apply=args.apply)
    except DatabaseError as e:
        print(f"FAIL: {e}")
        return 1
    print(f"need_human: {result}")
    return 0 if result in ("ready", "applied") else 2


if __name__ == "__main__":
    raise SystemExit(main())
