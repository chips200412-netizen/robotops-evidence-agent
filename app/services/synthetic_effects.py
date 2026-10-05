"""MOD-024 项目内合成 effect 适配器。

该模块只修改显式传入的本地 SQLite 文件，不连接真实机器人、ROS、
网络、MySQL 或 MCP。effect 返回与独立 readback 是两个不同方法；调用方
仍必须以权威 ledger CAS 作为唯一执行授权。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.agent.catalog import DEVICE_MODELS


SYNTHETIC_EFFECT_SCHEMA_VERSION = "1"
SYNTHETIC_EFFECT_TYPE = "restart_simulated_robot"
SYNTHETIC_EFFECT_SOURCE = "SYNTHETIC_LOCAL"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class SyntheticEffectError(RuntimeError):
    """对外只暴露稳定代码，不附带路径、SQL 或底层异常。"""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SyntheticEffectNotAppliedError(SyntheticEffectError):
    """在进入本地状态修改前已明确拒绝。"""


class SyntheticEffectOutcomeUnknownError(SyntheticEffectError):
    """调用已开始且无法安全证明是否应用；不得自动重试。"""


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE synthetic_effect_meta (
        key TEXT PRIMARY KEY NOT NULL CHECK (key = 'schema_version'),
        value TEXT NOT NULL CHECK (value = '1')
    )
    """,
    """
    CREATE TABLE synthetic_device_state (
        device_model TEXT PRIMARY KEY NOT NULL,
        boot_generation INTEGER NOT NULL CHECK (boot_generation >= 0),
        last_effect_key TEXT,
        last_binding_sha256 TEXT,
        CHECK (
            (last_effect_key IS NULL AND last_binding_sha256 IS NULL)
            OR (
                last_effect_key GLOB '[0-9a-f]*'
                AND length(last_effect_key) = 64
                AND last_binding_sha256 GLOB '[0-9a-f]*'
                AND length(last_binding_sha256) = 64
            )
        )
    )
    """,
    """
    CREATE TABLE synthetic_effect_applications (
        effect_key TEXT PRIMARY KEY NOT NULL,
        binding_sha256 TEXT NOT NULL,
        device_model TEXT NOT NULL,
        expected_boot_generation_delta INTEGER NOT NULL CHECK (
            expected_boot_generation_delta = 1
        ),
        boot_generation_before INTEGER NOT NULL CHECK (
            boot_generation_before >= 0
        ),
        boot_generation_after INTEGER NOT NULL CHECK (
            boot_generation_after = boot_generation_before + 1
        ),
        result_json TEXT NOT NULL,
        result_sha256 TEXT NOT NULL,
        FOREIGN KEY (device_model)
            REFERENCES synthetic_device_state(device_model)
    )
    """,
)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SyntheticEffectNotAppliedError("SYNTHETIC_EFFECT_INPUT_INVALID") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _HEX_64.fullmatch(value) is not None


def _validate_call(
    *,
    effect_key: object,
    binding_sha256: object,
    device_model: object,
    expected_boot_generation_delta: object,
) -> tuple[str, str, str]:
    if (
        not _valid_digest(effect_key)
        or not _valid_digest(binding_sha256)
        or not isinstance(device_model, str)
        or device_model not in DEVICE_MODELS
        or len(device_model) > 200
        or type(expected_boot_generation_delta) is not int
        or expected_boot_generation_delta != 1
    ):
        raise SyntheticEffectNotAppliedError("SYNTHETIC_EFFECT_INPUT_INVALID")
    return effect_key, binding_sha256, device_model


def _validate_read_device(device_model: object) -> str:
    if (
        not isinstance(device_model, str)
        or device_model not in DEVICE_MODELS
        or len(device_model) > 200
    ):
        raise SyntheticEffectNotAppliedError("SYNTHETIC_EFFECT_INPUT_INVALID")
    return device_model


def _validate_result(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "effect_type",
        "source",
        "device_model",
        "effect_key",
        "binding_sha256",
        "boot_generation_before",
        "boot_generation_after",
        "applied",
    }:
        raise SyntheticEffectOutcomeUnknownError("SYNTHETIC_EFFECT_RESULT_INVALID")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("effect_type") != SYNTHETIC_EFFECT_TYPE
        or value.get("source") != SYNTHETIC_EFFECT_SOURCE
        or not isinstance(value.get("device_model"), str)
        or not _valid_digest(value.get("effect_key"))
        or not _valid_digest(value.get("binding_sha256"))
        or type(value.get("boot_generation_before")) is not int
        or type(value.get("boot_generation_after")) is not int
        or value["boot_generation_before"] < 0
        or value["boot_generation_after"] != value["boot_generation_before"] + 1
        or value.get("applied") is not True
    ):
        raise SyntheticEffectOutcomeUnknownError("SYNTHETIC_EFFECT_RESULT_INVALID")
    return dict(value)


class SyntheticEffectStore:
    """单进程本地合成设备状态。

    `effect_key` 去重是 ledger CAS 之外的第二道防线，不取代上层授权。
    """

    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        try:
            path = Path(db_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = path
            self._connection = sqlite3.connect(
                str(path), check_same_thread=False, timeout=5.0
            )
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._setup_schema()
        except Exception as exc:
            self.close()
            if isinstance(exc, SyntheticEffectError):
                raise
            raise SyntheticEffectError("SYNTHETIC_EFFECT_STORE_INVALID") from exc

    @property
    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise SyntheticEffectError("SYNTHETIC_EFFECT_STORE_CLOSED")
        return self._connection

    @staticmethod
    def _stored_ddl(statement: str) -> str:
        return statement.strip().rstrip(";")

    def _setup_schema(self) -> None:
        db = self._db
        try:
            db.execute("BEGIN IMMEDIATE")
            objects = db.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall()
            if not objects:
                for statement in _SCHEMA_STATEMENTS:
                    db.execute(statement)
                db.execute(
                    "INSERT INTO synthetic_effect_meta (key, value) "
                    "VALUES ('schema_version', ?)",
                    (SYNTHETIC_EFFECT_SCHEMA_VERSION,),
                )
            self._validate_schema_locked()
            db.commit()
        except Exception as exc:
            try:
                db.rollback()
            except Exception:
                pass
            if isinstance(exc, SyntheticEffectError):
                raise
            raise SyntheticEffectError("SYNTHETIC_EFFECT_STORE_INVALID") from exc

    def _validate_schema_locked(self) -> None:
        db = self._db
        expected_tables = {
            "synthetic_effect_meta",
            "synthetic_device_state",
            "synthetic_effect_applications",
        }
        objects = db.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        if set(objects) != {("table", name) for name in expected_tables}:
            raise SyntheticEffectError("SYNTHETIC_EFFECT_STORE_INVALID")
        for statement, table in zip(_SCHEMA_STATEMENTS, (
            "synthetic_effect_meta",
            "synthetic_device_state",
            "synthetic_effect_applications",
        )):
            row = db.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if row is None or row[0] != self._stored_ddl(statement):
                raise SyntheticEffectError("SYNTHETIC_EFFECT_STORE_INVALID")
        meta = db.execute(
            "SELECT key, value FROM synthetic_effect_meta"
        ).fetchall()
        if meta != [("schema_version", SYNTHETIC_EFFECT_SCHEMA_VERSION)]:
            raise SyntheticEffectError("SYNTHETIC_EFFECT_STORE_INVALID")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise SyntheticEffectError("SYNTHETIC_EFFECT_STORE_INVALID")

    def restart_simulated_robot(
        self,
        *,
        effect_key: str,
        binding_sha256: str,
        device_model: str,
        expected_boot_generation_delta: int,
    ) -> dict[str, Any]:
        effect_key, binding_sha256, device_model = _validate_call(
            effect_key=effect_key,
            binding_sha256=binding_sha256,
            device_model=device_model,
            expected_boot_generation_delta=expected_boot_generation_delta,
        )
        db = self._db
        with self._lock:
            started = False
            try:
                db.execute("BEGIN IMMEDIATE")
                started = True
                existing = db.execute(
                    """
                    SELECT binding_sha256, device_model,
                           expected_boot_generation_delta,
                           boot_generation_before, boot_generation_after,
                           result_json, result_sha256
                    FROM synthetic_effect_applications WHERE effect_key = ?
                    """,
                    (effect_key,),
                ).fetchone()
                if existing is not None:
                    if (
                        existing[0] != binding_sha256
                        or existing[1] != device_model
                        or type(existing[2]) is not int
                        or existing[2] != expected_boot_generation_delta
                    ):
                        raise SyntheticEffectNotAppliedError(
                            "SYNTHETIC_EFFECT_BINDING_CONFLICT"
                        )
                    parsed = json.loads(str(existing[5]))
                    result = _validate_result(parsed)
                    if (
                        canonical_json(result) != str(existing[5])
                        or canonical_sha256(result) != str(existing[6])
                        or result["boot_generation_before"] != existing[3]
                        or result["boot_generation_after"] != existing[4]
                        or result["effect_key"] != effect_key
                        or result["binding_sha256"] != binding_sha256
                        or result["device_model"] != device_model
                    ):
                        raise SyntheticEffectOutcomeUnknownError(
                            "SYNTHETIC_EFFECT_RESULT_INVALID"
                        )
                    db.commit()
                    return result

                state = db.execute(
                    """
                    SELECT boot_generation, last_effect_key, last_binding_sha256
                    FROM synthetic_device_state WHERE device_model = ?
                    """,
                    (device_model,),
                ).fetchone()
                if state is None:
                    before = 0
                    inserted = db.execute(
                        """
                        INSERT INTO synthetic_device_state (
                            device_model, boot_generation,
                            last_effect_key, last_binding_sha256
                        ) VALUES (?, 0, NULL, NULL)
                        """,
                        (device_model,),
                    ).rowcount
                    if inserted != 1:
                        raise SyntheticEffectOutcomeUnknownError(
                            "SYNTHETIC_EFFECT_STORE_ERROR"
                        )
                else:
                    if (
                        type(state[0]) is not int
                        or state[0] < 0
                        or (state[1] is None) != (state[2] is None)
                        or (state[1] is not None and not _valid_digest(state[1]))
                        or (state[2] is not None and not _valid_digest(state[2]))
                    ):
                        raise SyntheticEffectOutcomeUnknownError(
                            "SYNTHETIC_EFFECT_STORE_ERROR"
                        )
                    before = state[0]
                after = before + 1
                result = {
                    "schema_version": 1,
                    "effect_type": SYNTHETIC_EFFECT_TYPE,
                    "source": SYNTHETIC_EFFECT_SOURCE,
                    "device_model": device_model,
                    "effect_key": effect_key,
                    "binding_sha256": binding_sha256,
                    "boot_generation_before": before,
                    "boot_generation_after": after,
                    "applied": True,
                }
                result_json = canonical_json(result)
                result_sha256 = canonical_sha256(result)
                changed = db.execute(
                    """
                    UPDATE synthetic_device_state
                    SET boot_generation = ?, last_effect_key = ?,
                        last_binding_sha256 = ?
                    WHERE device_model = ? AND boot_generation = ?
                    """,
                    (after, effect_key, binding_sha256, device_model, before),
                ).rowcount
                inserted = db.execute(
                    """
                    INSERT INTO synthetic_effect_applications (
                        effect_key, binding_sha256, device_model,
                        expected_boot_generation_delta,
                        boot_generation_before, boot_generation_after,
                        result_json, result_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        effect_key,
                        binding_sha256,
                        device_model,
                        expected_boot_generation_delta,
                        before,
                        after,
                        result_json,
                        result_sha256,
                    ),
                ).rowcount
                if (changed, inserted) != (1, 1):
                    raise SyntheticEffectOutcomeUnknownError(
                        "SYNTHETIC_EFFECT_STORE_ERROR"
                    )
                db.commit()
                return result
            except SyntheticEffectNotAppliedError:
                try:
                    db.rollback()
                except Exception:
                    pass
                raise
            except Exception as exc:
                try:
                    db.rollback()
                except Exception:
                    pass
                if isinstance(exc, SyntheticEffectOutcomeUnknownError):
                    raise
                if not started:
                    raise SyntheticEffectNotAppliedError(
                        "SYNTHETIC_EFFECT_STORE_ERROR"
                    ) from exc
                raise SyntheticEffectOutcomeUnknownError(
                    "SYNTHETIC_EFFECT_OUTCOME_UNKNOWN"
                ) from exc

    def read_device_state(self, *, device_model: str) -> dict[str, Any]:
        device_model = _validate_read_device(device_model)
        with self._lock:
            try:
                row = self._db.execute(
                    """
                    SELECT boot_generation, last_effect_key, last_binding_sha256
                    FROM synthetic_device_state WHERE device_model = ?
                    """,
                    (device_model,),
                ).fetchone()
            except Exception as exc:
                raise SyntheticEffectOutcomeUnknownError(
                    "SYNTHETIC_EFFECT_READBACK_UNKNOWN"
                ) from exc
        if row is None:
            generation = 0
            effect_key = None
            binding_sha256 = None
        else:
            generation, effect_key, binding_sha256 = row
            if (
                type(generation) is not int
                or generation < 0
                or (effect_key is None) != (binding_sha256 is None)
                or (effect_key is not None and not _valid_digest(effect_key))
                or (binding_sha256 is not None and not _valid_digest(binding_sha256))
            ):
                raise SyntheticEffectOutcomeUnknownError(
                    "SYNTHETIC_EFFECT_READBACK_UNKNOWN"
                )
        return {
            "schema_version": 1,
            "source": SYNTHETIC_EFFECT_SOURCE,
            "device_model": device_model,
            "boot_generation": generation,
            "last_effect_key": effect_key,
            "last_binding_sha256": binding_sha256,
        }

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def __enter__(self) -> "SyntheticEffectStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
