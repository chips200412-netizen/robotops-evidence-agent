"""Pure candidate rows and additive SQL contracts, not issued capabilities.

The caller owns transactions and independently sourced authority. This module
never opens a connection, advances a lifecycle, checks a clock or executes an
effect. The live service owns v8 migration, persisted lifecycle, and execution.
"""

from __future__ import annotations

import json

from app.agent.claim_sources import CONTEXT_FIELDS, _canonical, _json_copy
from app.services.effect_capabilities_v2 import (
    EFFECT_CAPABILITY_TTL_MS, MAX_MATERIAL_BYTES, validate_capability_material,
)


MAX_RECORDS_BYTES = 8_388_608
MAX_CONTEXT_BYTES = 32_768
MAX_ARGUMENTS_BYTES = 32_768
EFFECT_RECORD_V2_INVALID = "EFFECT_RECORD_V2_INVALID"
APPROVAL_COLUMNS = (
    "approval_id", "session_id", "run_id", "interrupt_id", "request_id",
    "turn_number", "context_json", "approval_kind", "execution_contract",
    "material_json", "effect_key", "binding_sha256", "arguments_sha256",
    "effect_type", "source", "nonce_digest", "issued_at", "expires_at",
    "status", "decision", "consumed_at",
)
LEDGER_COLUMNS = (
    "effect_key", "approval_id", "binding_sha256", "effect_type", "source",
    "arguments_json", "arguments_sha256", "status", "attempt_count",
    "result_json", "result_sha256", "verification_json", "verification_sha256",
    "created_at", "updated_at",
)
SCHEMA_STATEMENTS = (
    f"""
    CREATE TABLE diagnostic_claim_effect_approvals (
        approval_id TEXT PRIMARY KEY NOT NULL,
        session_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        interrupt_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        turn_number INTEGER NOT NULL CHECK (
            typeof(turn_number) = 'integer' AND turn_number BETWEEN 1 AND 1000000
        ),
        context_json TEXT NOT NULL,
        approval_kind TEXT NOT NULL CHECK (approval_kind = 'EFFECT_V2'),
        execution_contract TEXT NOT NULL CHECK (execution_contract = 'claim_effect_v2'),
        material_json TEXT NOT NULL,
        effect_key TEXT NOT NULL UNIQUE,
        binding_sha256 TEXT NOT NULL UNIQUE,
        arguments_sha256 TEXT NOT NULL,
        effect_type TEXT NOT NULL CHECK (effect_type = 'restart_simulated_robot'),
        source TEXT NOT NULL CHECK (source = 'SYNTHETIC_LOCAL'),
        nonce_digest TEXT NOT NULL UNIQUE,
        issued_at INTEGER NOT NULL CHECK (
            typeof(issued_at) = 'integer'
            AND issued_at BETWEEN 0 AND {2**63 - 1 - EFFECT_CAPABILITY_TTL_MS}
        ),
        expires_at INTEGER NOT NULL CHECK (
            typeof(expires_at) = 'integer'
            AND expires_at = issued_at + {EFFECT_CAPABILITY_TTL_MS}
        ),
        status TEXT NOT NULL CHECK (status IN ('PENDING', 'CONSUMED', 'CANCELLED', 'EXPIRED')),
        decision TEXT CHECK (decision IS NULL OR decision IN ('APPROVE', 'REJECT')),
        consumed_at INTEGER CHECK (
            consumed_at IS NULL OR (typeof(consumed_at) = 'integer' AND consumed_at >= 0)
        ),
        UNIQUE (effect_key, approval_id, binding_sha256, effect_type, source, arguments_sha256),
        FOREIGN KEY (approval_id, session_id, run_id, interrupt_id)
            REFERENCES diagnostic_interrupts(approval_id, session_id, run_id, interrupt_id),
        CHECK (
            (status = 'PENDING' AND decision IS NULL AND consumed_at IS NULL)
            OR (status = 'CONSUMED' AND decision IS NOT NULL AND consumed_at IS NOT NULL)
            OR (status IN ('CANCELLED', 'EXPIRED') AND decision IS NULL AND consumed_at IS NOT NULL)
        )
    )
    """,
    """
    CREATE TABLE diagnostic_claim_effect_ledger (
        effect_key TEXT PRIMARY KEY NOT NULL,
        approval_id TEXT NOT NULL UNIQUE,
        binding_sha256 TEXT NOT NULL,
        effect_type TEXT NOT NULL CHECK (effect_type = 'restart_simulated_robot'),
        source TEXT NOT NULL CHECK (source = 'SYNTHETIC_LOCAL'),
        arguments_json TEXT NOT NULL,
        arguments_sha256 TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('PREPARED', 'EXECUTING', 'APPLIED', 'VERIFIED',
                       'EXECUTION_FAILED', 'VERIFICATION_FAILED', 'UNKNOWN')
        ),
        attempt_count INTEGER NOT NULL CHECK (
            typeof(attempt_count) = 'integer' AND attempt_count IN (0, 1)
        ),
        result_json TEXT,
        result_sha256 TEXT,
        verification_json TEXT,
        verification_sha256 TEXT,
        created_at INTEGER NOT NULL CHECK (typeof(created_at) = 'integer' AND created_at >= 0),
        updated_at INTEGER NOT NULL CHECK (typeof(updated_at) = 'integer' AND updated_at >= created_at),
        FOREIGN KEY (effect_key, approval_id, binding_sha256, effect_type, source, arguments_sha256)
            REFERENCES diagnostic_claim_effect_approvals(
                effect_key, approval_id, binding_sha256, effect_type, source, arguments_sha256
            ),
        CHECK (
            (status = 'PREPARED' AND attempt_count = 0
                AND result_json IS NULL AND result_sha256 IS NULL
                AND verification_json IS NULL AND verification_sha256 IS NULL)
            OR (status = 'EXECUTING' AND attempt_count = 1
                AND result_json IS NULL AND result_sha256 IS NULL
                AND verification_json IS NULL AND verification_sha256 IS NULL)
            OR (status IN ('APPLIED', 'VERIFIED', 'EXECUTION_FAILED', 'VERIFICATION_FAILED', 'UNKNOWN')
                AND attempt_count = 1)
        ),
        CHECK ((result_json IS NULL) = (result_sha256 IS NULL)),
        CHECK ((verification_json IS NULL) = (verification_sha256 IS NULL))
    )
    """,
)


class EffectRecordV2Error(ValueError):
    """Stable redacted failure for invalid input or a failed pure dependency."""

    def __init__(self) -> None:
        self.code = EFFECT_RECORD_V2_INVALID
        super().__init__(EFFECT_RECORD_V2_INVALID)


def _reject() -> None:
    raise EffectRecordV2Error()


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _reject()
        result[key] = value
    return result


def _constant(_value: str) -> None:
    _reject()


def _read_json(value: object, *, maximum: int) -> dict:
    if type(value) is not str or len(value) > maximum:
        _reject()
    if len(value.encode("utf-8", errors="strict")) > maximum:
        _reject()
    parsed = json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)
    copied = _json_copy(parsed, maximum=maximum)
    if type(copied) is not dict or _canonical(copied) != value:
        _reject()
    return copied


def _rows(material: dict) -> dict:
    authority = material["proof"]["authority"]
    binding = material["binding"]
    proposal = material["proposal"]
    approval = {
        "approval_id": binding["approval_id"], "session_id": authority["session_id"],
        "run_id": authority["run_id"], "interrupt_id": binding["interrupt_id"],
        "request_id": authority["request_id"], "turn_number": authority["turn_number"],
        "context_json": _canonical({name: authority[name] for name in CONTEXT_FIELDS}),
        "approval_kind": material["approval_kind"],
        "execution_contract": binding["execution_contract"],
        "material_json": _canonical(material), "effect_key": material["effect_key"],
        "binding_sha256": material["binding_sha256"],
        "arguments_sha256": material["arguments_sha256"],
        "effect_type": proposal["effect_type"], "source": proposal["source"],
        "nonce_digest": binding["nonce_digest"], "issued_at": binding["issued_at"],
        "expires_at": binding["expires_at"], "status": "PENDING",
        "decision": None, "consumed_at": None,
    }
    ledger = {
        "effect_key": material["effect_key"], "approval_id": binding["approval_id"],
        "binding_sha256": material["binding_sha256"], "effect_type": proposal["effect_type"],
        "source": proposal["source"], "arguments_json": _canonical(proposal["arguments"]),
        "arguments_sha256": material["arguments_sha256"], "status": "PREPARED",
        "attempt_count": 0, "result_json": None, "result_sha256": None,
        "verification_json": None, "verification_sha256": None,
        "created_at": binding["issued_at"], "updated_at": binding["issued_at"],
    }
    rows = {"approval": approval, "ledger": ledger}
    _json_copy(rows, maximum=MAX_RECORDS_BYTES)
    _read_json(approval["material_json"], maximum=MAX_MATERIAL_BYTES)
    _read_json(approval["context_json"], maximum=MAX_CONTEXT_BYTES)
    _read_json(ledger["arguments_json"], maximum=MAX_ARGUMENTS_BYTES)
    return rows


def build_pending_records(material: object, *, expected_authority: object,
                          execution_contract: object, interrupt_id: object,
                          approval_id: object, expected_binding_sha256: object) -> dict:
    """Encode a fully replayed subject as unissued, initial candidate rows."""
    try:
        rebuilt = validate_capability_material(
            material, expected_authority=expected_authority,
            execution_contract=execution_contract, interrupt_id=interrupt_id,
            approval_id=approval_id, expected_binding_sha256=expected_binding_sha256,
        )
        return _rows(rebuilt)
    except EffectRecordV2Error:
        raise
    except Exception:
        raise EffectRecordV2Error() from None


def validate_pending_records(records: object, *, expected_authority: object,
                             execution_contract: object, interrupt_id: object,
                             approval_id: object, expected_binding_sha256: object) -> dict:
    """Decode only PENDING/PREPARED candidates using independently held facts."""
    try:
        raw = _json_copy(records, maximum=MAX_RECORDS_BYTES)
        if type(raw) is not dict or set(raw) != {"approval", "ledger"}:
            _reject()
        for name, columns in (("approval", APPROVAL_COLUMNS), ("ledger", LEDGER_COLUMNS)):
            if type(raw[name]) is not dict or set(raw[name]) != set(columns):
                _reject()
        material = _read_json(raw["approval"]["material_json"], maximum=MAX_MATERIAL_BYTES)
        _read_json(raw["approval"]["context_json"], maximum=MAX_CONTEXT_BYTES)
        _read_json(raw["ledger"]["arguments_json"], maximum=MAX_ARGUMENTS_BYTES)
        rebuilt = validate_capability_material(
            material, expected_authority=expected_authority,
            execution_contract=execution_contract, interrupt_id=interrupt_id,
            approval_id=approval_id, expected_binding_sha256=expected_binding_sha256,
        )
        wanted = _rows(rebuilt)
        for name in ("approval", "ledger"):
            for column, value in wanted[name].items():
                actual = raw[name][column]
                if type(actual) is not type(value) or actual != value:
                    _reject()
        return rebuilt
    except EffectRecordV2Error:
        raise
    except Exception:
        raise EffectRecordV2Error() from None
