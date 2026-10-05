"""Admit two bounded metadata declarations against one public context snapshot.

The receipt proves declaration compatibility only. No resource body is read,
and neither declared sizes nor hashes establish content or runtime readiness.
"""

from __future__ import annotations

import json
import sys

from app.evaluation import context_plan as _context
from app.evaluation.offline_runner import FairConfig


_CATALOG_KEYS = ("kind", "resource_id", "device_model", "start_ms", "end_ms")
_ROW_KEYS = {*_CATALOG_KEYS, "size_bytes", "sha256"}
_D1_CHECKPOINT = "33a99126720842381f156c442e586a3124c98d41"
_D6_CHECKPOINT = "5f4405fcb90497818395d1865c70726a9099e516"
_BLOCKERS = (
    "REAL_BASELINE_ADAPTERS_NOT_ACCEPTED", "MODEL_SOURCE_NOT_SELECTED",
    "EXECUTION_AUTHORIZATION_NOT_GRANTED",
    "SYNTHETIC_RESOURCE_RUNTIME_BINDING_NOT_ACCEPTED",
)


def _fail() -> ValueError:
    return ValueError("invalid resource declaration input")


def _snapshot(declarations: object) -> tuple[list[dict], str]:
    if type(declarations) is not list or len(declarations) != 2:
        raise _fail()
    rows = []
    for value in declarations:
        # Check native key types before set membership, hashing or equality.
        if (type(value) is not dict or len(value) != 7
                or any(type(key) is not str for key in value)
                or set(value) != _ROW_KEYS):
            raise _fail()
        row = {key: value[key] for key in _ROW_KEYS}
        if any(type(row[key]) is not str or not 1 <= len(row[key]) <= 128
               for key in _CATALOG_KEYS[:3]):
            raise _fail()
        if (any(type(row[key]) is not int for key in ("start_ms", "end_ms", "size_bytes"))
                or not 0 <= row["start_ms"] < row["end_ms"] <= 2**63 - 1
                or not 1 <= row["size_bytes"] <= 32768):
            raise _fail()
        declared_hash = row["sha256"]
        if (type(declared_hash) is not str or len(declared_hash) != 64
                or any(character not in "0123456789abcdef" for character in declared_hash)):
            raise _fail()
        rows.append(row)
    # Validate canonical UTF-8 and detach both rows before any source read.
    return rows, _context._digest(rows)


def admit_resource_declarations(declarations: object) -> dict[str, object]:
    """Return safe contract metadata for exactly two native declarations."""
    try:
        rows, declarations_digest = _snapshot(declarations)
        dataset, _, context = _context._prepare(FairConfig(
            model_id="synthetic-resource-declaration-contract", temperature="0",
            max_output_tokens=512, timeout_ms=30000, max_tool_steps=4, repetitions=1,
        ))
        catalog = context["resource_catalog"]
        for row, resource in zip(rows, catalog, strict=True):
            if any(type(row[key]) is not type(resource[key]) or row[key] != resource[key]
                   for key in _CATALOG_KEYS):
                raise _fail()
            if row["size_bytes"] > context["limits"]["max_bytes"]:
                raise _fail()
        binding = _context._extension_binding()
        return {
            "schema_version": "robotops-resource-declaration-v1",
            "report_kind": "LOCAL_RESOURCE_DECLARATION_CONTRACT_ONLY",
            "status": "RESOURCE_DECLARATION_CONTRACT_READY",
            "provenance": "DECLARATION_FIXTURE_ONLY",
            "license_scope": "LOCAL_EVALUATION_ONLY", "publication_status": "NOT_AUTHORIZED",
            "parent": {"d1_checkpoint": _D1_CHECKPOINT, "d6_checkpoint": _D6_CHECKPOINT,
                       "dataset_id": dataset.dataset_id, "dataset_version": dataset.version,
                       "context_version": context["version"], **binding},
            "context_fingerprint": _context._digest(binding),
            "catalog_sha256": _context._digest(catalog),
            "declarations_sha256": declarations_digest,
            "resource_count": len(rows),
            "declared_total_size_bytes": sum(row["size_bytes"] for row in rows),
            "content_verification": "NOT_PERFORMED", "runtime_binding": "NOT_ACCEPTED",
            "admission_status": "BLOCKED_FOR_REAL_MEASUREMENT", "blockers": list(_BLOCKERS),
            "metrics": {metric: "PENDING_REAL_MEASUREMENT" for metric in _context._METRICS},
            "undeclared_freeform_status": "PENDING_UNCOVERED",
        }
    except Exception:
        raise _fail() from None


def main() -> None:
    try:
        if sys.argv[1:]:
            raise _fail()
        # Handwritten public metadata fixtures; the hashes are declarations.
        declarations = [
            {"kind": "telemetry", "resource_id": "telemetry-edubot-a1-v1",
             "device_model": "EduBot A1", "start_ms": 1789344000000,
             "end_ms": 1789344600000, "size_bytes": 1, "sha256": "0" * 64},
            {"kind": "ros_log", "resource_id": "roslog-edubot-a1-v1",
             "device_model": "EduBot A1", "start_ms": 1789344000000,
             "end_ms": 1789344600000, "size_bytes": 1, "sha256": "1" * 64},
        ]
        result = {"fixture_profile": "DECLARED_METADATA_FIXTURE",
                  "receipt": admit_resource_declarations(declarations)}
        encoded = json.dumps(result, ensure_ascii=True, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
    except Exception:
        print("RESOURCE_DECLARATION_INVALID", file=sys.stderr)
        raise SystemExit(2) from None
    print(encoded)


if __name__ == "__main__":
    main()
