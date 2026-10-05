"""Bound native candidate fixtures to one verified public preparation snapshot.

Admission validates only the local contract. It executes no baseline, model,
resource or scoring function, and every real measurement remains blocked.
"""

from __future__ import annotations

from hashlib import sha256
import json
import sys

from app.evaluation import call_parameter_formula as _calls
from app.evaluation import context_plan as _context
from app.evaluation import formula_scoring as _candidate
from app.evaluation.offline_runner import BASELINES, FairConfig


_CASE_IDS = tuple(f"SYN-{number:03d}" for number in range(1, 51))
_ROW_KEYS = {"baseline", "case_id", "repetition", "status", "candidate", "calls", "binding"}
_CANDIDATE_KEYS = {"fault", "ranked_evidence_ids", "citations", "tools", "safety", "recovery"}
_BINDING_KEYS = (
    "cases_sha256", "corpus_sha256", "manifest_sha256", "parameter_formula_sha256",
    "context_sha256", "context_fingerprint", "config_fingerprint", "public_input_sha256",
)
_MAX_NODES = 65_536
_MAX_BYTES = 4_194_304
_D1_CHECKPOINT = "33a99126720842381f156c442e586a3124c98d41"
_D5_CHECKPOINT = "ba304ef2425e496e92a3feb8acdf83c146e090cb"


def _fail() -> ValueError:
    return ValueError("invalid candidate contract input")


def _fields(value: object, keys: set[str]) -> dict:
    # Never invoke a subclass's methods or hash/equality of non-native keys.
    if (type(value) is not dict or len(value) != len(keys)
            or any(type(key) is not str for key in value)
            or set(value) != keys):
        raise _fail()
    return value


def _snapshot_config(value: object) -> FairConfig:
    current = _fields(value, set(_context._CONFIG_KEYS))
    if (any(type(current[key]) is not str for key in _context._CONFIG_KEYS[:2])
            or any(type(current[key]) is not int for key in _context._CONFIG_KEYS[2:])
            or len(current["temperature"]) > 128):
        raise _fail()
    config = FairConfig(**{key: current[key] for key in _context._CONFIG_KEYS})
    # Verify normalization is representable before any source read, while
    # retaining the original temperature string in the public request.
    config.fingerprint
    return config


def _strings(value: object, maximum: int, allowed: frozenset[str] | None = None) -> list[str]:
    if type(value) is not list or len(value) > maximum:
        raise _fail()
    result = []
    seen = set()
    for item in value:
        if (type(item) is not str or len(item) > 20_000 or item in seen
                or (allowed is not None and item not in allowed)):
            raise _fail()
        seen.add(item)
        result.append(item)
    return result


def _snapshot_candidate(value: object) -> dict:
    current = _fields(value, _CANDIDATE_KEYS)
    for key, allowed in (("fault", _candidate._FAULTS), ("safety", _candidate._SAFETY),
                         ("recovery", _candidate._RECOVERY)):
        if type(current[key]) is not str or current[key] not in allowed:
            raise _fail()
    return {
        "fault": current["fault"], "safety": current["safety"], "recovery": current["recovery"],
        "ranked_evidence_ids": _strings(current["ranked_evidence_ids"], 15),
        "citations": _strings(current["citations"], 15),
        "tools": _strings(current["tools"], 5, _candidate._TOOLS),
    }


def _snapshot_binding(value: object) -> dict[str, str]:
    current = _fields(value, set(_BINDING_KEYS))
    result = {}
    for key in _BINDING_KEYS:
        item = current[key]
        if (type(item) is not str or len(item) != 64
                or any(character not in "0123456789abcdef" for character in item)):
            raise _fail()
        result[key] = item
    return result


def _node_count(value: object) -> int:
    """Count already trusted cell trees; aliases have been expanded by value."""
    if type(value) is dict:
        return 1 + sum(1 + _node_count(item) for item in value.values())
    if type(value) is list:
        return 1 + sum(_node_count(item) for item in value)
    return 1


def _encoded_chunks(value: object):
    # Match D1's canonical form without retaining a full activity serialization.
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(value):
        yield chunk.encode("utf-8")


def _snapshot_rows(rows: object, repetitions: int) -> tuple[list[dict], str, str]:
    if type(rows) is not list or len(rows) != 150 * repetitions:
        raise _fail()
    snapshots = []
    nodes = 0
    size = 2  # The enclosing list's opening and closing brackets.
    records_hash = sha256(b"[")
    index = 0
    for repetition in range(1, repetitions + 1):
        for case_id in _CASE_IDS:
            for baseline in BASELINES:
                current = _fields(rows[index], _ROW_KEYS)
                if (type(current["baseline"]) is not str or current["baseline"] != baseline
                        or type(current["case_id"]) is not str or current["case_id"] != case_id
                        or type(current["repetition"]) is not int or current["repetition"] != repetition
                        or type(current["status"]) is not str
                        or current["status"] not in {"COMPLETED", "ERROR"}):
                    raise _fail()
                binding = _snapshot_binding(current["binding"])
                if current["status"] == "ERROR":
                    if current["candidate"] is not None or current["calls"] is not None:
                        raise _fail()
                    candidate = calls = None
                else:
                    candidate = _snapshot_candidate(current["candidate"])
                    nodes += _node_count(candidate)
                    if nodes > _MAX_NODES:
                        raise _fail()
                    calls = _calls._candidate_snapshot(current["calls"])
                    nodes += _node_count(calls)
                    if nodes > _MAX_NODES:
                        raise _fail()
                row = {"baseline": baseline, "case_id": case_id, "repetition": repetition,
                       "status": current["status"], "candidate": candidate,
                       "calls": calls, "binding": binding}
                if index:
                    size += 1
                    records_hash.update(b",")
                for chunk in _encoded_chunks(row):
                    size += len(chunk)
                    if size > _MAX_BYTES:
                        raise _fail()
                    records_hash.update(chunk)
                snapshots.append(row)
                index += 1
    records_hash.update(b"]")
    bindings = [{key: row[key] for key in ("baseline", "case_id", "repetition", "binding")}
                for row in snapshots]
    return snapshots, records_hash.hexdigest(), _context._digest(bindings)


def _receipt(snapshot: tuple[list[dict], str, str], dataset, config: FairConfig,
             context: dict, plan: tuple) -> dict[str, object]:
    rows, records_digest, bindings_digest = snapshot
    evidence_ids = frozenset(doc.evidence_id for doc in dataset.corpus)
    groups = {baseline: {"completed_count": 0, "error_count": 0} for baseline in BASELINES}
    for row, run in zip(rows, plan, strict=True):
        if row["binding"] != {key: getattr(run, key) for key in _BINDING_KEYS}:
            raise _fail()
        if row["status"] == "COMPLETED":
            candidate = row["candidate"]
            if any(item not in evidence_ids for key in ("ranked_evidence_ids", "citations")
                   for item in candidate[key]):
                raise _fail()
        groups[row["baseline"]]["completed_count" if row["status"] == "COMPLETED" else "error_count"] += 1
    return {
        "schema_version": "robotops-candidate-contract-v1",
        "report_kind": "LOCAL_CANDIDATE_CONTRACT_ONLY", "status": "CANDIDATE_CONTRACT_READY",
        "provenance": "FIXTURE_INPUT_ONLY", "license_scope": "LOCAL_EVALUATION_ONLY",
        "publication_status": "NOT_AUTHORIZED", "baseline_label_role": "FIXTURE_GROUP_LABELS_ONLY",
        "parent": {"d1_checkpoint": _D1_CHECKPOINT, "d5_checkpoint": _D5_CHECKPOINT,
                   "dataset_id": dataset.dataset_id, "dataset_version": dataset.version,
                   "context_version": context["version"], **_context._extension_binding()},
        "config_fingerprint": config.fingerprint,
        "context_fingerprint": _context._digest(_context._extension_binding()),
        "repetitions": config.repetitions, "record_count": len(rows),
        "completed_count": sum(group["completed_count"] for group in groups.values()),
        "error_count": sum(group["error_count"] for group in groups.values()),
        "records_sha256": records_digest, "bindings_sha256": bindings_digest,
        "baselines": groups, "admission_status": "BLOCKED_FOR_REAL_MEASUREMENT",
        "blockers": list(_calls._BLOCKERS),
        "metrics": {metric: "PENDING_REAL_MEASUREMENT" for metric in _calls._METRICS},
        "undeclared_freeform_status": "PENDING_UNCOVERED",
    }


def admit_candidate_fixtures(rows: object, *, config: object) -> dict[str, object]:
    """Admit a complete native grid, returning only safe contract metadata."""
    try:
        safe_config = _snapshot_config(config)
        snapshot = _snapshot_rows(rows, safe_config.repetitions)
        dataset, safe_config, context = _context._prepare(safe_config)
        plan = _context._make_plan(dataset, safe_config, context)
        return _receipt(snapshot, dataset, safe_config, context, plan)
    except Exception:
        raise _fail() from None


def main() -> None:
    try:
        if sys.argv[1:]:
            raise _fail()
        config = _snapshot_config({
            "model_id": "synthetic-candidate-contract", "temperature": "0",
            "max_output_tokens": 512, "timeout_ms": 30000, "max_tool_steps": 4, "repetitions": 1,
        })
        dataset, config, context = _context._prepare(config)
        plan = _context._make_plan(dataset, config, context)
        rows = [{
            "baseline": run.baseline, "case_id": run.case_id, "repetition": run.repetition,
            "status": "COMPLETED", "candidate": {
                "fault": "UNKNOWN", "ranked_evidence_ids": [], "citations": [], "tools": [],
                "safety": "UNKNOWN", "recovery": "NONE",
            }, "calls": [], "binding": {key: getattr(run, key) for key in _BINDING_KEYS},
        } for run in plan]
        result = {"fixture_profile": "UNKNOWN_EMPTY_CONTRACT_FIXTURE",
                  "receipt": _receipt(_snapshot_rows(rows, config.repetitions),
                                      dataset, config, context, plan)}
        encoded = json.dumps(result, ensure_ascii=True, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
    except Exception:
        print("CANDIDATE_CONTRACT_INVALID", file=sys.stderr)
        raise SystemExit(2) from None
    print(encoded)


if __name__ == "__main__":
    main()
