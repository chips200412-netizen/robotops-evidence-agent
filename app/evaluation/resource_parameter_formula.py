"""Frozen resource-parameter formula over the declared full synthetic window.

Only scoring-side local sources are read. No resource body, tool, baseline,
provider or business module is executed. Each entry consumes the verified
in-memory D1 snapshot and the verified resource-rule bytes exactly once.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import sys

from app.evaluation.context_plan import _prepare as _prepare_context
from app.evaluation.offline_runner import FairConfig


_RULES_SHA256 = "2e5ffcbec611e76772c8f771056ba06fcecbdc5f4d5b0826821e3c61f710b136"
_VERSION = "synthetic-resource-parameter-2.0.0"
_SOURCE_MAX_BYTES = 131_072
_TOOLS = ("query_simulated_telemetry", "analyze_synthetic_ros_log")
_COMMON_KEYS = {"device_model", "resource_id", "start_ms", "end_ms", "max_bytes"}
_CAP_KEYS = {"query_simulated_telemetry": "max_points", "analyze_synthetic_ros_log": "max_entries"}
_RESOURCE_KINDS = {"query_simulated_telemetry": "telemetry", "analyze_synthetic_ros_log": "ros_log"}
_PARENT = {
    "dataset_id": "robotops-synthetic-v1", "dataset_version": "synthetic-1.0.0",
    "d1_checkpoint": "33a99126720842381f156c442e586a3124c98d41",
    "d2_checkpoint": "0a0132aa6215f22f3a3e3315c4c682b03d79b20a",
    "cases_sha256": "2b0466c5895f968bf82c50f5f0040f0a627bc66e1399145d2c16d9eeb04dbfd1",
    "corpus_sha256": "f563009109aabad3a9e00606c3942cd320dae15a4a5cdce6d957fabec93eb6e9",
    "manifest_sha256": "6003197d724dd0f30ee223d230e3cdd50bd946a5f698e663530ff60f36453a87",
    "parameter_formula_sha256": "c3f4f1cb2c3863c686940ebefbc68eb31d0f7ba45b39e6ab38ece46fd05a636e",
    "context_sha256": "65be3869d835caf517b84e5858097972d805f82448966ba5ac16e7b92595973f",
}
_POLICY = {
    "window_policy": "EXACT_FULL_DECLARED_WINDOW",
    "window_interval": "START_INCLUSIVE_END_EXCLUSIVE",
    "resource_catalog_binding": "PUBLIC_DECLARATION_ONLY", "runtime_binding": "NOT_ACCEPTED",
}
_BLOCKERS = (
    "FULL_CALL_PARAMETER_CONTRACT_NOT_FROZEN", "REAL_BASELINE_ADAPTERS_NOT_ACCEPTED",
    "MODEL_SOURCE_NOT_SELECTED", "EXECUTION_AUTHORIZATION_NOT_GRANTED",
    "SYNTHETIC_RESOURCE_RUNTIME_BINDING_NOT_ACCEPTED",
)
_METRICS = (
    "top1_fault_accuracy", "retrieval_recall_at_k", "retrieval_mrr", "citation_correctness",
    "tool_selection_correctness", "tool_parameter_correctness", "dangerous_action_block_rate",
    "failure_recovery_rate", "wall_p50_ms", "wall_p95_ms", "provider_tokens", "cost_per_success",
)


def _fail() -> ValueError:
    return ValueError("invalid resource parameter formula input")


def _fixed_root() -> Path:
    # Keep lexical parents so a Windows junction cannot disappear via resolve().
    return Path(__file__).absolute().parents[2]


def _regular_stat(path: Path, *, directory: bool) -> os.stat_result:
    value = path.lstat()
    if (
        stat.S_ISLNK(value.st_mode)
        or getattr(value, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        or not (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode))
    ):
        raise _fail()
    return value


def _guard_parents(path: Path) -> None:
    for parent in reversed(path.parents):
        _regular_stat(parent, directory=True)


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _source_bytes() -> bytes:
    path = _fixed_root() / "eval" / "resource_parameter_rules_v2.json"
    _guard_parents(path)
    before = _regular_stat(path, directory=False)
    if not 1 <= before.st_size <= _SOURCE_MAX_BYTES:
        raise _fail()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or getattr(opened, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            or _identity(opened) != _identity(before)
        ):
            raise _fail()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(_SOURCE_MAX_BYTES + 1)
        _guard_parents(path)
        after = _regular_stat(path, directory=False)
        if (
            not 1 <= len(raw) <= _SOURCE_MAX_BYTES or len(raw) != before.st_size
            or _identity(after) != _identity(before) or sha256(raw).hexdigest() != _RULES_SHA256
        ):
            raise _fail()
        return raw
    finally:
        os.close(descriptor)


def _reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _fail()
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _fail()


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise _fail()
    return number


def _native_tree(value: object) -> bool:
    if type(value) is dict:
        return all(type(key) is str and _native_tree(item) for key, item in value.items())
    if type(value) is list:
        return all(_native_tree(item) for item in value)
    if type(value) is float:
        return math.isfinite(value)
    return type(value) in {str, int, bool, type(None)}


def _decode_json(raw: bytes) -> object:
    if type(raw) is not bytes:
        raise _fail()
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_pairs,
                       parse_constant=_reject_constant, parse_float=_finite_float)
    if not _native_tree(value):
        raise _fail()
    return value


def _fields(value: object, keys: set[str]) -> dict:
    if (type(value) is not dict or any(type(key) is not str for key in value)
            or set(value) != keys or not _native_tree(value)):
        raise _fail()
    return value


def _same_native(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return (all(type(key) is str for key in value) and set(value) == set(expected)
                and all(_same_native(value[key], expected[key]) for key in expected))
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _same_native(item, wanted) for item, wanted in zip(value, expected, strict=True)
        )
    return type(expected) in {str, int, bool, type(None)} and value == expected


@dataclass(frozen=True)
class _CaseRule:
    case_id: str
    resource_tool: str
    expected_arguments: tuple[tuple[str, str | int], ...]


def _compile_rules(value: object, dataset, context: dict) -> tuple[_CaseRule, ...]:
    rule = _fields(value, {
        "schema_version", "version", "status", "provenance", "license_scope",
        "publication_status", "scoring_scope", "parent", "policy", "cases",
    })
    header = {
        "schema_version": "robotops-resource-parameter-formula-v2", "version": _VERSION,
        "status": "FROZEN_LOCAL_FORMULA_ONLY", "provenance": "SELF_AUTHORED_SYNTHETIC",
        "license_scope": "LOCAL_EVALUATION_ONLY", "publication_status": "NOT_AUTHORIZED",
        "scoring_scope": "RESOURCE_PARAMETERS_ONLY", "parent": _PARENT, "policy": _POLICY,
    }
    if not _same_native({key: rule[key] for key in header}, header):
        raise _fail()
    if (dataset.dataset_id != _PARENT["dataset_id"] or dataset.version != _PARENT["dataset_version"]
            or dataset.cases_sha256 != _PARENT["cases_sha256"]
            or dataset.corpus_sha256 != _PARENT["corpus_sha256"]
            or context["version"] != "synthetic-context-2.0.0"):
        raise _fail()
    expected_cases = [case for case in dataset.cases if any(tool in _TOOLS for tool in case.gold.tools)]
    if type(rule["cases"]) is not list or len(rule["cases"]) != 18 or len(expected_cases) != 18:
        raise _fail()
    models = {row["case_id"]: row["device_model"] for row in context["cases"]}
    catalog = {row["kind"]: row for row in context["resource_catalog"]}
    compiled = []
    for row, case in zip(rule["cases"], expected_cases, strict=True):
        row = _fields(row, {"case_id", "public_input_sha256", "resource_tool", "expected_arguments", "review_status"})
        tools = [tool for tool in case.gold.tools if tool in _TOOLS]
        if len(tools) != 1:
            raise _fail()
        tool = tools[0]
        resource = catalog[_RESOURCE_KINDS[tool]]
        cap = _CAP_KEYS[tool]
        arguments = {
            "device_model": models[case.case_id], "resource_id": resource["resource_id"],
            "start_ms": resource["start_ms"], "end_ms": resource["end_ms"],
            cap: context["limits"][cap], "max_bytes": context["limits"]["max_bytes"],
        }
        expected_row = {
            "case_id": case.case_id, "public_input_sha256": sha256(case.input.encode("utf-8")).hexdigest(),
            "resource_tool": tool, "expected_arguments": arguments, "review_status": "CONTROLLER_REVIEWED_LOCAL",
        }
        if (models[case.case_id] != resource["device_model"]
                or resource["end_ms"] - resource["start_ms"] != context["limits"]["window_max_ms"]
                or not _same_native(row, expected_row)):
            raise _fail()
        compiled.append(_CaseRule(case.case_id, tool, tuple(arguments.items())))
    counts = Counter(row.resource_tool for row in compiled)
    if dict(counts) != {"query_simulated_telemetry": 1, "analyze_synthetic_ros_log": 17}:
        raise _fail()
    return tuple(compiled)


def _prepare() -> tuple[tuple[str, ...], tuple[_CaseRule, ...]]:
    try:
        dataset, _, context = _prepare_context(FairConfig(
            model_id="synthetic-resource-parameter-formula", temperature="0", max_output_tokens=512,
            timeout_ms=30000, max_tool_steps=4, repetitions=1,
        ))
        rules = _compile_rules(_decode_json(_source_bytes()), dataset, context)
        return tuple(case.case_id for case in dataset.cases), rules
    except Exception:
        raise _fail() from None


def score_resource_parameter_formula(case_id: str, tool_name: str, arguments: dict) -> dict[str, object]:
    """Compare one native candidate; no caller-selected source or tool execution."""
    try:
        if (type(case_id) is not str or type(tool_name) is not str or tool_name not in _TOOLS
                or type(arguments) is not dict or any(type(key) is not str for key in arguments)
                or set(arguments) != _COMMON_KEYS | {_CAP_KEYS[tool_name]}
                or any(type(arguments[key]) is not str for key in ("device_model", "resource_id"))
                or any(type(arguments[key]) is not int for key in ("start_ms", "end_ms", _CAP_KEYS[tool_name], "max_bytes"))):
            raise _fail()
        case_ids, rules = _prepare()
        if case_id not in case_ids:
            raise _fail()
        rule = next((row for row in rules if row.case_id == case_id), None)
        return {
            "case_id": case_id, "tool_name": tool_name, "scoring_scope": "RESOURCE_PARAMETERS_ONLY",
            "scoring_mode": "FORMULA_TEST_ONLY", "rule_version": _VERSION, "rules_sha256": _RULES_SHA256,
            "applicability": "NOT_APPLICABLE_RESOURCE_PARAMETERS_ONLY" if rule is None else "APPLICABLE_RESOURCE_PARAMETERS_ONLY",
            "correct": None if rule is None else tool_name == rule.resource_tool
                       and _same_native(arguments, dict(rule.expected_arguments)),
        }
    except Exception:
        raise _fail() from None


def build_resource_parameter_readiness_report() -> dict[str, object]:
    """Report local static coverage while every real metric remains pending."""
    case_ids, rules = _prepare()
    counts = Counter(row.resource_tool for row in rules)
    return {
        "report_kind": "LOCAL_DATA_CONTRACT_ONLY", "scoring_mode": "FORMULA_TEST_ONLY",
        "scoring_scope": "RESOURCE_PARAMETERS_ONLY", "rules_status": "RESOURCE_PARAMETER_FORMULA_READY",
        "admission_status": "BLOCKED_FOR_REAL_MEASUREMENT", "blockers": list(_BLOCKERS),
        "rule_version": _VERSION, "rules_sha256": _RULES_SHA256, **_PARENT,
        "case_count": len(case_ids), "resource_parameter_case_counts": {"applicable": len(rules), "not_applicable": len(case_ids) - len(rules)},
        "expected_resource_tool_selection_items": {
            "total": sum(counts.values()), "counts": {tool: counts[tool] for tool in _TOOLS},
            "status": "EXPECTED_SELECTION_ONLY",
        },
        "metrics": {metric: "PENDING_REAL_MEASUREMENT" for metric in _METRICS},
    }


def main() -> None:
    try:
        if sys.argv[1:]:
            raise _fail()
        report = build_resource_parameter_readiness_report()
    except Exception:
        print("RESOURCE_PARAMETER_FORMULA_INVALID", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
