"""Frozen whole-case call-parameter formula; no tool or model is executed.

The eight local sources are each read once per entry. D2 and D3 compile from
the same verified D1 objects, and scoring uses only those compiled snapshots.
Coverage describes finite phrases and declared full windows, not an agent's
natural-language accuracy or actual tool execution.
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

from app.evaluation import context_plan as _context
from app.evaluation import resource_parameter_formula as _resources
from app.evaluation import search_query_formula as _queries
from app.evaluation.offline_runner import FairConfig


_RULES_SHA256 = "b53bbe03e5a58523838795deae0a600fba2a4a61516acb0b337aabc52945de3b"
_VERSION = "synthetic-call-parameter-2.0.0"
_SOURCE_MAX_BYTES = 131_072
_CASE_IDS = tuple(f"SYN-{number:03d}" for number in range(1, 51))
_TOOLS = (
    "search_manual", "search_fault_cases", "analyze_synthetic_ros_log",
    "query_simulated_telemetry", "query_device_information",
)
_PARENT = {
    "dataset_id": "robotops-synthetic-v1", "dataset_version": "synthetic-1.0.0",
    "d1_checkpoint": "33a99126720842381f156c442e586a3124c98d41",
    "d2_checkpoint": "0a0132aa6215f22f3a3e3315c4c682b03d79b20a",
    "d3_checkpoint": "de7a38a4e70a5319b37c53a063a17f605ed4ead9",
    "cases_sha256": "2b0466c5895f968bf82c50f5f0040f0a627bc66e1399145d2c16d9eeb04dbfd1",
    "corpus_sha256": "f563009109aabad3a9e00606c3942cd320dae15a4a5cdce6d957fabec93eb6e9",
    "manifest_sha256": "6003197d724dd0f30ee223d230e3cdd50bd946a5f698e663530ff60f36453a87",
    "parameter_formula_sha256": "c3f4f1cb2c3863c686940ebefbc68eb31d0f7ba45b39e6ab38ece46fd05a636e",
    "context_sha256": "65be3869d835caf517b84e5858097972d805f82448966ba5ac16e7b92595973f",
    "search_query_rules_sha256": "46f6bd90face97c394e20df82a81c82098b23b9bab2dd74443a48d8f6d619ec8",
    "resource_parameter_rules_sha256": "2e5ffcbec611e76772c8f771056ba06fcecbdc5f4d5b0826821e3c61f710b136",
}
_POLICY = {
    "case_policy": "ALL_EXPECTED_CALLS_CORRECT", "tool_order": "UNORDERED",
    "duplicate_policy": "CASE_FALSE", "missing_or_extra_policy": "CASE_FALSE",
    "no_expected_tools": "NOT_APPLICABLE", "argument_error_policy": "CASE_FALSE_FOR_NATIVE_JSON",
    "call_envelope_invalid_policy": "FIXED_VALUE_ERROR", "max_candidate_calls": 64,
    "candidate_max_depth": 8, "candidate_max_nodes": 8192, "candidate_max_string_chars": 20000,
}
_BLOCKERS = (
    "REAL_BASELINE_ADAPTERS_NOT_ACCEPTED", "MODEL_SOURCE_NOT_SELECTED",
    "EXECUTION_AUTHORIZATION_NOT_GRANTED", "SYNTHETIC_RESOURCE_RUNTIME_BINDING_NOT_ACCEPTED",
)
_METRICS = (
    "top1_fault_accuracy", "retrieval_recall_at_k", "retrieval_mrr", "citation_correctness",
    "tool_selection_correctness", "tool_parameter_correctness", "dangerous_action_block_rate",
    "failure_recovery_rate", "wall_p50_ms", "wall_p95_ms", "provider_tokens", "cost_per_success",
)


def _fail() -> ValueError:
    return ValueError("invalid call parameter formula input")


def _fixed_root() -> Path:
    # Preserve lexical parents so symlinks and Windows junctions stay visible.
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
    path = _fixed_root() / "eval" / "call_parameter_rules_v2.json"
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


@dataclass(frozen=True)
class _CaseRule:
    case_id: str
    expected_tool_names: tuple[str, ...]
    query_rule: _queries._CaseRule | None
    resource_rule: _resources._CaseRule | None


def _compile_rules(value: object, dataset, context: dict, query_rules: tuple,
                   resource_rules: tuple) -> tuple[_CaseRule, ...]:
    rule = _fields(value, {
        "schema_version", "version", "status", "provenance", "license_scope",
        "publication_status", "scoring_scope", "parent", "policy", "cases",
    })
    header = {
        "schema_version": "robotops-call-parameter-formula-v2", "version": _VERSION,
        "status": "FROZEN_LOCAL_FORMULA_ONLY", "provenance": "SELF_AUTHORED_SYNTHETIC",
        "license_scope": "LOCAL_EVALUATION_ONLY", "publication_status": "NOT_AUTHORIZED",
        "scoring_scope": "FULL_CALL_PARAMETERS_ONLY", "parent": _PARENT, "policy": _POLICY,
    }
    if not _resources._same_native({key: rule[key] for key in header}, header):
        raise _fail()
    if (dataset.dataset_id != _PARENT["dataset_id"] or dataset.version != _PARENT["dataset_version"]
            or dataset.cases_sha256 != _PARENT["cases_sha256"]
            or dataset.corpus_sha256 != _PARENT["corpus_sha256"]
            or context["version"] != "synthetic-context-2.0.0"
            or tuple(case.case_id for case in dataset.cases) != _CASE_IDS):
        raise _fail()
    expected = [case for case in dataset.cases if case.gold.tools]
    if type(rule["cases"]) is not list or len(rule["cases"]) != 40 or len(expected) != 40:
        raise _fail()
    queries = {row.case_id: row for row in query_rules}
    resources = {row.case_id: row for row in resource_rules}
    expected_ids = {case.case_id for case in expected}
    if (len(query_rules) != 37 or len(queries) != 37 or len(resource_rules) != 18
            or len(resources) != 18 or set(queries) | set(resources) != expected_ids
            or len(set(queries) & set(resources)) != 15):
        raise _fail()
    compiled = []
    for row, case in zip(rule["cases"], expected, strict=True):
        row = _fields(row, {"case_id", "public_input_sha256", "expected_tool_names", "review_status"})
        if not _resources._same_native(row, {
            "case_id": case.case_id, "public_input_sha256": sha256(case.input.encode("utf-8")).hexdigest(),
            "expected_tool_names": list(case.gold.tools), "review_status": "CONTROLLER_REVIEWED_LOCAL",
        }):
            raise _fail()
        query = queries.get(case.case_id)
        resource = resources.get(case.case_id)
        query_tools = tuple(tool for tool in case.gold.tools if tool in _queries._TOOLS)
        resource_tools = tuple(tool for tool in case.gold.tools if tool in _resources._TOOLS)
        if (
            (query is None and query_tools) or (query is not None and query.search_tools != query_tools)
            or (resource is None and resource_tools)
            or (resource is not None and (resource.resource_tool,) != resource_tools)
            or len(query_tools) + len(resource_tools) != len(case.gold.tools)
        ):
            raise _fail()
        compiled.append(_CaseRule(case.case_id, tuple(case.gold.tools), query, resource))
    counts = Counter(tool for row in compiled for tool in row.expected_tool_names)
    if (dict(counts) != {"search_manual": 30, "search_fault_cases": 17,
                         "analyze_synthetic_ros_log": 17, "query_simulated_telemetry": 1}
            or dict(Counter(len(case.gold.tools) for case in dataset.cases)) != {0: 10, 1: 15, 2: 25}):
        raise _fail()
    return tuple(compiled)


def _prepare() -> tuple[tuple[str, ...], tuple[_CaseRule, ...]]:
    try:
        dataset, _, context = _context._prepare(FairConfig(
            model_id="synthetic-call-parameter-formula", temperature="0", max_output_tokens=512,
            timeout_ms=30000, max_tool_steps=4, repetitions=1,
        ))
        query_raw = _queries._source_bytes()
        resource_raw = _resources._source_bytes()
        if (sha256(query_raw).hexdigest() != _PARENT["search_query_rules_sha256"]
                or sha256(resource_raw).hexdigest() != _PARENT["resource_parameter_rules_sha256"]):
            raise _fail()
        query_rules = _queries._compile_rules(_queries._decode_json(query_raw), dataset, context)
        resource_rules = _resources._compile_rules(_resources._decode_json(resource_raw), dataset, context)
        rules = _compile_rules(_decode_json(_source_bytes()), dataset, context, query_rules, resource_rules)
        return tuple(case.case_id for case in dataset.cases), rules
    except Exception:
        raise _fail() from None


def _candidate_snapshot(calls: object) -> list:
    """Copy exact JSON types; count nodes and bound each string individually.

    Root calls has depth zero. Repeated references are copied separately;
    references to an ancestor are rejected as cycles before any recursion.
    """
    if type(calls) is not list or len(calls) > _POLICY["max_candidate_calls"]:
        raise _fail()
    nodes = 0
    ancestors: set[int] = set()

    def copy(value: object, depth: int) -> object:
        nonlocal nodes
        nodes += 1
        if depth > _POLICY["candidate_max_depth"] or nodes > _POLICY["candidate_max_nodes"]:
            raise _fail()
        kind = type(value)
        if kind is str:
            if len(value) > _POLICY["candidate_max_string_chars"]:
                raise _fail()
            return value
        if kind is int or kind is bool or kind is type(None):
            return value
        if kind is float:
            if not math.isfinite(value):
                raise _fail()
            return value
        if (kind is not list and kind is not dict) or id(value) in ancestors:
            raise _fail()
        ancestors.add(id(value))
        try:
            if kind is list:
                return [copy(item, depth + 1) for item in value]
            result = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise _fail()
                result[copy(key, depth + 1)] = copy(item, depth + 1)
            return result
        finally:
            ancestors.remove(id(value))

    snapshot = copy(calls, 0)
    for call in snapshot:
        if (type(call) is not dict or set(call) != {"tool_name", "arguments"}
                or type(call["tool_name"]) is not str):
            raise _fail()
    return snapshot


def _arguments_match(call: dict, rule: _CaseRule) -> bool:
    name, arguments = call["tool_name"], call["arguments"]
    if name in _queries._TOOLS:
        # The pure D2 helper alone equates 5.0 and 5; keep the native seam here.
        return (
            type(arguments) is dict and set(arguments) == {"query", "top_k"}
            and type(arguments["query"]) is str and type(arguments["top_k"]) is int
            and rule.query_rule is not None and name in rule.query_rule.search_tools
            and _queries._matches(arguments["query"], arguments["top_k"], rule.query_rule)
        )
    return (
        rule.resource_rule is not None and name == rule.resource_rule.resource_tool
        and _resources._same_native(arguments, dict(rule.resource_rule.expected_arguments))
    )


def score_call_parameter_formula(case_id: str, calls: list) -> dict[str, object]:
    """Score a whole native candidate set, with fixed errors for invalid APIs."""
    try:
        if type(case_id) is not str or case_id not in _CASE_IDS:
            raise _fail()
        snapshot = _candidate_snapshot(calls)
        case_ids, rules = _prepare()
        if case_id not in case_ids:
            raise _fail()
        rule = next((row for row in rules if row.case_id == case_id), None)
        correct = None
        if rule is not None:
            names = [call["tool_name"] for call in snapshot]
            correct = (len(names) == len(rule.expected_tool_names) and len(set(names)) == len(names)
                       and set(names) == set(rule.expected_tool_names)
                       and all(_arguments_match(call, rule) for call in snapshot))
        return {
            "case_id": case_id, "scoring_scope": "FULL_CALL_PARAMETERS_ONLY",
            "scoring_mode": "FORMULA_TEST_ONLY", "rule_version": _VERSION, "rules_sha256": _RULES_SHA256,
            "applicability": "NOT_APPLICABLE_FULL_CALL_PARAMETERS_ONLY" if rule is None
                             else "APPLICABLE_FULL_CALL_PARAMETERS_ONLY",
            "correct": correct,
        }
    except Exception:
        raise _fail() from None


def build_call_parameter_readiness_report() -> dict[str, object]:
    """Report frozen coverage, with four actual admission blockers and no scores."""
    case_ids, rules = _prepare()
    counts = Counter(tool for row in rules for tool in row.expected_tool_names)
    return {
        "report_kind": "LOCAL_DATA_CONTRACT_ONLY", "scoring_mode": "FORMULA_TEST_ONLY",
        "scoring_scope": "FULL_CALL_PARAMETERS_ONLY", "rules_status": "CALL_PARAMETER_FORMULA_READY",
        "admission_status": "BLOCKED_FOR_REAL_MEASUREMENT", "blockers": list(_BLOCKERS),
        "rule_version": _VERSION, "rules_sha256": _RULES_SHA256, **_PARENT,
        "case_count": len(case_ids),
        "call_parameter_case_counts": {"applicable": len(rules), "not_applicable": len(case_ids) - len(rules)},
        "expected_tool_selection_items": {
            "total": sum(counts.values()), "counts": {tool: counts[tool] for tool in _TOOLS},
            "status": "EXPECTED_SELECTION_ONLY",
        },
        "undeclared_freeform_status": "PENDING_UNCOVERED",
        "metrics": {metric: "PENDING_REAL_MEASUREMENT" for metric in _METRICS},
    }


def main() -> None:
    try:
        if sys.argv[1:]:
            raise _fail()
        report = build_call_parameter_readiness_report()
    except Exception:
        print("CALL_PARAMETER_FORMULA_INVALID", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
