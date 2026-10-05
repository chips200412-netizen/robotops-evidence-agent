"""Exact whole-call parameter math for a local fixture activity only.

All candidates are bounded and copied before the eight frozen sources are
read once. No baseline adapter, model, resource body or tool is executed.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
import sys

from app.evaluation import call_parameter_formula as _calls
from app.evaluation import context_plan as _context
from app.evaluation import resource_parameter_formula as _resources
from app.evaluation import search_query_formula as _queries
from app.evaluation.offline_runner import BASELINES, FairConfig


_CASE_IDS = tuple(f"SYN-{number:03d}" for number in range(1, 51))
_ROW_KEYS = {"baseline", "case_id", "repetition", "status", "calls"}
_ACTIVITY_MAX_NODES = 65_536
_PARENT = {
    **_calls._PARENT,
    "d4_checkpoint": "e8832e237b89f95fd3abd08aa5d03a2f2f8b89f8",
    "call_parameter_rules_sha256": "b53bbe03e5a58523838795deae0a600fba2a4a61516acb0b337aabc52945de3b",
}
_NOT_APPLICABLE = "NOT_APPLICABLE"


def _fail() -> ValueError:
    return ValueError("invalid call parameter activity input")


def _node_count(snapshot: object) -> int:
    """Count only trusted D4 snapshots, including each container and dict key."""
    if type(snapshot) is list:
        return 1 + sum(_node_count(item) for item in snapshot)
    if type(snapshot) is dict:
        return 1 + sum(1 + _node_count(item) for item in snapshot.values())
    return 1


def _snapshot_rows(rows: object, repetitions: int) -> list[dict]:
    if type(rows) is not list or len(rows) != 150 * repetitions:
        raise _fail()
    snapshots = []
    nodes = 0
    index = 0
    for repetition in range(1, repetitions + 1):
        for case_id in _CASE_IDS:
            for baseline in BASELINES:
                row = rows[index]
                index += 1
                if (
                    type(row) is not dict or any(type(key) is not str for key in row)
                    or set(row) != _ROW_KEYS
                    or type(row["baseline"]) is not str or row["baseline"] != baseline
                    or type(row["case_id"]) is not str or row["case_id"] != case_id
                    or type(row["repetition"]) is not int or row["repetition"] != repetition
                    or type(row["status"]) is not str or row["status"] not in {"COMPLETED", "ERROR"}
                ):
                    raise _fail()
                if row["status"] == "ERROR":
                    if row["calls"] is not None:
                        raise _fail()
                    calls = None
                else:
                    calls = _calls._candidate_snapshot(row["calls"])
                    # Check immediately after each bounded cell, before copying
                    # another cell; an overflow retains at most one extra cell.
                    nodes += _node_count(calls)
                    if nodes > _ACTIVITY_MAX_NODES:
                        raise _fail()
                snapshots.append({
                    "baseline": baseline, "case_id": case_id,
                    "repetition": repetition, "status": row["status"], "calls": calls,
                })
    return snapshots


def _prepare_activity(repetitions: int) -> tuple[FairConfig, tuple]:
    dataset, safe_config, context = _context._prepare(FairConfig(
        model_id="synthetic-call-parameter-activity", temperature="0", max_output_tokens=512,
        timeout_ms=30000, max_tool_steps=4, repetitions=repetitions,
    ))
    query_raw = _queries._source_bytes()
    resource_raw = _resources._source_bytes()
    call_raw = _calls._source_bytes()
    if (
        sha256(query_raw).hexdigest() != _PARENT["search_query_rules_sha256"]
        or sha256(resource_raw).hexdigest() != _PARENT["resource_parameter_rules_sha256"]
        or sha256(call_raw).hexdigest() != _PARENT["call_parameter_rules_sha256"]
    ):
        raise _fail()
    query_rules = _queries._compile_rules(_queries._decode_json(query_raw), dataset, context)
    resource_rules = _resources._compile_rules(_resources._decode_json(resource_raw), dataset, context)
    rules = _calls._compile_rules(
        _calls._decode_json(call_raw), dataset, context, query_rules, resource_rules,
    )
    return safe_config, rules


def _correct(calls: list, rule) -> bool:
    names = [call["tool_name"] for call in calls]
    return (
        len(names) == len(rule.expected_tool_names) and len(set(names)) == len(names)
        and set(names) == set(rule.expected_tool_names)
        and all(_calls._arguments_match(call, rule) for call in calls)
    )


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _aggregate_verified_fixtures(snapshots: list[dict], rules: tuple, *, repetitions: int) -> tuple[dict, dict]:
    """Apply the original activity math to copied rows and compiled rules."""
    by_case = {rule.case_id: rule for rule in rules}
    buckets = {
        baseline: {"completed": 0, "error": 0, "correct": [0] * repetitions}
        for baseline in BASELINES
    }
    for row in snapshots:
        bucket = buckets[row["baseline"]]
        bucket["completed" if row["status"] == "COMPLETED" else "error"] += 1
        rule = by_case.get(row["case_id"])
        if (rule is not None and row["status"] == "COMPLETED"
                and _correct(row["calls"], rule)):
            bucket["correct"][row["repetition"] - 1] += 1
    baselines = {}
    for baseline, bucket in buckets.items():
        correct = sum(bucket["correct"])
        denominator = 40 * repetitions
        repeat_rates = [Fraction(value, 40) for value in bucket["correct"]]
        variance: dict | str = _NOT_APPLICABLE
        if repetitions >= 2:
            mean = sum(repeat_rates, Fraction(0, 1)) / repetitions
            variance = _fraction(
                sum(((rate - mean) ** 2 for rate in repeat_rates), Fraction(0, 1)) / repetitions,
            )
        baselines[baseline] = {
            "completed_count": bucket["completed"], "error_count": bucket["error"],
            "parameter_applicable_count": denominator,
            "parameter_not_applicable_count": 10 * repetitions,
            "parameter_correct_count": correct, "parameter_failed_count": denominator - correct,
            "parameter_metric": {
                "numerator": correct, "denominator": denominator,
                "rate": _fraction(Fraction(correct, denominator)),
                "repeat_rates": [_fraction(rate) for rate in repeat_rates], "variance": variance,
            },
        }
    counts = Counter(tool for rule in rules for tool in rule.expected_tool_names)
    return baselines, {
        "total": sum(counts.values()), "counts": {tool: counts[tool] for tool in _calls._TOOLS},
        "status": "EXPECTED_SELECTION_ONLY",
    }


def aggregate_call_parameter_fixtures(rows, *, repetitions=1) -> dict[str, object]:
    """Recompute a complete native fixture grid with exact 40-case denominators."""
    try:
        if type(repetitions) is not int or not 1 <= repetitions <= 3:
            raise _fail()
        snapshots = _snapshot_rows(rows, repetitions)
        safe_config, rules = _prepare_activity(repetitions)
        baselines, expected = _aggregate_verified_fixtures(snapshots, rules, repetitions=repetitions)
        return {
            "schema_version": "robotops-call-parameter-activity-v1",
            "report_kind": "LOCAL_FORMULA_ACTIVITY_ONLY", "status": "FORMULA_TEST_ONLY",
            "scoring_scope": "FULL_CALL_PARAMETERS_ONLY", "provenance": "FIXTURE_INPUT_ONLY",
            "license_scope": "LOCAL_EVALUATION_ONLY", "publication_status": "NOT_AUTHORIZED",
            "baseline_label_role": "FIXTURE_GROUP_LABELS_ONLY", "parent": dict(_PARENT),
            "config_fingerprint": safe_config.fingerprint, "repetitions": repetitions,
            "run_count": len(snapshots), "case_count": 50,
            "expected_tool_selection_items": expected,
            "baselines": baselines, "admission_status": "BLOCKED_FOR_REAL_MEASUREMENT",
            "blockers": list(_calls._BLOCKERS),
            "metrics": {metric: "PENDING_REAL_MEASUREMENT" for metric in _calls._METRICS},
            "undeclared_freeform_status": "PENDING_UNCOVERED",
        }
    except Exception:
        raise _fail() from None


def main() -> None:
    try:
        if sys.argv[1:]:
            raise _fail()
        rows = [
            {"baseline": baseline, "case_id": case_id, "repetition": 1,
             "status": "COMPLETED", "calls": []}
            for case_id in _CASE_IDS for baseline in BASELINES
        ]
        report = {
            "fixture_profile": "EMPTY_CALLS_CONTRACT_FIXTURE",
            "report": aggregate_call_parameter_fixtures(rows),
        }
        encoded = json.dumps(report, ensure_ascii=True, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
    except Exception:
        print("CALL_PARAMETER_ACTIVITY_INVALID", file=sys.stderr)
        raise SystemExit(2) from None
    print(encoded)


if __name__ == "__main__":
    main()
