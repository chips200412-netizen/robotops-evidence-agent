"""Recompute two existing fixture formulas from one verified source snapshot.

The candidate and whole-call columns are independent local fixture math.
They do not execute tools or establish any real measurement authority.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
import sys

from app.evaluation import call_parameter_aggregate as _call_activity
from app.evaluation import call_parameter_formula as _calls
from app.evaluation import candidate_contract as _admission
from app.evaluation import context_plan as _context
from app.evaluation import formula_aggregate as _aggregate
from app.evaluation import formula_scoring as _scoring
from app.evaluation import resource_parameter_formula as _resources
from app.evaluation import search_query_formula as _queries


_D6_CHECKPOINT = "5f4405fcb90497818395d1865c70726a9099e516"
_D7_CHECKPOINT = "e32554f869aec0ed63b63e8f19e95fbe7a780679"
_RULE_PINS = {
    key: _call_activity._PARENT[key]
    for key in ("search_query_rules_sha256", "resource_parameter_rules_sha256",
                "call_parameter_rules_sha256")
}


def _fail() -> ValueError:
    return ValueError("invalid parallel formula activity input")


def _prepare_rules(dataset, context: dict) -> tuple:
    # Keep D5's read, pin and compile order, with D1's very same objects.
    query_raw = _queries._source_bytes()
    resource_raw = _resources._source_bytes()
    call_raw = _calls._source_bytes()
    if (
        sha256(query_raw).hexdigest() != _RULE_PINS["search_query_rules_sha256"]
        or sha256(resource_raw).hexdigest() != _RULE_PINS["resource_parameter_rules_sha256"]
        or sha256(call_raw).hexdigest() != _RULE_PINS["call_parameter_rules_sha256"]
    ):
        raise _fail()
    query_rules = _queries._compile_rules(_queries._decode_json(query_raw), dataset, context)
    resource_rules = _resources._compile_rules(_resources._decode_json(resource_raw), dataset, context)
    return _calls._compile_rules(
        _calls._decode_json(call_raw), dataset, context, query_rules, resource_rules,
    )


def _exact(value):
    """Project rational arithmetic into JSON without rounding its numerator."""
    if type(value) is Fraction:
        return {"numerator": value.numerator, "denominator": value.denominator}
    if type(value) is tuple:
        return [_exact(item) for item in value]
    return value


def _activity(snapshot: tuple, dataset, config, context: dict, plan: tuple) -> dict[str, object]:
    receipt = _admission._receipt(snapshot, dataset, config, context, plan)
    rules = _prepare_rules(dataset, context)
    rows = snapshot[0]
    fixtures = tuple(_aggregate.FixtureRun(
        baseline=row["baseline"], case_id=row["case_id"], repetition=row["repetition"],
        status=row["status"], candidate=None if row["candidate"] is None else _scoring.FormulaCandidate(
            **{key: tuple(value) if type(value) is list else value
               for key, value in row["candidate"].items()},
        ),
    ) for row in rows)
    candidates = _aggregate._aggregate_verified_fixtures(
        dataset, fixtures, repetitions=config.repetitions, scorer=_scoring._score_verified_fixture,
    )
    call_rows = [{key: row[key] for key in ("baseline", "case_id", "repetition", "status", "calls")}
                 for row in rows]
    call_baselines, expected = _call_activity._aggregate_verified_fixtures(
        call_rows, rules, repetitions=config.repetitions,
    )
    baselines = {}
    for baseline, candidate in candidates.baselines.items():
        candidate_summary = {
            "completed_count": candidate.completed_count, "error_count": candidate.error_count,
            "metrics": {metric: {
                "numerator": _exact(summary.numerator), "denominator": summary.denominator,
                "rate": _exact(summary.rate), "repeat_rates": _exact(summary.repeat_rates),
                "variance": _exact(summary.variance),
            } for metric, summary in candidate.metrics.items()},
        }
        call_summary = call_baselines[baseline]
        for key in ("completed_count", "error_count"):
            if candidate_summary[key] != receipt["baselines"][baseline][key] or (
                call_summary[key] != receipt["baselines"][baseline][key]
            ):
                raise _fail()
        baselines[baseline] = {"candidate_formula": candidate_summary,
                               "call_parameter_formula": call_summary}
    return {
        "schema_version": "robotops-parallel-formula-activity-v1",
        "report_kind": "LOCAL_PARALLEL_FORMULA_ACTIVITY_ONLY", "status": "FORMULA_TEST_ONLY",
        "scoring_scope": "PARALLEL_EXISTING_FORMULAS_ONLY", "provenance": "FIXTURE_INPUT_ONLY",
        "license_scope": "LOCAL_EVALUATION_ONLY", "publication_status": "NOT_AUTHORIZED",
        "baseline_label_role": "FIXTURE_GROUP_LABELS_ONLY",
        "parent": {**receipt["parent"], "d6_checkpoint": _D6_CHECKPOINT,
                   "d7_checkpoint": _D7_CHECKPOINT, **_RULE_PINS},
        "config_fingerprint": receipt["config_fingerprint"],
        "context_fingerprint": receipt["context_fingerprint"],
        "records_sha256": receipt["records_sha256"], "bindings_sha256": receipt["bindings_sha256"],
        "repetitions": config.repetitions, "run_count": len(rows), "case_count": len(dataset.cases),
        "baselines": baselines, "expected_tool_selection_items": expected,
        "admission_status": receipt["admission_status"], "blockers": receipt["blockers"],
        "metrics": receipt["metrics"], "undeclared_freeform_status": receipt["undeclared_freeform_status"],
    }


def aggregate_parallel_formula_fixtures(rows: object, *, config: object) -> dict[str, object]:
    """Admit a complete D6 grid and recompute both existing formulas independently."""
    try:
        safe_config = _admission._snapshot_config(config)
        snapshot = _admission._snapshot_rows(rows, safe_config.repetitions)
        dataset, safe_config, context = _context._prepare(safe_config)
        plan = _context._make_plan(dataset, safe_config, context)
        return _activity(snapshot, dataset, safe_config, context, plan)
    except Exception:
        raise _fail() from None


def main() -> None:
    try:
        if sys.argv[1:]:
            raise _fail()
        config = _admission._snapshot_config({
            "model_id": "synthetic-parallel-formula-activity", "temperature": "0",
            "max_output_tokens": 512, "timeout_ms": 30000, "max_tool_steps": 4, "repetitions": 1,
        })
        dataset, config, context = _context._prepare(config)
        plan = _context._make_plan(dataset, config, context)
        rows = [{
            "baseline": run.baseline, "case_id": run.case_id, "repetition": run.repetition,
            "status": "COMPLETED", "candidate": {
                "fault": "UNKNOWN", "ranked_evidence_ids": [], "citations": [], "tools": [],
                "safety": "UNKNOWN", "recovery": "NONE",
            }, "calls": [], "binding": {key: getattr(run, key) for key in _admission._BINDING_KEYS},
        } for run in plan]
        result = {"fixture_profile": "UNKNOWN_EMPTY_PARALLEL_FORMULA_FIXTURE",
                  "report": _activity(_admission._snapshot_rows(rows, config.repetitions),
                                      dataset, config, context, plan)}
        encoded = json.dumps(result, ensure_ascii=True, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
    except Exception:
        print("PARALLEL_FORMULA_ACTIVITY_INVALID", file=sys.stderr)
        raise SystemExit(2) from None
    print(encoded)


if __name__ == "__main__":
    main()
