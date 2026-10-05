"""Exact aggregate math for synthetic formula fixtures, never live metrics."""

from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction

from app.evaluation.dataset import FrozenDataset
from app.evaluation.formula_scoring import (
    FormulaCandidate, NOT_APPLICABLE, score_formula_fixture,
)
from app.evaluation.offline_runner import BASELINES, verify_frozen_dataset


_METRICS = (
    "top1", "recall_at_5", "mrr", "citation", "tool_selection",
    "safety", "dangerous_block", "recovery",
)
_UNMEASURED = (
    "tool_parameter_correctness", "wall_p50_ms", "wall_p95_ms",
    "provider_tokens", "cost_per_success",
)


@dataclass(frozen=True)
class FixtureRun:
    baseline: str
    case_id: str
    repetition: int
    status: str
    candidate: FormulaCandidate | None


@dataclass(frozen=True)
class MetricSummary:
    numerator: Fraction
    denominator: int
    rate: Fraction | str
    repeat_rates: tuple[Fraction | str, ...]
    variance: Fraction | str


@dataclass(frozen=True)
class BaselineSummary:
    completed_count: int
    error_count: int
    metrics: dict[str, MetricSummary]


@dataclass(frozen=True)
class FormulaAggregate:
    status: str
    dataset_version: str
    cases_sha256: str
    corpus_sha256: str
    repetitions: int
    run_count: int
    baselines: dict[str, BaselineSummary]
    unmeasured: dict[str, str]


def _exact_run(row: object, baseline: str, case_id: str, repetition: int) -> bool:
    return (
        type(row) is FixtureRun
        and set(vars(row)) == {field.name for field in fields(FixtureRun)}
        and type(row.baseline) is str and row.baseline == baseline
        and type(row.case_id) is str and row.case_id == case_id
        and type(row.repetition) is int and row.repetition == repetition
        and type(row.status) is str and row.status in {"COMPLETED", "ERROR"}
        and (
            (row.status == "ERROR" and row.candidate is None)
            or (row.status == "COMPLETED" and type(row.candidate) is FormulaCandidate)
        )
    )


def _eligible(case, metric: str) -> bool:
    if metric in {"recall_at_5", "mrr"}:
        return bool(case.gold.evidence_ids)
    if metric == "dangerous_block":
        return case.scenario in {"HIGH_RISK", "INJECTION"}
    if metric == "recovery":
        return case.scenario in {"TOOL_FAILURE", "RECOVERY"}
    return True


def _score_value(result, metric: str) -> Fraction:
    value = {
        "top1": result.top1_correct,
        "recall_at_5": result.recall_at_5,
        "mrr": result.mrr,
        "citation": result.citation_correct,
        "tool_selection": result.tool_selection_correct,
        "safety": result.safety_correct,
        "dangerous_block": result.dangerous_block,
        "recovery": result.recovery_correct,
    }[metric]
    if type(value) is bool:
        return Fraction(int(value), 1)
    if type(value) is Fraction:
        return value
    raise ValueError("invalid formula-test metric applicability")


def aggregate_formula_fixtures(
    dataset: FrozenDataset,
    rows: tuple[FixtureRun, ...],
    *,
    repetitions: int,
) -> FormulaAggregate:
    """Recompute every fixture score, including zero scores for error rows."""
    trusted = verify_frozen_dataset(dataset)
    return _aggregate_verified_fixtures(
        trusted, rows, repetitions=repetitions, scorer=score_formula_fixture,
    )


def _aggregate_verified_fixtures(
    trusted: FrozenDataset,
    rows: tuple[FixtureRun, ...],
    *,
    repetitions: int,
    scorer,
) -> FormulaAggregate:
    """Aggregate already verified data; each caller explicitly selects its scorer."""
    if type(repetitions) is not int or not 1 <= repetitions <= 3:
        raise ValueError("invalid formula-test repetition count")
    if type(rows) is not tuple or len(rows) != len(trusted.cases) * len(BASELINES) * repetitions:
        raise ValueError("incomplete formula-test grid")
    accum = {
        baseline: {
            "completed": 0, "error": 0,
            "numerators": {metric: [Fraction(0, 1) for _ in range(repetitions)] for metric in _METRICS},
            "denominators": {metric: [0 for _ in range(repetitions)] for metric in _METRICS},
        }
        for baseline in BASELINES
    }
    index = 0
    for repetition in range(1, repetitions + 1):
        for case in trusted.cases:
            for baseline in BASELINES:
                row = rows[index]
                index += 1
                if not _exact_run(row, baseline, case.case_id, repetition):
                    raise ValueError("invalid formula-test grid row")
                bucket = accum[baseline]
                if row.status == "COMPLETED":
                    result = scorer(trusted, case.case_id, row.candidate)
                    bucket["completed"] += 1
                else:
                    result = None
                    bucket["error"] += 1
                for metric in _METRICS:
                    if _eligible(case, metric):
                        bucket["denominators"][metric][repetition - 1] += 1
                        if result is not None:
                            bucket["numerators"][metric][repetition - 1] += _score_value(result, metric)
    summaries = {}
    for baseline in BASELINES:
        bucket = accum[baseline]
        metrics = {}
        for metric in _METRICS:
            numerators = bucket["numerators"][metric]
            denominators = bucket["denominators"][metric]
            numerator = sum(numerators, Fraction(0, 1))
            denominator = sum(denominators)
            repeat_rates = tuple(
                value / count if count else NOT_APPLICABLE
                for value, count in zip(numerators, denominators, strict=True)
            )
            rate = numerator / denominator if denominator else NOT_APPLICABLE
            variance: Fraction | str = NOT_APPLICABLE
            if repetitions >= 2 and denominator and all(type(value) is Fraction for value in repeat_rates):
                mean = sum(repeat_rates, Fraction(0, 1)) / repetitions
                variance = sum(((value - mean) ** 2 for value in repeat_rates), Fraction(0, 1)) / repetitions
            metrics[metric] = MetricSummary(numerator, denominator, rate, repeat_rates, variance)
        summaries[baseline] = BaselineSummary(bucket["completed"], bucket["error"], metrics)
    return FormulaAggregate(
        status="FORMULA_TEST_ONLY", dataset_version=trusted.version,
        cases_sha256=trusted.cases_sha256, corpus_sha256=trusted.corpus_sha256,
        repetitions=repetitions, run_count=len(rows), baselines=summaries,
        unmeasured={metric: "PENDING_REAL_MEASUREMENT" for metric in _UNMEASURED},
    )
