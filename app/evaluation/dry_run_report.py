"""Reproducible metadata report for inert three-baseline contract runs.

No synthetic stub candidate is scored as a model prediction. This module
imports no provider, observability exporter, database, or production corpus.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
import sys

from app.evaluation.dataset import FrozenDataset, load_frozen_dataset
from app.evaluation.offline_runner import (
    BASELINES, CandidateResult, FairConfig, LocalSyntheticAdapter,
    PlannedRun, RawRun, build_fair_plan, run_offline_campaign,
)


PENDING = "PENDING_REAL_MEASUREMENT"
_METRICS = (
    "top1_fault_accuracy", "retrieval_recall_at_k", "retrieval_mrr",
    "citation_correctness", "tool_selection_correctness",
    "tool_parameter_correctness", "dangerous_action_block_rate",
    "failure_recovery_rate", "wall_p50_ms", "wall_p95_ms",
    "provider_tokens", "cost_per_success",
)
_STUB_CANDIDATE = CandidateResult(
    fault="UNKNOWN", evidence_ids=(), tools=(),
    safety="UNKNOWN", recovery="NONE",
)


def _exact_record(value: object, expected_type: type) -> bool:
    return type(value) is expected_type and set(vars(value)) == {
        field.name for field in fields(expected_type)
    }


def _valid_stub_candidate(value: object) -> bool:
    return (
        _exact_record(value, CandidateResult)
        and type(value.fault) is str and value.fault == _STUB_CANDIDATE.fault
        and type(value.evidence_ids) is tuple and value.evidence_ids == ()
        and type(value.tools) is tuple and value.tools == ()
        and type(value.safety) is str and value.safety == _STUB_CANDIDATE.safety
        and type(value.recovery) is str and value.recovery == _STUB_CANDIDATE.recovery
    )


def _valid_row_metadata(row: RawRun, planned: PlannedRun) -> bool:
    return (
        type(row.baseline) is str and row.baseline == planned.baseline
        and type(row.case_id) is str and row.case_id == planned.case_id
        and type(row.repetition) is int and row.repetition == planned.repetition
        and type(row.config_fingerprint) is str
        and row.config_fingerprint == planned.config_fingerprint
        and type(row.dataset_id) is str and row.dataset_id == planned.dataset_id
        and type(row.dataset_version) is str and row.dataset_version == planned.dataset_version
        and type(row.cases_sha256) is str and row.cases_sha256 == planned.cases_sha256
        and type(row.corpus_sha256) is str and row.corpus_sha256 == planned.corpus_sha256
        and type(row.source) is str and row.source == "SYNTHETIC_STUB"
        and type(row.execution_mode) is str and row.execution_mode == "OFFLINE_SYNTHETIC"
        and type(row.measurement_status) is str
        and row.measurement_status == "DRY_RUN_NOT_MODEL_MEASURED"
        and type(row.status) is str and row.status in {"COMPLETED", "ERROR"}
    )


def build_dry_run_report(
    dataset: FrozenDataset,
    config: FairConfig,
    runs: tuple[RawRun, ...],
) -> dict[str, object]:
    """Fail closed on raw-run drift and keep every model metric pending."""
    expected = build_fair_plan(dataset, config)
    if type(runs) is not tuple or len(runs) != len(expected):
        raise ValueError("incomplete offline dry-run campaign")
    counts = {baseline: {"COMPLETED": 0, "ERROR": 0} for baseline in BASELINES}
    for planned, row in zip(expected, runs, strict=True):
        if (
            not _exact_record(row, RawRun)
            or not _valid_row_metadata(row, planned)
            or (row.status == "COMPLETED" and not _valid_stub_candidate(row.candidate))
            or (row.status == "ERROR" and row.candidate is not None)
        ):
            raise ValueError("invalid offline dry-run result")
        counts[row.baseline][row.status] += 1
    return {
        "report_kind": "LOCAL_CONTRACT_ONLY",
        "measurement_status": PENDING,
        "display_status_zh": "待实测",
        "dataset_id": expected[0].dataset_id,
        "dataset_version": expected[0].dataset_version,
        "cases_sha256": expected[0].cases_sha256,
        "corpus_sha256": expected[0].corpus_sha256,
        "config_fingerprint": expected[0].config_fingerprint,
        "fair_config": {
            "model_id": config.model_id,
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "timeout_ms": config.timeout_ms,
            "max_tool_steps": config.max_tool_steps,
            "repetitions": config.repetitions,
        },
        "case_count": len(dataset.cases),
        "repetitions": config.repetitions,
        "run_count": len(runs),
        "baseline_counts": counts,
        "metrics": {name: PENDING for name in _METRICS},
    }


def main() -> None:
    """Run a deterministic local contract smoke test and print one JSON line."""
    try:
        dataset = load_frozen_dataset(Path(__file__).resolve().parents[2] / "eval")
        config = FairConfig.from_mapping({
            "model_id": "synthetic-model", "temperature": "0",
            "max_output_tokens": 512, "timeout_ms": 30000,
            "max_tool_steps": 4, "repetitions": 1,
        })
        adapters = tuple(LocalSyntheticAdapter(baseline) for baseline in BASELINES)
        report = build_dry_run_report(dataset, config, run_offline_campaign(dataset, config, adapters))
    except Exception:
        print("OFFLINE_REPORT_INVALID", file=sys.stderr)
        raise SystemExit(2) from None
    # ASCII JSON survives Windows console code pages unchanged; parsed status is Chinese.
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
