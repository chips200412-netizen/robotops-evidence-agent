"""Fair, local-only three-baseline dry runs over the frozen synthetic set.

This module deliberately has no provider adapter, network client, or scorer.
Its output is an execution-contract fixture, never a model benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping

from app.evaluation.dataset import (
    EvaluationCase, EvidenceDocument, FrozenDataset, Gold,
    load_frozen_dataset,
)


BASELINES = ("DIRECT_LLM", "SINGLE_TURN_RAG", "FULL_AGENT")
_CONFIG_KEYS = frozenset({
    "model_id", "temperature", "max_output_tokens", "timeout_ms",
    "max_tool_steps", "repetitions",
})
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,79}\Z")
_MODEL_SECRET_SHAPE = re.compile(
    r"sk[-_]|api[_-]?key|secret|password|token|bearer|credential|"
    r"^gh[pousr]_|^glpat-|^xox[baprs]-|^(?:AKIA|ASIA)[A-Z0-9]{16}$|"
    r"^AIza|^ya29[._-]|^[pr]k_live_",
    re.I,
)
_CASE_ID = re.compile(r"SYN-\d{3}\Z")


def _exact_fields(value: object, expected_type: type) -> bool:
    """Reject extra/missing dataclass fields before any attacker-defined equality."""
    return type(value) is expected_type and set(vars(value)) == {
        field.name for field in fields(expected_type)
    }


def _str_tuple(value: object) -> bool:
    return type(value) is tuple and all(type(item) is str for item in value)


def _safe_dataset_shape(dataset: FrozenDataset) -> bool:
    if not _exact_fields(dataset, FrozenDataset):
        return False
    if any(type(getattr(dataset, name)) is not str for name in (
        "dataset_id", "version", "cases_sha256", "corpus_sha256", "license_scope",
    )):
        return False
    if (
        type(dataset.cases) is not tuple or type(dataset.corpus) is not tuple
        or type(dataset.scenario_counts) is not dict
        or any(type(key) is not str or type(value) is not int
               for key, value in dataset.scenario_counts.items())
    ):
        return False
    for case in dataset.cases:
        if not _exact_fields(case, EvaluationCase):
            return False
        if any(type(getattr(case, name)) is not str for name in (
            "case_id", "version", "scenario", "input", "provenance", "license_scope",
        )) or not _exact_fields(case.gold, Gold):
            return False
        if (
            any(type(getattr(case.gold, name)) is not str
                for name in ("fault", "safety", "recovery"))
            or not _str_tuple(case.gold.evidence_ids)
            or not _str_tuple(case.gold.tools)
        ):
            return False
    return all(
        _exact_fields(doc, EvidenceDocument)
        and all(type(getattr(doc, name)) is str for name in ("evidence_id", "kind", "text"))
        for doc in dataset.corpus
    )


@dataclass(frozen=True)
class FairConfig:
    model_id: str
    temperature: str
    max_output_tokens: int
    timeout_ms: int
    max_tool_steps: int
    repetitions: int

    def __post_init__(self) -> None:
        try:
            temp = Decimal(self.temperature) if type(self.temperature) is str else None
        except InvalidOperation as exc:
            raise ValueError("invalid offline evaluation configuration") from exc
        if (
            type(self.model_id) is not str or _MODEL_ID.fullmatch(self.model_id) is None
            or _MODEL_SECRET_SHAPE.search(self.model_id) is not None
            or temp is None or not temp.is_finite() or not Decimal(0) <= temp <= Decimal(2)
            or type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= 4096
            or type(self.timeout_ms) is not int or not 1 <= self.timeout_ms <= 120000
            or type(self.max_tool_steps) is not int or not 1 <= self.max_tool_steps <= 8
            or type(self.repetitions) is not int or not 1 <= self.repetitions <= 3
        ):
            raise ValueError("invalid offline evaluation configuration")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FairConfig:
        if type(value) is not dict or set(value) != _CONFIG_KEYS:
            raise ValueError("invalid offline evaluation configuration")
        return cls(**value)

    @property
    def fingerprint(self) -> str:
        payload = {
            "model_id": self.model_id, "temperature": str(Decimal(self.temperature).normalize()),
            "max_output_tokens": self.max_output_tokens, "timeout_ms": self.timeout_ms,
            "max_tool_steps": self.max_tool_steps, "repetitions": self.repetitions,
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PublicEvidence:
    evidence_id: str
    text: str


@dataclass(frozen=True)
class BaselineRequest:
    input: str
    corpus: tuple[PublicEvidence, ...]
    config: FairConfig


@dataclass(frozen=True)
class PlannedRun:
    baseline: str
    case_id: str
    repetition: int
    request: BaselineRequest
    config_fingerprint: str
    dataset_id: str
    dataset_version: str
    cases_sha256: str
    corpus_sha256: str
    execution_mode: str = "OFFLINE_SYNTHETIC"


@dataclass(frozen=True)
class CandidateResult:
    fault: str
    evidence_ids: tuple[str, ...]
    tools: tuple[str, ...]
    safety: str
    recovery: str


@dataclass(frozen=True)
class RawRun:
    baseline: str
    case_id: str
    repetition: int
    status: str
    candidate: CandidateResult | None
    config_fingerprint: str
    dataset_id: str
    dataset_version: str
    cases_sha256: str
    corpus_sha256: str
    source: str = "SYNTHETIC_STUB"
    execution_mode: str = "OFFLINE_SYNTHETIC"
    measurement_status: str = "DRY_RUN_NOT_MODEL_MEASURED"


@dataclass(frozen=True)
class LocalSyntheticAdapter:
    baseline: str
    failure_case_ids: tuple[str, ...] = ()
    mode: str = "SYNTHETIC_STUB"

    def __post_init__(self) -> None:
        if (
            self.baseline not in BASELINES or self.mode != "SYNTHETIC_STUB"
            or type(self.failure_case_ids) is not tuple
            or any(type(case_id) is not str or _CASE_ID.fullmatch(case_id) is None
                   for case_id in self.failure_case_ids)
            or len(set(self.failure_case_ids)) != len(self.failure_case_ids)
        ):
            raise ValueError("invalid local synthetic adapter")

    def run(self, request: BaselineRequest, *, case_id: str) -> CandidateResult:
        if case_id in self.failure_case_ids:
            raise RuntimeError("synthetic failure")
        return CandidateResult(
            fault="UNKNOWN", evidence_ids=(), tools=(),
            safety="UNKNOWN", recovery="NONE",
        )


def verify_frozen_dataset(dataset: FrozenDataset) -> FrozenDataset:
    """Return the trusted project-local copy only when the caller's copy matches."""
    if type(dataset) is not FrozenDataset:
        raise ValueError("invalid offline evaluation input")
    # A dataclass can be replaced in memory without changing its declared SHA.
    # Re-read the sole project-local, byte-checked source before making requests.
    trusted = load_frozen_dataset(Path(__file__).resolve().parents[2] / "eval")
    if not _safe_dataset_shape(dataset) or dataset != trusted:
        raise ValueError("offline dataset differs from frozen source")
    return trusted


def build_fair_plan(dataset: FrozenDataset, config: FairConfig) -> tuple[PlannedRun, ...]:
    if type(config) is not FairConfig:
        raise ValueError("invalid offline evaluation input")
    FairConfig.__post_init__(config)
    trusted = verify_frozen_dataset(dataset)
    public_cases = trusted.public_case_inputs()
    public_docs = trusted.public_corpus()
    if len(public_cases) != 50 or len(public_docs) != 15:
        raise ValueError("invalid frozen evaluation coverage")
    corpus = tuple(PublicEvidence(evidence_id=doc["evidence_id"], text=doc["text"])
                   for doc in public_docs)
    plan: list[PlannedRun] = []
    for repetition in range(1, config.repetitions + 1):
        for case in public_cases:
            for baseline in BASELINES:
                plan.append(PlannedRun(
                    baseline=baseline, case_id=case["case_id"], repetition=repetition,
                    request=BaselineRequest(input=case["input"], corpus=() if baseline == "DIRECT_LLM" else corpus, config=config),
                    config_fingerprint=config.fingerprint, dataset_id=trusted.dataset_id,
                    dataset_version=trusted.version, cases_sha256=trusted.cases_sha256,
                    corpus_sha256=trusted.corpus_sha256,
                ))
    return tuple(plan)


def run_offline_campaign(
    dataset: FrozenDataset,
    config: FairConfig,
    adapters: tuple[LocalSyntheticAdapter, ...],
) -> tuple[RawRun, ...]:
    """Execute only built-in inert stubs; never infer model quality from these rows."""
    if (
        type(adapters) is not tuple or len(adapters) != len(BASELINES)
        or any(type(adapter) is not LocalSyntheticAdapter for adapter in adapters)
    ):
        raise ValueError("three unique local synthetic adapters required")
    for adapter in adapters:
        LocalSyntheticAdapter.__post_init__(adapter)
    if {adapter.baseline for adapter in adapters} != set(BASELINES):
        raise ValueError("three unique local synthetic adapters required")
    by_baseline = {adapter.baseline: adapter for adapter in adapters}
    plan = build_fair_plan(dataset, config)
    valid_case_ids = {item.case_id for item in plan}
    if any(case_id not in valid_case_ids for adapter in adapters for case_id in adapter.failure_case_ids):
        raise ValueError("invalid local synthetic failure target")
    rows: list[RawRun] = []
    for item in plan:
        try:
            candidate = LocalSyntheticAdapter.run(
                by_baseline[item.baseline], item.request, case_id=item.case_id,
            )
            status = "COMPLETED"
        except Exception:
            candidate = None
            status = "ERROR"
        rows.append(RawRun(
            baseline=item.baseline, case_id=item.case_id,
            repetition=item.repetition, status=status, candidate=candidate,
            config_fingerprint=item.config_fingerprint,
            dataset_id=item.dataset_id, dataset_version=item.dataset_version,
            cases_sha256=item.cases_sha256, corpus_sha256=item.corpus_sha256,
        ))
    if len(rows) != len(plan) or len({(row.case_id, row.repetition, row.baseline) for row in rows}) != len(plan):
        raise ValueError("incomplete offline evaluation campaign")
    return tuple(rows)
