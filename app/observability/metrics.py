"""Deterministic, offline campaign denominators over explicit completed runs.

This consumes synthetic or separately audited measurements; it does not read
an exporter, database, environment variable, or provider. Its output is an
estimate, not a bill or a claim about live model performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import re
from typing import Mapping, Sequence

from app.observability.costs import PriceCard, estimate_attempt_cost


_INT64_MAX = 2**63 - 1
_CASE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
_PER_SUCCESS_QUANTUM = Decimal("0.000000000000000000000000000001")


@dataclass(frozen=True)
class AttemptObservation:
    elapsed_ms: int
    attributes: Mapping[str, object]
    model_id: str
    at_utc: str


@dataclass(frozen=True)
class RunObservation:
    case_id: str
    terminal_status: str
    task_success: bool
    wall_ms: int
    active_ms: int
    attempts_complete: bool
    attempts: tuple[AttemptObservation, ...]


@dataclass(frozen=True)
class Percentiles:
    status: str
    count: int
    p50_ms: int | None
    p95_ms: int | None
    method: str = "nearest_rank"


@dataclass(frozen=True)
class SuccessCost:
    status: str
    reason: str | None
    task_count: int
    success_count: int
    attempt_count: int
    priced_attempt_count: int
    unknown_attempt_count: int
    known_subtotal: str
    total_estimate: str | None
    per_success_estimate: str | None
    currency: str | None
    price_card_id: str | None
    price_card_digest: str | None
    source_url: str | None


@dataclass(frozen=True)
class CampaignMetrics:
    task_count: int
    success_count: int
    attempt_count: int
    wall: Percentiles
    active: Percentiles
    sdk: Percentiles
    cost: SuccessCost


def _valid_ms(value: object) -> bool:
    return type(value) is int and 0 <= value <= _INT64_MAX


def _percentiles(values: list[int]) -> Percentiles:
    if not values:
        return Percentiles(status="UNKNOWN", count=0, p50_ms=None, p95_ms=None)
    ordered = sorted(values)
    n = len(ordered)
    return Percentiles(
        status="KNOWN", count=n,
        p50_ms=ordered[(n + 1) // 2 - 1],
        p95_ms=ordered[(95 * n + 99) // 100 - 1],
    )


def _decimal_text(value: Decimal) -> str:
    # normalize() inherits the caller's Decimal context and may round a large
    # exact subtotal; fixed-point formatting preserves its coefficient.
    formatted = format(value, "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def summarize_runs(
    runs: Sequence[RunObservation], *, price_card: PriceCard | None,
) -> CampaignMetrics:
    """Count each final task once and each actual model SDK attempt once."""
    if not isinstance(runs, (list, tuple)):
        raise ValueError("invalid run observations")
    seen: set[str] = set()
    walls: list[int] = []
    active: list[int] = []
    sdk: list[int] = []
    successes = 0
    complete = True
    priced = 0
    unknown = 0
    subtotal = Decimal(0)
    card_digests: set[str] = set()
    for run in runs:
        if (
            type(run) is not RunObservation
            or type(run.case_id) is not str or _CASE_ID.fullmatch(run.case_id) is None
            or run.case_id in seen
            or type(run.terminal_status) is not str
            or run.terminal_status not in _TERMINAL
            or type(run.task_success) is not bool
            or (run.task_success and run.terminal_status != "COMPLETED")
            or not _valid_ms(run.wall_ms) or not _valid_ms(run.active_ms)
            or run.active_ms > run.wall_ms
            or type(run.attempts_complete) is not bool
            or type(run.attempts) is not tuple
        ):
            raise ValueError("invalid run observations")
        seen.add(run.case_id)
        walls.append(run.wall_ms)
        active.append(run.active_ms)
        successes += int(run.task_success)
        complete = complete and run.attempts_complete
        for attempt in run.attempts:
            if (
                type(attempt) is not AttemptObservation
                or not _valid_ms(attempt.elapsed_ms)
                or not isinstance(attempt.attributes, Mapping)
                or type(attempt.model_id) is not str
                or type(attempt.at_utc) is not str
            ):
                raise ValueError("invalid attempt observations")
            sdk.append(attempt.elapsed_ms)
            estimate = estimate_attempt_cost(
                attempt.attributes, model_id=attempt.model_id,
                at_utc=attempt.at_utc, price_card=price_card,
            )
            if estimate.status == "ESTIMATE" and estimate.amount is not None:
                priced += 1
                with localcontext() as context:
                    context.prec = 100
                    subtotal += Decimal(estimate.amount)
                if estimate.price_card_digest is not None:
                    card_digests.add(estimate.price_card_digest)
            else:
                unknown += 1
    task_count = len(walls)
    attempt_count = len(sdk)
    reason: str | None = None
    if not complete:
        reason = "ATTEMPTS_INCOMPLETE"
    elif price_card is None:
        reason = "PRICE_CARD_MISSING"
    elif attempt_count == 0:
        reason = "NO_ATTEMPTS"
    elif unknown:
        reason = "ATTEMPT_COST_UNKNOWN"
    elif successes == 0:
        reason = "NO_SUCCESS"
    elif len(card_digests) != 1:
        reason = "PRICE_CARD_INCONSISTENT"
    if reason is None:
        with localcontext() as context:
            context.prec = 100
            per_success = (subtotal / Decimal(successes)).quantize(
                _PER_SUCCESS_QUANTUM, rounding=ROUND_HALF_EVEN,
            )
        cost = SuccessCost(
            status="ESTIMATE", reason=None, task_count=task_count,
            success_count=successes, attempt_count=attempt_count,
            priced_attempt_count=priced, unknown_attempt_count=unknown,
            known_subtotal=_decimal_text(subtotal),
            total_estimate=_decimal_text(subtotal),
            per_success_estimate=_decimal_text(per_success),
            currency=price_card.currency, price_card_id=price_card.price_card_id,
            price_card_digest=price_card.digest, source_url=price_card.source_url,
        )
    else:
        cost = SuccessCost(
            status="UNKNOWN", reason=reason, task_count=task_count,
            success_count=successes, attempt_count=attempt_count,
            priced_attempt_count=priced, unknown_attempt_count=unknown,
            known_subtotal=_decimal_text(subtotal), total_estimate=None,
            per_success_estimate=None, currency=None, price_card_id=None,
            price_card_digest=None, source_url=None,
        )
    return CampaignMetrics(
        task_count=task_count, success_count=successes,
        attempt_count=attempt_count, wall=_percentiles(walls),
        active=_percentiles(active), sdk=_percentiles(sdk), cost=cost,
    )
