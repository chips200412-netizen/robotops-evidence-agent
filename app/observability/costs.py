"""Offline, source-labelled price-card estimates for provider-sourced attempts.

This module has no network, environment, or provider dependency. A result is
never presented as an invoice; missing evidence remains UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException, localcontext
import hashlib
import json
import re
from typing import Mapping
from urllib.parse import urlsplit


_INT64_MAX = 2**63 - 1
_CARD_KEYS = frozenset({
    "price_card_id", "model_id", "currency", "source_url",
    "retrieved_at_utc", "valid_from_utc", "valid_until_utc",
    "input_cache_hit_per_million", "input_cache_miss_per_million",
    "output_per_million",
})
_IDENTIFIER = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_RATE = re.compile(r"\d{1,12}(?:\.\d{1,18})?\Z")
_SUPPORTED_CURRENCIES = frozenset({"USD", "CNY"})


def _utc_timestamp(value: object) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError("invalid UTC timestamp")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("invalid UTC timestamp") from exc
    if result.utcoffset() != timedelta(0):
        raise ValueError("invalid UTC timestamp")
    return result


def _rate(value: object) -> Decimal:
    if type(value) is not str or _RATE.fullmatch(value) is None:
        raise ValueError("invalid price rate")
    rate = Decimal(value)
    if not rate.is_finite() or rate < 0:
        raise ValueError("invalid price rate")
    return rate


def _price_card_digest(value: Mapping[str, object]) -> str:
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_card_values(
    *, card_id: str, model_id: str, currency: str, source_url: str,
    retrieved: datetime, start: datetime, end: datetime,
    hit_rate: Decimal, miss_rate: Decimal, output_rate: Decimal,
) -> dict[str, str]:
    return {
        "price_card_id": card_id, "model_id": model_id,
        "currency": currency, "source_url": source_url,
        "retrieved_at_utc": retrieved.isoformat().replace("+00:00", "Z"),
        "valid_from_utc": start.isoformat().replace("+00:00", "Z"),
        "valid_until_utc": end.isoformat().replace("+00:00", "Z"),
        "input_cache_hit_per_million": format(hit_rate, "f"),
        "input_cache_miss_per_million": format(miss_rate, "f"),
        "output_per_million": format(output_rate, "f"),
    }


@dataclass(frozen=True)
class PriceCard:
    price_card_id: str
    model_id: str
    currency: str
    source_url: str
    retrieved_at_utc: datetime
    valid_from_utc: datetime
    valid_until_utc: datetime
    input_cache_hit_per_million: Decimal
    input_cache_miss_per_million: Decimal
    output_per_million: Decimal
    digest: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PriceCard":
        """Validate an explicit immutable price snapshot; never fetch a live rate."""
        if not isinstance(value, Mapping) or frozenset(value) != _CARD_KEYS:
            raise ValueError("invalid price card")
        card_id = value["price_card_id"]
        model_id = value["model_id"]
        currency = value["currency"]
        source_url = value["source_url"]
        if (
            type(card_id) is not str or _IDENTIFIER.fullmatch(card_id) is None
            or type(model_id) is not str or _IDENTIFIER.fullmatch(model_id) is None
            or type(currency) is not str or currency not in _SUPPORTED_CURRENCIES
            or type(source_url) is not str or len(source_url) > 2048
        ):
            raise ValueError("invalid price card")
        try:
            parsed = urlsplit(source_url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("invalid price card") from exc
        if (
            parsed.scheme != "https" or not parsed.hostname
            or port == 0
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or any(ch.isspace() for ch in source_url)
        ):
            raise ValueError("invalid price card")
        retrieved = _utc_timestamp(value["retrieved_at_utc"])
        start = _utc_timestamp(value["valid_from_utc"])
        end = _utc_timestamp(value["valid_until_utc"])
        if end <= start:
            raise ValueError("invalid price card")
        hit_rate = _rate(value["input_cache_hit_per_million"])
        miss_rate = _rate(value["input_cache_miss_per_million"])
        output_rate = _rate(value["output_per_million"])
        digest = _price_card_digest(_canonical_card_values(
            card_id=card_id, model_id=model_id, currency=currency,
            source_url=source_url, retrieved=retrieved, start=start, end=end,
            hit_rate=hit_rate, miss_rate=miss_rate, output_rate=output_rate,
        ))
        return cls(
            price_card_id=card_id, model_id=model_id, currency=currency,
            source_url=source_url, retrieved_at_utc=retrieved,
            valid_from_utc=start, valid_until_utc=end,
            input_cache_hit_per_million=hit_rate,
            input_cache_miss_per_million=miss_rate,
            output_per_million=output_rate, digest=digest,
        )


@dataclass(frozen=True)
class CostResult:
    status: str
    amount: str | None = None
    currency: str | None = None
    price_card_id: str | None = None
    price_card_digest: str | None = None
    source_url: str | None = None
    reason: str | None = None


def _unknown(reason: str) -> CostResult:
    return CostResult(status="UNKNOWN", reason=reason)


def _token(value: object) -> bool:
    return type(value) is int and 0 <= value <= _INT64_MAX


def _valid_price_card(card: PriceCard) -> bool:
    """Recheck public dataclass instances, including dataclasses.replace copies."""
    try:
        dates = (
            card.retrieved_at_utc, card.valid_from_utc,
            card.valid_until_utc,
        )
        rates = (
            card.input_cache_hit_per_million,
            card.input_cache_miss_per_million, card.output_per_million,
        )
        if (
            not all(type(value) is datetime and value.utcoffset() == timedelta(0)
                    for value in dates)
            or not all(type(value) is Decimal and value.is_finite()
                       and 0 <= value < Decimal(10**12)
                       and -18 <= value.as_tuple().exponent <= 12
                       for value in rates)
            or type(card.digest) is not str
        ):
            return False
        values = _canonical_card_values(
            card_id=card.price_card_id, model_id=card.model_id,
            currency=card.currency, source_url=card.source_url,
            retrieved=card.retrieved_at_utc, start=card.valid_from_utc,
            end=card.valid_until_utc,
            hit_rate=card.input_cache_hit_per_million,
            miss_rate=card.input_cache_miss_per_million,
            output_rate=card.output_per_million,
        )
        validated = PriceCard.from_mapping(values)
        return validated.digest == card.digest
    except (AttributeError, DecimalException, TypeError, ValueError):
        return False


def estimate_attempt_cost(
    attributes: Mapping[str, object], *, model_id: str, at_utc: str,
    price_card: PriceCard | None,
) -> CostResult:
    """Calculate a reproducible estimate, or state why it cannot be calculated."""
    if not isinstance(attributes, Mapping):
        return _unknown("INVALID_USAGE")
    if (attributes.get("robotops.usage_status") != "KNOWN"
            or attributes.get("robotops.usage_source") != "PROVIDER"):
        return _unknown("USAGE_UNKNOWN")
    if (attributes.get("robotops.cache_status") != "KNOWN"
            or attributes.get("robotops.cache_source") != "PROVIDER"):
        return _unknown("CACHE_UNKNOWN")
    input_tokens = attributes.get("robotops.input_tokens")
    output_tokens = attributes.get("robotops.output_tokens")
    total_tokens = attributes.get("robotops.total_tokens")
    cache_hit = attributes.get("robotops.cache_hit_tokens")
    cache_miss = attributes.get("robotops.cache_miss_tokens")
    if (
        not all(_token(item) for item in (
            input_tokens, output_tokens, total_tokens, cache_hit, cache_miss,
        ))
        or cache_hit + cache_miss != input_tokens
        or total_tokens < input_tokens + output_tokens
    ):
        return _unknown("INVALID_USAGE")
    if price_card is None:
        return _unknown("PRICE_CARD_MISSING")
    if type(price_card) is not PriceCard or not _valid_price_card(price_card):
        return _unknown("PRICE_CARD_INVALID")
    if model_id != price_card.model_id:
        return _unknown("MODEL_MISMATCH")
    try:
        timestamp = _utc_timestamp(at_utc)
    except ValueError:
        return _unknown("INVALID_TIME")
    if not price_card.valid_from_utc <= timestamp < price_card.valid_until_utc:
        return _unknown("PRICE_CARD_OUT_OF_WINDOW")
    with localcontext() as context:
        context.prec = 80
        amount = (
            cache_hit * price_card.input_cache_hit_per_million
            + cache_miss * price_card.input_cache_miss_per_million
            + output_tokens * price_card.output_per_million
        ) / Decimal(1_000_000)
        amount_text = format(amount.normalize(), "f")
    return CostResult(
        status="ESTIMATE", amount=amount_text, currency=price_card.currency,
        price_card_id=price_card.price_card_id,
        price_card_digest=price_card.digest,
        source_url=price_card.source_url,
    )
