"""Private runtime boundary for independently authorized synthetic effects.

The context is supplied by the authoritative service through LangGraph's
``context=`` argument.  It is deliberately separate from ``AgentState`` so a
model, client payload, or restored checkpoint cannot enable an effect surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


ASYNC_AUTHORITY_SURFACE = "ASYNC_AUTHORITY_V1"
SYNTHETIC_RESTART_EFFECT_MODE = "SYNTHETIC_RESTART_DEMO_V1"


@runtime_checkable
class EffectAuthority(Protocol):
    """Service-owned durable execution and independent readback boundary."""

    def execute_effect(
        self,
        *,
        session_id: str,
        run_id: str,
        approval_id: str,
        proposal: Mapping[str, Any],
        proposal_sha256: str,
    ) -> Mapping[str, Any]: ...

    def verify_effect(
        self,
        *,
        session_id: str,
        run_id: str,
        approval_id: str,
        proposal: Mapping[str, Any],
        proposal_sha256: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    """Non-checkpointed, server-injected execution context."""

    surface: str
    effect_mode: str | None
    session_id: str
    run_id: str
    approval_id: str | None
    effect_authority: EffectAuthority | None
    # Persisted service identity; old callers retain their V1 defaults.
    execution_contract: str | None = None
    request_id: str | None = None
    turn_number: int | None = None
