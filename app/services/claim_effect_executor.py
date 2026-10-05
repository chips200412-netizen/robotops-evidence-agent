"""Service-owned V2 synthetic effect authority; the graph carries no permission.

The registry CAS is committed before the only device call. Once an attempt is
declared, a missing response or an orphaned EXECUTING row is never retried.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping
from typing import Any

from app.services import effect_capabilities
from app.observability.tracing import (
    safe_span, set_safe_outcome, set_safe_readback_matched,
)
from app.services.diagnostic_sessions import (
    EFFECT_MODE_SYNTHETIC_RESTART,
    SANITIZED_STORE_ERROR,
    SessionStoreError,
)
from app.services.synthetic_effects import SyntheticEffectNotAppliedError


class ServiceClaimEffectAuthority:
    """Narrow execution/verification surface supplied only by session service."""

    __slots__ = ("__service",)

    def __init__(self, service: Any) -> None:
        self.__service = service

    def _authorized_locked(
        self, *, session_id: str, run_id: str, approval_id: str,
        proposal: Mapping[str, Any], proposal_sha256: str,
    ) -> dict[str, Any]:
        service = self.__service
        try:
            if (
                type(session_id) is not str or type(run_id) is not str
                or type(approval_id) is not str or type(proposal_sha256) is not str
                or not isinstance(proposal, Mapping)
                or service._effect_mode != EFFECT_MODE_SYNTHETIC_RESTART
            ):
                raise ValueError()
            adapter = service._synthetic_effect_adapter
            if (adapter is None
                    or not callable(getattr(adapter, "restart_simulated_robot", None))
                    or not callable(getattr(adapter, "read_device_state", None))):
                raise ValueError()
            checked = service._validate_claim_effect_binding_locked(
                approval_id, require_checkpoint=False,
            )
            approval, ledger, material = (
                checked["approval"], checked["ledger"], checked["material"]
            )
            run = service._select_run_locked(run_id)
            encoded = effect_capabilities.canonical_json(dict(proposal))
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            expected_encoded = effect_capabilities.canonical_json(material["proposal"])
            if (
                run is None or run.session_id != session_id
                or approval["session_id"] != session_id
                or approval["run_id"] != run_id
                or approval["approval_id"] != approval_id
                or approval["status"] != "CONSUMED"
                or approval["decision"] != "APPROVE"
                or not secrets.compare_digest(proposal_sha256, digest)
                or not secrets.compare_digest(material["proposal_sha256"], digest)
                or encoded != expected_encoded
                or approval["effect_key"] != ledger["effect_key"]
                or approval["effect_key"] != material["effect_key"]
                or approval["binding_sha256"] != ledger["binding_sha256"]
                or approval["binding_sha256"] != material["binding_sha256"]
            ):
                raise ValueError()
            return {"run": run, "approval": approval, "ledger": ledger,
                    "material": material, "runtime": checked["runtime"],
                    "result": checked["result"], "adapter": adapter}
        except Exception:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from None

    @staticmethod
    def _runtime(checked: dict[str, Any]) -> dict[str, Any]:
        return checked["runtime"]

    def _force_unknown(self, approval_id: str) -> dict[str, Any]:
        service = self.__service
        try:
            with service._registry_transaction():
                service._registry.execute("BEGIN IMMEDIATE")
                checked = service._validate_claim_effect_binding_locked(
                    approval_id, require_checkpoint=False,
                )
                ledger = checked["ledger"]
                if ledger["status"] in {"EXECUTING", "APPLIED"}:
                    changed = service._registry.execute(
                        "UPDATE diagnostic_claim_effect_ledger "
                        "SET status='UNKNOWN',updated_at=? "
                        "WHERE effect_key=? AND binding_sha256=? "
                        "AND status=? AND attempt_count=1",
                        (max(service._typed_now_ms(), ledger["updated_at"]),
                         ledger["effect_key"], ledger["binding_sha256"], ledger["status"]),
                    ).rowcount
                    if changed != 1:
                        raise ValueError()
                    checked = service._validate_claim_effect_binding_locked(
                        approval_id, require_checkpoint=False,
                    )
                service._registry.commit()
                return self._runtime(checked)
        except Exception:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from None

    def _persist_execution(
        self, approval_id: str, *, status: str, result: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        service = self.__service
        try:
            if status not in {"APPLIED", "EXECUTION_FAILED", "UNKNOWN"}:
                raise ValueError()
            result_json = (effect_capabilities.canonical_json(dict(result))
                           if result is not None else None)
            result_sha256 = (hashlib.sha256(result_json.encode("utf-8")).hexdigest()
                             if result_json is not None else None)
            with service._registry_transaction():
                service._registry.execute("BEGIN IMMEDIATE")
                checked = service._validate_claim_effect_binding_locked(
                    approval_id, require_checkpoint=False,
                )
                ledger = checked["ledger"]
                if ledger["status"] != "EXECUTING":
                    service._registry.commit()
                    return self._runtime(checked)
                changed = service._registry.execute(
                    "UPDATE diagnostic_claim_effect_ledger "
                    "SET status=?,result_json=?,result_sha256=?,"
                    "verification_json=NULL,verification_sha256=NULL,updated_at=? "
                    "WHERE effect_key=? AND binding_sha256=? "
                    "AND status='EXECUTING' AND attempt_count=1",
                    (status, result_json, result_sha256,
                     max(service._typed_now_ms(), ledger["updated_at"]),
                     ledger["effect_key"], ledger["binding_sha256"]),
                ).rowcount
                if changed != 1:
                    raise ValueError()
                checked = service._validate_claim_effect_binding_locked(
                    approval_id, require_checkpoint=False,
                )
                service._registry.commit()
                return self._runtime(checked)
        except Exception:
            return self._force_unknown(approval_id)

    def execute_effect(
        self, *, session_id: str, run_id: str, approval_id: str,
        proposal: Mapping[str, Any], proposal_sha256: str,
    ) -> Mapping[str, Any]:
        service = self.__service
        try:
            with service._registry_transaction():
                service._registry.execute("BEGIN IMMEDIATE")
                checked = self._authorized_locked(
                    session_id=session_id, run_id=run_id, approval_id=approval_id,
                    proposal=proposal, proposal_sha256=proposal_sha256,
                )
                ledger = checked["ledger"]
                if ledger["status"] != "PREPARED":
                    service._registry.commit()
                    if ledger["status"] == "EXECUTING":
                        return self._force_unknown(approval_id)
                    return self._runtime(checked)
                session = service._registry.execute(
                    "SELECT status,active_run_id FROM diagnostic_sessions WHERE session_id=?",
                    (session_id,),
                ).fetchone()
                if (checked["run"].status != "RUNNING"
                        or session != ("RUNNING", run_id)
                        or ledger["attempt_count"] != 0):
                    raise ValueError()
                changed = service._registry.execute(
                    "UPDATE diagnostic_claim_effect_ledger "
                    "SET status='EXECUTING',attempt_count=1,updated_at=? "
                    "WHERE effect_key=? AND binding_sha256=? "
                    "AND status='PREPARED' AND attempt_count=0",
                    (max(service._typed_now_ms(), ledger["updated_at"]),
                     ledger["effect_key"], ledger["binding_sha256"]),
                ).rowcount
                if changed != 1:
                    raise ValueError()
                service._registry.commit()
        except Exception:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from None

        adapter = checked["adapter"]
        approval = checked["approval"]
        arguments = checked["material"]["proposal"]["arguments"]
        try:
            with safe_span("robotops.effect.execute") as span:
                raw = adapter.restart_simulated_robot(
                    effect_key=approval["effect_key"],
                    binding_sha256=approval["binding_sha256"],
                    device_model=arguments["device_model"],
                    expected_boot_generation_delta=arguments["expected_boot_generation_delta"],
                )
                result = service._validate_synthetic_result(
                    raw, effect_key=approval["effect_key"],
                    binding_sha256=approval["binding_sha256"],
                    device_model=arguments["device_model"],
                )
                set_safe_outcome(span, "COMPLETED")
        except SyntheticEffectNotAppliedError:
            return self._persist_execution(approval_id, status="EXECUTION_FAILED", result=None)
        except Exception:
            return self._persist_execution(approval_id, status="UNKNOWN", result=None)
        return self._persist_execution(approval_id, status="APPLIED", result=result)

    def _persist_verification(
        self, approval_id: str, *, status: str, verification: Mapping[str, Any],
    ) -> dict[str, Any]:
        service = self.__service
        try:
            if status not in {"VERIFIED", "VERIFICATION_FAILED", "UNKNOWN"}:
                raise ValueError()
            encoded = effect_capabilities.canonical_json(dict(verification))
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            with service._registry_transaction():
                service._registry.execute("BEGIN IMMEDIATE")
                checked = service._validate_claim_effect_binding_locked(
                    approval_id, require_checkpoint=False,
                )
                ledger = checked["ledger"]
                if ledger["status"] != "APPLIED":
                    service._registry.commit()
                    return self._runtime(checked)
                changed = service._registry.execute(
                    "UPDATE diagnostic_claim_effect_ledger "
                    "SET status=?,verification_json=?,verification_sha256=?,updated_at=? "
                    "WHERE effect_key=? AND binding_sha256=? "
                    "AND status='APPLIED' AND attempt_count=1",
                    (status, encoded, digest,
                     max(service._typed_now_ms(), ledger["updated_at"]),
                     ledger["effect_key"], ledger["binding_sha256"]),
                ).rowcount
                if changed != 1:
                    raise ValueError()
                checked = service._validate_claim_effect_binding_locked(
                    approval_id, require_checkpoint=False,
                )
                service._registry.commit()
                return self._runtime(checked)
        except Exception:
            return self._force_unknown(approval_id)

    def verify_effect(
        self, *, session_id: str, run_id: str, approval_id: str,
        proposal: Mapping[str, Any], proposal_sha256: str,
    ) -> Mapping[str, Any]:
        service = self.__service
        with service._registry_lock:
            checked = self._authorized_locked(
                session_id=session_id, run_id=run_id, approval_id=approval_id,
                proposal=proposal, proposal_sha256=proposal_sha256,
            )
        ledger = checked["ledger"]
        if ledger["status"] == "EXECUTING":
            return self._force_unknown(approval_id)
        if ledger["status"] != "APPLIED":
            return self._runtime(checked)
        result = checked["result"]
        if result is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        approval = checked["approval"]
        device_model = checked["material"]["proposal"]["arguments"]["device_model"]
        try:
            with safe_span("robotops.effect.readback") as span:
                readback = service._validate_synthetic_readback(
                    checked["adapter"].read_device_state(device_model=device_model),
                    device_model=device_model,
                )
                matched = (
                    readback["boot_generation"] == result["boot_generation_after"]
                    and readback["last_effect_key"] == approval["effect_key"]
                    and readback["last_binding_sha256"] == approval["binding_sha256"]
                )
                set_safe_readback_matched(span, matched)
                set_safe_outcome(span, "COMPLETED")
            status = "VERIFIED" if matched else "VERIFICATION_FAILED"
        except Exception:
            readback = None
            status = "UNKNOWN"
        try:
            verification = service._verification_record(
                status=status, effect_key=approval["effect_key"],
                binding_sha256=approval["binding_sha256"],
                device_model=device_model,
                expected_boot_generation=result["boot_generation_after"],
                readback=readback,
            )
            service._validate_synthetic_verification(
                verification, effect_key=approval["effect_key"],
                binding_sha256=approval["binding_sha256"],
                device_model=device_model,
                expected_boot_generation=result["boot_generation_after"],
            )
        except Exception:
            return self._force_unknown(approval_id)
        return self._persist_verification(approval_id, status=status,
                                          verification=verification)
