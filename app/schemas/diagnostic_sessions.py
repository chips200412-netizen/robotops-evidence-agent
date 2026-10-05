"""MOD-019～021：服务端权威会话、run、消息与草稿合同。"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictApiModel(BaseModel):
    """拒绝客户端夹带历史、身份或其他权威状态字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ExactApiModel(BaseModel):
    """拒绝额外字段，也不改写用于权威绑定的 token/decision。"""

    model_config = ConfigDict(extra="forbid")


class ExactResponseModel(BaseModel):
    """Reject response coercion as well as fields outside the public contract."""

    model_config = ConfigDict(extra="forbid", strict=True)


class DiagnosticSessionStatus(str, Enum):
    OPEN = "OPEN"
    RUNNING = "RUNNING"
    AWAITING_USER = "AWAITING_USER"
    INTERRUPTED = "INTERRUPTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class DiagnosticRunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class FailureDisposition(str, Enum):
    RETRYABLE = "RETRYABLE"
    FINAL = "FINAL"


class DiagnosticMessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class DiagnosticMessageKind(str, Enum):
    INPUT = "INPUT"
    FOLLOW_UP = "FOLLOW_UP"
    RESULT = "RESULT"


class DiagnosticDraftStatus(str, Enum):
    COLLECTING = "COLLECTING"
    READY = "READY"
    PROJECTION_FAILED = "PROJECTION_FAILED"
    PROJECTED = "PROJECTED"


class DiagnosticHistorySource(str, Enum):
    DRAFT = "DRAFT"
    TICKET = "TICKET"
    CASE = "CASE"


class ResumeDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class DiagnosticEffectApprovalStatus(str, Enum):
    PENDING = "PENDING"
    CONSUMED = "CONSUMED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class DiagnosticEffectStatus(str, Enum):
    PREPARED = "PREPARED"
    EXECUTING = "EXECUTING"
    APPLIED = "APPLIED"
    VERIFIED = "VERIFIED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    UNKNOWN = "UNKNOWN"


class CreateDiagnosticSessionRequest(StrictApiModel):
    """只允许客户端提供幂等键；会话身份与状态仍全部由服务端生成。"""

    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._~-]+$",
    )


class CreateDiagnosticSessionResponse(StrictApiModel):
    session_id: str
    claim_evidence_profile: Literal["disabled_v0", "robotops_agent_claim_evidence_v1"]
    status: DiagnosticSessionStatus
    created_at: datetime
    updated_at: datetime


class DiagnosticTurnRequest(StrictApiModel):
    message: str = Field(min_length=1, max_length=8_000)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


class DiagnosticTurnResponse(StrictApiModel):
    session_id: str
    status: DiagnosticSessionStatus
    turn_number: int = Field(ge=1)
    reply: str
    updated_at: datetime
    references: list[dict] = Field(default_factory=list)
    nodes: list[str] = Field(
        default_factory=list,
        description="兼容 UI 的节点快照推断；MOD-027 前不得作为真实 trace 证据",
    )
    follow_up: list[dict] = Field(default_factory=list)
    diagnosis: dict = Field(default_factory=dict)
    ticket_id: str | None = None
    ticket_note: str | None = None


class DiagnosticSessionResponse(StrictApiModel):
    session_id: str
    claim_evidence_profile: Literal["disabled_v0", "robotops_agent_claim_evidence_v1"]
    status: DiagnosticSessionStatus
    turn_count: int = Field(ge=0)
    active_run_id: str | None = None
    latest_run_id: str | None = None
    parent_session_id: str | None = None
    created_at: datetime
    updated_at: datetime


class DiagnosticRunRequest(StrictApiModel):
    """页面固定传 request_id；缺省仅用于兼容旧调用，由服务端补生成。"""

    message: str = Field(min_length=1, max_length=8_000)
    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._~-]+$",
    )


class ResumeDiagnosticRunRequest(ExactApiModel):
    interrupt_id: str = Field(min_length=1, max_length=256)
    approval_id: str = Field(
        min_length=32,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]{32}$",
    )
    decision: ResumeDecision


class CancelDiagnosticRunRequest(StrictApiModel):
    """取消请求故意为空；目标只能由 path 中的 session/run 决定。"""


class ReopenDiagnosticSessionRequest(StrictApiModel):
    """复开请求故意为空；父子关系完全由服务端创建。"""


class RetryDiagnosticRunRequest(StrictApiModel):
    """重试请求故意为空；服务端只允许尚未进入图的原 run。"""


class RetryDiagnosticDraftProjectionRequest(StrictApiModel):
    """只允许重试响应里公开的同一草稿版本。"""

    version: int = Field(ge=1)


class DiagnosticInterruptResponse(ExactApiModel):
    interrupt_id: str
    approval_id: str
    payload: dict[str, Any]
    created_at: datetime


class DiagnosticRunErrorResponse(StrictApiModel):
    code: str
    message: str


DiagnosticVerifierCheckName = Literal[
    "plan",
    "trajectory",
    "authoritative_arguments",
    "normalized_results",
    "evidence_binding",
    "stop_preconditions",
]


class DiagnosticEffectArgumentsResponse(ExactResponseModel):
    device_model: str = Field(min_length=1, max_length=200)
    expected_boot_generation_delta: int = Field(strict=True, ge=1, le=1)


class DiagnosticEffectRiskResponse(ExactResponseModel):
    risk_level: Literal["低", "中"]
    urgency: Literal["低", "中"]
    need_human: bool = Field(strict=True)
    risk_check_failed: bool = Field(strict=True)
    llm_failed: bool = Field(strict=True)
    tool_loop_failed: bool = Field(strict=True)

    @model_validator(mode="after")
    def exact_safe_risk_gate(self) -> "DiagnosticEffectRiskResponse":
        if any(
            (
                self.need_human,
                self.risk_check_failed,
                self.llm_failed,
                self.tool_loop_failed,
            )
        ):
            raise ValueError("effect risk gate is not eligible for approval")
        return self


class DiagnosticEffectVerifierChecksResponse(ExactResponseModel):
    plan: bool
    trajectory: bool
    authoritative_arguments: bool
    normalized_results: bool
    evidence_binding: bool
    stop_preconditions: bool


class DiagnosticEffectVerifierResponse(ExactResponseModel):
    status: Literal["PASSED"]
    checks: DiagnosticEffectVerifierChecksResponse
    reason_codes: list[str]
    supporting_evidence: list[DiagnosticVerifierCheckName]
    missing_evidence: list[DiagnosticVerifierCheckName]
    verifier_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: dict[str, Any]

    @model_validator(mode="after")
    def consistent_check_lists(self) -> "DiagnosticEffectVerifierResponse":
        checks = self.checks.model_dump()
        supporting = sorted(name for name, passed in checks.items() if passed)
        missing = sorted(name for name, passed in checks.items() if not passed)
        if (
            self.supporting_evidence != supporting
            or self.missing_evidence != missing
            or bool(missing)
            or bool(self.reason_codes)
        ):
            raise ValueError("verifier summary is inconsistent")
        return self


class DiagnosticEffectProposalResponse(ExactResponseModel):
    schema_version: int = Field(strict=True, ge=1, le=1)
    effect_type: Literal["restart_simulated_robot"]
    source: Literal["SYNTHETIC_LOCAL"]
    arguments: DiagnosticEffectArgumentsResponse


class DiagnosticEffectApprovalPayloadResponse(ExactResponseModel):
    """Exact public EFFECT_V1 interrupt payload used for cross-envelope binding."""

    approval_kind: Literal["EFFECT_V1"]
    effect_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal: DiagnosticEffectProposalResponse
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk: DiagnosticEffectRiskResponse
    verifier: DiagnosticEffectVerifierResponse
    issued_at: int = Field(strict=True, ge=0)
    expires_at: int = Field(strict=True, ge=0)
    ttl_ms: int = Field(strict=True, ge=1)
    status: Literal["PENDING", "EXPIRED"]
    decision: None = None
    can_decide: bool = Field(strict=True)
    server_time: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def exact_capability_window(self) -> "DiagnosticEffectApprovalPayloadResponse":
        if self.expires_at - self.issued_at != self.ttl_ms:
            raise ValueError("approval TTL does not match its capability window")
        expected_can_decide = (
            self.status == "PENDING"
            and self.issued_at <= self.server_time < self.expires_at
        )
        if self.can_decide is not expected_can_decide:
            raise ValueError("approval availability is inconsistent")
        return self


class DiagnosticEffectResultResponse(ExactResponseModel):
    schema_version: int = Field(strict=True, ge=1, le=1)
    effect_type: Literal["restart_simulated_robot"]
    source: Literal["SYNTHETIC_LOCAL"]
    device_model: str = Field(min_length=1, max_length=200)
    effect_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    boot_generation_before: int = Field(strict=True, ge=0)
    boot_generation_after: int = Field(strict=True, ge=1)
    applied: bool = Field(strict=True)

    @model_validator(mode="after")
    def generation_incremented_once(self) -> "DiagnosticEffectResultResponse":
        if (
            self.applied is not True
            or self.boot_generation_after != self.boot_generation_before + 1
        ):
            raise ValueError("boot generation delta must equal one")
        return self


class DiagnosticEffectVerificationResponse(ExactResponseModel):
    schema_version: int = Field(strict=True, ge=1, le=1)
    status: Literal["VERIFIED", "VERIFICATION_FAILED", "UNKNOWN"]
    source: Literal["SYNTHETIC_LOCAL"]
    device_model: str = Field(min_length=1, max_length=200)
    effect_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_boot_generation: int = Field(strict=True, ge=1)
    observed_boot_generation: int | None = Field(default=None, strict=True, ge=0)
    observed_effect_key: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    observed_binding_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    matched: bool = Field(strict=True)
    reason_code: str

    @model_validator(mode="after")
    def exact_readback_outcome(self) -> "DiagnosticEffectVerificationResponse":
        if self.status == "UNKNOWN":
            valid = (
                self.observed_boot_generation is None
                and self.observed_effect_key is None
                and self.observed_binding_sha256 is None
                and self.matched is False
                and self.reason_code == "READBACK_UNAVAILABLE"
            )
        else:
            observed_binding_complete = (
                self.observed_effect_key is None
            ) == (self.observed_binding_sha256 is None)
            matched = (
                self.observed_boot_generation == self.expected_boot_generation
                and self.observed_effect_key == self.effect_key
                and self.observed_binding_sha256 == self.binding_sha256
            )
            valid = (
                self.observed_boot_generation is not None
                and observed_binding_complete
                and self.reason_code == ""
                and self.matched is matched
                and (self.status == "VERIFIED") is matched
            )
        if not valid:
            raise ValueError("verification outcome is inconsistent")
        return self


class DiagnosticEffectResponse(ExactResponseModel):
    """GET-safe projection of the typed approval and durable effect ledger."""

    effect_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    effect_type: Literal["restart_simulated_robot"]
    source: Literal["SYNTHETIC_LOCAL"]
    arguments: DiagnosticEffectArgumentsResponse
    risk: DiagnosticEffectRiskResponse
    verifier: DiagnosticEffectVerifierResponse
    approval_status: DiagnosticEffectApprovalStatus
    decision: ResumeDecision | None = None
    issued_at: int = Field(strict=True, ge=0)
    expires_at: int = Field(strict=True, ge=0)
    can_decide: bool = Field(strict=True)
    status: DiagnosticEffectStatus
    attempt_count: int = Field(strict=True, ge=0, le=1)
    result: DiagnosticEffectResultResponse | None = None
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verification: DiagnosticEffectVerificationResponse | None = None
    verification_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    updated_at: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def exact_effect_lifecycle(self) -> "DiagnosticEffectResponse":
        result_present = self.result is not None and self.result_sha256 is not None
        result_absent = self.result is None and self.result_sha256 is None
        verification_present = (
            self.verification is not None and self.verification_sha256 is not None
        )
        verification_absent = (
            self.verification is None and self.verification_sha256 is None
        )
        if self.status is DiagnosticEffectStatus.PREPARED:
            runtime_valid = (
                self.attempt_count == 0 and result_absent and verification_absent
            )
        elif self.status in {
            DiagnosticEffectStatus.EXECUTING,
            DiagnosticEffectStatus.EXECUTION_FAILED,
        }:
            runtime_valid = (
                self.attempt_count == 1 and result_absent and verification_absent
            )
        elif self.status is DiagnosticEffectStatus.APPLIED:
            runtime_valid = (
                self.attempt_count == 1 and result_present and verification_absent
            )
        elif self.status in {
            DiagnosticEffectStatus.VERIFIED,
            DiagnosticEffectStatus.VERIFICATION_FAILED,
        }:
            runtime_valid = (
                self.attempt_count == 1 and result_present and verification_present
            )
        else:
            runtime_valid = (
                self.attempt_count == 1
                and not (self.result is None and self.verification is not None)
                and (self.result is None) == (self.result_sha256 is None)
                and (self.verification is None)
                == (self.verification_sha256 is None)
            )

        approval_valid = (
            self.approval_status is DiagnosticEffectApprovalStatus.PENDING
            and self.decision is None
            and self.status is DiagnosticEffectStatus.PREPARED
            or self.approval_status
            in {
                DiagnosticEffectApprovalStatus.CANCELLED,
                DiagnosticEffectApprovalStatus.EXPIRED,
            }
            and self.decision is None
            and self.can_decide is False
            and self.status is DiagnosticEffectStatus.PREPARED
            or self.approval_status is DiagnosticEffectApprovalStatus.CONSUMED
            and self.decision is ResumeDecision.REJECT
            and self.can_decide is False
            and self.status is DiagnosticEffectStatus.PREPARED
            or self.approval_status is DiagnosticEffectApprovalStatus.CONSUMED
            and self.decision is ResumeDecision.APPROVE
            and self.can_decide is False
        )
        if self.approval_status is DiagnosticEffectApprovalStatus.PENDING:
            approval_valid = approval_valid and self.can_decide
        if self.expires_at <= self.issued_at:
            runtime_valid = False

        if self.result is not None:
            runtime_valid = runtime_valid and (
                self.result.effect_type == self.effect_type
                and self.result.source == self.source
                and self.result.device_model == self.arguments.device_model
                and self.result.effect_key == self.effect_key
                and self.result.binding_sha256 == self.binding_sha256
            )
        if self.verification is not None:
            runtime_valid = runtime_valid and self.result is not None and (
                self.verification.status == self.status.value
                and self.verification.source == self.source
                and self.verification.device_model == self.arguments.device_model
                and self.verification.effect_key == self.effect_key
                and self.verification.binding_sha256 == self.binding_sha256
                and self.verification.expected_boot_generation
                == self.result.boot_generation_after
            )
        if not runtime_valid or not approval_valid:
            raise ValueError("effect lifecycle is inconsistent")
        return self


class DiagnosticEffectAuthorityV2Response(ExactResponseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    turn_number: int = Field(ge=1, le=1_000_000)
    user_query: str
    device_model: str
    fault_category: str
    fault_description: str

    @model_validator(mode="after")
    def canonical_context(self) -> "DiagnosticEffectAuthorityV2Response":
        from app.agent.claim_sources import CONTEXT_FIELDS, _validate_context
        _validate_context({key: getattr(self, key) for key in CONTEXT_FIELDS})
        return self


class DiagnosticEffectClaimSummaryV2Response(ExactResponseModel):
    status: Literal["VERIFIED"]
    verification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_ids: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def distinct_claims(self) -> "DiagnosticEffectClaimSummaryV2Response":
        if any(not item for item in self.claim_ids) or len(set(self.claim_ids)) != len(self.claim_ids):
            raise ValueError("claim summary is inconsistent")
        return self


class DiagnosticEffectActionBasisV2Response(ExactResponseModel):
    kind: Literal["SYNTHETIC_DEMO_POLICY"]
    policy: Literal["SYNTHETIC_BAD_TELEMETRY_DEMO_V2"]
    telemetry_evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_evidence(self) -> "DiagnosticEffectActionBasisV2Response":
        ids = self.telemetry_evidence_ids
        if any(not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("action basis is inconsistent")
        return self


class DiagnosticEffectProposalV2Response(ExactResponseModel):
    schema_version: int = Field(ge=2, le=2)
    effect_type: Literal["restart_simulated_robot"]
    source: Literal["SYNTHETIC_LOCAL"]
    arguments: DiagnosticEffectArgumentsResponse


class _DiagnosticEffectSubjectV2Response(ExactResponseModel):
    approval_kind: Literal["EFFECT_V2"]
    execution_contract: Literal["claim_effect_v2"]
    authority: DiagnosticEffectAuthorityV2Response
    interrupt_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    approval_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    effect_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proof_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal: DiagnosticEffectProposalV2Response
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk: DiagnosticEffectRiskResponse
    claim_summary: DiagnosticEffectClaimSummaryV2Response
    action_basis: DiagnosticEffectActionBasisV2Response
    issued_at: int = Field(ge=0, le=2**63 - 1 - 900_000)
    expires_at: int = Field(ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def consistent_subject(self) -> "_DiagnosticEffectSubjectV2Response":
        from app.agent.claim_sources import _digest
        if (
            self.expires_at != self.issued_at + 900_000
            or self.proposal.arguments.device_model != self.authority.device_model
            or _digest(self.proposal.model_dump(mode="json")) != self.proposal_sha256
        ):
            raise ValueError("V2 effect subject is inconsistent")
        return self


class DiagnosticEffectApprovalPayloadV2Response(_DiagnosticEffectSubjectV2Response):
    ttl_ms: int = Field(ge=900_000, le=900_000)
    status: Literal["PENDING", "EXPIRED"]
    decision: None
    can_decide: bool
    server_time: int = Field(ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def consistent_current_window(self) -> "DiagnosticEffectApprovalPayloadV2Response":
        expected = self.issued_at <= self.server_time < self.expires_at
        if self.server_time < self.issued_at or (
            self.status != ("PENDING" if expected else "EXPIRED")
            or self.can_decide is not expected
        ):
            raise ValueError("V2 approval availability is inconsistent")
        return self


class DiagnosticEffectV2Response(_DiagnosticEffectSubjectV2Response):
    """The V2 approval subject and separately bound schema1 execution facts."""

    approval_status: DiagnosticEffectApprovalStatus
    decision: ResumeDecision | None
    can_decide: bool
    status: DiagnosticEffectStatus
    attempt_count: int = Field(ge=0, le=1)
    result: DiagnosticEffectResultResponse | None
    result_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    verification: DiagnosticEffectVerificationResponse | None
    verification_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    updated_at: int = Field(ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def consistent_lifecycle(self) -> "DiagnosticEffectV2Response":
        from app.agent.claim_sources import _digest
        status = self.status.value
        approval = self.approval_status.value
        decision = self.decision.value if self.decision else None
        has_result = self.result is not None
        has_verification = self.verification is not None
        valid = (
            has_result == (self.result_sha256 is not None)
            and has_verification == (self.verification_sha256 is not None)
            and self.updated_at >= self.issued_at
        )
        if status == "PREPARED":
            valid &= self.attempt_count == 0 and not has_result and not has_verification
        elif status in {"EXECUTING", "EXECUTION_FAILED"}:
            valid &= self.attempt_count == 1 and not has_result and not has_verification
        elif status == "APPLIED":
            valid &= self.attempt_count == 1 and has_result and not has_verification
        elif status in {"VERIFIED", "VERIFICATION_FAILED"}:
            valid &= self.attempt_count == 1 and has_result and has_verification
        else:
            valid &= self.attempt_count == 1 and (has_result or not has_verification)
        if approval == "PENDING":
            valid &= decision is None and status == "PREPARED" and self.can_decide
        elif approval in {"CANCELLED", "EXPIRED"}:
            valid &= decision is None and status == "PREPARED" and not self.can_decide
        elif decision == "REJECT":
            valid &= status == "PREPARED" and not self.can_decide
        else:
            valid &= decision == "APPROVE" and not self.can_decide
        if self.result is not None:
            result = self.result
            valid &= (
                result.device_model == self.authority.device_model
                and result.effect_key == self.effect_key
                and result.binding_sha256 == self.binding_sha256
                and _digest(result.model_dump(mode="json")) == self.result_sha256
            )
        if self.verification is not None:
            verification = self.verification
            valid &= (
                self.result is not None and verification.status == status
                and verification.device_model == self.authority.device_model
                and verification.effect_key == self.effect_key
                and verification.binding_sha256 == self.binding_sha256
                and verification.expected_boot_generation == self.result.boot_generation_after
                and _digest(verification.model_dump(mode="json")) == self.verification_sha256
            )
        if not valid:
            raise ValueError("V2 effect lifecycle is inconsistent")
        return self


def _validate_v2_run_effect(run: "DiagnosticRunResponse") -> None:
    effect = run.effect
    if not isinstance(effect, DiagnosticEffectV2Response):
        raise ValueError("typed V2 interrupt is missing its effect projection")
    if any(getattr(effect.authority, key) != getattr(run, key) for key in (
        "session_id", "run_id", "request_id", "turn_number",
    )):
        raise ValueError("V2 effect belongs to another run")
    if run.result is not None and run.result.turn_number != run.turn_number:
        raise ValueError("V2 result belongs to another turn")
    if run.status is DiagnosticRunStatus.INTERRUPTED:
        if run.interrupt is None:
            raise ValueError("V2 interrupted run is missing its interrupt")
        payload = DiagnosticEffectApprovalPayloadV2Response.model_validate(run.interrupt.payload)
        public = payload.model_dump(mode="json")
        projected = effect.model_dump(mode="json")
        if (
            run.interrupt.interrupt_id != effect.interrupt_id
            or run.interrupt.approval_id != effect.approval_id
            or public["status"] != projected["approval_status"]
            or public["decision"] != projected["decision"]
            or public["can_decide"] != projected["can_decide"]
            or any(public[key] != projected[key] for key in _DiagnosticEffectSubjectV2Response.model_fields)
        ):
            raise ValueError("V2 interrupt and effect binding do not match")
    elif effect.can_decide or effect.approval_status is DiagnosticEffectApprovalStatus.PENDING:
        raise ValueError("decidable V2 effect requires an interrupted run")
    if effect.approval_status is DiagnosticEffectApprovalStatus.EXPIRED and run.status not in {
        DiagnosticRunStatus.INTERRUPTED, DiagnosticRunStatus.CANCELLED,
    }:
        raise ValueError("expired V2 effect requires an interrupted or cancelled run")
    if run.status is DiagnosticRunStatus.COMPLETED:
        terminal = effect.approval_status is DiagnosticEffectApprovalStatus.CONSUMED and (
            effect.decision is ResumeDecision.REJECT and effect.status is DiagnosticEffectStatus.PREPARED
            or effect.decision is ResumeDecision.APPROVE and effect.status in {
                DiagnosticEffectStatus.VERIFIED, DiagnosticEffectStatus.VERIFICATION_FAILED,
                DiagnosticEffectStatus.EXECUTION_FAILED, DiagnosticEffectStatus.UNKNOWN,
            }
        )
        if not terminal:
            raise ValueError("completed run has an active or incomplete V2 effect")


class DiagnosticRunResponse(StrictApiModel):
    session_id: str
    run_id: str
    request_id: str
    status: DiagnosticRunStatus
    session_status: DiagnosticSessionStatus
    turn_number: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    result: DiagnosticTurnResponse | None = None
    interrupt: DiagnosticInterruptResponse | None = None
    error: DiagnosticRunErrorResponse | None = None
    effect: DiagnosticEffectResponse | DiagnosticEffectV2Response | None = None
    tool_iterations: int = Field(default=0, ge=0, le=4)
    max_tool_iterations: int = Field(default=4, ge=4, le=4)
    stop_reason: str | None = None
    failure_disposition: FailureDisposition | None = None

    @model_validator(mode="after")
    def mutually_exclusive_outcomes(self) -> "DiagnosticRunResponse":
        outcomes = sum(
            value is not None for value in (self.result, self.interrupt, self.error)
        )
        if self.status is DiagnosticRunStatus.INTERRUPTED:
            valid = self.interrupt is not None and outcomes == 1
        elif self.status is DiagnosticRunStatus.COMPLETED:
            valid = self.result is not None and outcomes == 1
        elif self.status is DiagnosticRunStatus.FAILED:
            valid = (
                self.error is not None
                and outcomes == 1
                and self.failure_disposition is not None
            )
        else:
            valid = outcomes == 0 and self.failure_disposition is None
        if self.status is not DiagnosticRunStatus.FAILED and self.failure_disposition is not None:
            valid = False
        if not valid:
            raise ValueError("run outcome does not match its status")
        if self.result is not None:
            if self.result.session_id != self.session_id:
                raise ValueError("nested result belongs to another session")
            if self.result.status not in {
                DiagnosticSessionStatus.AWAITING_USER,
                DiagnosticSessionStatus.COMPLETED,
            }:
                raise ValueError("nested result has an invalid stable session status")
        if isinstance(self.effect, DiagnosticEffectV2Response) or (
            self.interrupt is not None and self.interrupt.payload.get("approval_kind") == "EFFECT_V2"
        ):
            _validate_v2_run_effect(self)
            return self
        typed_payload_requested = (
            self.interrupt is not None
            and isinstance(self.interrupt.payload, dict)
            and self.interrupt.payload.get("approval_kind") == "EFFECT_V1"
        )
        if typed_payload_requested and self.effect is None:
            raise ValueError("typed interrupt is missing its effect projection")
        if self.effect is not None:
            effect = self.effect
            if effect.approval_status is DiagnosticEffectApprovalStatus.PENDING or (
                effect.can_decide
            ):
                if (
                    self.status is not DiagnosticRunStatus.INTERRUPTED
                    or self.interrupt is None
                ):
                    raise ValueError("decidable effect requires an interrupted run")
            if effect.approval_status is DiagnosticEffectApprovalStatus.EXPIRED and (
                self.status
                not in {DiagnosticRunStatus.INTERRUPTED, DiagnosticRunStatus.CANCELLED}
                or (
                    self.status is DiagnosticRunStatus.INTERRUPTED
                    and self.interrupt is None
                )
            ):
                raise ValueError("expired effect requires an interrupted or cancelled run")

            if self.status is DiagnosticRunStatus.INTERRUPTED:
                if self.interrupt is None:
                    raise ValueError("typed interrupted run is missing its interrupt")
                try:
                    payload = DiagnosticEffectApprovalPayloadResponse.model_validate(
                        self.interrupt.payload
                    )
                except ValueError as exc:
                    raise ValueError("typed interrupt payload is invalid") from exc
                payload_matches = (
                    effect.approval_status.value == payload.status
                    and effect.decision is None
                    and effect.status is DiagnosticEffectStatus.PREPARED
                    and effect.attempt_count == 0
                    and effect.result is None
                    and effect.verification is None
                    and effect.effect_key == payload.effect_key
                    and effect.binding_sha256 == payload.binding_sha256
                    and effect.proposal_sha256 == payload.proposal_sha256
                    and effect.effect_type == payload.proposal.effect_type
                    and effect.source == payload.proposal.source
                    and effect.arguments == payload.proposal.arguments
                    and effect.risk == payload.risk
                    and effect.verifier == payload.verifier
                    and effect.issued_at == payload.issued_at
                    and effect.expires_at == payload.expires_at
                    and effect.can_decide is payload.can_decide
                )
                if not payload_matches:
                    raise ValueError("typed interrupt and effect binding do not match")

            if self.status is DiagnosticRunStatus.COMPLETED:
                approved_terminal = (
                    effect.approval_status
                    is DiagnosticEffectApprovalStatus.CONSUMED
                    and effect.decision is ResumeDecision.APPROVE
                    and effect.can_decide is False
                    and effect.status
                    in {
                        DiagnosticEffectStatus.VERIFIED,
                        DiagnosticEffectStatus.VERIFICATION_FAILED,
                        DiagnosticEffectStatus.EXECUTION_FAILED,
                        DiagnosticEffectStatus.UNKNOWN,
                    }
                )
                rejected_without_execution = (
                    effect.approval_status
                    is DiagnosticEffectApprovalStatus.CONSUMED
                    and effect.decision is ResumeDecision.REJECT
                    and effect.can_decide is False
                    and effect.status is DiagnosticEffectStatus.PREPARED
                    and effect.attempt_count == 0
                )
                if not (approved_terminal or rejected_without_execution):
                    raise ValueError("completed run has an active or incomplete effect")
            elif self.status in {
                DiagnosticRunStatus.CANCELLED,
                DiagnosticRunStatus.FAILED,
            } and (
                effect.approval_status is DiagnosticEffectApprovalStatus.PENDING
                or effect.can_decide
            ):
                raise ValueError("terminal run cannot retain a decidable effect")
        return self


class DiagnosticMessageResponse(StrictApiModel):
    message_id: str
    session_id: str
    sequence_no: int = Field(ge=1)
    turn_number: int = Field(ge=1)
    role: DiagnosticMessageRole
    kind: DiagnosticMessageKind
    content: str
    run_id: str
    request_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class DiagnosticMessagesResponse(StrictApiModel):
    session_id: str
    items: list[DiagnosticMessageResponse] = Field(default_factory=list)


class DiagnosticDraftTicketResponse(StrictApiModel):
    draft_id: str
    session_id: str
    source_run_id: str
    version: int = Field(ge=1)
    status: DiagnosticDraftStatus
    device_model: str
    fault_category: str
    fault_description: str
    diagnosis: list[str] = Field(default_factory=list)
    solution_steps: list[str] = Field(default_factory=list)
    risk_level: str
    need_human: bool
    external_ticket_id: str | None = None
    projection_note: str | None = None
    created_at: datetime
    updated_at: datetime


class DiagnosticSimilarHistoryItemResponse(StrictApiModel):
    source_id: str
    source: DiagnosticHistorySource
    summary: str
    similarity: float = Field(ge=0.0, le=1.0)


class DiagnosticSimilarHistoryResponse(StrictApiModel):
    session_id: str
    current_draft_id: str | None = None
    items: list[DiagnosticSimilarHistoryItemResponse] = Field(default_factory=list)
