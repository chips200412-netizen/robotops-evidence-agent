"""MOD-019/020：服务端权威会话、durable checkpoint 与 run 生命周期。"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import secrets
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions, Checkpoint, CheckpointMetadata, CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphDrained, GraphRecursionError
from langgraph.runtime import RunControl
from langgraph.types import Command

from app.agent import tool_loop
from app.agent.diagnostic_facts import merge_authoritative_facts
from app.agent.graph import build_graph
from app.agent.runtime_context import (
    ASYNC_AUTHORITY_SURFACE,
    SYNTHETIC_RESTART_EFFECT_MODE,
    AgentRuntimeContext,
)
from app.agent.state import (
    MAX_FOLLOW_UP_ROUNDS,
    MAX_TOOL_ITERATIONS,
    initial_state,
)
from app.observability.tracing import (
    safe_span, set_safe_approval_decision, set_safe_outcome,
    set_safe_persisted_event, set_safe_readback_matched, set_safe_run_correlation,
)
from app.schemas.diagnostic_sessions import (
    CreateDiagnosticSessionResponse,
    DiagnosticDraftStatus,
    DiagnosticDraftTicketResponse,
    DiagnosticEffectApprovalStatus,
    DiagnosticEffectResponse,
    DiagnosticEffectStatus,
    DiagnosticHistorySource,
    DiagnosticInterruptResponse,
    DiagnosticMessageKind,
    DiagnosticMessageResponse,
    DiagnosticMessageRole,
    DiagnosticMessagesResponse,
    DiagnosticRunErrorResponse,
    DiagnosticRunResponse,
    DiagnosticRunStatus,
    DiagnosticSessionResponse,
    DiagnosticSessionStatus,
    DiagnosticSimilarHistoryItemResponse,
    DiagnosticSimilarHistoryResponse,
    DiagnosticTurnResponse,
    FailureDisposition,
    ResumeDecision,
)
from app.services import effect_capabilities
from app.services.synthetic_effects import (
    SYNTHETIC_EFFECT_SOURCE,
    SYNTHETIC_EFFECT_TYPE,
    SyntheticEffectNotAppliedError,
    SyntheticEffectStore,
)


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32}$")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._~-]{1,128}$")
STOP_REASON_PATTERN = re.compile(r"^(?:[A-Z][A-Z0-9_]{0,79})?$")
SANITIZED_STORE_ERROR = "诊断会话暂时不可用，请稍后重试"
SESSION_NOT_FOUND = "诊断会话不存在"
SESSION_CONFLICT = "诊断会话当前不可写入"
RUN_NOT_FOUND = "诊断运行不存在"
RUN_CONFLICT = "诊断运行状态或审批绑定冲突"
GRAPH_RECURSION_LIMIT = 32
REGISTRY_SCHEMA_VERSION = "8"
DISABLED_CLAIM_EVIDENCE_PROFILE = "disabled_v0"
ACTIVE_CLAIM_EVIDENCE_PROFILE = "robotops_agent_claim_evidence_v1"
LEGACY_EXECUTION_CONTRACT = "legacy_effect_v1"
READONLY_EXECUTION_CONTRACT = "readonly_v1"
CLAIM_EXECUTION_CONTRACT = "claim_effect_v2"
EFFECT_MODE_SYNTHETIC_RESTART = SYNTHETIC_RESTART_EFFECT_MODE
EFFECT_CAPABILITY_TTL_MS = 900_000
EXECUTION_LIMIT_ERROR = "诊断执行已达到安全上限，请联系人工支持"
RUN_EXECUTION_ERROR = "诊断运行失败，请稍后重试或联系人工支持"
DRAFT_NOT_FOUND = "诊断会话尚无草稿工单"
DRAFT_CONFLICT = "草稿工单状态、标识或版本冲突"
REQUEST_NOT_FOUND = "诊断请求不存在"
FOLLOW_UP_LIMIT_REPLY = (
    "已达到最大追问轮数，当前信息仍不足。请停止进一步操作并联系人工支持。"
)

_WRITABLE_STATUSES = {
    DiagnosticSessionStatus.OPEN.value,
    DiagnosticSessionStatus.AWAITING_USER.value,
}
_RISK_RANK = {"": 0, "低": 1, "中": 2, "高": 3}
_RESTORED_STRING_FIELDS = (
    "user_query",
    "device_model",
    "fault_category",
    "fault_description",
    "risk_level",
    "final_answer",
    "urgency",
    "ticket_id",
    "ticket_note",
    "ticket_mode",
    "route",
    "risk_note",
    "llm_failure_node",
    "execution_stop_reason",
    "agent_route",
    "tool_capability_profile",
    "verification_status",
    "verifier_result_sha256",
    "effect_policy_status",
    "effect_proposal_sha256",
)
_RESTORED_LIST_FIELDS = (
    "retrieved_documents",
    "missing_information",
    "diagnosis",
    "solution_steps",
    "references",
    "follow_up_questions",
    "conversation",
    "completed_plan_steps",
    "tool_trajectory",
)
_RESTORED_DICT_FIELDS = (
    "diagnostic_plan",
    "pending_tool_action",
    "tool_result_evidence",
    "verifier_result",
    "effect_proposal",
    "effect_runtime",
)
_RESTORED_BOOL_FIELDS = (
    "need_human",
    "risk_check_failed",
    "llm_failed",
    "agent_tool_loop_enabled",
    "tool_loop_failed",
)
_ACTIVE_RUN_STATUSES = {
    DiagnosticRunStatus.QUEUED.value,
    DiagnosticRunStatus.RUNNING.value,
    DiagnosticRunStatus.INTERRUPTED.value,
    DiagnosticRunStatus.CANCEL_REQUESTED.value,
}
_RETRYABLE_RUN_ERRORS = {"INITIAL_RUN_SCHEDULING_FAILED"}
_TYPED_APPROVAL_COLUMNS = (
    "approval_id", "session_id", "run_id", "interrupt_id", "approval_kind",
    "effect_key", "effect_type", "source", "proposal_json", "proposal_sha256",
    "arguments_json", "arguments_sha256", "risk_summary_json",
    "risk_summary_sha256", "verifier_summary_json", "verifier_summary_sha256",
    "verifier_result_sha256", "binding_json", "binding_sha256", "nonce_digest",
    "issued_at", "expires_at", "status", "decision", "consumed_at",
)
_EFFECT_LEDGER_COLUMNS = (
    "effect_key", "approval_id", "binding_sha256", "effect_type", "source",
    "arguments_json", "arguments_sha256", "status", "attempt_count",
    "result_json", "result_sha256", "verification_json", "verification_sha256",
    "created_at", "updated_at",
)
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EFFECT_RUNTIME_FIELDS = {
    "schema_version",
    "effect_key",
    "binding_sha256",
    "status",
    "attempt_count",
    "result_sha256",
    "verification_sha256",
}
_EFFECT_ATTEMPT_TERMINAL_STATUSES = {
    "APPLIED",
    "VERIFIED",
    "EXECUTION_FAILED",
    "VERIFICATION_FAILED",
    "UNKNOWN",
}

# 本地实现明确只承诺单进程；同一进程内可能存在多个 service 实例，因此
# RunControl 必须按数据库与 run 路由到真正执行者，而不能只放在实例字段中。
_PROCESS_CONTROLS_LOCK = threading.RLock()
_PROCESS_RUN_CONTROLS: dict[tuple[str, str], RunControl] = {}
_PROCESS_SYNC_SESSION_OWNERS: dict[tuple[str, str], object] = {}


class InvalidSessionIdError(ValueError):
    """会话标识不符合公开 token 合同。"""


class SessionNotFoundError(LookupError):
    """合法格式的会话不存在。"""


class SessionConflictError(RuntimeError):
    """会话正在运行或已经进入终态。"""


class SessionStoreError(RuntimeError):
    """持久层或 checkpoint 不可安全读取；对外不得附带底层异常。"""


class InvalidRunIdError(ValueError):
    """run 标识不符合公开 token 合同。"""


class RunNotFoundError(LookupError):
    """合法格式的 run 不存在。"""


class RunConflictError(RuntimeError):
    """run 状态或精确审批绑定冲突。"""


class RequestNotFoundError(LookupError):
    """合法 request_id 在指定会话中不存在。"""


class DraftNotFoundError(LookupError):
    """会话存在，但尚未形成草稿工单。"""


class InvalidDraftIdError(ValueError):
    """草稿标识不符合公开 token 合同。"""


class DraftConflictError(RuntimeError):
    """草稿状态、版本或投影绑定冲突。"""


class RunPersistenceError(SessionStoreError):
    """图已产出结果，但本地终态事务未能完整提交。"""


class RunCheckpointError(SessionStoreError):
    """权威 checkpoint 读取或写入失败。"""


@dataclass(frozen=True)
class _SessionRecord:
    session_id: str
    thread_id: str
    status: str
    created_at: str
    updated_at: str
    turn_count: int
    claim_evidence_profile: str = DISABLED_CLAIM_EVIDENCE_PROFILE


@dataclass(frozen=True)
class _RunRecord:
    run_id: str
    session_id: str
    turn_number: int
    status: str
    previous_session_status: str
    base_turn_count: int
    created_at: str
    updated_at: str
    result_json: str | None
    error_code: str | None
    tool_iterations: int
    max_tool_iterations: int
    stop_reason: str | None
    current_interrupt_id: str | None


class _ServiceEffectAuthority:
    """Expose only the two authority operations required by graph runtime context."""

    __slots__ = ("__service",)

    def __init__(self, service: "DiagnosticSessionService") -> None:
        self.__service = service

    def execute_effect(
        self,
        *,
        session_id: str,
        run_id: str,
        approval_id: str,
        proposal: Mapping[str, Any],
        proposal_sha256: str,
    ) -> Mapping[str, Any]:
        return self.__service._execute_authorized_effect(
            session_id=session_id,
            run_id=run_id,
            approval_id=approval_id,
            proposal=proposal,
            proposal_sha256=proposal_sha256,
        )

    def verify_effect(
        self,
        *,
        session_id: str,
        run_id: str,
        approval_id: str,
        proposal: Mapping[str, Any],
        proposal_sha256: str,
    ) -> Mapping[str, Any]:
        return self.__service._verify_authorized_effect(
            session_id=session_id,
            run_id=run_id,
            approval_id=approval_id,
            proposal=proposal,
            proposal_sha256=proposal_sha256,
        )


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _event_token_digest(token: str) -> str:
    """为审计去重生成不可逆标识，不把一次性审批 capability 写入事件。"""
    return hashlib.sha256(f"robotops-event-approval-v1:{token}".encode("utf-8")).hexdigest()


def _higher_risk(previous: object, current: object) -> str:
    if previous not in _RISK_RANK or current not in _RISK_RANK:
        raise ValueError("invalid persisted risk level")
    return str(previous if _RISK_RANK[previous] >= _RISK_RANK[current] else current)


def select_similar_history(
    candidates: Iterable[Mapping[str, Any]],
    *,
    exclude_draft_id: str | None,
    exclude_ticket_id: str | None,
    limit: int,
) -> list[DiagnosticSimilarHistoryItemResponse]:
    """先排除当前草稿/外部工单，再排序和 LIMIT。"""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("invalid history limit")
    normalized: list[DiagnosticSimilarHistoryItemResponse] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        draft_id = candidate.get("draft_id")
        ticket_id = candidate.get("external_ticket_id") or candidate.get("ticket_id")
        case_id = candidate.get("case_id") or candidate.get("id")
        draft_id = draft_id if isinstance(draft_id, str) else ""
        ticket_id = ticket_id if isinstance(ticket_id, str) else ""
        case_id = case_id if isinstance(case_id, str) else ""
        if exclude_draft_id and draft_id == exclude_draft_id:
            continue
        if exclude_ticket_id and ticket_id == exclude_ticket_id:
            continue
        if draft_id:
            source_id = draft_id
            source = DiagnosticHistorySource.DRAFT
        elif ticket_id:
            source_id = ticket_id
            source = DiagnosticHistorySource.TICKET
        elif case_id:
            source_id = case_id
            source = DiagnosticHistorySource.CASE
        else:
            continue
        raw_similarity = candidate.get("similarity", 0.0)
        if isinstance(raw_similarity, bool) or not isinstance(
            raw_similarity, (int, float)
        ):
            continue
        similarity = float(raw_similarity)
        if not 0.0 <= similarity <= 1.0:
            continue
        summary = candidate.get("summary") or candidate.get("fault_description")
        if not isinstance(summary, str) or not summary.strip():
            summary = "历史诊断记录"
        normalized.append(
            DiagnosticSimilarHistoryItemResponse(
                source_id=source_id,
                source=source,
                summary=summary.strip(),
                similarity=similarity,
            )
        )
    normalized.sort(key=lambda item: (-item.similarity, item.source_id))
    return normalized[:limit]


def merge_sticky_safety(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict:
    """跨轮合并安全状态：风险、人工与失败标记只能保持或升级。"""
    merged = dict(current)
    merged["risk_level"] = _higher_risk(
        previous.get("risk_level", ""), current.get("risk_level", "")
    )
    merged["urgency"] = _higher_risk(
        previous.get("urgency", ""), current.get("urgency", "")
    )
    for field in ("need_human", "llm_failed", "risk_check_failed"):
        old_value = previous.get(field, False)
        new_value = current.get(field, False)
        if not isinstance(old_value, bool) or not isinstance(new_value, bool):
            raise ValueError("invalid persisted safety flag")
        merged[field] = old_value or new_value
    if merged["risk_level"] == "高" or merged["urgency"] == "高":
        merged["need_human"] = True
    if previous.get("llm_failed") is True:
        merged["llm_failure_node"] = previous.get("llm_failure_node", "")
    else:
        merged["llm_failure_node"] = current.get("llm_failure_node", "")
    if previous.get("risk_check_failed") is True:
        merged["risk_note"] = previous.get("risk_note", "")
    else:
        merged["risk_note"] = current.get("risk_note", "")
    return merged


def _follow_up_reply(questions: list[dict]) -> str:
    lines = ["【请补充信息】，请补充以下信息："]
    for index, question in enumerate(questions, 1):
        options = " / ".join(question.get("options") or [])
        suffix = f"（可选：{options}）" if options else ""
        lines.append(f"{index}. {question.get('question', '')}{suffix}")
    return "\n".join(lines)


def _infer_nodes(result: Mapping[str, Any]) -> list[str]:
    nodes = ["parse", "identify", "classify", "retrieve", "judge"]
    if result.get("route") == "ask":
        return [*nodes, "ask"]
    if result.get("llm_failure_node") == "ask":
        nodes.append("ask")
    nodes.append("generate")
    if result.get("risk_level"):
        nodes.append("risk")
    if result.get("ticket_id") or (
        result.get("ticket_mode") == "draft" and result.get("route") != "ask"
    ):
        nodes.append("ticket")
    nodes.append("answer")
    return nodes


def _diagnosis(result: Mapping[str, Any]) -> dict:
    return {
        "device_model": result.get("device_model") or "",
        "fault_category": result.get("fault_category") or "",
        "risk_level": result.get("risk_level") or "",
        "causes": result.get("diagnosis") or [],
        "steps": result.get("solution_steps") or [],
        "need_human": result.get("need_human") is True,
    }


class _ObservedSqliteSaver(SqliteSaver):
    """Observe native calls without changing serialization or transaction ownership.

    COMPLETED means the superclass returned, including absent reads and deduped
    writes. ERROR does not imply rollback: the native cursor commits in finally.
    """

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        with safe_span("robotops.checkpoint.get_tuple") as span:
            result = super().get_tuple(config)
            set_safe_outcome(span, "COMPLETED")
            return result

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with safe_span("robotops.checkpoint.put") as span:
            result = super().put(config, checkpoint, metadata, new_versions)
            set_safe_outcome(span, "COMPLETED")
            return result

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with safe_span("robotops.checkpoint.put_writes") as span:
            result = super().put_writes(config, writes, task_id, task_path)
            set_safe_outcome(span, "COMPLETED")
            return result


class DiagnosticSessionService:
    """SQLite-backed 单进程实现；生产多 worker 后端留待 ADR。"""

    def __init__(
        self,
        db_path: str | Path,
        *,
        graph_factory: Callable[..., Any] = build_graph,
        history_provider: Callable[..., Iterable[Mapping[str, Any]]]
        | None = None,
        draft_projector: Callable[
            [DiagnosticDraftTicketResponse, str], Mapping[str, Any]
        ]
        | None = None,
        enable_agent_tool_loop: bool = False,
        tool_capability_profile: str = tool_loop.CORE_TOOL_PROFILE,
        effect_mode: str | None = None,
        synthetic_effect_adapter: object | None = None,
        synthetic_effect_db_path: str | Path | None = None,
        utc_now_ms: Callable[[], int] | None = None,
        claim_evidence_profile: str = DISABLED_CLAIM_EVIDENCE_PROFILE,
        execution_contract: str | None = None,
    ) -> None:
        # New-session selection is private; historical sessions retain their profile.
        if (
            type(claim_evidence_profile) is not str
            or claim_evidence_profile not in {
                DISABLED_CLAIM_EVIDENCE_PROFILE, ACTIVE_CLAIM_EVIDENCE_PROFILE,
            }
            or (claim_evidence_profile == ACTIVE_CLAIM_EVIDENCE_PROFILE and (
                enable_agent_tool_loop is not True
                or type(tool_capability_profile) is not str
                or tool_capability_profile != tool_loop.SYNTHETIC_EVIDENCE_TOOL_PROFILE
            ))
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        default_contract = (LEGACY_EXECUTION_CONTRACT
                            if claim_evidence_profile == DISABLED_CLAIM_EVIDENCE_PROFILE
                            else READONLY_EXECUTION_CONTRACT)
        selected_contract = default_contract if execution_contract is None else execution_contract
        if (type(selected_contract) is not str
                or selected_contract not in {default_contract, CLAIM_EXECUTION_CONTRACT}
                or (selected_contract == CLAIM_EXECUTION_CONTRACT and (
                    claim_evidence_profile != ACTIVE_CLAIM_EVIDENCE_PROFILE
                    or effect_mode != EFFECT_MODE_SYNTHETIC_RESTART
                ))):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        self._new_execution_contract = selected_contract
        self._registry_lock = threading.RLock()
        self._committed_run_observer_local = threading.local()
        self._worker_lock = threading.RLock()
        self._sync_condition = threading.Condition(self._worker_lock)
        self._closing = False
        self._active_controls: dict[str, RunControl] = {}
        self._sync_owner_tokens: dict[str, object] = {}
        self._futures: set[Future[Any]] = set()
        self._executor: ThreadPoolExecutor | None = None
        self._registry: sqlite3.Connection | None = None
        self._checkpoint_connection: sqlite3.Connection | None = None
        self._synthetic_effect_adapter: object | None = None
        self._owns_synthetic_effect_adapter = False
        self._graph_accepts_context = False
        self._history_provider = history_provider
        self._draft_projector = draft_projector
        if not isinstance(enable_agent_tool_loop, bool):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        try:
            tool_loop.registry_for_profile(tool_capability_profile)
        except tool_loop.ToolLoopError as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        if (
            not enable_agent_tool_loop
            and tool_capability_profile != tool_loop.CORE_TOOL_PROFILE
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if effect_mode not in {None, EFFECT_MODE_SYNTHETIC_RESTART}:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if effect_mode is not None and (
            not enable_agent_tool_loop
            or tool_capability_profile != tool_loop.SYNTHETIC_EVIDENCE_TOOL_PROFILE
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if effect_mode is None and (
            synthetic_effect_adapter is not None
            or synthetic_effect_db_path is not None
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if (
            synthetic_effect_adapter is not None
            and synthetic_effect_db_path is not None
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if utc_now_ms is not None and not callable(utc_now_ms):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        self._enable_agent_tool_loop = enable_agent_tool_loop
        self._tool_capability_profile = tool_capability_profile
        self._claim_evidence_profile = claim_evidence_profile
        self._effect_mode = effect_mode
        self._utc_now_ms = utc_now_ms
        try:
            self.db_path = Path(db_path).resolve()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._registry = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=5.0
            )
            self._checkpoint_connection = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=5.0
            )
            self._configure_connection(self._registry)
            self._configure_connection(self._checkpoint_connection)
            self._setup_registry()
            strict_serde = JsonPlusSerializer(allowed_msgpack_modules=None)
            self._checkpointer = _ObservedSqliteSaver(
                self._checkpoint_connection, serde=strict_serde
            )
            self._checkpointer.setup()
            self._graph = graph_factory(checkpointer=self._checkpointer)
            self._graph_accepts_context = self._accepts_keyword(
                getattr(self._graph, "invoke", None), "context"
            )
            if self._effect_mode is not None and not self._graph_accepts_context:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if self._effect_mode == EFFECT_MODE_SYNTHETIC_RESTART:
                if synthetic_effect_adapter is None:
                    adapter_path = (
                        Path(synthetic_effect_db_path).resolve()
                        if synthetic_effect_db_path is not None
                        else self.db_path.with_name(
                            f"{self.db_path.stem}-synthetic-effects.sqlite3"
                        )
                    )
                    self._synthetic_effect_adapter = SyntheticEffectStore(
                        adapter_path
                    )
                    self._owns_synthetic_effect_adapter = True
                else:
                    self._synthetic_effect_adapter = synthetic_effect_adapter
                if not all(
                    callable(getattr(self._synthetic_effect_adapter, name, None))
                    for name in ("restart_simulated_robot", "read_device_state")
                ):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
            self._effect_authority = _ServiceEffectAuthority(self)
            self._recover_applied_effects()
            self._recover_applied_claim_effects()
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="robotops-run"
            )
        except Exception as exc:
            self._close_synthetic_effect_adapter()
            self._close_connections()
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    @staticmethod
    def _accepts_keyword(callable_value: object, keyword: str) -> bool:
        if not callable(callable_value):
            return False
        try:
            parameters = inspect.signature(callable_value).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or parameter.name == keyword
            for parameter in parameters
        )

    @staticmethod
    def _configure_connection(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=5000")

    def _typed_now_ms(self) -> int:
        try:
            value = (
                self._utc_now_ms()
                if self._utc_now_ms is not None
                else int(datetime.now(timezone.utc).timestamp() * 1000)
            )
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        if type(value) is not int or value < 0:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return value

    @staticmethod
    def _observe_committed_run(
        run_id: str, event_type: str,
        *, name: str = "robotops.persistence.run_state",
    ) -> None:
        """Short post-commit fact, never a transaction duration or write count."""
        try:
            with safe_span(name) as span:
                set_safe_run_correlation(span, run_id)
                set_safe_persisted_event(span, event_type)
                set_safe_outcome(span, "COMPLETED")
        except Exception:
            pass

    @contextmanager
    def _committed_run_observations(self) -> Iterator[list[tuple[str, str, str]]]:
        """Collect explicit committed facts and emit after the transaction lock exits.

        Each owner appends only after its own commit returns. The enclosing
        registry context exits first, even on early return or later exceptions.
        No replay or failed commit appends a fact.
        """
        # Thread-local state is not copied into LangGraph's background executor.
        # Nested owners (shutdown/early resume cancellation) defer to the current
        # outer scope without adding arguments to the existing business helpers.
        try:
            pending = getattr(self._committed_run_observer_local, "pending", None)
        except Exception:
            yield []
            return
        if (type(pending) is list and len(pending) == 2 and pending[0] is True
                and type(pending[1]) is list):
            yield pending[1]
            return
        committed: list[tuple[str, str, str]] = []
        scope = [True, committed]
        try:
            self._committed_run_observer_local.pending = scope
        except Exception:
            yield []
            return
        try:
            yield committed
        finally:
            # Invalidate and clear before any observer work. Even a failed local
            # cleanup cannot make the next run inherit an active collector.
            scope[0] = False
            facts = tuple(committed)
            committed.clear()
            try:
                self._committed_run_observer_local.pending = None
            except Exception:
                try:
                    del self._committed_run_observer_local.pending
                except Exception:
                    pass
            for run_id, event_type, name in facts:
                self._observe_committed_run(run_id, event_type, name=name)

    def _setup_registry(self) -> None:
        legacy_running: list[str] = []
        committed_orphans: list[tuple[str, str]] = []
        try:
            self._registry.execute("BEGIN IMMEDIATE")
            meta_exists = self._table_exists("diagnostic_registry_meta")
            version_row = (
                self._registry.execute(
                    "SELECT value FROM diagnostic_registry_meta "
                    "WHERE key = 'schema_version'"
                ).fetchone()
                if meta_exists
                else None
            )
            if meta_exists and version_row is None:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            version = None if version_row is None else str(version_row[0])
            if version not in {None, "2", "3", "4", "5", "6", "7", REGISTRY_SCHEMA_VERSION}:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if version not in {"6", "7", REGISTRY_SCHEMA_VERSION} and self._v6_objects_present():
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if version not in {"7", REGISTRY_SCHEMA_VERSION} and self._v7_objects_present():
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if version != REGISTRY_SCHEMA_VERSION and self._v8_objects_present():
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if version is None and not meta_exists:
                diagnostic_tables = {
                    str(row[0])
                    for row in self._registry.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' "
                        "AND name LIKE 'diagnostic_%'"
                    ).fetchall()
                }
                if diagnostic_tables not in (set(), {"diagnostic_sessions"}):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                if diagnostic_tables == {"diagnostic_sessions"}:
                    legacy_columns = {
                        "session_id", "thread_id", "status", "created_at",
                        "updated_at", "turn_count",
                    }
                    existing_columns = {
                        str(row[1])
                        for row in self._registry.execute(
                            "PRAGMA table_info(diagnostic_sessions)"
                        ).fetchall()
                    }
                    if existing_columns != legacy_columns:
                        raise SessionStoreError(SANITIZED_STORE_ERROR)

            table = self._registry.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'diagnostic_sessions'"
            ).fetchone()
            if table is None:
                if version is not None:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                self._create_session_table("diagnostic_sessions")
            else:
                columns = {
                    row[1]
                    for row in self._registry.execute(
                        "PRAGMA table_info(diagnostic_sessions)"
                    )
                }
                legacy_columns = {
                    "session_id",
                    "thread_id",
                    "status",
                    "created_at",
                    "updated_at",
                    "turn_count",
                }
                current_columns = {
                    *legacy_columns,
                    "parent_session_id",
                    "active_run_id",
                }
                sql = str(table[0] or "")
                if columns == legacy_columns:
                    if version is not None:
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    legacy_running = [
                        row[0]
                        for row in self._registry.execute(
                            "SELECT session_id FROM diagnostic_sessions "
                            "WHERE status = 'RUNNING'"
                        )
                    ]
                    self._migrate_mod019_sessions()
                elif columns != current_columns or "CANCEL_REQUESTED" not in sql:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)

            if version == REGISTRY_SCHEMA_VERSION:
                # A declared current schema is never repaired by backfilling.
                self._validate_v8_schema()
            else:
                if version == "7":
                    self._validate_v7_schema()
                else:
                    if version == "6":
                        self._validate_v6_schema()
                    else:
                        if version == "5":
                            self._validate_v5_schema()
                        else:
                            if self._v5_objects_present():
                                raise SessionStoreError(SANITIZED_STORE_ERROR)
                            if version == "4":
                                # Validate the predecessor before any additive migration.
                                self._validate_v4_schema()
                            else:
                                self._create_run_tables()
                                self._validate_v4_schema()
                            self._create_v5_tables()
                            self._validate_v5_schema()
                        self._create_v6_profiles()
                        self._validate_v6_schema()
                    self._create_v7_contracts()
                    self._validate_v7_schema()
                self._create_v8_tables()
                self._validate_v8_schema()

            if version == "2":
                self._backfill_mod020_requests()
            self._reconcile_orphaned_effects()
            self._reconcile_orphaned_claim_effects()
            committed_orphans = self._reconcile_orphaned_runs()
            self._reconcile_orphaned_projections()
            for session_id in legacy_running:
                self._append_event(
                    session_id=session_id,
                    run_id=None,
                    event_type="MIGRATION_ORPHAN_FAILED",
                    from_status="RUNNING",
                    to_status="FAILED",
                    reason_code="LEGACY_RUNNING_WITHOUT_RUN",
                    dedupe_key=f"migration:{session_id}:legacy-running",
                )
            violations = self._registry.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if version is None:
                self._registry.execute(
                    "INSERT INTO diagnostic_registry_meta(key, value) "
                    "VALUES ('schema_version', ?)"
                    , (REGISTRY_SCHEMA_VERSION,)
                )
            elif version in {"2", "3", "4", "5", "6", "7"}:
                changed = self._registry.execute(
                    "UPDATE diagnostic_registry_meta SET value = ? "
                    "WHERE key = 'schema_version' AND value = ?",
                    (REGISTRY_SCHEMA_VERSION, version),
                ).rowcount
                if changed != 1:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
            self._registry.commit()
        except Exception as exc:
            try:
                self._registry.rollback()
            except Exception:
                pass
            if isinstance(exc, SessionStoreError):
                raise
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        for run_id, event_type in committed_orphans:
            self._observe_committed_run(
                run_id, event_type, name="robotops.persistence.orphan_state",
            )

    def _table_exists(self, name: str) -> bool:
        return (
            self._registry.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (name,),
            ).fetchone()
            is not None
        )

    @contextmanager
    def _registry_transaction(self) -> Iterator[None]:
        """Keep failure cleanup inside ownership of the shared connection.

        Callers retain their explicit BEGIN/commit decisions. Read helpers must
        continue to use the bare reentrant lock, never a committing context.
        No caller may repeat rollback after leaving this write scope.
        """
        with self._registry_lock:
            try:
                yield
            except BaseException:
                try:
                    self._registry.rollback()
                except Exception:
                    pass
                raise

    @staticmethod
    def _v6_profile_schema_statement() -> str:
        return """
            CREATE TABLE diagnostic_session_claim_profiles (
                session_id TEXT PRIMARY KEY NOT NULL
                    REFERENCES diagnostic_sessions(session_id),
                claim_evidence_profile TEXT NOT NULL CHECK (
                    claim_evidence_profile IN (
                        'disabled_v0', 'robotops_agent_claim_evidence_v1'
                    )
                )
            )
            """

    def _v6_objects_present(self) -> bool:
        return any(
            str(row[0]).casefold().startswith("diagnostic_session_claim_profile")
            or str(row[0]).casefold().endswith(("_v6", "_mod026"))
            for row in self._registry.execute(
                "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
        )

    def _create_v6_profiles(self) -> None:
        self._registry.execute(self._v6_profile_schema_statement())
        self._registry.execute(
            "INSERT INTO diagnostic_session_claim_profiles "
            "(session_id, claim_evidence_profile) "
            "SELECT session_id, ? FROM diagnostic_sessions",
            (DISABLED_CLAIM_EVIDENCE_PROFILE,),
        )

    def _validate_v6_schema(
        self, *, allow_v7_tables: bool = False, allow_v8_tables: bool = False,
    ) -> None:
        """Validate the additive table without relaxing any v5 business contract."""
        self._validate_v5_schema(allow_v6_tables=True, allow_v7_tables=allow_v7_tables,
                                 allow_v8_tables=allow_v8_tables)
        objects = self._registry.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        profile_objects = [
            row for row in objects
            if str(row[1]).casefold().startswith("diagnostic_session_claim_profile")
            or str(row[1]).casefold().endswith(("_v6", "_mod026"))
        ]
        if (
            len(profile_objects) != 1
            or profile_objects[0][0] != "table"
            or profile_objects[0][1] != "diagnostic_session_claim_profiles"
            or profile_objects[0][2]
            != self._v6_profile_schema_statement().strip()
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        columns = tuple(
            tuple(row) for row in self._registry.execute(
                "PRAGMA table_info(diagnostic_session_claim_profiles)"
            ).fetchall()
        )
        if columns != (
            (0, "session_id", "TEXT", 1, None, 1),
            (1, "claim_evidence_profile", "TEXT", 1, None, 0),
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        foreign_keys = tuple(
            tuple(row) for row in self._registry.execute(
                "PRAGMA foreign_key_list(diagnostic_session_claim_profiles)"
            ).fetchall()
        )
        if foreign_keys != (
            (0, 0, "diagnostic_sessions", "session_id", "session_id",
             "NO ACTION", "NO ACTION", "NONE"),
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        indexes = self._registry.execute(
            "PRAGMA index_list(diagnostic_session_claim_profiles)"
        ).fetchall()
        if len(indexes) != 1 or tuple(indexes[0][2:]) != (1, "pk", 0):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        index_columns = self._registry.execute(
            "PRAGMA index_info(sqlite_autoindex_diagnostic_session_claim_profiles_1)"
        ).fetchall()
        if [tuple(row) for row in index_columns] != [(0, 0, "session_id")]:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if self._registry.execute(
            "SELECT 1 FROM diagnostic_sessions AS sessions "
            "LEFT JOIN diagnostic_session_claim_profiles AS profiles "
            "ON profiles.session_id = sessions.session_id "
            "WHERE profiles.session_id IS NULL LIMIT 1"
        ).fetchone() is not None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if self._registry.execute(
            "SELECT 1 FROM diagnostic_session_claim_profiles AS profiles "
            "LEFT JOIN diagnostic_sessions AS sessions "
            "ON sessions.session_id = profiles.session_id "
            "WHERE sessions.session_id IS NULL "
            "OR profiles.claim_evidence_profile NOT IN (?, ?) LIMIT 1",
            (DISABLED_CLAIM_EVIDENCE_PROFILE, ACTIVE_CLAIM_EVIDENCE_PROFILE),
        ).fetchone() is not None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        active = self._registry.execute(
            "SELECT session_id FROM diagnostic_session_claim_profiles "
            "WHERE claim_evidence_profile = ?",
            (ACTIVE_CLAIM_EVIDENCE_PROFILE,),
        ).fetchall()
        for row in active:
            self._read_v6_claim_profile_locked(row[0])
        if self._registry.execute("PRAGMA foreign_key_check").fetchall():
            raise SessionStoreError(SANITIZED_STORE_ERROR)

    def _read_v6_claim_profile_locked(self, session_id: str) -> str:
        """Read the predecessor's authority without depending on a v7 expansion.

        Only the v6 validator and the complete runtime guard use this seam.
        Runtime callers must not bypass the persisted execution pair below.
        """
        with self._registry_lock:
            rows = self._registry.execute(
                "SELECT profiles.claim_evidence_profile "
                "FROM diagnostic_sessions AS sessions "
                "LEFT JOIN diagnostic_session_claim_profiles AS profiles "
                "ON profiles.session_id = sessions.session_id "
                "WHERE sessions.session_id = ? LIMIT 2",
                (session_id,),
            ).fetchall()
        if (
            len(rows) != 1
            or type(rows[0][0]) is not str
            or rows[0][0] not in {
                DISABLED_CLAIM_EVIDENCE_PROFILE, ACTIVE_CLAIM_EVIDENCE_PROFILE,
            }
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if rows[0][0] == ACTIVE_CLAIM_EVIDENCE_PROFILE:
            if (self._enable_agent_tool_loop is not True
                    or self._tool_capability_profile != tool_loop.SYNTHETIC_EVIDENCE_TOOL_PROFILE):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            with self._registry_lock:
                if self._registry.execute(
                    "SELECT 1 FROM diagnostic_effect_approvals WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone() is not None:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
        return rows[0][0]

    def _claim_profile_locked(self, session_id: str) -> str:
        """Read both persisted axes; missing data is corruption, never a default."""
        with self._registry_lock:
            profile = self._read_v6_claim_profile_locked(session_id)
            self._execution_contract_locked(session_id, claim_profile=profile)
            return profile

    def _execution_contract_locked(
        self, session_id: str, *, claim_profile: str | None = None,
        allow_claim_effect: bool = True,
    ) -> str:
        with self._registry_lock:
            if claim_profile is None:
                claim_profile = self._read_v6_claim_profile_locked(session_id)
            rows = self._registry.execute(
                "SELECT profiles.claim_evidence_profile, contracts.execution_contract "
                "FROM diagnostic_sessions AS sessions "
                "LEFT JOIN diagnostic_session_claim_profiles AS profiles "
                "ON profiles.session_id = sessions.session_id "
                "LEFT JOIN diagnostic_session_execution_contracts AS contracts "
                "ON contracts.session_id = sessions.session_id "
                "WHERE sessions.session_id = ? LIMIT 2", (session_id,),
            ).fetchall()
        allowed_pairs = {
            (DISABLED_CLAIM_EVIDENCE_PROFILE, LEGACY_EXECUTION_CONTRACT),
            (ACTIVE_CLAIM_EVIDENCE_PROFILE, READONLY_EXECUTION_CONTRACT),
        }
        if allow_claim_effect and self._effect_mode == EFFECT_MODE_SYNTHETIC_RESTART:
            allowed_pairs.add((ACTIVE_CLAIM_EVIDENCE_PROFILE, CLAIM_EXECUTION_CONTRACT))
        if (len(rows) != 1 or type(rows[0][0]) is not str or type(rows[0][1]) is not str
                or rows[0][0] != claim_profile
                or (rows[0][0], rows[0][1]) not in allowed_pairs):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return rows[0][1]

    def _insert_claim_profile_locked(self, session_id: str) -> str:
        self._registry.execute(
            "INSERT INTO diagnostic_session_claim_profiles "
            "(session_id, claim_evidence_profile) VALUES (?, ?)",
            (session_id, self._claim_evidence_profile),
        )
        contract = self._new_execution_contract
        self._registry.execute(
            "INSERT INTO diagnostic_session_execution_contracts "
            "(session_id, execution_contract) VALUES (?, ?)", (session_id, contract),
        )
        return self._claim_profile_locked(session_id)

    @staticmethod
    def _v7_contract_schema_statement() -> str:
        return """
            CREATE TABLE diagnostic_session_execution_contracts (
                session_id TEXT PRIMARY KEY NOT NULL
                    REFERENCES diagnostic_sessions(session_id),
                execution_contract TEXT NOT NULL CHECK (
                    execution_contract IN (
                        'legacy_effect_v1', 'readonly_v1', 'claim_effect_v2'
                    )
                )
            )
            """

    def _v7_objects_present(self) -> bool:
        return any(
            str(row[0]).casefold().startswith("diagnostic_session_execution_contract")
            or str(row[0]).casefold().endswith("_v7")
            for row in self._registry.execute(
                "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
        )

    def _create_v7_contracts(self) -> None:
        self._registry.execute(self._v7_contract_schema_statement())
        # The validated predecessor, never deployment configuration, owns this map.
        self._registry.execute(
            "INSERT INTO diagnostic_session_execution_contracts "
            "(session_id, execution_contract) "
            "SELECT session_id, CASE claim_evidence_profile WHEN ? THEN ? ELSE ? END "
            "FROM diagnostic_session_claim_profiles",
            (DISABLED_CLAIM_EVIDENCE_PROFILE, LEGACY_EXECUTION_CONTRACT, READONLY_EXECUTION_CONTRACT),
        )

    def _validate_v7_schema(self, *, allow_v8_tables: bool = False) -> None:
        self._validate_v6_schema(allow_v7_tables=True, allow_v8_tables=allow_v8_tables)
        objects = [
            row for row in self._registry.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if str(row[1]).casefold().startswith("diagnostic_session_execution_contract")
            or str(row[1]).casefold().endswith("_v7")
        ]
        if (len(objects) != 1 or tuple(objects[0]) != (
                "table", "diagnostic_session_execution_contracts",
                self._v7_contract_schema_statement().strip())):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        columns = self._registry.execute(
            "PRAGMA table_info(diagnostic_session_execution_contracts)"
        ).fetchall()
        if [tuple(row) for row in columns] != [
            (0, "session_id", "TEXT", 1, None, 1),
            (1, "execution_contract", "TEXT", 1, None, 0),
        ]:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        foreign_keys = self._registry.execute(
            "PRAGMA foreign_key_list(diagnostic_session_execution_contracts)"
        ).fetchall()
        if [tuple(row) for row in foreign_keys] != [
            (0, 0, "diagnostic_sessions", "session_id", "session_id",
             "NO ACTION", "NO ACTION", "NONE"),
        ]:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        indexes = self._registry.execute(
            "PRAGMA index_list(diagnostic_session_execution_contracts)"
        ).fetchall()
        if len(indexes) != 1 or tuple(indexes[0][2:]) != (1, "pk", 0):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        index_columns = self._registry.execute(
            "PRAGMA index_info(sqlite_autoindex_diagnostic_session_execution_contracts_1)"
        ).fetchall()
        if [tuple(row) for row in index_columns] != [(0, 0, "session_id")]:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if self._registry.execute(
            "SELECT 1 FROM diagnostic_session_execution_contracts AS contracts "
            "LEFT JOIN diagnostic_sessions AS sessions ON sessions.session_id = contracts.session_id "
            "WHERE sessions.session_id IS NULL LIMIT 1"
        ).fetchone() is not None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        for row in self._registry.execute("SELECT session_id FROM diagnostic_sessions").fetchall():
            self._execution_contract_locked(row[0], allow_claim_effect=allow_v8_tables)
        if self._registry.execute("PRAGMA foreign_key_check").fetchall():
            raise SessionStoreError(SANITIZED_STORE_ERROR)

    def _v8_objects_present(self) -> bool:
        return any(
            str(row[0]).casefold().startswith("diagnostic_claim_effect_")
            or str(row[0]).casefold().endswith("_v8")
            for row in self._registry.execute(
                "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
        )

    def _create_v8_tables(self) -> None:
        from app.services.effect_records_v2 import SCHEMA_STATEMENTS
        for statement in SCHEMA_STATEMENTS:
            self._registry.execute(statement)

    def _validate_v8_schema(self) -> None:
        from app.services.effect_records_v2 import (
            APPROVAL_COLUMNS, LEDGER_COLUMNS, SCHEMA_STATEMENTS,
        )
        self._validate_v7_schema(allow_v8_tables=True)
        approval_table = "diagnostic_claim_effect_approvals"
        ledger_table = "diagnostic_claim_effect_ledger"
        names = (approval_table, ledger_table)
        objects = [tuple(row) for row in self._registry.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall() if str(row[1]).casefold().startswith("diagnostic_claim_effect_")
            or str(row[1]).casefold().endswith("_v8") or str(row[2]).casefold() in names]
        expected_objects = [
            ("table", name, name, statement.strip())
            for name, statement in zip(names, SCHEMA_STATEMENTS, strict=True)
        ]
        if set(objects) != set(expected_objects) or len(objects) != 2:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        approval_key = ("approval_id", "session_id", "run_id", "interrupt_id")
        ledger_key = ("effect_key", "approval_id", "binding_sha256", "effect_type", "source", "arguments_sha256")
        shapes = (
            (approval_table, APPROVAL_COLUMNS, {"turn_number", "issued_at", "expires_at", "consumed_at"},
             {"decision", "consumed_at"}, "approval_id", "diagnostic_interrupts", approval_key,
             {("approval_id",), ("effect_key",), ("binding_sha256",), ("nonce_digest",), ledger_key}),
            (ledger_table, LEDGER_COLUMNS, {"attempt_count", "created_at", "updated_at"},
             {"result_json", "result_sha256", "verification_json", "verification_sha256"},
             "effect_key", approval_table, ledger_key, {("effect_key",), ("approval_id",)}),
        )
        for table, columns, integers, nullable, primary, parent, fk_columns, unique in shapes:
            info = [tuple(row) for row in self._registry.execute(f"PRAGMA table_info({table})").fetchall()]
            if info != [
                (index, column, "INTEGER" if column in integers else "TEXT",
                 int(column not in nullable), None, int(column == primary))
                for index, column in enumerate(columns)
            ]:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            foreign_keys = [tuple(row) for row in self._registry.execute(f"PRAGMA foreign_key_list({table})")]
            if foreign_keys != [
                (0, index, parent, column, column, "NO ACTION", "NO ACTION", "NONE")
                for index, column in enumerate(fk_columns)
            ]:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            indexes = self._registry.execute(f"PRAGMA index_list({table})").fetchall()
            actual_unique = set()
            for index in indexes:
                if (index[2] != 1 or index[3] not in {"pk", "u"} or index[4] != 0
                        or not str(index[1]).startswith(f"sqlite_autoindex_{table}_")):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                # Object names came from SQLite and must match its restricted autoindex prefix.
                index_name = str(index[1])
                if not re.fullmatch(r"[A-Za-z0-9_]+", index_name):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                actual_unique.add(tuple(row[2] for row in self._registry.execute(f"PRAGMA index_info({index_name})")))
            if actual_unique != unique or len(indexes) != len(unique):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
        if self._registry.execute("PRAGMA foreign_key_check").fetchall():
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        # This check will be the single V2 binding entry before recovery; a
        # nonempty store cannot pass while its runtime integration is absent.
        for row in self._registry.execute(f"SELECT approval_id FROM {approval_table}").fetchall():
            self._validate_claim_effect_binding_locked(str(row[0]), require_checkpoint=False)
        for approval_id, run_id in self._registry.execute(
            "SELECT approval_id,run_id FROM diagnostic_interrupts"
        ).fetchall():
            run = self._select_run_locked(str(run_id))
            if run is None:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            self._effect_family_locked(self._effect_parent_locked(str(approval_id)), run)
        if self._registry.execute(
            f"SELECT 1 FROM {ledger_table} AS ledger LEFT JOIN {approval_table} AS approval "
            "ON approval.approval_id=ledger.approval_id WHERE approval.approval_id IS NULL LIMIT 1"
        ).fetchone() is not None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)

    def _v5_objects_present(self) -> bool:
        names = {
            str(row[1])
            for row in self._registry.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        return any(
            name.casefold() in {
                "diagnostic_effect_approvals",
                "diagnostic_effect_ledger",
                "ux_diagnostic_interrupt_exact_binding",
            }
            or name.casefold().startswith("diagnostic_effect_")
            or name.casefold().startswith(
                "ux_diagnostic_interrupt_exact_binding"
            )
            or name.casefold().endswith(("_v5", "_mod024", "_migration"))
            for name in names
        )

    def _validate_v4_schema(
        self, *, allow_v5_tables: bool = False, allow_v6_tables: bool = False,
        allow_v7_tables: bool = False, allow_v8_tables: bool = False,
    ) -> None:
        if self._registry.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
            "AND tbl_name LIKE 'diagnostic_%' LIMIT 1"
        ).fetchone() is not None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        expected_columns = {
            "diagnostic_sessions": (
                "session_id", "thread_id", "status", "created_at", "updated_at",
                "turn_count", "parent_session_id", "active_run_id",
            ),
            "diagnostic_registry_meta": ("key", "value"),
            "diagnostic_session_creations": (
                "request_id", "session_id", "created_at",
            ),
            "diagnostic_runs": (
                "run_id", "session_id", "turn_number", "status",
                "previous_session_status", "base_turn_count", "created_at",
                "updated_at", "result_json", "error_code", "tool_iterations",
                "max_tool_iterations", "stop_reason", "current_interrupt_id",
            ),
            "diagnostic_interrupts": (
                "approval_id", "session_id", "run_id", "interrupt_id",
                "payload_json", "payload_sha256", "status", "decision",
                "created_at", "consumed_at",
            ),
            "diagnostic_session_events": (
                "event_id", "session_id", "run_id", "event_type", "from_status",
                "to_status", "reason_code", "dedupe_key", "created_at",
            ),
            "diagnostic_requests": (
                "session_id", "request_id", "operation", "payload_sha256",
                "payload_json", "run_id", "attempt_count", "created_at",
                "updated_at",
            ),
            "diagnostic_messages": (
                "message_id", "session_id", "sequence_no", "turn_number", "role",
                "kind", "content", "run_id", "request_id", "payload_json",
                "created_at",
            ),
            "diagnostic_draft_tickets": (
                "draft_id", "session_id", "source_run_id", "version", "status",
                "device_model", "fault_category", "fault_description",
                "diagnosis_json", "solution_steps_json", "risk_level",
                "need_human", "external_ticket_id", "projection_note",
                "created_at", "updated_at",
            ),
            "diagnostic_draft_projections": (
                "draft_id", "version", "payload_json", "payload_sha256",
                "idempotency_key", "status", "attempt_count", "last_error_code",
                "external_ticket_id", "created_at", "updated_at",
            ),
        }
        present = {
            str(row[0]).casefold()
            for row in self._registry.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'diagnostic_%'"
            ).fetchall()
        }
        expected_tables = set(expected_columns)
        if allow_v5_tables:
            expected_tables.update(
                {"diagnostic_effect_approvals", "diagnostic_effect_ledger"}
            )
        if allow_v6_tables:
            expected_tables.add("diagnostic_session_claim_profiles")
        if allow_v7_tables:
            expected_tables.add("diagnostic_session_execution_contracts")
        if allow_v8_tables:
            expected_tables.update({"diagnostic_claim_effect_approvals", "diagnostic_claim_effect_ledger"})
        if present != expected_tables:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        for table_name, expected in expected_columns.items():
            actual = tuple(
                str(row[1])
                for row in self._registry.execute(
                    f"PRAGMA table_info({table_name})"
                ).fetchall()
            )
            if actual != expected:
                raise SessionStoreError(SANITIZED_STORE_ERROR)

        expected_objects = (
            ("table", "diagnostic_registry_meta"),
            ("table", "diagnostic_session_creations"),
            ("table", "diagnostic_runs"),
            ("table", "diagnostic_interrupts"),
            ("table", "diagnostic_session_events"),
            ("table", "diagnostic_requests"),
            ("table", "diagnostic_messages"),
            ("table", "diagnostic_draft_tickets"),
            ("table", "diagnostic_draft_projections"),
            ("index", "ux_diagnostic_session_parent"),
            ("index", "ux_diagnostic_active_run"),
            ("index", "ux_diagnostic_pending_interrupt"),
        )
        statements = self._v4_schema_statements()
        if len(expected_objects) != len(statements):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        def stored_ddl(value: str) -> str:
            return value.replace(" IF NOT EXISTS ", " ", 1).strip().rstrip(";")

        session_sql = self._registry.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'diagnostic_sessions'"
        ).fetchone()
        if (
            session_sql is None
            or not isinstance(session_sql[0], str)
            or stored_ddl(
                session_sql[0].replace(
                    'CREATE TABLE "diagnostic_sessions"',
                    "CREATE TABLE diagnostic_sessions",
                    1,
                )
            )
            != stored_ddl(self._session_table_statement("diagnostic_sessions"))
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        for (object_type, object_name), expected_sql in zip(
            expected_objects, statements
        ):
            row = self._registry.execute(
                "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
                (object_type, object_name),
            ).fetchone()
            if (
                row is None
                or not isinstance(row[0], str)
                or stored_ddl(row[0]) != stored_ddl(expected_sql)
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)

        explicit_indexes = {
            str(row[0]).casefold()
            for row in self._registry.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND name NOT LIKE 'sqlite_%' "
                "AND tbl_name LIKE 'diagnostic_%'"
            ).fetchall()
        }
        expected_explicit_indexes = {
            "ux_diagnostic_session_parent",
            "ux_diagnostic_active_run",
            "ux_diagnostic_pending_interrupt",
        }
        if allow_v5_tables:
            expected_explicit_indexes.add(
                "ux_diagnostic_interrupt_exact_binding"
            )
        if explicit_indexes != expected_explicit_indexes:
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        expected_named_indexes = {
            "ux_diagnostic_session_parent": (
                "diagnostic_sessions", ("parent_session_id",), 1,
            ),
            "ux_diagnostic_active_run": (
                "diagnostic_runs", ("session_id",), 1,
            ),
            "ux_diagnostic_pending_interrupt": (
                "diagnostic_interrupts", ("run_id",), 1,
            ),
        }
        for index_name, (table_name, columns, partial) in expected_named_indexes.items():
            metadata = next(
                (
                    row
                    for row in self._registry.execute(
                        f"PRAGMA index_list({table_name})"
                    ).fetchall()
                    if str(row[1]) == index_name
                ),
                None,
            )
            actual_columns = tuple(
                str(row[2])
                for row in self._registry.execute(
                    f"PRAGMA index_info({index_name})"
                ).fetchall()
            )
            if (
                metadata is None
                or int(metadata[2]) != 1
                or str(metadata[3]) != "c"
                or (len(metadata) <= 4 or int(metadata[4]) != partial)
                or actual_columns != columns
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)

        def foreign_key_signatures(
            table_name: str,
        ) -> set[tuple[str, tuple[tuple[str, str], ...]]]:
            grouped: dict[int, list[tuple[Any, ...]]] = {}
            for row in self._registry.execute(
                f"PRAGMA foreign_key_list({table_name})"
            ).fetchall():
                if (str(row[5]), str(row[6]), str(row[7])) != (
                    "NO ACTION", "NO ACTION", "NONE"
                ):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                grouped.setdefault(int(row[0]), []).append(tuple(row))
            return {
                (
                    str(rows[0][2]),
                    tuple(
                        (str(row[3]), str(row[4]))
                        for row in sorted(rows, key=lambda item: int(item[1]))
                    ),
                )
                for rows in grouped.values()
            }

        expected_foreign_keys = {
            "diagnostic_sessions": set(),
            "diagnostic_registry_meta": set(),
            "diagnostic_session_creations": {
                ("diagnostic_sessions", (("session_id", "session_id"),)),
            },
            "diagnostic_runs": {
                ("diagnostic_sessions", (("session_id", "session_id"),)),
            },
            "diagnostic_interrupts": {
                (
                    "diagnostic_runs",
                    (("run_id", "run_id"), ("session_id", "session_id")),
                ),
            },
            "diagnostic_session_events": set(),
            "diagnostic_requests": {
                ("diagnostic_sessions", (("session_id", "session_id"),)),
                (
                    "diagnostic_runs",
                    (("run_id", "run_id"), ("session_id", "session_id")),
                ),
            },
            "diagnostic_messages": {
                (
                    "diagnostic_requests",
                    (("session_id", "session_id"), ("request_id", "request_id")),
                ),
                (
                    "diagnostic_runs",
                    (("run_id", "run_id"), ("session_id", "session_id")),
                ),
            },
            "diagnostic_draft_tickets": {
                ("diagnostic_sessions", (("session_id", "session_id"),)),
                (
                    "diagnostic_runs",
                    (
                        ("source_run_id", "run_id"),
                        ("session_id", "session_id"),
                    ),
                ),
            },
            "diagnostic_draft_projections": {
                ("diagnostic_draft_tickets", (("draft_id", "draft_id"),)),
            },
        }
        for table_name, expected in expected_foreign_keys.items():
            if foreign_key_signatures(table_name) != expected:
                raise SessionStoreError(SANITIZED_STORE_ERROR)

    @staticmethod
    def _v5_schema_statements() -> tuple[str, str, str]:
        return (
            """
            CREATE UNIQUE INDEX ux_diagnostic_interrupt_exact_binding
            ON diagnostic_interrupts(approval_id, session_id, run_id, interrupt_id)
            """,
            f"""
            CREATE TABLE diagnostic_effect_approvals (
                approval_id TEXT PRIMARY KEY NOT NULL,
                session_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                interrupt_id TEXT NOT NULL,
                approval_kind TEXT NOT NULL CHECK (approval_kind = 'EFFECT_V1'),
                effect_key TEXT NOT NULL UNIQUE,
                effect_type TEXT NOT NULL CHECK (
                    effect_type = 'restart_simulated_robot'
                ),
                source TEXT NOT NULL CHECK (source = 'SYNTHETIC_LOCAL'),
                proposal_json TEXT NOT NULL,
                proposal_sha256 TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                arguments_sha256 TEXT NOT NULL,
                risk_summary_json TEXT NOT NULL,
                risk_summary_sha256 TEXT NOT NULL,
                verifier_summary_json TEXT NOT NULL,
                verifier_summary_sha256 TEXT NOT NULL,
                verifier_result_sha256 TEXT NOT NULL,
                binding_json TEXT NOT NULL,
                binding_sha256 TEXT NOT NULL UNIQUE,
                nonce_digest TEXT NOT NULL UNIQUE,
                issued_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('PENDING', 'CONSUMED', 'CANCELLED', 'EXPIRED')
                ),
                decision TEXT CHECK (
                    decision IS NULL OR decision IN ('APPROVE', 'REJECT')
                ),
                consumed_at INTEGER,
                UNIQUE (
                    effect_key, approval_id, binding_sha256, effect_type,
                    source, arguments_sha256
                ),
                FOREIGN KEY (approval_id, session_id, run_id, interrupt_id)
                    REFERENCES diagnostic_interrupts(
                        approval_id, session_id, run_id, interrupt_id
                    ),
                CHECK (expires_at = issued_at + {EFFECT_CAPABILITY_TTL_MS}),
                CHECK (
                    (status = 'PENDING' AND decision IS NULL AND consumed_at IS NULL)
                    OR (status = 'CONSUMED' AND decision IS NOT NULL
                        AND consumed_at IS NOT NULL)
                    OR (status IN ('CANCELLED', 'EXPIRED') AND decision IS NULL
                        AND consumed_at IS NOT NULL)
                )
            )
            """,
            """
            CREATE TABLE diagnostic_effect_ledger (
                effect_key TEXT PRIMARY KEY NOT NULL,
                approval_id TEXT NOT NULL UNIQUE,
                binding_sha256 TEXT NOT NULL,
                effect_type TEXT NOT NULL CHECK (
                    effect_type = 'restart_simulated_robot'
                ),
                source TEXT NOT NULL CHECK (source = 'SYNTHETIC_LOCAL'),
                arguments_json TEXT NOT NULL,
                arguments_sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN (
                        'PREPARED', 'EXECUTING', 'APPLIED', 'VERIFIED',
                        'EXECUTION_FAILED', 'VERIFICATION_FAILED', 'UNKNOWN'
                    )
                ),
                attempt_count INTEGER NOT NULL CHECK (attempt_count IN (0, 1)),
                result_json TEXT,
                result_sha256 TEXT,
                verification_json TEXT,
                verification_sha256 TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (
                    effect_key, approval_id, binding_sha256, effect_type,
                    source, arguments_sha256
                ) REFERENCES diagnostic_effect_approvals(
                    effect_key, approval_id, binding_sha256, effect_type,
                    source, arguments_sha256
                ),
                CHECK (
                    (status = 'PREPARED' AND attempt_count = 0
                        AND result_json IS NULL AND result_sha256 IS NULL
                        AND verification_json IS NULL
                        AND verification_sha256 IS NULL)
                    OR (status = 'EXECUTING' AND attempt_count = 1
                        AND result_json IS NULL AND result_sha256 IS NULL
                        AND verification_json IS NULL
                        AND verification_sha256 IS NULL)
                    OR (status IN (
                            'APPLIED', 'VERIFIED', 'EXECUTION_FAILED',
                            'VERIFICATION_FAILED', 'UNKNOWN'
                        ) AND attempt_count = 1)
                ),
                CHECK ((result_json IS NULL) = (result_sha256 IS NULL)),
                CHECK (
                    (verification_json IS NULL) = (verification_sha256 IS NULL)
                )
            )
            """,
        )

    def _create_v5_tables(self) -> None:
        for statement in self._v5_schema_statements():
            self._registry.execute(statement)

    def _validate_v5_schema(
        self, *, allow_v6_tables: bool = False, allow_v7_tables: bool = False,
        allow_v8_tables: bool = False,
    ) -> None:
        self._validate_v4_schema(
            allow_v5_tables=True, allow_v6_tables=allow_v6_tables, allow_v7_tables=allow_v7_tables,
            allow_v8_tables=allow_v8_tables,
        )
        required_base_tables = {
            "diagnostic_sessions",
            "diagnostic_registry_meta",
            "diagnostic_session_creations",
            "diagnostic_runs",
            "diagnostic_interrupts",
            "diagnostic_session_events",
            "diagnostic_requests",
            "diagnostic_messages",
            "diagnostic_draft_tickets",
            "diagnostic_draft_projections",
            "diagnostic_effect_approvals",
            "diagnostic_effect_ledger",
        }
        present = {
            str(row[0]).casefold()
            for row in self._registry.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not required_base_tables.issubset(present):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        schema_objects = {
            (str(row[0]), str(row[1]), str(row[2]))
            for row in self._registry.execute(
                "SELECT type, name, tbl_name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        if any(
            name.casefold().endswith(("_v5", "_mod024", "_migration"))
            or (
                name.casefold().startswith("diagnostic_effect_")
                and name.casefold() not in {
                    "diagnostic_effect_approvals",
                    "diagnostic_effect_ledger",
                }
            )
            or (
                object_type == "trigger"
                and table_name.casefold().startswith("diagnostic_")
            )
            for object_type, name, table_name in schema_objects
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        expected_columns = {
            "diagnostic_interrupts": (
                "approval_id", "session_id", "run_id", "interrupt_id",
                "payload_json", "payload_sha256", "status", "decision",
                "created_at", "consumed_at",
            ),
            "diagnostic_effect_approvals": (
                "approval_id", "session_id", "run_id", "interrupt_id",
                "approval_kind", "effect_key", "effect_type", "source",
                "proposal_json", "proposal_sha256", "arguments_json",
                "arguments_sha256", "risk_summary_json", "risk_summary_sha256",
                "verifier_summary_json", "verifier_summary_sha256",
                "verifier_result_sha256", "binding_json", "binding_sha256",
                "nonce_digest", "issued_at", "expires_at", "status",
                "decision", "consumed_at",
            ),
            "diagnostic_effect_ledger": (
                "effect_key", "approval_id", "binding_sha256", "effect_type",
                "source", "arguments_json", "arguments_sha256", "status",
                "attempt_count", "result_json", "result_sha256",
                "verification_json", "verification_sha256", "created_at",
                "updated_at",
            ),
        }
        table_info: dict[str, tuple[tuple[Any, ...], ...]] = {}
        for table_name, expected in expected_columns.items():
            rows = tuple(
                tuple(row)
                for row in self._registry.execute(
                    f"PRAGMA table_info({table_name})"
                ).fetchall()
            )
            table_info[table_name] = rows
            actual = tuple(str(row[1]) for row in rows)
            if actual != expected:
                raise SessionStoreError(SANITIZED_STORE_ERROR)

        expected_shapes = {
            "diagnostic_interrupts": (
                ("approval_id", "TEXT", 0, 1),
                ("session_id", "TEXT", 1, 0),
                ("run_id", "TEXT", 1, 0),
                ("interrupt_id", "TEXT", 1, 0),
                ("payload_json", "TEXT", 1, 0),
                ("payload_sha256", "TEXT", 1, 0),
                ("status", "TEXT", 1, 0),
                ("decision", "TEXT", 0, 0),
                ("created_at", "TEXT", 1, 0),
                ("consumed_at", "TEXT", 0, 0),
            ),
            "diagnostic_effect_approvals": (
                ("approval_id", "TEXT", 1, 1),
                ("session_id", "TEXT", 1, 0),
                ("run_id", "TEXT", 1, 0),
                ("interrupt_id", "TEXT", 1, 0),
                ("approval_kind", "TEXT", 1, 0),
                ("effect_key", "TEXT", 1, 0),
                ("effect_type", "TEXT", 1, 0),
                ("source", "TEXT", 1, 0),
                ("proposal_json", "TEXT", 1, 0),
                ("proposal_sha256", "TEXT", 1, 0),
                ("arguments_json", "TEXT", 1, 0),
                ("arguments_sha256", "TEXT", 1, 0),
                ("risk_summary_json", "TEXT", 1, 0),
                ("risk_summary_sha256", "TEXT", 1, 0),
                ("verifier_summary_json", "TEXT", 1, 0),
                ("verifier_summary_sha256", "TEXT", 1, 0),
                ("verifier_result_sha256", "TEXT", 1, 0),
                ("binding_json", "TEXT", 1, 0),
                ("binding_sha256", "TEXT", 1, 0),
                ("nonce_digest", "TEXT", 1, 0),
                ("issued_at", "INTEGER", 1, 0),
                ("expires_at", "INTEGER", 1, 0),
                ("status", "TEXT", 1, 0),
                ("decision", "TEXT", 0, 0),
                ("consumed_at", "INTEGER", 0, 0),
            ),
            "diagnostic_effect_ledger": (
                ("effect_key", "TEXT", 1, 1),
                ("approval_id", "TEXT", 1, 0),
                ("binding_sha256", "TEXT", 1, 0),
                ("effect_type", "TEXT", 1, 0),
                ("source", "TEXT", 1, 0),
                ("arguments_json", "TEXT", 1, 0),
                ("arguments_sha256", "TEXT", 1, 0),
                ("status", "TEXT", 1, 0),
                ("attempt_count", "INTEGER", 1, 0),
                ("result_json", "TEXT", 0, 0),
                ("result_sha256", "TEXT", 0, 0),
                ("verification_json", "TEXT", 0, 0),
                ("verification_sha256", "TEXT", 0, 0),
                ("created_at", "INTEGER", 1, 0),
                ("updated_at", "INTEGER", 1, 0),
            ),
        }
        for table_name, expected in expected_shapes.items():
            actual = tuple(
                (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
                for row in table_info[table_name]
            )
            if actual != expected or any(row[4] is not None for row in table_info[table_name]):
                raise SessionStoreError(SANITIZED_STORE_ERROR)

        index_row = self._registry.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' "
            "AND name = 'ux_diagnostic_interrupt_exact_binding'"
        ).fetchone()
        index_columns = tuple(
            str(row[2])
            for row in sorted(
                self._registry.execute(
                    "PRAGMA index_info(ux_diagnostic_interrupt_exact_binding)"
                ).fetchall(),
                key=lambda item: int(item[0]),
            )
        )
        parent_index_metadata = next(
            (
                row
                for row in self._registry.execute(
                    "PRAGMA index_list(diagnostic_interrupts)"
                ).fetchall()
                if row[1] == "ux_diagnostic_interrupt_exact_binding"
            ),
            None,
        )
        if (
            index_row is None
            or not index_row[0]
            or parent_index_metadata is None
            or int(parent_index_metadata[2]) != 1
            or str(parent_index_metadata[3]) != "c"
            or len(parent_index_metadata) <= 4
            or int(parent_index_metadata[4]) != 0
            or index_columns != (
                "approval_id", "session_id", "run_id", "interrupt_id"
            )
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        def index_signatures(
            table_name: str,
        ) -> list[tuple[str, int, str, int, tuple[str, ...]]]:
            signatures: list[tuple[str, int, str, int, tuple[str, ...]]] = []
            for row in self._registry.execute(
                f"PRAGMA index_list({table_name})"
            ).fetchall():
                if len(row) <= 4:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                info = sorted(
                    self._registry.execute(
                        f"PRAGMA index_info({row[1]})"
                    ).fetchall(),
                    key=lambda item: int(item[0]),
                )
                if [int(item[0]) for item in info] != list(range(len(info))):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                origin = str(row[3])
                signatures.append(
                    (
                        str(row[1]) if origin == "c" else "",
                        int(row[2]),
                        origin,
                        int(row[4]),
                        tuple(str(item[2]) for item in info),
                    )
                )
            return sorted(signatures)

        if index_signatures("diagnostic_effect_approvals") != sorted(
            [
                ("", 1, "pk", 0, ("approval_id",)),
                ("", 1, "u", 0, ("effect_key",)),
                ("", 1, "u", 0, ("binding_sha256",)),
                ("", 1, "u", 0, ("nonce_digest",)),
                (
                    "", 1, "u", 0,
                    (
                        "effect_key", "approval_id", "binding_sha256",
                        "effect_type", "source", "arguments_sha256",
                    ),
                ),
            ]
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if index_signatures("diagnostic_effect_ledger") != sorted(
            [
                ("", 1, "pk", 0, ("effect_key",)),
                ("", 1, "u", 0, ("approval_id",)),
            ]
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        def foreign_key_signatures(
            table_name: str,
        ) -> list[tuple[str, tuple[tuple[str, str], ...], str, str, str]]:
            grouped: dict[int, list[tuple[Any, ...]]] = {}
            for row in self._registry.execute(
                f"PRAGMA foreign_key_list({table_name})"
            ).fetchall():
                grouped.setdefault(int(row[0]), []).append(tuple(row))
            signatures: list[
                tuple[str, tuple[tuple[str, str], ...], str, str, str]
            ] = []
            for rows in grouped.values():
                ordered = sorted(rows, key=lambda item: int(item[1]))
                if [int(row[1]) for row in ordered] != list(range(len(ordered))):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                if any(
                    (str(row[5]), str(row[6]), str(row[7]))
                    != ("NO ACTION", "NO ACTION", "NONE")
                    for row in ordered
                ):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                signatures.append(
                    (
                        str(ordered[0][2]),
                        tuple((str(row[3]), str(row[4])) for row in ordered),
                        str(ordered[0][5]),
                        str(ordered[0][6]),
                        str(ordered[0][7]),
                    )
                )
            return sorted(signatures)

        if foreign_key_signatures("diagnostic_effect_approvals") != [
            (
                "diagnostic_interrupts",
                (
                    ("approval_id", "approval_id"),
                    ("session_id", "session_id"),
                    ("run_id", "run_id"),
                    ("interrupt_id", "interrupt_id"),
                ),
                "NO ACTION",
                "NO ACTION",
                "NONE",
            )
        ]:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if foreign_key_signatures("diagnostic_effect_ledger") != [
            (
                "diagnostic_effect_approvals",
                (
                    ("effect_key", "effect_key"),
                    ("approval_id", "approval_id"),
                    ("binding_sha256", "binding_sha256"),
                    ("effect_type", "effect_type"),
                    ("source", "source"),
                    ("arguments_sha256", "arguments_sha256"),
                ),
                "NO ACTION",
                "NO ACTION",
                "NONE",
            )
        ]:
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        approval_sql = str(
            self._registry.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' "
                "AND name = 'diagnostic_effect_approvals'"
            ).fetchone()[0]
        )
        ledger_sql = str(
            self._registry.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' "
                "AND name = 'diagnostic_effect_ledger'"
            ).fetchone()[0]
        )
        expected_index_sql, expected_approval_sql, expected_ledger_sql = (
            self._v5_schema_statements()
        )

        def stored_ddl(value: str) -> str:
            return value.strip().rstrip(";")

        if (
            stored_ddl(str(index_row[0])) != stored_ddl(expected_index_sql)
            or stored_ddl(approval_sql) != stored_ddl(expected_approval_sql)
            or stored_ddl(ledger_sql) != stored_ddl(expected_ledger_sql)
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

    @staticmethod
    def _session_table_statement(table_name: str) -> str:
        if table_name not in {"diagnostic_sessions", "diagnostic_sessions_mod020"}:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return f"""
            CREATE TABLE {table_name} (
                session_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK (
                    status IN (
                        'OPEN', 'RUNNING', 'AWAITING_USER', 'INTERRUPTED',
                        'CANCEL_REQUESTED', 'COMPLETED', 'CANCELLED', 'FAILED'
                    )
                ),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                turn_count INTEGER NOT NULL DEFAULT 0 CHECK (turn_count >= 0),
                parent_session_id TEXT,
                active_run_id TEXT
            )
            """

    def _create_session_table(self, table_name: str) -> None:
        self._registry.execute(self._session_table_statement(table_name))

    def _migrate_mod019_sessions(self) -> None:
        old_rows = self._registry.execute(
            """
            SELECT session_id, thread_id, status, created_at, updated_at, turn_count
            FROM diagnostic_sessions ORDER BY session_id
            """
        ).fetchall()
        self._registry.execute("DROP TABLE IF EXISTS diagnostic_sessions_mod020")
        self._create_session_table("diagnostic_sessions_mod020")
        self._registry.execute(
            """
            INSERT INTO diagnostic_sessions_mod020 (
                session_id, thread_id, status, created_at, updated_at, turn_count,
                parent_session_id, active_run_id
            )
            SELECT session_id, thread_id,
                   CASE WHEN status = 'RUNNING' THEN 'FAILED' ELSE status END,
                   created_at, updated_at, turn_count, NULL, NULL
            FROM diagnostic_sessions
            """
        )
        new_rows = self._registry.execute(
            """
            SELECT session_id, thread_id, status, created_at, updated_at, turn_count
            FROM diagnostic_sessions_mod020 ORDER BY session_id
            """
        ).fetchall()
        expected_rows = [
            (
                row[0],
                row[1],
                "FAILED" if row[2] == "RUNNING" else row[2],
                row[3],
                row[4],
                row[5],
            )
            for row in old_rows
        ]
        if new_rows != expected_rows:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        self._registry.execute("DROP TABLE diagnostic_sessions")
        self._registry.execute(
            "ALTER TABLE diagnostic_sessions_mod020 RENAME TO diagnostic_sessions"
        )

    @staticmethod
    def _v4_schema_statements() -> tuple[str, ...]:
        return (
            """
            CREATE TABLE IF NOT EXISTS diagnostic_registry_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS diagnostic_session_creations (
                request_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES diagnostic_sessions(session_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS diagnostic_runs (
                run_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL CHECK (turn_number >= 1),
                status TEXT NOT NULL CHECK (
                    status IN (
                        'QUEUED', 'RUNNING', 'INTERRUPTED', 'CANCEL_REQUESTED',
                        'COMPLETED', 'CANCELLED', 'FAILED'
                    )
                ),
                previous_session_status TEXT NOT NULL CHECK (
                    previous_session_status IN ('OPEN', 'AWAITING_USER')
                ),
                base_turn_count INTEGER NOT NULL CHECK (base_turn_count >= 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_json TEXT,
                error_code TEXT,
                tool_iterations INTEGER NOT NULL DEFAULT 0 CHECK (
                    tool_iterations BETWEEN 0 AND max_tool_iterations
                ),
                max_tool_iterations INTEGER NOT NULL DEFAULT 4 CHECK (
                    max_tool_iterations = 4
                ),
                stop_reason TEXT,
                current_interrupt_id TEXT,
                UNIQUE (run_id, session_id),
                UNIQUE (session_id, turn_number),
                FOREIGN KEY (session_id) REFERENCES diagnostic_sessions(session_id),
                CHECK (
                    (status = 'COMPLETED' AND result_json IS NOT NULL AND error_code IS NULL)
                    OR (status = 'FAILED' AND result_json IS NULL AND error_code IS NOT NULL)
                    OR (status NOT IN ('COMPLETED', 'FAILED')
                        AND result_json IS NULL AND error_code IS NULL)
                )
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS diagnostic_interrupts (
                approval_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                interrupt_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('PENDING', 'CONSUMED', 'CANCELLED')
                ),
                decision TEXT CHECK (decision IS NULL OR decision IN ('APPROVE', 'REJECT')),
                created_at TEXT NOT NULL,
                consumed_at TEXT,
                FOREIGN KEY (run_id, session_id)
                    REFERENCES diagnostic_runs(run_id, session_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS diagnostic_session_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                run_id TEXT,
                event_type TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                reason_code TEXT,
                dedupe_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS diagnostic_requests (
                session_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                operation TEXT NOT NULL CHECK (operation IN ('RUN', 'LEGACY_RUN')),
                payload_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                run_id TEXT NOT NULL UNIQUE,
                attempt_count INTEGER NOT NULL DEFAULT 1 CHECK (attempt_count >= 1),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, request_id),
                FOREIGN KEY (session_id) REFERENCES diagnostic_sessions(session_id),
                FOREIGN KEY (run_id, session_id)
                    REFERENCES diagnostic_runs(run_id, session_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS diagnostic_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence_no INTEGER NOT NULL CHECK (sequence_no >= 1),
                turn_number INTEGER NOT NULL CHECK (turn_number >= 1),
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                kind TEXT NOT NULL CHECK (kind IN ('INPUT', 'FOLLOW_UP', 'RESULT')),
                content TEXT NOT NULL,
                run_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE (session_id, sequence_no),
                UNIQUE (run_id, role),
                FOREIGN KEY (session_id, request_id)
                    REFERENCES diagnostic_requests(session_id, request_id),
                FOREIGN KEY (run_id, session_id)
                    REFERENCES diagnostic_runs(run_id, session_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS diagnostic_draft_tickets (
                draft_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                source_run_id TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                status TEXT NOT NULL CHECK (
                    status IN ('COLLECTING', 'READY', 'PROJECTION_FAILED', 'PROJECTED')
                ),
                device_model TEXT NOT NULL,
                fault_category TEXT NOT NULL,
                fault_description TEXT NOT NULL,
                diagnosis_json TEXT NOT NULL,
                solution_steps_json TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                need_human INTEGER NOT NULL CHECK (need_human IN (0, 1)),
                external_ticket_id TEXT,
                projection_note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES diagnostic_sessions(session_id),
                FOREIGN KEY (source_run_id, session_id)
                    REFERENCES diagnostic_runs(run_id, session_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS diagnostic_draft_projections (
                draft_id TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK (
                    status IN ('FAILED', 'IN_PROGRESS', 'PROJECTED')
                ),
                attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                last_error_code TEXT,
                external_ticket_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (draft_id, version),
                FOREIGN KEY (draft_id)
                    REFERENCES diagnostic_draft_tickets(draft_id)
            )
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_diagnostic_session_parent
            ON diagnostic_sessions(parent_session_id)
            WHERE parent_session_id IS NOT NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_diagnostic_active_run
            ON diagnostic_runs(session_id)
            WHERE status IN ('QUEUED', 'RUNNING', 'INTERRUPTED', 'CANCEL_REQUESTED')
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_diagnostic_pending_interrupt
            ON diagnostic_interrupts(run_id)
            WHERE status = 'PENDING'
            """,
        )

    def _create_run_tables(self) -> None:
        for statement in self._v4_schema_statements():
            self._registry.execute(statement)

    def _backfill_mod020_requests(self) -> None:
        """为既有 v2 run 建立不可重放的只读映射，保留恢复能力且不伪造消息。"""
        rows = self._registry.execute(
            """
            SELECT run_id, session_id, created_at, updated_at
            FROM diagnostic_runs AS runs
            WHERE NOT EXISTS (
                SELECT 1 FROM diagnostic_requests AS requests
                WHERE requests.run_id = runs.run_id
            )
            ORDER BY run_id
            """
        ).fetchall()
        for run_id, session_id, created_at, updated_at in rows:
            payload_json = json.dumps(
                {"legacy_run_id": run_id},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            self._registry.execute(
                """
                INSERT INTO diagnostic_requests (
                    session_id, request_id, operation, payload_sha256, payload_json,
                    run_id, attempt_count, created_at, updated_at
                ) VALUES (?, ?, 'LEGACY_RUN', ?, ?, ?, 1, ?, ?)
                """,
                (
                    session_id,
                    self._new_token(),
                    digest,
                    payload_json,
                    run_id,
                    created_at,
                    updated_at,
                ),
            )

    def _reconcile_orphaned_effects(self) -> None:
        """启动时把执行结果未知的遗留 effect 收口，永不自动重试。"""
        now_ms = self._typed_now_ms()
        owned_run_ids = self._process_owned_run_ids()
        parameters: list[Any] = [now_ms]
        ownership_clause = ""
        if owned_run_ids:
            placeholders = ", ".join("?" for _ in owned_run_ids)
            ownership_clause = f" AND approvals.run_id NOT IN ({placeholders})"
            parameters.extend(sorted(owned_run_ids))
        self._registry.execute(
            """
            UPDATE diagnostic_effect_ledger
            SET status = 'UNKNOWN', updated_at = ?
            WHERE status = 'EXECUTING' AND attempt_count = 1
              AND effect_key IN (
                  SELECT ledger.effect_key
                  FROM diagnostic_effect_ledger AS ledger
                  JOIN diagnostic_effect_approvals AS approvals
                    ON approvals.effect_key = ledger.effect_key
                  WHERE ledger.status = 'EXECUTING'
            """
            + ownership_clause
            + ")",
            parameters,
        )

    def _reconcile_orphaned_claim_effects(self) -> None:
        """An interrupted synthetic attempt has an unknown outcome, never retry it."""
        owned = self._process_owned_run_ids()
        rows = self._registry.execute(
            "SELECT approvals.approval_id,approvals.run_id,ledger.effect_key "
            "FROM diagnostic_claim_effect_ledger AS ledger "
            "JOIN diagnostic_claim_effect_approvals AS approvals "
            "ON approvals.approval_id=ledger.approval_id "
            "WHERE ledger.status='EXECUTING' AND ledger.attempt_count=1"
        ).fetchall()
        for approval_id, run_id, effect_key in rows:
            if str(run_id) in owned:
                continue
            self._validate_claim_effect_binding_locked(str(approval_id), require_checkpoint=False)
            changed = self._registry.execute(
                "UPDATE diagnostic_claim_effect_ledger "
                "SET status='UNKNOWN', updated_at=? "
                "WHERE effect_key=? AND status='EXECUTING' AND attempt_count=1",
                (self._typed_now_ms(), effect_key),
            ).rowcount
            if changed != 1:
                raise SessionStoreError(SANITIZED_STORE_ERROR)

    def _recover_applied_claim_effects(self) -> None:
        """Read back durable APPLIED rows after adapter creation, without another restart."""
        owned = self._process_owned_run_ids()
        with self._registry_lock:
            rows = self._registry.execute(
                "SELECT approvals.approval_id,approvals.run_id,approvals.session_id "
                "FROM diagnostic_claim_effect_ledger AS ledger "
                "JOIN diagnostic_claim_effect_approvals AS approvals "
                "ON approvals.approval_id=ledger.approval_id "
                "WHERE ledger.status='APPLIED' AND ledger.attempt_count=1"
            ).fetchall()
            for approval_id, run_id, _session_id in rows:
                self._validate_claim_effect_binding_locked(str(approval_id), require_checkpoint=False)
        if not rows:
            return
        from app.services.claim_effect_executor import ServiceClaimEffectAuthority
        authority = ServiceClaimEffectAuthority(self)
        for approval_id, run_id, session_id in rows:
            if str(run_id) in owned:
                continue
            with self._registry_lock:
                checked = self._validate_claim_effect_binding_locked(
                    str(approval_id), require_checkpoint=False)
            authority.verify_effect(
                session_id=str(session_id), run_id=str(run_id), approval_id=str(approval_id),
                proposal=checked["material"]["proposal"],
                proposal_sha256=checked["material"]["proposal_sha256"],
            )

    def _recover_applied_effects(self) -> None:
        """Read back orphaned APPLIED effects once; never repeat the effect."""
        owned_run_ids = self._process_owned_run_ids()
        parameters: list[Any] = []
        ownership_clause = ""
        if owned_run_ids:
            placeholders = ", ".join("?" for _ in owned_run_ids)
            ownership_clause = f" AND approvals.run_id NOT IN ({placeholders})"
            parameters.extend(sorted(owned_run_ids))
        try:
            with self._registry_transaction():
                rows = self._registry.execute(
                    """
                    SELECT approvals.session_id, approvals.run_id,
                           approvals.approval_id, approvals.proposal_json,
                           approvals.proposal_sha256, ledger.effect_key
                    FROM diagnostic_effect_ledger AS ledger
                    JOIN diagnostic_effect_approvals AS approvals
                      ON approvals.effect_key = ledger.effect_key
                    WHERE ledger.status = 'APPLIED'
                    """
                    + ownership_clause
                    + " ORDER BY ledger.effect_key",
                    parameters,
                ).fetchall()
                for row in rows:
                    ledger = self._select_effect_ledger_locked(str(row[5]))
                    if ledger is None:
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    self._validated_effect_ledger_locked(ledger)
                if self._synthetic_effect_adapter is None:
                    if rows:
                        self._registry.execute("BEGIN IMMEDIATE")
                        for row in rows:
                            changed = self._registry.execute(
                                """
                                UPDATE diagnostic_effect_ledger
                                SET status = 'UNKNOWN', updated_at = ?
                                WHERE effect_key = ? AND status = 'APPLIED'
                                  AND attempt_count = 1
                                """,
                                (self._typed_now_ms(), row[5]),
                            ).rowcount
                            if changed != 1:
                                raise SessionStoreError(SANITIZED_STORE_ERROR)
                        self._registry.commit()
                    return
            for session_id, run_id, approval_id, proposal_json, proposal_sha256, _ in rows:
                proposal, canonical = self._canonical_stored_json(proposal_json)
                if not isinstance(proposal, dict) or canonical != proposal_json:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                self._verify_authorized_effect(
                    session_id=str(session_id),
                    run_id=str(run_id),
                    approval_id=str(approval_id),
                    proposal=proposal,
                    proposal_sha256=str(proposal_sha256),
                )
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def _process_owned_run_ids(self) -> set[str]:
        database_key = str(self.db_path)
        with _PROCESS_CONTROLS_LOCK:
            return {
                run_id
                for (db_path, run_id), _control in _PROCESS_RUN_CONTROLS.items()
                if db_path == database_key
            }

    def _process_owned_sync_session_ids(self) -> set[str]:
        database_key = str(self.db_path)
        with _PROCESS_CONTROLS_LOCK:
            return {
                session_id
                for db_path, session_id in _PROCESS_SYNC_SESSION_OWNERS
                if db_path == database_key
            }

    def _reconcile_orphaned_runs(self) -> list[tuple[str, str]]:
        """启动时失败关闭没有本进程执行者的 v2 活动态。

        `INTERRUPTED` 是可恢复的稳定状态，必须保留。当前实现只承诺单进程；
        同进程 peer service 的 live owner 通过进程级 control registry 排除。
        """
        committed_orphans: list[tuple[str, str]] = []
        owned_run_ids = self._process_owned_run_ids()
        rows = self._registry.execute(
            """
            SELECT run_id FROM diagnostic_runs
            WHERE status IN ('QUEUED', 'RUNNING', 'CANCEL_REQUESTED')
            ORDER BY run_id
            """
        ).fetchall()
        for row in rows:
            run_id = str(row[0])
            if run_id in owned_run_ids:
                continue
            run = self._select_run_locked(run_id)
            if run is None:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            now = _now_text()
            if run.status == DiagnosticRunStatus.CANCEL_REQUESTED.value:
                self._set_cancelled_locked(
                    run,
                    from_status=run.status,
                    now=now,
                    reason_code="SERVICE_RESTARTED_AFTER_CANCEL",
                )
                committed_orphans.append((run_id, "RUN_CANCELLED"))
                continue
            self._set_failed_locked(
                run,
                from_status=run.status,
                now=now,
                error_code="SERVICE_RESTARTED_DURING_EXECUTION",
            )
            committed_orphans.append((run_id, "RUN_FAILED"))
        owned_sync_sessions = self._process_owned_sync_session_ids()
        session_rows = self._registry.execute(
            """
            SELECT session_id, status FROM diagnostic_sessions
            WHERE status IN ('RUNNING', 'CANCEL_REQUESTED')
              AND active_run_id IS NULL
            ORDER BY session_id
            """
        ).fetchall()
        for session_id, previous_status in session_rows:
            session_id = str(session_id)
            self._claim_profile_locked(session_id)
            if session_id in owned_sync_sessions:
                continue
            now = _now_text()
            changed = self._registry.execute(
                """
                UPDATE diagnostic_sessions
                SET status = 'FAILED', updated_at = ?
                WHERE session_id = ? AND status = ? AND active_run_id IS NULL
                """,
                (now, session_id, previous_status),
            ).rowcount
            if changed != 1:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            self._append_event(
                session_id=session_id,
                run_id=None,
                event_type="SESSION_FAILED",
                from_status=str(previous_status),
                to_status=DiagnosticSessionStatus.FAILED.value,
                reason_code="SERVICE_RESTARTED_DURING_SYNC_TURN",
                dedupe_key=f"session:{session_id}:sync-orphan-failed",
                created_at=now,
            )
        return committed_orphans

    def _reconcile_orphaned_projections(self) -> None:
        """把进程退出时遗留的投影抢占恢复为可重试失败态。"""
        rows = self._registry.execute(
            """
            SELECT draft_id, version FROM diagnostic_draft_projections
            WHERE status = 'IN_PROGRESS' ORDER BY draft_id, version
            """
        ).fetchall()
        for draft_id, version in rows:
            now = _now_text()
            changed = self._registry.execute(
                """
                UPDATE diagnostic_draft_projections
                SET status = 'FAILED', last_error_code = ?, updated_at = ?
                WHERE draft_id = ? AND version = ? AND status = 'IN_PROGRESS'
                """,
                (
                    "SERVICE_RESTARTED_DURING_PROJECTION",
                    now,
                    draft_id,
                    version,
                ),
            ).rowcount
            draft_changed = self._registry.execute(
                """
                UPDATE diagnostic_draft_tickets
                SET status = 'PROJECTION_FAILED', projection_note = ?, updated_at = ?
                WHERE draft_id = ? AND version = ?
                  AND status = 'PROJECTION_FAILED'
                """,
                (
                    "外部工单投影在服务重启时中断，可安全重试",
                    now,
                    draft_id,
                    version,
                ),
            ).rowcount
            if changed != 1 or draft_changed != 1:
                raise SessionStoreError(SANITIZED_STORE_ERROR)

    def _append_event(
        self,
        *,
        session_id: str,
        run_id: str | None,
        event_type: str,
        from_status: str | None,
        to_status: str,
        dedupe_key: str,
        reason_code: str | None = None,
        created_at: str | None = None,
    ) -> None:
        cursor = self._registry.execute(
            """
            INSERT INTO diagnostic_session_events (
                session_id, run_id, event_type, from_status, to_status,
                reason_code, dedupe_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dedupe_key) DO NOTHING
            """,
            (
                session_id,
                run_id,
                event_type,
                from_status,
                to_status,
                reason_code,
                dedupe_key,
                created_at or _now_text(),
            ),
        )
        if cursor.rowcount == 1:
            return
        existing = self._registry.execute(
            """
            SELECT session_id, run_id, event_type, from_status, to_status, reason_code
            FROM diagnostic_session_events WHERE dedupe_key = ?
            """,
            (dedupe_key,),
        ).fetchone()
        expected = (
            session_id,
            run_id,
            event_type,
            from_status,
            to_status,
            reason_code,
        )
        if existing != expected:
            raise SessionStoreError(SANITIZED_STORE_ERROR)

    def _close_connections(self) -> None:
        for attribute in ("_checkpoint_connection", "_registry"):
            connection = getattr(self, attribute, None)
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
                setattr(self, attribute, None)

    def _close_synthetic_effect_adapter(self) -> None:
        adapter = self._synthetic_effect_adapter
        self._synthetic_effect_adapter = None
        owned = self._owns_synthetic_effect_adapter
        self._owns_synthetic_effect_adapter = False
        if owned and adapter is not None:
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _control_key(self, run_id: str) -> tuple[str, str]:
        return str(self.db_path), run_id

    def _register_control(self, run_id: str, control: RunControl) -> None:
        """在状态提交前登记 owner，关闭 commit→schedule 的取消竞态窗口。"""
        key = self._control_key(run_id)
        with _PROCESS_CONTROLS_LOCK:
            with self._worker_lock:
                if self._executor is None:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                # INTERRUPTED 发布后旧 segment 可能尚在 finally；新 segment 可替换它，
                # 注销时以对象身份核验，旧 segment 不会误删新 owner。
                self._active_controls[run_id] = control
                _PROCESS_RUN_CONTROLS[key] = control

    def _unregister_control(self, run_id: str, control: RunControl) -> None:
        key = self._control_key(run_id)
        with _PROCESS_CONTROLS_LOCK:
            with self._worker_lock:
                if self._active_controls.get(run_id) is control:
                    self._active_controls.pop(run_id, None)
                if _PROCESS_RUN_CONTROLS.get(key) is control:
                    _PROCESS_RUN_CONTROLS.pop(key, None)

    def _process_control(self, run_id: str) -> RunControl | None:
        with _PROCESS_CONTROLS_LOCK:
            return _PROCESS_RUN_CONTROLS.get(self._control_key(run_id))

    def _register_sync_session_owner(self, session_id: str) -> object:
        token = object()
        with _PROCESS_CONTROLS_LOCK:
            with self._sync_condition:
                if self._closing:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                self._sync_owner_tokens[session_id] = token
                _PROCESS_SYNC_SESSION_OWNERS[(str(self.db_path), session_id)] = token
        return token

    def _unregister_sync_session_owner(self, session_id: str, token: object) -> None:
        key = (str(self.db_path), session_id)
        with _PROCESS_CONTROLS_LOCK:
            with self._sync_condition:
                if self._sync_owner_tokens.get(session_id) is token:
                    self._sync_owner_tokens.pop(session_id, None)
                if _PROCESS_SYNC_SESSION_OWNERS.get(key) is token:
                    _PROCESS_SYNC_SESSION_OWNERS.pop(key, None)
                self._sync_condition.notify_all()

    def close(self) -> None:
        """显式释放 SQLite 连接，便于测试真实模拟进程重建。"""
        with self._worker_lock:
            self._closing = True
            active = list(self._active_controls.items())
            executor = self._executor
            self._executor = None
        shutdown_error: SessionStoreError | None = None
        for run_id, control in active:
            if self._process_control(run_id) is not control:
                continue
            with self._committed_run_observations() as committed:
                try:
                    self._request_shutdown_drain(run_id)
                except SessionStoreError as exc:
                    shutdown_error = exc
                # Keep the existing safety signal ahead of new observer work.
                control.request_drain("shutdown")
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        with self._sync_condition:
            while self._sync_owner_tokens:
                self._sync_condition.wait()
        with self._registry_lock:
            self._close_synthetic_effect_adapter()
            self._close_connections()
        if shutdown_error is not None:
            raise shutdown_error

    def _request_shutdown_drain(self, run_id: str) -> None:
        """关闭前把内存 drain 信号先持久化，避免留下无归属 RUNNING。"""
        now = _now_text()
        try:
            with self._committed_run_observations() as committed, self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                run = self._select_run_locked(run_id)
                if run is None or run.status not in {
                    DiagnosticRunStatus.QUEUED.value,
                    DiagnosticRunStatus.RUNNING.value,
                    DiagnosticRunStatus.CANCEL_REQUESTED.value,
                }:
                    self._registry.commit()
                    return
                if run.status == DiagnosticRunStatus.QUEUED.value:
                    self._set_cancelled_locked(
                        run,
                        from_status=run.status,
                        now=now,
                        reason_code="SERVICE_SHUTDOWN",
                    )
                elif run.status == DiagnosticRunStatus.RUNNING.value:
                    run_changed = self._registry.execute(
                        """
                        UPDATE diagnostic_runs
                        SET status = 'CANCEL_REQUESTED', updated_at = ?,
                            stop_reason = 'SERVICE_SHUTDOWN'
                        WHERE run_id = ? AND status = 'RUNNING'
                        """,
                        (now, run_id),
                    ).rowcount
                    session_changed = self._registry.execute(
                        """
                        UPDATE diagnostic_sessions
                        SET status = 'CANCEL_REQUESTED', updated_at = ?
                        WHERE session_id = ? AND active_run_id = ?
                          AND status = 'RUNNING'
                        """,
                        (now, run.session_id, run_id),
                    ).rowcount
                    if (run_changed, session_changed) != (1, 1):
                        self._registry.rollback()
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    self._append_event(
                        session_id=run.session_id,
                        run_id=run_id,
                        event_type="RUN_CANCEL_REQUESTED",
                        from_status=DiagnosticRunStatus.RUNNING.value,
                        to_status=DiagnosticRunStatus.CANCEL_REQUESTED.value,
                        reason_code="SERVICE_SHUTDOWN",
                        dedupe_key=f"run:{run_id}:cancel-requested",
                        created_at=now,
                    )
                    self._append_event(
                        session_id=run.session_id,
                        run_id=run_id,
                        event_type="SESSION_CANCEL_REQUESTED",
                        from_status=DiagnosticSessionStatus.RUNNING.value,
                        to_status=DiagnosticSessionStatus.CANCEL_REQUESTED.value,
                        reason_code="SERVICE_SHUTDOWN",
                        dedupe_key=f"run:{run_id}:session-cancel-requested",
                        created_at=now,
                    )
                self._registry.commit()
                if run.status != DiagnosticRunStatus.CANCEL_REQUESTED.value:
                    committed.append((
                        run_id,
                        "RUN_CANCEL_REQUESTED" if run.status == DiagnosticRunStatus.RUNNING.value
                        else "RUN_CANCELLED",
                        "robotops.persistence.run_state",
                    ))
        except Exception as exc:
            if isinstance(exc, SessionStoreError):
                raise
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if (
            not isinstance(session_id, str)
            or not SESSION_ID_PATTERN.fullmatch(session_id)
        ):
            raise InvalidSessionIdError("invalid session id")

    @staticmethod
    def _new_token() -> str:
        # 24 个随机字节 = 192 bit，token_urlsafe 后固定为 32 个无填充字符。
        return secrets.token_urlsafe(24)

    def create_session(
        self, request_id: str | None = None
    ) -> CreateDiagnosticSessionResponse:
        """生成服务端 token；可选请求键使丢失的创建响应可安全重放。"""
        if request_id is not None:
            self._validate_request_id(request_id)
        for _ in range(5):
            session_id = self._new_token()
            thread_id = self._new_token()
            now = _now_text()
            try:
                with self._registry_transaction():
                    self._registry.execute("BEGIN IMMEDIATE")
                    if request_id is not None:
                        existing = self._registry.execute(
                            """
                            SELECT sessions.session_id, sessions.status,
                                   sessions.created_at, sessions.updated_at
                            FROM diagnostic_session_creations AS creations
                            JOIN diagnostic_sessions AS sessions
                              ON sessions.session_id = creations.session_id
                            WHERE creations.request_id = ?
                            """,
                            (request_id,),
                        ).fetchone()
                        if existing is not None:
                            profile = self._claim_profile_locked(existing[0])
                            self._registry.commit()
                            return CreateDiagnosticSessionResponse(
                                session_id=existing[0],
                                claim_evidence_profile=profile,
                                status=DiagnosticSessionStatus(existing[1]),
                                created_at=_parse_time(existing[2]),
                                updated_at=_parse_time(existing[3]),
                            )
                    self._registry.execute(
                        """
                        INSERT INTO diagnostic_sessions (
                            session_id, thread_id, status, created_at, updated_at, turn_count
                        ) VALUES (?, ?, 'OPEN', ?, ?, 0)
                        """,
                        (session_id, thread_id, now, now),
                    )
                    profile = self._insert_claim_profile_locked(session_id)
                    if request_id is not None:
                        self._registry.execute(
                            """
                            INSERT INTO diagnostic_session_creations (
                                request_id, session_id, created_at
                            ) VALUES (?, ?, ?)
                            """,
                            (request_id, session_id, now),
                        )
                    self._append_event(
                        session_id=session_id,
                        run_id=None,
                        event_type="SESSION_CREATED",
                        from_status=None,
                        to_status=DiagnosticSessionStatus.OPEN.value,
                        dedupe_key=f"session:{session_id}:created",
                        created_at=now,
                    )
                    self._registry.commit()
            except sqlite3.IntegrityError:
                continue
            except Exception as exc:
                raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
            return CreateDiagnosticSessionResponse(
                session_id=session_id,
                claim_evidence_profile=profile,
                status=DiagnosticSessionStatus.OPEN,
                created_at=_parse_time(now),
                updated_at=_parse_time(now),
            )
        raise SessionStoreError(SANITIZED_STORE_ERROR)

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not isinstance(run_id, str) or not SESSION_ID_PATTERN.fullmatch(run_id):
            raise InvalidRunIdError("invalid run id")

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        if not isinstance(request_id, str) or not REQUEST_ID_PATTERN.fullmatch(request_id):
            raise RunConflictError(RUN_CONFLICT)

    @staticmethod
    def _validate_draft_id(draft_id: str) -> None:
        if not isinstance(draft_id, str) or not SESSION_ID_PATTERN.fullmatch(draft_id):
            raise InvalidDraftIdError("invalid draft id")

    @staticmethod
    def _request_payload(message: str) -> tuple[str, str]:
        payload_json = json.dumps(
            {"message": message, "operation": "RUN"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return payload_json, hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    def _select_request_locked(
        self, session_id: str, request_id: str
    ) -> sqlite3.Row | tuple | None:
        return self._registry.execute(
            """
            SELECT session_id, request_id, operation, payload_sha256, payload_json,
                   run_id, attempt_count, created_at, updated_at
            FROM diagnostic_requests
            WHERE session_id = ? AND request_id = ?
            """,
            (session_id, request_id),
        ).fetchone()

    def _request_id_for_run_locked(self, run_id: str) -> str:
        row = self._registry.execute(
            "SELECT request_id FROM diagnostic_requests WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None or not isinstance(row[0], str):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return str(row[0])

    def _run_attempt_suffix_locked(self, run_id: str) -> str:
        row = self._registry.execute(
            "SELECT attempt_count FROM diagnostic_requests WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        # MOD-020 旧 registry 可能含没有 request 映射的 orphan run；启动收口
        # 仍须成功，且首个失败事件沿用原 dedupe key。
        if row is None:
            return ""
        if not isinstance(row[0], int) or row[0] < 1:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return "" if row[0] == 1 else f":attempt:{row[0]}"

    def _append_message_locked(
        self,
        *,
        session_id: str,
        sequence_no: int,
        turn_number: int,
        role: DiagnosticMessageRole,
        kind: DiagnosticMessageKind,
        content: str,
        run_id: str,
        request_id: str,
        payload: Mapping[str, Any] | None,
        created_at: str,
    ) -> None:
        payload_json = json.dumps(
            dict(payload or {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._registry.execute(
            """
            INSERT INTO diagnostic_messages (
                message_id, session_id, sequence_no, turn_number, role, kind,
                content, run_id, request_id, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._new_token(),
                session_id,
                sequence_no,
                turn_number,
                role.value,
                kind.value,
                content,
                run_id,
                request_id,
                payload_json,
                created_at,
            ),
        )

    def _session_response_from_row(
        self, row: sqlite3.Row | tuple
    ) -> DiagnosticSessionResponse:
        session_id = str(row[0])
        with self._registry_lock:
            contract = self._execution_contract_locked(session_id)
            if contract == CLAIM_EXECUTION_CONTRACT:
                for approval_id, run_id in self._registry.execute(
                    "SELECT approval_id,run_id FROM diagnostic_interrupts WHERE session_id=?",
                    (session_id,),
                ).fetchall():
                    run = self._select_run_locked(str(run_id))
                    if run is None or run.session_id != session_id:
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    family = self._effect_family_locked(
                        self._effect_parent_locked(str(approval_id)), run,
                    )
                    if family == "EFFECT_V2":
                        self._validate_claim_effect_binding_locked(
                            str(approval_id), require_checkpoint=run.status == "INTERRUPTED",
                        )
                for run_id, in self._registry.execute(
                    "SELECT run_id FROM diagnostic_runs WHERE session_id=? AND status='COMPLETED'",
                    (session_id,),
                ).fetchall():
                    run = self._select_run_locked(str(run_id))
                    if run is None:
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    self._decoded_claim_run_locked(run)
        return DiagnosticSessionResponse(
            session_id=session_id,
            claim_evidence_profile=self._claim_profile_locked(session_id),
            status=DiagnosticSessionStatus(row[1]),
            turn_count=row[2],
            parent_session_id=row[3],
            active_run_id=row[4],
            latest_run_id=row[5],
            created_at=_parse_time(row[6]),
            updated_at=_parse_time(row[7]),
        )

    def get_session(self, session_id: str) -> DiagnosticSessionResponse:
        self._validate_session_id(session_id)
        try:
            with self._registry_lock:
                row = self._registry.execute(
                    """
                    SELECT sessions.session_id, sessions.status, sessions.turn_count,
                           sessions.parent_session_id, sessions.active_run_id,
                           (
                               SELECT run_id FROM diagnostic_runs
                               WHERE session_id = sessions.session_id
                               ORDER BY turn_number DESC LIMIT 1
                           ) AS latest_run_id,
                           sessions.created_at, sessions.updated_at
                    FROM diagnostic_sessions AS sessions WHERE sessions.session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
            if row is None:
                raise SessionNotFoundError(SESSION_NOT_FOUND)
            return self._session_response_from_row(row)
        except (InvalidSessionIdError, SessionNotFoundError):
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def reopen_session(
        self, parent_session_id: str
    ) -> tuple[DiagnosticSessionResponse, bool]:
        """创建唯一空白 successor；父 COMPLETED 行保持逐字段不可变。"""
        self._validate_session_id(parent_session_id)
        try:
            # The connection rolls back *before* releasing the shared lock,
            # including domain validation failures from persisted profiles.
            with self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                parent = self._registry.execute(
                    "SELECT status FROM diagnostic_sessions WHERE session_id = ?",
                    (parent_session_id,),
                ).fetchone()
                if parent is None:
                    self._registry.rollback()
                    raise SessionNotFoundError(SESSION_NOT_FOUND)
                self._claim_profile_locked(parent_session_id)
                if parent[0] != DiagnosticSessionStatus.COMPLETED.value:
                    self._registry.rollback()
                    raise SessionConflictError(SESSION_CONFLICT)
                existing = self._registry.execute(
                    """
                    SELECT sessions.session_id, sessions.status, sessions.turn_count,
                           sessions.parent_session_id, sessions.active_run_id,
                           NULL AS latest_run_id, sessions.created_at, sessions.updated_at
                    FROM diagnostic_sessions AS sessions WHERE parent_session_id = ?
                    """,
                    (parent_session_id,),
                ).fetchone()
                if existing is not None:
                    response = self._session_response_from_row(existing)
                    self._registry.commit()
                    return response, False

                for _ in range(5):
                    session_id = self._new_token()
                    thread_id = self._new_token()
                    now = _now_text()
                    try:
                        self._registry.execute(
                            """
                            INSERT INTO diagnostic_sessions (
                                session_id, thread_id, status, created_at, updated_at,
                                turn_count, parent_session_id, active_run_id
                            ) VALUES (?, ?, 'OPEN', ?, ?, 0, ?, NULL)
                            """,
                            (session_id, thread_id, now, now, parent_session_id),
                        )
                    except sqlite3.IntegrityError:
                        continue
                    profile = self._insert_claim_profile_locked(session_id)
                    self._append_event(
                        session_id=session_id,
                        run_id=None,
                        event_type="SESSION_REOPENED",
                        from_status=None,
                        to_status=DiagnosticSessionStatus.OPEN.value,
                        dedupe_key=f"session:{parent_session_id}:reopen",
                        reason_code="LINKED_SUCCESSOR",
                        created_at=now,
                    )
                    self._registry.commit()
                    return (
                        DiagnosticSessionResponse(
                            session_id=session_id,
                            claim_evidence_profile=profile,
                            status=DiagnosticSessionStatus.OPEN,
                            turn_count=0,
                            parent_session_id=parent_session_id,
                            active_run_id=None,
                            latest_run_id=None,
                            created_at=_parse_time(now),
                            updated_at=_parse_time(now),
                        ),
                        True,
                    )
                self._registry.rollback()
                raise SessionStoreError(SANITIZED_STORE_ERROR)
        except (InvalidSessionIdError, SessionNotFoundError, SessionConflictError):
            raise
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def _row_to_record(self, row: sqlite3.Row | tuple) -> _SessionRecord:
        profile = self._claim_profile_locked(row[0])
        return _SessionRecord(*row, claim_evidence_profile=profile)

    def _claim_session(self, session_id: str) -> tuple[_SessionRecord, object]:
        self._validate_session_id(session_id)
        owner_token: object | None = None
        try:
            with self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                row = self._registry.execute(
                    """
                    SELECT session_id, thread_id, status, created_at, updated_at, turn_count
                    FROM diagnostic_sessions WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if row is None:
                    self._registry.rollback()
                    raise SessionNotFoundError(SESSION_NOT_FOUND)
                record = self._row_to_record(row)
                if record.status not in _WRITABLE_STATUSES:
                    self._registry.rollback()
                    raise SessionConflictError(SESSION_CONFLICT)
                changed = self._registry.execute(
                    """
                    UPDATE diagnostic_sessions SET status = 'RUNNING', updated_at = ?
                    WHERE session_id = ? AND status = ? AND active_run_id IS NULL
                    """,
                    (_now_text(), session_id, record.status),
                ).rowcount
                if changed != 1:
                    self._registry.rollback()
                    raise SessionConflictError(SESSION_CONFLICT)
                owner_token = self._register_sync_session_owner(session_id)
                self._registry.commit()
                return record, owner_token
        except (InvalidSessionIdError, SessionNotFoundError, SessionConflictError):
            if owner_token is not None:
                self._unregister_sync_session_owner(session_id, owner_token)
            raise
        except Exception as exc:
            if owner_token is not None:
                self._unregister_sync_session_owner(session_id, owner_token)
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }

    def _async_runtime_context(
        self,
        *,
        session_id: str,
        run_id: str,
        approval_id: str | None,
    ) -> AgentRuntimeContext:
        """Create a fresh, non-checkpointed authority context for each segment."""
        with safe_span("robotops.session.load") as span:
            contract = self._execution_contract_locked(session_id)
            legacy = contract == LEGACY_EXECUTION_CONTRACT
            claim_effect = contract == CLAIM_EXECUTION_CONTRACT
            run = self._select_run_locked(run_id)
            if run is None or run.session_id != session_id:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if claim_effect:
                from app.services.claim_effect_executor import ServiceClaimEffectAuthority
            context = AgentRuntimeContext(
                surface=ASYNC_AUTHORITY_SURFACE,
                effect_mode=self._effect_mode if legacy or claim_effect else None,
                session_id=session_id,
                run_id=run_id,
                approval_id=approval_id,
                effect_authority=(
                    ServiceClaimEffectAuthority(self) if claim_effect else self._effect_authority
                    if legacy and self._effect_mode == EFFECT_MODE_SYNTHETIC_RESTART
                    else None
                ),
                execution_contract=contract,
                request_id=self._request_id_for_run_locked(run_id),
                turn_number=run.turn_number,
            )

            set_safe_outcome(span, "COMPLETED")
            return context
    def _invoke_async_graph(
        self,
        value: object,
        *,
        config: dict,
        session_id: str,
        run_id: str,
        approval_id: str | None,
        control: RunControl,
    ) -> object:
        kwargs: dict[str, Any] = {
            "config": config,
            "durability": "sync",
            "control": control,
        }
        if self._graph_accepts_context:
            kwargs["context"] = self._async_runtime_context(
                session_id=session_id,
                run_id=run_id,
                approval_id=approval_id,
            )
        elif self._effect_mode is not None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        with safe_span("robotops.graph.invoke"):
            return self._graph.invoke(value, **kwargs)

    def _authoritative_resume_update(
        self, checkpoint: Mapping[str, Any], *,
        claim_profile: str = DISABLED_CLAIM_EVIDENCE_PROFILE,
    ) -> dict[str, Any]:
        """补齐旧 checkpoint，并以当前 service 策略覆盖私有执行边界。"""
        if not isinstance(checkpoint, Mapping):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        normalized = dict(checkpoint)
        self._validate_claim_checkpoint_profile(normalized, claim_profile, terminal=False)
        defaults: dict[str, Any] = {
            "tool_iterations": 0,
            "max_tool_iterations": MAX_TOOL_ITERATIONS,
            "execution_stop_reason": "",
            "diagnostic_plan": {},
            "pending_tool_action": {},
            "completed_plan_steps": [],
            "tool_trajectory": [],
            "agent_route": "",
            "tool_loop_failed": False,
            "tool_capability_profile": tool_loop.CORE_TOOL_PROFILE,
            "tool_result_evidence": {},
            "verification_status": "NOT_STARTED",
            "verifier_result": {},
            "verifier_result_sha256": "",
            "effect_policy_status": "NOT_STARTED",
            "effect_proposal": {},
            "effect_proposal_sha256": "",
            "effect_runtime": {},
        }
        for field, value in defaults.items():
            if field not in normalized:
                normalized[field] = value.copy() if isinstance(value, (dict, list)) else value
        existing_string_fields = tuple(
            field
            for field in _RESTORED_STRING_FIELDS
            if field
            not in {
                "ticket_mode",
                "execution_stop_reason",
                "agent_route",
                "verification_status",
                "verifier_result_sha256",
                "effect_policy_status",
                "effect_proposal_sha256",
            }
        )
        existing_list_fields = tuple(
            field
            for field in _RESTORED_LIST_FIELDS
            if field not in {"completed_plan_steps", "tool_trajectory"}
        )
        existing_bool_fields = tuple(
            field
            for field in _RESTORED_BOOL_FIELDS
            if field not in {"agent_tool_loop_enabled", "tool_loop_failed"}
        )
        if (
            any(field not in normalized for field in existing_string_fields)
            or any(field not in normalized for field in existing_list_fields)
            or any(field not in normalized for field in existing_bool_fields)
            or "follow_up_rounds" not in normalized
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        # 这是服务端私有能力位：checkpoint 只保存历史状态，不能覆盖当前策略。
        normalized["agent_tool_loop_enabled"] = self._enable_agent_tool_loop
        normalized["ticket_mode"] = "draft"
        if (
            any(
                not isinstance(normalized.get(field), str)
                for field in _RESTORED_STRING_FIELDS
            )
            or any(
                not isinstance(normalized.get(field), list)
                for field in _RESTORED_LIST_FIELDS
            )
            or any(
                not isinstance(normalized.get(field), dict)
                for field in _RESTORED_DICT_FIELDS
            )
            or any(
                not isinstance(normalized.get(field), bool)
                for field in _RESTORED_BOOL_FIELDS
            )
            or not normalized["user_query"]
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        rounds = normalized["follow_up_rounds"]
        if (
            not isinstance(rounds, int)
            or isinstance(rounds, bool)
            or not 0 <= rounds <= MAX_FOLLOW_UP_ROUNDS
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        conversation = self._conversation(normalized)
        expected_roles = ["user", "assistant"] * rounds + ["user"]
        authoritative_query = "\n".join(
            item["content"] for item in conversation if item["role"] == "user"
        )
        if (
            [item["role"] for item in conversation] != expected_roles
            or authoritative_query != normalized["user_query"]
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if not STOP_REASON_PATTERN.fullmatch(normalized["execution_stop_reason"]):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        agent_fields = self._validate_agent_checkpoint_state(
            normalized,
            allow_active=True,
        )
        return {
            "ticket_mode": "draft",
            "tool_iterations": normalized["tool_iterations"],
            "max_tool_iterations": MAX_TOOL_ITERATIONS,
            "execution_stop_reason": normalized["execution_stop_reason"],
            "risk_level": normalized["risk_level"],
            "urgency": normalized["urgency"],
            "need_human": normalized["need_human"],
            "risk_check_failed": normalized["risk_check_failed"],
            "llm_failed": normalized["llm_failed"],
            "agent_tool_loop_enabled": self._enable_agent_tool_loop,
            **agent_fields,
        }

    def _validate_claim_checkpoint_profile(
        self, state: Mapping[str, Any], profile: str, *, terminal: bool,
        session_id: str | None = None, turn_number: int | None = None,
    ) -> None:
        from app.agent.claim_sources import CONTEXT_FIELDS, _json_copy, _validate_context, MAX_SNAPSHOT_BYTES
        try:
            if session_id is not None:
                self._validate_claim_effect_state_domain(state, session_id)
            elif profile == DISABLED_CLAIM_EVIDENCE_PROFILE and any(
                state.get(key, default) != default for key, default in (
                    ("claim_effect_intent", {}), ("claim_effect_runtime", {}),
                    ("claim_effect_decision", ""),
                )
            ):
                raise ValueError()
            values = _json_copy({key: state.get(key) for key in (
                "claim_evidence_profile", "claim_context_seed", "claim_draft",
                "claim_verification", "claim_publication",
            )}, maximum=MAX_SNAPSHOT_BYTES * 3)
            if profile == DISABLED_CLAIM_EVIDENCE_PROFILE:
                if "claim_evidence_profile" in state and values["claim_evidence_profile"] != DISABLED_CLAIM_EVIDENCE_PROFILE:
                    raise ValueError()
                if values["claim_evidence_profile"] not in {None, DISABLED_CLAIM_EVIDENCE_PROFILE}:
                    raise ValueError()
                if any(key in state and (type(values[key]) is not dict or values[key])
                       for key in values if key != "claim_evidence_profile"):
                    raise ValueError()
                return
            if (values["claim_evidence_profile"] != ACTIVE_CLAIM_EVIDENCE_PROFILE
                    or any(type(values[key]) is not dict for key in values if key != "claim_evidence_profile")
                    or state.get("effect_policy_status") != "NOT_STARTED"
                    or state.get("effect_proposal") != {}
                    or state.get("effect_proposal_sha256") != ""
                    or state.get("effect_runtime") != {}):
                raise ValueError()
            seed = values["claim_context_seed"]
            _validate_context(seed)
            if seed["user_query"] != state.get("user_query"):
                raise ValueError()
            if not terminal:
                if values["claim_publication"]:
                    raise ValueError()
                if values["claim_verification"]:
                    from app.agent.claim_verification import validate_claim_verification
                    validate_claim_verification(values["claim_verification"],
                        expected_context={key: state[key] for key in CONTEXT_FIELDS})
                return
            from app.services import claim_sessions
            authority = {"session_id": session_id, "turn_number": turn_number,
                         "run_id": None, "request_id": None,
                         **{key: state[key] for key in CONTEXT_FIELDS}}
            with self._registry_lock:
                row = self._registry.execute(
                    "SELECT run_id FROM diagnostic_runs WHERE session_id = ? "
                    "AND turn_number = ? AND status = 'COMPLETED'",
                    (session_id, turn_number),
                ).fetchone()
                if row is not None:
                    run = self._select_run_locked(row[0])
                    decoded = self._decoded_claim_run_locked(run)
                    authority = decoded["authority"]
                    from app.agent.claim_sources import _canonical
                    if _canonical(values["claim_publication"]) != _canonical(decoded["publication"]):
                        raise ValueError()
            claim_sessions.validate_claim_terminal_state(state, expected_authority=authority)
        except Exception:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from None

    def _load_previous_state(self, record: _SessionRecord) -> dict:
        if record.status == DiagnosticSessionStatus.OPEN.value:
            if record.turn_count != 0:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            return {}
        if (
            record.status != DiagnosticSessionStatus.AWAITING_USER.value
            or not isinstance(record.turn_count, int)
            or isinstance(record.turn_count, bool)
            or record.turn_count < 1
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        with safe_span("robotops.checkpoint.restore") as span:
            try:
                snapshot = self._graph.get_state(self._config(record.thread_id))
                values = snapshot.values
            except Exception as exc:
                raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
            if not isinstance(values, Mapping) or not values:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            restored = dict(values)
            self._validate_claim_checkpoint_profile(
                restored, record.claim_evidence_profile, terminal=True,
                session_id=record.session_id, turn_number=record.turn_count,
            )
            # MOD-019 checkpoint 向前兼容：新增计数缺失时由服务端补权威默认值。
            restored.setdefault("tool_iterations", 0)
            restored.setdefault("max_tool_iterations", MAX_TOOL_ITERATIONS)
            restored.setdefault("execution_stop_reason", "")
            restored["agent_tool_loop_enabled"] = self._enable_agent_tool_loop
            restored.setdefault("diagnostic_plan", {})
            restored.setdefault("pending_tool_action", {})
            restored.setdefault("completed_plan_steps", [])
            restored.setdefault("tool_trajectory", [])
            restored.setdefault("agent_route", "")
            restored.setdefault("tool_loop_failed", False)
            restored.setdefault("tool_capability_profile", tool_loop.CORE_TOOL_PROFILE)
            restored.setdefault("tool_result_evidence", {})
            restored.setdefault("verification_status", "NOT_STARTED")
            restored.setdefault("verifier_result", {})
            restored.setdefault("verifier_result_sha256", "")
            restored.setdefault("effect_policy_status", "NOT_STARTED")
            restored.setdefault("effect_proposal", {})
            restored.setdefault("effect_proposal_sha256", "")
            restored.setdefault("effect_runtime", {})
            restored["ticket_mode"] = "draft"
            if any(
                field not in restored or not isinstance(restored[field], str)
                for field in _RESTORED_STRING_FIELDS
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if any(
                field not in restored or not isinstance(restored[field], list)
                for field in _RESTORED_LIST_FIELDS
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if any(
                field not in restored or not isinstance(restored[field], dict)
                for field in _RESTORED_DICT_FIELDS
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if any(
                field not in restored or not isinstance(restored[field], bool)
                for field in _RESTORED_BOOL_FIELDS
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            tool_iterations = restored["tool_iterations"]
            max_tool_iterations = restored["max_tool_iterations"]
            if (
                not isinstance(tool_iterations, int)
                or isinstance(tool_iterations, bool)
                or not 0 <= tool_iterations <= MAX_TOOL_ITERATIONS
                or max_tool_iterations != MAX_TOOL_ITERATIONS
                or isinstance(max_tool_iterations, bool)
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            conversation = self._conversation(restored)
            expected_roles = ["user", "assistant"] * record.turn_count
            if [item["role"] for item in conversation] != expected_roles:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            authoritative_query = "\n".join(
                item["content"] for item in conversation if item["role"] == "user"
            )
            if restored["user_query"] != authoritative_query:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            rounds = restored.get("follow_up_rounds")
            if (
                not isinstance(rounds, int)
                or isinstance(rounds, bool)
                or rounds != record.turn_count
                or rounds > MAX_FOLLOW_UP_ROUNDS
                or restored["route"] != "ask"
                or not restored["follow_up_questions"]
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            self._validate_agent_checkpoint_state(restored, allow_active=False)
            set_safe_outcome(span, "COMPLETED")
            return restored

    @staticmethod
    def _conversation(previous: Mapping[str, Any]) -> list[dict]:
        raw = previous.get("conversation", [])
        if not isinstance(raw, list):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        conversation: list[dict] = []
        for item in raw:
            if (
                not isinstance(item, dict)
                or item.get("role") not in {"user", "assistant"}
                or not isinstance(item.get("content"), str)
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            conversation.append({"role": item["role"], "content": item["content"]})
        return conversation

    def _prepare_state(
        self, previous: Mapping[str, Any], message: str, *,
        claim_profile: str = DISABLED_CLAIM_EVIDENCE_PROFILE,
    ) -> dict:
        conversation = self._conversation(previous)
        conversation.append({"role": "user", "content": message})
        rounds = previous.get("follow_up_rounds", 0)
        if (
            not isinstance(rounds, int)
            or isinstance(rounds, bool)
            or not 0 <= rounds <= MAX_FOLLOW_UP_ROUNDS
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        combined_query = "\n".join(
            item["content"] for item in conversation if item["role"] == "user"
        )
        state = initial_state(
            combined_query,
            conversation=conversation,
            follow_up_rounds=rounds,
        )
        state["ticket_mode"] = "draft"
        state["agent_tool_loop_enabled"] = self._enable_agent_tool_loop
        state["tool_capability_profile"] = (
            self._authorized_tool_capability_profile(
                previous.get(
                    "tool_capability_profile", tool_loop.CORE_TOOL_PROFILE
                )
            )
            if previous
            else self._tool_capability_profile
        )
        state["tool_result_evidence"] = {}
        for field in ("device_model", "fault_category", "fault_description"):
            value = previous.get(field, "")
            if not isinstance(value, str):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            state[field] = value
        sticky = merge_sticky_safety(previous, state)
        state.update(sticky)
        state["claim_evidence_profile"] = claim_profile
        if claim_profile == ACTIVE_CLAIM_EVIDENCE_PROFILE:
            from app.agent.claim_graph import validate_claim_context
            try:
                validate_claim_context(state)
            except Exception:
                raise SessionStoreError(SANITIZED_STORE_ERROR) from None
        return state

    def _authorized_tool_capability_profile(self, candidate: object) -> str:
        """只允许 checkpoint 保持在当前 service 能力集合的子集内。"""
        try:
            configured_tools = set(
                tool_loop.registry_for_profile(self._tool_capability_profile)
            )
            candidate_tools = set(tool_loop.registry_for_profile(candidate))
        except tool_loop.ToolLoopError as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        if not candidate_tools.issubset(configured_tools):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return str(candidate)

    def _finish_registry(
        self, session_id: str, status: DiagnosticSessionStatus, turn_count: int, now: str,
        *, claim_state: dict | None = None, claim_response: DiagnosticTurnResponse | None = None,
    ) -> None:
        try:
            with self._registry_transaction():
                profile = self._claim_profile_locked(session_id)
                if profile == ACTIVE_CLAIM_EVIDENCE_PROFILE or claim_response is not None:
                    if profile != ACTIVE_CLAIM_EVIDENCE_PROFILE or claim_state is None or claim_response is None:
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    from app.agent.claim_sources import CONTEXT_FIELDS
                    from app.services.claim_sessions import validate_claim_terminal_state, validate_claim_response
                    authority = {"session_id": session_id, "run_id": None, "request_id": None,
                                 "turn_number": turn_count,
                                 **{key: claim_state[key] for key in CONTEXT_FIELDS}}
                    validate_claim_terminal_state(claim_state, expected_authority=authority)
                    validate_claim_response(claim_response.model_dump(mode="json"), expected_authority=authority)
                    if claim_response.status != status:
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                changed = self._registry.execute(
                    """
                    UPDATE diagnostic_sessions
                    SET status = ?, updated_at = ?, turn_count = ?
                    WHERE session_id = ? AND status = 'RUNNING'
                    """,
                    (status.value, now, turn_count, session_id),
                ).rowcount
                if changed != 1:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                self._registry.commit()
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def _mark_failed(self, session_id: str) -> None:
        now = _now_text()
        try:
            with self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                changed = self._registry.execute(
                    """
                    UPDATE diagnostic_sessions SET status = 'FAILED', updated_at = ?
                    WHERE session_id = ? AND status = 'RUNNING'
                      AND active_run_id IS NULL
                    """,
                    (now, session_id),
                ).rowcount
                if changed != 1:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                self._append_event(
                    session_id=session_id,
                    run_id=None,
                    event_type="SESSION_FAILED",
                    from_status=DiagnosticSessionStatus.RUNNING.value,
                    to_status=DiagnosticSessionStatus.FAILED.value,
                    reason_code="SYNC_TURN_FAILED",
                    dedupe_key=f"session:{session_id}:sync-turn-failed",
                    created_at=now,
                )
                self._registry.commit()
        except Exception as exc:
            if isinstance(exc, SessionStoreError):
                raise
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def submit_turn(
        self, session_id: str, message: str, request_id: str | None = None
    ) -> DiagnosticTurnResponse:
        """合并当前文本与权威 checkpoint，执行一轮并持久化助手回复。"""
        del request_id  # MOD-021 才赋予请求幂等语义；2A 只冻结输入接缝。
        record, owner_token = self._claim_session(session_id)
        try:
            previous = self._load_previous_state(record)
            prepared = self._prepare_state(previous, message,
                                           claim_profile=record.claim_evidence_profile)
            config = self._config(record.thread_id)
            result = self._graph.invoke(prepared, config=config)
            if not isinstance(result, Mapping):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if (
                "__interrupt__" in result
                or result.get("effect_policy_status", "NOT_STARTED")
                != "NOT_STARTED"
                or bool(result.get("effect_proposal"))
                or bool(result.get("effect_proposal_sha256"))
            ):
                # typed HITL/effect 只属于异步权威 run；同步兼容入口不得
                # 把 proposal 写进 checkpoint，也不得把 interrupt 当终态发布。
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if record.claim_evidence_profile == ACTIVE_CLAIM_EVIDENCE_PROFILE:
                response, _persisted = self._materialize_claim_result(
                    session_id=session_id, turn_number=record.turn_count + 1,
                    previous=previous, source=prepared, raw_result=result, config=config,
                )
                self._finish_registry(session_id, response.status, response.turn_number,
                                      response.updated_at.isoformat().replace("+00:00", "Z"),
                                      claim_state=_persisted, claim_response=response)
                return response
            self._validate_claim_checkpoint_profile(
                result, record.claim_evidence_profile, terminal=True)
            result = merge_authoritative_facts(
                previous,
                result,
                cumulative_query=prepared["user_query"],
            )
            merged = merge_sticky_safety(previous, result)
            result = {**result, **merged}
            result["agent_tool_loop_enabled"] = self._enable_agent_tool_loop
            result["tool_capability_profile"] = prepared[
                "tool_capability_profile"
            ]
            result.setdefault(
                "tool_result_evidence", prepared.get("tool_result_evidence", {})
            )
            result["ticket_mode"] = "draft"
            result["ticket_id"] = ""
            result["ticket_note"] = ""

            questions = result.get("follow_up_questions") or []
            if not isinstance(questions, list):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            asked_for_more = result.get("route") == "ask" and bool(questions)
            limit_reached = (
                asked_for_more
                and prepared["follow_up_rounds"] >= MAX_FOLLOW_UP_ROUNDS
            )
            awaiting = asked_for_more and not limit_reached
            if limit_reached:
                result["need_human"] = True
                questions = []
            status = (
                DiagnosticSessionStatus.AWAITING_USER
                if awaiting
                else DiagnosticSessionStatus.COMPLETED
            )
            reply = (
                FOLLOW_UP_LIMIT_REPLY
                if limit_reached
                else (
                    _follow_up_reply(questions)
                    if awaiting
                    else result.get("final_answer")
                    or "诊断工作流未生成有效回答，请联系人工支持。"
                )
            )
            conversation = list(prepared["conversation"])
            conversation.append({"role": "assistant", "content": reply})
            rounds = prepared["follow_up_rounds"] + (1 if awaiting else 0)
            persisted = {
                **result,
                "user_query": prepared["user_query"],
                "conversation": conversation,
                "follow_up_rounds": rounds,
            }
            self._validate_agent_checkpoint_state(persisted, allow_active=False)
            self._graph.update_state(config, persisted)

            turn_number = record.turn_count + 1
            now = _now_text()
            self._finish_registry(session_id, status, turn_number, now)
            return DiagnosticTurnResponse(
                session_id=session_id,
                status=status,
                turn_number=turn_number,
                reply=reply,
                updated_at=_parse_time(now),
                references=result.get("references") or [],
                nodes=_infer_nodes(result),
                follow_up=questions if awaiting else [],
                diagnosis=_diagnosis(result),
                ticket_id=result.get("ticket_id") or None,
                ticket_note=result.get("ticket_note") or None,
            )
        except (SessionNotFoundError, SessionConflictError, InvalidSessionIdError):
            raise
        except Exception as exc:
            self._mark_failed(session_id)
            if isinstance(exc, SessionStoreError):
                raise
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        finally:
            self._unregister_sync_session_owner(session_id, owner_token)

    @staticmethod
    def _row_to_run_record(row: sqlite3.Row | tuple) -> _RunRecord:
        return _RunRecord(*row)

    def _select_run_locked(self, run_id: str) -> _RunRecord | None:
        row = self._registry.execute(
            """
            SELECT run_id, session_id, turn_number, status,
                   previous_session_status, base_turn_count, created_at, updated_at,
                   result_json, error_code, tool_iterations, max_tool_iterations,
                   stop_reason, current_interrupt_id
            FROM diagnostic_runs WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        self._claim_profile_locked(row[1])
        return self._row_to_run_record(row)

    def _bound_run_locked(
        self, session_id: str, run_id: str
    ) -> tuple[str, _RunRecord]:
        session = self._registry.execute(
            "SELECT status FROM diagnostic_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if session is None:
            raise SessionNotFoundError(SESSION_NOT_FOUND)
        self._claim_profile_locked(session_id)
        run = self._select_run_locked(run_id)
        if run is None:
            raise RunNotFoundError(RUN_NOT_FOUND)
        if run.session_id != session_id:
            raise RunConflictError(RUN_CONFLICT)
        return str(session[0]), run

    @staticmethod
    def _canonical_stored_json(value: object) -> tuple[object, str]:
        if not isinstance(value, str):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        try:
            parsed = json.loads(value)
            canonical = effect_capabilities.canonical_json(parsed)
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        if canonical != value:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return parsed, canonical

    def _select_typed_approval_locked(self, approval_id: str) -> dict[str, Any] | None:
        row = self._registry.execute(
            "SELECT " + ", ".join(_TYPED_APPROVAL_COLUMNS)
            + " FROM diagnostic_effect_approvals WHERE approval_id = ?",
            (approval_id,),
        ).fetchone()
        return None if row is None else dict(zip(_TYPED_APPROVAL_COLUMNS, row))

    def _select_effect_ledger_locked(self, effect_key: str) -> dict[str, Any] | None:
        row = self._registry.execute(
            "SELECT " + ", ".join(_EFFECT_LEDGER_COLUMNS)
            + " FROM diagnostic_effect_ledger WHERE effect_key = ?",
            (effect_key,),
        ).fetchone()
        return None if row is None else dict(zip(_EFFECT_LEDGER_COLUMNS, row))

    def _select_claim_effect_records_locked(self, approval_id: str) -> dict | None:
        from app.services.effect_records_v2 import APPROVAL_COLUMNS, LEDGER_COLUMNS
        approval = self._registry.execute(
            "SELECT " + ",".join(APPROVAL_COLUMNS)
            + " FROM diagnostic_claim_effect_approvals WHERE approval_id=?", (approval_id,),
        ).fetchone()
        ledger = self._registry.execute(
            "SELECT " + ",".join(LEDGER_COLUMNS)
            + " FROM diagnostic_claim_effect_ledger WHERE approval_id=?", (approval_id,),
        ).fetchone()
        if approval is None and ledger is None:
            return None
        if approval is None or ledger is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return {"approval": dict(zip(APPROVAL_COLUMNS, approval)),
                "ledger": dict(zip(LEDGER_COLUMNS, ledger))}

    def _validate_claim_effect_state_domain(self, state: Mapping[str, Any], session_id: str) -> None:
        intent = state.get("claim_effect_intent", {})
        runtime = state.get("claim_effect_runtime", {})
        decision = state.get("claim_effect_decision", "")
        if (type(intent) is not dict or type(runtime) is not dict
                or type(decision) is not str or decision not in {"", "APPROVE", "REJECT"}):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if (intent or runtime or decision) and (
            self._execution_contract_locked(session_id) != CLAIM_EXECUTION_CONTRACT
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

    def _effect_family_locked(self, parent: Mapping[str, Any], run: _RunRecord) -> str | None:
        """Dispatch only a registry-authorized kind with exactly its own sidecar."""
        payload, canonical = self._canonical_stored_json(parent["payload_json"])
        if (type(payload) is not dict or canonical != parent["payload_json"]
                or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != parent["payload_sha256"]):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        approval_id = str(parent["approval_id"])
        v1 = self._select_typed_approval_locked(approval_id)
        v2 = self._select_claim_effect_records_locked(approval_id)
        kind = payload.get("approval_kind")
        contract = self._execution_contract_locked(run.session_id)
        if kind == "EFFECT_V1":
            valid = v1 is not None and v2 is None and contract == LEGACY_EXECUTION_CONTRACT
        elif kind == "EFFECT_V2":
            valid = v2 is not None and v1 is None and contract == CLAIM_EXECUTION_CONTRACT
        else:
            if (v1 is not None or v2 is not None
                    or (type(kind) is str and kind.startswith("EFFECT_"))):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            return None
        if not valid:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return kind

    def _claim_effect_response_locked(self, run: _RunRecord, *, terminal: bool = False):
        from app.services import claim_effect_results
        from app.schemas.diagnostic_sessions import DiagnosticEffectV2Response
        ids = self._registry.execute(
            "SELECT approval_id FROM diagnostic_claim_effect_approvals WHERE run_id=?", (run.run_id,),
        ).fetchall()
        if not ids:
            return None
        if len(ids) != 1:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        approval_id = str(ids[0][0])
        validated = self._validate_claim_effect_binding_locked(
            approval_id, require_checkpoint=run.status == "INTERRUPTED")
        expected = self._claim_effect_expected_locked(validated["approval"], run)
        response = claim_effect_results.build_effect_response(
            {"approval": validated["approval"], "ledger": validated["ledger"]}, **expected,
            now_ms=self._typed_now_ms(), run_status="COMPLETED" if terminal else run.status,
        )
        return DiagnosticEffectV2Response.model_validate_json(
            effect_capabilities.canonical_json(response), strict=True)

    def _claim_effect_expected_locked(self, approval: dict, run: _RunRecord) -> dict:
        from app.agent.claim_sources import CONTEXT_FIELDS
        if self._execution_contract_locked(run.session_id) != CLAIM_EXECUTION_CONTRACT:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        message = self._claim_request_message_locked(run)
        context, _text = self._canonical_stored_json(approval["context_json"])
        if type(context) is not dict or set(context) != set(CONTEXT_FIELDS):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        query = context["user_query"]
        if type(query) is not str or (query != message and not query.endswith("\n" + message)):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        authority = {"session_id": run.session_id, "run_id": run.run_id,
                     "request_id": self._request_id_for_run_locked(run.run_id),
                     "turn_number": run.turn_number, **context}
        return {"expected_authority": authority, "execution_contract": CLAIM_EXECUTION_CONTRACT,
                "interrupt_id": approval["interrupt_id"], "approval_id": approval["approval_id"],
                "expected_binding_sha256": approval["binding_sha256"]}

    def _validate_claim_effect_binding_locked(
        self, approval_id: str, *, require_checkpoint: bool,
    ) -> dict:
        """Join caller-independent registry facts before replaying frozen V2 rows."""
        from app.services import claim_effect_results
        from app.agent.claim_sources import CONTEXT_FIELDS, _canonical
        try:
            records = self._select_claim_effect_records_locked(approval_id)
            if records is None or self._select_typed_approval_locked(approval_id) is not None:
                raise ValueError()
            approval = records["approval"]
            run = self._select_run_locked(approval["run_id"])
            parent = self._effect_parent_locked(approval_id)
            if run is None or any(
                approval[key] != parent[key] for key in ("approval_id", "session_id", "run_id", "interrupt_id")
            ) or approval["session_id"] != run.session_id:
                raise ValueError()
            expected = self._claim_effect_expected_locked(approval, run)
            expected.update(interrupt_id=parent["interrupt_id"], approval_id=parent["approval_id"])
            validated = claim_effect_results.validate_effect_records(records, **expected)
            material = validated["material"]
            if parent["status"] == "PENDING":
                lifecycle_ok = (approval["status"] in {"PENDING", "EXPIRED"}
                                and parent["decision"] is None and approval["decision"] is None)
            elif parent["status"] == "CONSUMED":
                lifecycle_ok = (approval["status"] == "CONSUMED"
                                and parent["decision"] == approval["decision"]
                                and parent["decision"] in {"APPROVE", "REJECT"})
            elif parent["status"] == "CANCELLED":
                lifecycle_ok = (approval["status"] in {"CANCELLED", "EXPIRED"}
                                and parent["decision"] is None and approval["decision"] is None)
            else:
                lifecycle_ok = False
            if not lifecycle_ok:
                raise ValueError()
            payload, payload_json = self._canonical_stored_json(parent["payload_json"])
            issued_payload = {**material["public_payload"], "status": "PENDING",
                              "decision": None, "can_decide": True}
            if (_canonical(payload) != _canonical(issued_payload)
                    or hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != parent["payload_sha256"]):
                raise ValueError()
            if require_checkpoint:
                with safe_span("robotops.checkpoint.restore") as span:
                    set_safe_run_correlation(span, run.run_id)
                    checkpoint = self._typed_checkpoint_locked(run)
                    if run.status != "INTERRUPTED" or run.current_interrupt_id != parent["interrupt_id"]:
                        raise ValueError()
                    wanted_intent = {"approval_kind": "EFFECT_V2", "intent_profile": "robotops_claim_effect_intent_v2",
                                     "proposal": material["proposal"],
                                     "verification_sha256": material["proof"]["verification"]["verification_sha256"]}
                    if (checkpoint.get("claim_effect_decision", "") != ""
                            or checkpoint.get("claim_effect_runtime", {}) != {}
                            or _canonical(checkpoint.get("claim_effect_intent")) != _canonical(wanted_intent)
                            or _canonical(checkpoint.get("claim_verification")) != _canonical(material["proof"]["verification"])
                            or _canonical({key: checkpoint[key] for key in CONTEXT_FIELDS})
                            != _canonical({key: expected["expected_authority"][key] for key in CONTEXT_FIELDS})):
                        raise ValueError()
                    self._validate_agent_checkpoint_state(checkpoint, allow_active=True,
                                                          expected_profile=tool_loop.SYNTHETIC_EVIDENCE_TOOL_PROFILE)
                    set_safe_outcome(span, "COMPLETED")
            return validated
        except SessionStoreError:
            raise
        except Exception:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from None

    @staticmethod
    def _valid_digest(value: object) -> bool:
        return isinstance(value, str) and _DIGEST_PATTERN.fullmatch(value) is not None

    @staticmethod
    def _validate_synthetic_result(
        value: object,
        *,
        effect_key: str,
        binding_sha256: str,
        device_model: str,
    ) -> dict[str, Any]:
        expected_fields = {
            "schema_version",
            "effect_type",
            "source",
            "device_model",
            "effect_key",
            "binding_sha256",
            "boot_generation_before",
            "boot_generation_after",
            "applied",
        }
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        result = dict(value)
        before = result.get("boot_generation_before")
        after = result.get("boot_generation_after")
        if (
            type(result.get("schema_version")) is not int
            or result["schema_version"] != 1
            or result.get("effect_type") != SYNTHETIC_EFFECT_TYPE
            or result.get("source") != SYNTHETIC_EFFECT_SOURCE
            or result.get("device_model") != device_model
            or result.get("effect_key") != effect_key
            or result.get("binding_sha256") != binding_sha256
            or type(before) is not int
            or before < 0
            or type(after) is not int
            or after != before + 1
            or result.get("applied") is not True
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return result

    @staticmethod
    def _validate_synthetic_readback(
        value: object, *, device_model: str
    ) -> dict[str, Any]:
        expected_fields = {
            "schema_version",
            "source",
            "device_model",
            "boot_generation",
            "last_effect_key",
            "last_binding_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        readback = dict(value)
        generation = readback.get("boot_generation")
        effect_key = readback.get("last_effect_key")
        binding_sha256 = readback.get("last_binding_sha256")
        if (
            type(readback.get("schema_version")) is not int
            or readback["schema_version"] != 1
            or readback.get("source") != SYNTHETIC_EFFECT_SOURCE
            or readback.get("device_model") != device_model
            or type(generation) is not int
            or generation < 0
            or (effect_key is None) != (binding_sha256 is None)
            or (effect_key is not None and not DiagnosticSessionService._valid_digest(effect_key))
            or (
                binding_sha256 is not None
                and not DiagnosticSessionService._valid_digest(binding_sha256)
            )
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return readback

    @staticmethod
    def _validate_synthetic_verification(
        value: object,
        *,
        effect_key: str,
        binding_sha256: str,
        device_model: str,
        expected_boot_generation: int,
    ) -> dict[str, Any]:
        expected_fields = {
            "schema_version",
            "status",
            "source",
            "device_model",
            "effect_key",
            "binding_sha256",
            "expected_boot_generation",
            "observed_boot_generation",
            "observed_effect_key",
            "observed_binding_sha256",
            "matched",
            "reason_code",
        }
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        verification = dict(value)
        status = verification.get("status")
        observed_generation = verification.get("observed_boot_generation")
        observed_effect_key = verification.get("observed_effect_key")
        observed_binding = verification.get("observed_binding_sha256")
        if (
            type(verification.get("schema_version")) is not int
            or verification["schema_version"] != 1
            or status not in {"VERIFIED", "VERIFICATION_FAILED", "UNKNOWN"}
            or verification.get("source") != SYNTHETIC_EFFECT_SOURCE
            or verification.get("device_model") != device_model
            or verification.get("effect_key") != effect_key
            or verification.get("binding_sha256") != binding_sha256
            or verification.get("expected_boot_generation")
            != expected_boot_generation
            or type(verification.get("matched")) is not bool
            or not isinstance(verification.get("reason_code"), str)
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if status == "UNKNOWN":
            if (
                observed_generation is not None
                or observed_effect_key is not None
                or observed_binding is not None
                or verification["matched"] is not False
                or verification["reason_code"] != "READBACK_UNAVAILABLE"
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            return verification
        if (
            type(observed_generation) is not int
            or observed_generation < 0
            or observed_effect_key is not None
            and not DiagnosticSessionService._valid_digest(observed_effect_key)
            or observed_binding is not None
            and not DiagnosticSessionService._valid_digest(observed_binding)
            or (observed_effect_key is None) != (observed_binding is None)
            or verification["reason_code"] != ""
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        matched = (
            observed_generation == expected_boot_generation
            and observed_effect_key == effect_key
            and observed_binding == binding_sha256
        )
        if (
            verification["matched"] is not matched
            or (status == "VERIFIED") is not matched
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return verification

    def _validated_effect_ledger_locked(
        self, ledger: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        if not isinstance(ledger, Mapping) or set(ledger) != set(_EFFECT_LEDGER_COLUMNS):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        normalized = dict(ledger)
        status = normalized.get("status")
        attempt_count = normalized.get("attempt_count")
        if (
            not self._valid_digest(normalized.get("effect_key"))
            or not self._valid_digest(normalized.get("binding_sha256"))
            or normalized.get("effect_type") != SYNTHETIC_EFFECT_TYPE
            or normalized.get("source") != SYNTHETIC_EFFECT_SOURCE
            or status
            not in {
                "PREPARED",
                "EXECUTING",
                "APPLIED",
                "VERIFIED",
                "EXECUTION_FAILED",
                "VERIFICATION_FAILED",
                "UNKNOWN",
            }
            or type(attempt_count) is not int
            or type(normalized.get("created_at")) is not int
            or type(normalized.get("updated_at")) is not int
            or normalized["created_at"] < 0
            or normalized["updated_at"] < normalized["created_at"]
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        arguments, arguments_json = self._canonical_stored_json(
            normalized.get("arguments_json")
        )
        if (
            not isinstance(arguments, dict)
            or set(arguments)
            != {"device_model", "expected_boot_generation_delta"}
            or not isinstance(arguments.get("device_model"), str)
            or not arguments["device_model"]
            or type(arguments.get("expected_boot_generation_delta")) is not int
            or arguments["expected_boot_generation_delta"] != 1
            or not self._valid_digest(normalized.get("arguments_sha256"))
            or not secrets.compare_digest(
                str(normalized["arguments_sha256"]),
                effect_capabilities.canonical_sha256(arguments),
            )
            or arguments_json != normalized["arguments_json"]
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        def optional_json(
            json_value: object, digest_value: object
        ) -> dict[str, Any] | None:
            if json_value is None and digest_value is None:
                return None
            if json_value is None or not self._valid_digest(digest_value):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            parsed, canonical = self._canonical_stored_json(json_value)
            if (
                not isinstance(parsed, dict)
                or not secrets.compare_digest(
                    str(digest_value),
                    hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                )
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            return parsed

        result = optional_json(
            normalized.get("result_json"), normalized.get("result_sha256")
        )
        verification = optional_json(
            normalized.get("verification_json"),
            normalized.get("verification_sha256"),
        )
        if status == "PREPARED":
            valid_shape = attempt_count == 0 and result is None and verification is None
        elif status == "EXECUTING":
            valid_shape = attempt_count == 1 and result is None and verification is None
        elif status == "APPLIED":
            valid_shape = attempt_count == 1 and result is not None and verification is None
        elif status in {"VERIFIED", "VERIFICATION_FAILED"}:
            valid_shape = attempt_count == 1 and result is not None and verification is not None
        elif status == "EXECUTION_FAILED":
            valid_shape = attempt_count == 1 and result is None and verification is None
        else:
            valid_shape = attempt_count == 1 and not (
                result is None and verification is not None
            )
        if not valid_shape:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if result is not None:
            result = self._validate_synthetic_result(
                result,
                effect_key=str(normalized["effect_key"]),
                binding_sha256=str(normalized["binding_sha256"]),
                device_model=str(arguments["device_model"]),
            )
        if verification is not None:
            if result is None:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            verification = self._validate_synthetic_verification(
                verification,
                effect_key=str(normalized["effect_key"]),
                binding_sha256=str(normalized["binding_sha256"]),
                device_model=str(arguments["device_model"]),
                expected_boot_generation=int(result["boot_generation_after"]),
            )
            if status != verification["status"]:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
        runtime = {
            "schema_version": 1,
            "effect_key": normalized["effect_key"],
            "binding_sha256": normalized["binding_sha256"],
            "status": status,
            "attempt_count": attempt_count,
            "result_sha256": normalized.get("result_sha256"),
            "verification_sha256": normalized.get("verification_sha256"),
        }
        if set(runtime) != _EFFECT_RUNTIME_FIELDS:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return runtime, arguments, result, verification

    def _effect_parent_locked(self, approval_id: str) -> dict[str, Any]:
        row = self._registry.execute(
            """
            SELECT approval_id, session_id, run_id, interrupt_id,
                   payload_json, payload_sha256, status, decision, consumed_at
            FROM diagnostic_interrupts WHERE approval_id = ?
            """,
            (approval_id,),
        ).fetchone()
        if row is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        keys = (
            "approval_id",
            "session_id",
            "run_id",
            "interrupt_id",
            "payload_json",
            "payload_sha256",
            "status",
            "decision",
            "consumed_at",
        )
        return dict(zip(keys, row))

    def _authorized_effect_material_locked(
        self,
        *,
        session_id: str,
        run_id: str,
        approval_id: str,
        proposal: Mapping[str, Any],
        proposal_sha256: str,
    ) -> tuple[_RunRecord, dict[str, Any], dict[str, Any], dict[str, Any]]:
        if (
            self._effect_mode != EFFECT_MODE_SYNTHETIC_RESTART
            or self._synthetic_effect_adapter is None
            or not SESSION_ID_PATTERN.fullmatch(session_id)
            or not SESSION_ID_PATTERN.fullmatch(run_id)
            or not SESSION_ID_PATTERN.fullmatch(approval_id)
            or not isinstance(proposal, Mapping)
            or not self._valid_digest(proposal_sha256)
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        run = self._select_run_locked(run_id)
        if run is None or run.session_id != session_id:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        parent = self._effect_parent_locked(approval_id)
        if (
            parent["session_id"] != session_id
            or parent["run_id"] != run_id
            or parent["status"] != "CONSUMED"
            or parent["decision"] != ResumeDecision.APPROVE.value
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        approval, _public = self._validate_typed_approval_locked(
            parent=parent,
            run=run,
            require_checkpoint=False,
        )
        if (
            approval["status"] != "CONSUMED"
            or approval["decision"] != ResumeDecision.APPROVE.value
            or not secrets.compare_digest(
                str(approval["proposal_sha256"]), proposal_sha256
            )
            or effect_capabilities.canonical_json(dict(proposal))
            != approval["proposal_json"]
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        ledger = self._select_effect_ledger_locked(str(approval["effect_key"]))
        if ledger is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        _runtime, arguments, _result, _verification = (
            self._validated_effect_ledger_locked(ledger)
        )
        return run, approval, ledger, arguments

    def _force_effect_unknown(self, effect_key: str) -> dict[str, Any]:
        """Best-effort fail-close after an effect attempt has begun."""
        try:
            with self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                ledger = self._select_effect_ledger_locked(effect_key)
                if ledger is None:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                runtime, _arguments, _result, _verification = (
                    self._validated_effect_ledger_locked(ledger)
                )
                if runtime["status"] in {"EXECUTING", "APPLIED"}:
                    changed = self._registry.execute(
                        """
                        UPDATE diagnostic_effect_ledger
                        SET status = 'UNKNOWN', updated_at = ?
                        WHERE effect_key = ? AND status = ? AND attempt_count = 1
                        """,
                        (self._typed_now_ms(), effect_key, runtime["status"]),
                    ).rowcount
                    if changed != 1:
                        self._registry.rollback()
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    ledger = self._select_effect_ledger_locked(effect_key)
                    if ledger is None:
                        self._registry.rollback()
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    runtime, _arguments, _result, _verification = (
                        self._validated_effect_ledger_locked(ledger)
                    )
                self._registry.commit()
                return runtime
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def _persist_effect_execution_outcome(
        self,
        *,
        effect_key: str,
        status: str,
        result: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if status not in {"APPLIED", "EXECUTION_FAILED", "UNKNOWN"}:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        result_json = (
            effect_capabilities.canonical_json(dict(result))
            if result is not None
            else None
        )
        result_sha256 = (
            hashlib.sha256(result_json.encode("utf-8")).hexdigest()
            if result_json is not None
            else None
        )
        try:
            with self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                changed = self._registry.execute(
                    """
                    UPDATE diagnostic_effect_ledger
                    SET status = ?, result_json = ?, result_sha256 = ?,
                        verification_json = NULL, verification_sha256 = NULL,
                        updated_at = ?
                    WHERE effect_key = ? AND status = 'EXECUTING'
                      AND attempt_count = 1
                    """,
                    (
                        status,
                        result_json,
                        result_sha256,
                        self._typed_now_ms(),
                        effect_key,
                    ),
                ).rowcount
                if changed != 1:
                    self._registry.rollback()
                    return self._force_effect_unknown(effect_key)
                ledger = self._select_effect_ledger_locked(effect_key)
                if ledger is None:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                runtime, _arguments, _stored_result, _verification = (
                    self._validated_effect_ledger_locked(ledger)
                )
                self._registry.commit()
                return runtime
        except Exception:
            return self._force_effect_unknown(effect_key)

    def _execute_authorized_effect(
        self,
        *,
        session_id: str,
        run_id: str,
        approval_id: str,
        proposal: Mapping[str, Any],
        proposal_sha256: str,
    ) -> dict[str, Any]:
        adapter = self._synthetic_effect_adapter
        if adapter is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        try:
            with self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                run, approval, ledger, arguments = (
                    self._authorized_effect_material_locked(
                        session_id=session_id,
                        run_id=run_id,
                        approval_id=approval_id,
                        proposal=proposal,
                        proposal_sha256=proposal_sha256,
                    )
                )
                runtime, _arguments, _result, _verification = (
                    self._validated_effect_ledger_locked(ledger)
                )
                if runtime["status"] == "EXECUTING":
                    self._registry.rollback()
                    return self._force_effect_unknown(str(approval["effect_key"]))
                if runtime["status"] != "PREPARED":
                    self._registry.commit()
                    return runtime
                session_row = self._registry.execute(
                    "SELECT status, active_run_id FROM diagnostic_sessions "
                    "WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                self._claim_profile_locked(session_id)
                if (
                    run.status != DiagnosticRunStatus.RUNNING.value
                    or session_row
                    != (DiagnosticSessionStatus.RUNNING.value, run_id)
                ):
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                changed = self._registry.execute(
                    """
                    UPDATE diagnostic_effect_ledger
                    SET status = 'EXECUTING', attempt_count = 1, updated_at = ?
                    WHERE effect_key = ? AND binding_sha256 = ?
                      AND status = 'PREPARED' AND attempt_count = 0
                    """,
                    (
                        self._typed_now_ms(),
                        approval["effect_key"],
                        approval["binding_sha256"],
                    ),
                ).rowcount
                if changed != 1:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                self._registry.commit()
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

        try:
            with safe_span("robotops.effect.execute") as span:
                raw_result = adapter.restart_simulated_robot(
                    effect_key=str(approval["effect_key"]),
                    binding_sha256=str(approval["binding_sha256"]),
                    device_model=str(arguments["device_model"]),
                    expected_boot_generation_delta=int(
                        arguments["expected_boot_generation_delta"]
                    ),
                )
                result = self._validate_synthetic_result(
                    raw_result,
                    effect_key=str(approval["effect_key"]),
                    binding_sha256=str(approval["binding_sha256"]),
                    device_model=str(arguments["device_model"]),
                )
                set_safe_outcome(span, "COMPLETED")
        except SyntheticEffectNotAppliedError:
            return self._persist_effect_execution_outcome(
                effect_key=str(approval["effect_key"]),
                status="EXECUTION_FAILED",
                result=None,
            )
        except Exception:
            return self._persist_effect_execution_outcome(
                effect_key=str(approval["effect_key"]),
                status="UNKNOWN",
                result=None,
            )
        return self._persist_effect_execution_outcome(
            effect_key=str(approval["effect_key"]),
            status="APPLIED",
            result=result,
        )

    @staticmethod
    def _verification_record(
        *,
        status: str,
        effect_key: str,
        binding_sha256: str,
        device_model: str,
        expected_boot_generation: int,
        readback: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": status,
            "source": SYNTHETIC_EFFECT_SOURCE,
            "device_model": device_model,
            "effect_key": effect_key,
            "binding_sha256": binding_sha256,
            "expected_boot_generation": expected_boot_generation,
            "observed_boot_generation": (
                readback.get("boot_generation") if readback is not None else None
            ),
            "observed_effect_key": (
                readback.get("last_effect_key") if readback is not None else None
            ),
            "observed_binding_sha256": (
                readback.get("last_binding_sha256")
                if readback is not None
                else None
            ),
            "matched": status == "VERIFIED",
            "reason_code": "READBACK_UNAVAILABLE" if status == "UNKNOWN" else "",
        }

    def _persist_effect_verification(
        self,
        *,
        effect_key: str,
        status: str,
        verification: Mapping[str, Any],
    ) -> dict[str, Any]:
        if status not in {"VERIFIED", "VERIFICATION_FAILED", "UNKNOWN"}:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        verification_json = effect_capabilities.canonical_json(dict(verification))
        verification_sha256 = hashlib.sha256(
            verification_json.encode("utf-8")
        ).hexdigest()
        try:
            with self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                changed = self._registry.execute(
                    """
                    UPDATE diagnostic_effect_ledger
                    SET status = ?, verification_json = ?,
                        verification_sha256 = ?, updated_at = ?
                    WHERE effect_key = ? AND status = 'APPLIED'
                      AND attempt_count = 1
                    """,
                    (
                        status,
                        verification_json,
                        verification_sha256,
                        self._typed_now_ms(),
                        effect_key,
                    ),
                ).rowcount
                if changed != 1:
                    self._registry.rollback()
                    return self._force_effect_unknown(effect_key)
                ledger = self._select_effect_ledger_locked(effect_key)
                if ledger is None:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                runtime, _arguments, _result, _stored_verification = (
                    self._validated_effect_ledger_locked(ledger)
                )
                self._registry.commit()
                return runtime
        except Exception:
            return self._force_effect_unknown(effect_key)

    def _verify_authorized_effect(
        self,
        *,
        session_id: str,
        run_id: str,
        approval_id: str,
        proposal: Mapping[str, Any],
        proposal_sha256: str,
    ) -> dict[str, Any]:
        adapter = self._synthetic_effect_adapter
        if adapter is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        with self._registry_lock:
            _run, approval, ledger, arguments = self._authorized_effect_material_locked(
                session_id=session_id,
                run_id=run_id,
                approval_id=approval_id,
                proposal=proposal,
                proposal_sha256=proposal_sha256,
            )
            runtime, _arguments, result, _verification = (
                self._validated_effect_ledger_locked(ledger)
            )
        if runtime["status"] == "EXECUTING":
            return self._force_effect_unknown(str(approval["effect_key"]))
        if runtime["status"] != "APPLIED":
            return runtime
        if result is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        try:
            with safe_span("robotops.effect.readback") as span:
                readback = self._validate_synthetic_readback(
                    adapter.read_device_state(
                        device_model=str(arguments["device_model"])
                    ),
                    device_model=str(arguments["device_model"]),
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
        verification = self._verification_record(
            status=status,
            effect_key=str(approval["effect_key"]),
            binding_sha256=str(approval["binding_sha256"]),
            device_model=str(arguments["device_model"]),
            expected_boot_generation=int(result["boot_generation_after"]),
            readback=readback,
        )
        self._validate_synthetic_verification(
            verification,
            effect_key=str(approval["effect_key"]),
            binding_sha256=str(approval["binding_sha256"]),
            device_model=str(arguments["device_model"]),
            expected_boot_generation=int(result["boot_generation_after"]),
        )
        return self._persist_effect_verification(
            effect_key=str(approval["effect_key"]),
            status=status,
            verification=verification,
        )

    def _typed_checkpoint_locked(self, run: _RunRecord) -> dict[str, Any]:
        self._claim_profile_locked(run.session_id)
        row = self._registry.execute(
            "SELECT thread_id FROM diagnostic_sessions WHERE session_id = ?",
            (run.session_id,),
        ).fetchone()
        if row is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        try:
            snapshot = self._graph.get_state(self._config(str(row[0])))
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        if not isinstance(snapshot.values, Mapping) or not snapshot.values:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return dict(snapshot.values)

    def _validate_typed_approval_locked(
        self,
        *,
        parent: Mapping[str, Any],
        run: _RunRecord,
        require_checkpoint: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._execution_contract_locked(run.session_id) != LEGACY_EXECUTION_CONTRACT:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        approval = self._select_typed_approval_locked(str(parent["approval_id"]))
        if approval is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if (
            approval["approval_id"] != parent["approval_id"]
            or approval["session_id"] != run.session_id
            or approval["run_id"] != run.run_id
            or approval["interrupt_id"] != parent["interrupt_id"]
            or approval["approval_kind"]
            != effect_capabilities.APPROVAL_KIND_EFFECT_V1
            or approval["effect_type"]
            != tool_loop.EFFECT_TYPE_RESTART_SIMULATED_ROBOT
            or approval["source"] != tool_loop.EFFECT_SOURCE_SYNTHETIC_LOCAL
            or type(approval["issued_at"]) is not int
            or type(approval["expires_at"]) is not int
            or approval["expires_at"] - approval["issued_at"]
            != EFFECT_CAPABILITY_TTL_MS
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        parent_status = parent.get("status")
        parent_decision = parent.get("decision")
        if parent_status == "PENDING":
            lifecycle_valid = (
                parent_decision is None
                and approval["status"] in {"PENDING", "EXPIRED"}
                and approval["decision"] is None
            )
        elif parent_status == "CONSUMED":
            lifecycle_valid = (
                approval["status"] == "CONSUMED"
                and parent_decision in {"APPROVE", "REJECT"}
                and approval["decision"] == parent_decision
            )
        elif parent_status == "CANCELLED":
            lifecycle_valid = (
                parent_decision is None
                and approval["status"] in {"CANCELLED", "EXPIRED"}
                and approval["decision"] is None
            )
        else:
            lifecycle_valid = False
        if not lifecycle_valid:
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        proposal, proposal_json = self._canonical_stored_json(
            approval["proposal_json"]
        )
        arguments, arguments_json = self._canonical_stored_json(
            approval["arguments_json"]
        )
        risk, risk_json = self._canonical_stored_json(approval["risk_summary_json"])
        verifier, verifier_json = self._canonical_stored_json(
            approval["verifier_summary_json"]
        )
        binding, binding_json = self._canonical_stored_json(approval["binding_json"])
        try:
            effect_capabilities.build_effect_interrupt_request(proposal)
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        if (
            proposal.get("effect_type") != approval["effect_type"]
            or proposal.get("source") != approval["source"]
            or effect_capabilities.canonical_json(proposal.get("arguments"))
            != arguments_json
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        digests = {
            "proposal_sha256": effect_capabilities.canonical_sha256(proposal),
            "arguments_sha256": effect_capabilities.canonical_sha256(arguments),
            "risk_summary_sha256": effect_capabilities.canonical_sha256(risk),
            "verifier_summary_sha256": effect_capabilities.canonical_sha256(verifier),
            "binding_sha256": effect_capabilities.canonical_sha256(binding),
        }
        if any(
            not secrets.compare_digest(str(approval[name]), value)
            for name, value in digests.items()
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        expected_effect_key = effect_capabilities.stable_effect_key(
            session_id=run.session_id,
            run_id=run.run_id,
            interrupt_id=str(parent["interrupt_id"]),
            proposal_sha256=str(approval["proposal_sha256"]),
            effect_type=str(approval["effect_type"]),
            arguments_sha256=str(approval["arguments_sha256"]),
        )
        if not secrets.compare_digest(str(approval["effect_key"]), expected_effect_key):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        expected_binding = effect_capabilities.build_binding(
            session_id=run.session_id,
            run_id=run.run_id,
            interrupt_id=str(parent["interrupt_id"]),
            proposal_sha256=str(approval["proposal_sha256"]),
            effect_key=str(approval["effect_key"]),
            effect_type=str(approval["effect_type"]),
            source=str(approval["source"]),
            arguments_sha256=str(approval["arguments_sha256"]),
            risk_summary_sha256=str(approval["risk_summary_sha256"]),
            verifier_summary_sha256=str(approval["verifier_summary_sha256"]),
            verifier_result_sha256=str(approval["verifier_result_sha256"]),
            nonce_digest_value=str(approval["nonce_digest"]),
            issued_at=int(approval["issued_at"]),
            expires_at=int(approval["expires_at"]),
        )
        if binding != expected_binding or binding_json != effect_capabilities.canonical_json(
            expected_binding
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        ledger = self._select_effect_ledger_locked(str(approval["effect_key"]))
        if ledger is None or any(
            ledger[field] != approval[field]
            for field in (
                "effect_key", "approval_id", "binding_sha256", "effect_type",
                "source", "arguments_json", "arguments_sha256",
            )
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        self._canonical_stored_json(ledger["arguments_json"])
        if (
            approval["status"] != "CONSUMED"
            or approval["decision"] == "REJECT"
        ) and (
            ledger["status"] != "PREPARED" or ledger["attempt_count"] != 0
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        expected_public_payload = {
            "approval_kind": effect_capabilities.APPROVAL_KIND_EFFECT_V1,
            "effect_key": approval["effect_key"],
            "binding_sha256": approval["binding_sha256"],
            "proposal": proposal,
            "proposal_sha256": approval["proposal_sha256"],
            "risk": risk,
            "verifier": verifier,
            "issued_at": approval["issued_at"],
            "expires_at": approval["expires_at"],
            "ttl_ms": EFFECT_CAPABILITY_TTL_MS,
            "status": "PENDING",
            "decision": None,
            "can_decide": True,
        }
        parent_payload, parent_payload_json = self._canonical_stored_json(
            parent["payload_json"]
        )
        if (
            parent_payload_json
            != effect_capabilities.canonical_json(expected_public_payload)
            or not secrets.compare_digest(
                hashlib.sha256(parent_payload_json.encode("utf-8")).hexdigest(),
                str(parent["payload_sha256"]),
            )
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        if require_checkpoint:
            with safe_span("robotops.checkpoint.restore") as span:
                set_safe_run_correlation(span, run.run_id)
                checkpoint = self._typed_checkpoint_locked(run)
                progress = self._validate_agent_checkpoint_state(
                    checkpoint,
                    allow_active=True,
                    expected_profile=tool_loop.SYNTHETIC_EVIDENCE_TOOL_PROFILE,
                )
                if (
                    progress.get("effect_policy_status") != "PROPOSED"
                    or effect_capabilities.canonical_json(
                        progress.get("effect_proposal")
                    )
                    != proposal_json
                    or progress.get("effect_proposal_sha256")
                    != approval["proposal_sha256"]
                    or effect_capabilities.canonical_json(
                        effect_capabilities.risk_summary_for_state(checkpoint)
                    )
                    != risk_json
                    or effect_capabilities.canonical_json(
                        effect_capabilities.verifier_summary_for_state(checkpoint)
                    )
                    != verifier_json
                ):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)

                set_safe_outcome(span, "COMPLETED")
        now_ms = self._typed_now_ms()
        displayed_status = str(approval["status"])
        if displayed_status == "PENDING" and now_ms >= int(approval["expires_at"]):
            displayed_status = "EXPIRED"
        public_payload = dict(expected_public_payload)
        public_payload.update(
            status=displayed_status,
            decision=approval["decision"],
            can_decide=(
                approval["status"] == "PENDING"
                and int(approval["issued_at"]) <= now_ms < int(approval["expires_at"])
            ),
            server_time=now_ms,
        )
        return approval, public_payload

    def _effect_response_locked(
        self, run: _RunRecord
    ) -> DiagnosticEffectResponse | None:
        for row in self._registry.execute(
            "SELECT approval_id FROM diagnostic_interrupts WHERE run_id=?", (run.run_id,),
        ).fetchall():
            self._effect_family_locked(self._effect_parent_locked(str(row[0])), run)
        if self._execution_contract_locked(run.session_id) == CLAIM_EXECUTION_CONTRACT:
            return self._claim_effect_response_locked(run)
        approval_ids = self._registry.execute(
            "SELECT approval_id FROM diagnostic_effect_approvals "
            "WHERE run_id = ? ORDER BY approval_id",
            (run.run_id,),
        ).fetchall()
        if not approval_ids:
            return None
        if len(approval_ids) != 1:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        parent = self._effect_parent_locked(str(approval_ids[0][0]))
        if parent["session_id"] != run.session_id or parent["run_id"] != run.run_id:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        approval, public = self._validate_typed_approval_locked(
            parent=parent,
            run=run,
            require_checkpoint=(
                run.status == DiagnosticRunStatus.INTERRUPTED.value
                and parent["status"] == "PENDING"
            ),
        )
        ledger = self._select_effect_ledger_locked(str(approval["effect_key"]))
        if ledger is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        runtime, arguments, result, verification = (
            self._validated_effect_ledger_locked(ledger)
        )
        now_ms = self._typed_now_ms()
        approval_status = str(approval["status"])
        if approval_status == "PENDING" and now_ms >= int(approval["expires_at"]):
            approval_status = "EXPIRED"
        return DiagnosticEffectResponse(
            effect_key=str(approval["effect_key"]),
            proposal_sha256=str(approval["proposal_sha256"]),
            binding_sha256=str(approval["binding_sha256"]),
            effect_type=str(approval["effect_type"]),
            source=str(approval["source"]),
            arguments=arguments,
            risk=dict(public["risk"]),
            verifier=dict(public["verifier"]),
            approval_status=DiagnosticEffectApprovalStatus(approval_status),
            decision=(
                ResumeDecision(str(approval["decision"]))
                if approval["decision"] is not None
                else None
            ),
            issued_at=int(approval["issued_at"]),
            expires_at=int(approval["expires_at"]),
            can_decide=(
                approval["status"] == "PENDING"
                and int(approval["issued_at"]) <= now_ms < int(approval["expires_at"])
                and run.status == DiagnosticRunStatus.INTERRUPTED.value
            ),
            status=DiagnosticEffectStatus(str(runtime["status"])),
            attempt_count=int(runtime["attempt_count"]),
            result=result,
            result_sha256=runtime["result_sha256"],
            verification=verification,
            verification_sha256=runtime["verification_sha256"],
            updated_at=int(ledger["updated_at"]),
        )

    def _decoded_claim_run_locked(self, run: _RunRecord) -> dict:
        from app.services import claim_sessions
        if (self._claim_profile_locked(run.session_id) != ACTIVE_CLAIM_EVIDENCE_PROFILE
                or run.status != DiagnosticRunStatus.COMPLETED.value):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        try:
            message = self._claim_request_message_locked(run)
            decoder = claim_sessions.decode_claim_run_result
            claim_effect = self._execution_contract_locked(run.session_id) == CLAIM_EXECUTION_CONTRACT
            if claim_effect:
                from app.services.claim_effect_results import decode_claim_effect_run_result
                decoder = decode_claim_effect_run_result
            decoded = decoder(run.result_json, expected_identity={
                "session_id": run.session_id, "run_id": run.run_id,
                "request_id": self._request_id_for_run_locked(run.run_id),
                "turn_number": run.turn_number,
            })
            query = decoded["authority"]["user_query"]
            if query != message and not query.endswith("\n" + message):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            if claim_effect:
                effect = self._claim_effect_response_locked(run, terminal=True)
                if decoded["action"] != (effect.model_dump(mode="json") if effect is not None else None):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
            return decoded
        except Exception:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from None

    def _run_response_locked(
        self, session_status: str, run: _RunRecord
    ) -> DiagnosticRunResponse:
        claim_profile = self._claim_profile_locked(run.session_id)
        result: DiagnosticTurnResponse | None = None
        interrupt_response: DiagnosticInterruptResponse | None = None
        error: DiagnosticRunErrorResponse | None = None
        failure_disposition: FailureDisposition | None = None
        if run.status == DiagnosticRunStatus.COMPLETED.value:
            if not run.result_json:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            try:
                if claim_profile == ACTIVE_CLAIM_EVIDENCE_PROFILE:
                    result = DiagnosticTurnResponse.model_validate(
                        self._decoded_claim_run_locked(run)["response"])
                else:
                    result = DiagnosticTurnResponse.model_validate_json(run.result_json)
            except Exception as exc:
                raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        elif run.status == DiagnosticRunStatus.INTERRUPTED.value:
            if not run.current_interrupt_id:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            row = self._registry.execute(
                """
                SELECT interrupt_id, approval_id, payload_json, payload_sha256,
                       created_at, status, decision, consumed_at
                FROM diagnostic_interrupts
                WHERE run_id = ? AND interrupt_id = ? AND status = 'PENDING'
                """,
                (run.run_id, run.current_interrupt_id),
            ).fetchone()
            if row is None:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            parent = {
                "interrupt_id": str(row[0]),
                "approval_id": str(row[1]),
                "payload_json": str(row[2]),
                "payload_sha256": str(row[3]),
                "status": str(row[5]),
                "decision": row[6],
                "consumed_at": row[7],
            }
            payload_json = parent["payload_json"]
            digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            if not secrets.compare_digest(digest, parent["payload_sha256"]):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            try:
                payload = json.loads(payload_json)
            except Exception as exc:
                raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
            if not isinstance(payload, dict):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            kind = self._effect_family_locked(parent, run)
            if kind == "EFFECT_V2":
                from app.services.claim_effect_results import build_effect_approval_payload
                validated = self._validate_claim_effect_binding_locked(
                    parent["approval_id"], require_checkpoint=True)
                payload = build_effect_approval_payload(
                    {"approval": validated["approval"], "ledger": validated["ledger"]},
                    **self._claim_effect_expected_locked(validated["approval"], run),
                    now_ms=self._typed_now_ms(), run_status=run.status,
                )
            elif kind == "EFFECT_V1":
                _approval, payload = self._validate_typed_approval_locked(
                    parent=parent,
                    run=run,
                    require_checkpoint=True,
                )
            interrupt_response = DiagnosticInterruptResponse(
                interrupt_id=row[0],
                approval_id=row[1],
                payload=payload,
                created_at=_parse_time(row[4]),
            )
        elif run.status == DiagnosticRunStatus.FAILED.value:
            if not run.error_code:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            message = (
                EXECUTION_LIMIT_ERROR
                if run.error_code == "EXECUTION_LIMIT_EXCEEDED"
                else RUN_EXECUTION_ERROR
            )
            error = DiagnosticRunErrorResponse(code=run.error_code, message=message)
            failure_disposition = (
                FailureDisposition.RETRYABLE
                if run.error_code in _RETRYABLE_RUN_ERRORS
                else FailureDisposition.FINAL
            )
        return DiagnosticRunResponse(
            session_id=run.session_id,
            run_id=run.run_id,
            request_id=self._request_id_for_run_locked(run.run_id),
            status=DiagnosticRunStatus(run.status),
            session_status=DiagnosticSessionStatus(session_status),
            turn_number=run.turn_number,
            created_at=_parse_time(run.created_at),
            updated_at=_parse_time(run.updated_at),
            result=result,
            interrupt=interrupt_response,
            error=error,
            effect=self._effect_response_locked(run),
            tool_iterations=run.tool_iterations,
            max_tool_iterations=run.max_tool_iterations,
            stop_reason=run.stop_reason,
            failure_disposition=failure_disposition,
        )

    def get_run(self, session_id: str, run_id: str) -> DiagnosticRunResponse:
        self._validate_session_id(session_id)
        self._validate_run_id(run_id)
        try:
            with self._registry_lock:
                session_status, run = self._bound_run_locked(session_id, run_id)
                return self._run_response_locked(session_status, run)
        except (
            InvalidSessionIdError,
            InvalidRunIdError,
            SessionNotFoundError,
            RunNotFoundError,
            RunConflictError,
        ):
            raise
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def get_run_by_request(
        self, session_id: str, request_id: str
    ) -> DiagnosticRunResponse:
        """只读对账 request_id，供响应丢失后的页面恢复使用。"""
        self._validate_session_id(session_id)
        self._validate_request_id(request_id)
        try:
            with self._registry_lock:
                session = self._registry.execute(
                    "SELECT status FROM diagnostic_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise SessionNotFoundError(SESSION_NOT_FOUND)
                self._claim_profile_locked(session_id)
                request = self._select_request_locked(session_id, request_id)
                if request is None:
                    raise RequestNotFoundError(REQUEST_NOT_FOUND)
                run = self._select_run_locked(str(request[5]))
                if run is None or run.session_id != session_id:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                return self._run_response_locked(str(session[0]), run)
        except (
            InvalidSessionIdError,
            RunConflictError,
            SessionNotFoundError,
            RequestNotFoundError,
        ):
            raise
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    @staticmethod
    def _claim_json(value: object) -> object:
        from app.agent.claim_sources import _json_copy
        from app.services.claim_sessions import MAX_RUN_RESULT_BYTES, _object_pairs, _invalid_constant
        try:
            if type(value) is not str or len(value.encode("utf-8")) > MAX_RUN_RESULT_BYTES:
                raise ValueError("invalid stored JSON")
            return _json_copy(json.loads(value, object_pairs_hook=_object_pairs,
                                        parse_constant=_invalid_constant),
                              maximum=MAX_RUN_RESULT_BYTES)
        except Exception:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from None

    def _claim_request_message_locked(self, run: _RunRecord) -> str:
        request_id = self._request_id_for_run_locked(run.run_id)
        row = self._select_request_locked(run.session_id, request_id)
        if (row is None or row[0] != run.session_id or row[1] != request_id
                or row[2] != "RUN" or row[5] != run.run_id
                or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id) is None):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        payload = self._claim_json(row[4])
        if (type(payload) is not dict or set(payload) != {"message", "operation"}
                or type(payload["message"]) is not str or not payload["message"]
                or payload["operation"] != "RUN"):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        canonical, digest = self._request_payload(payload["message"])
        if row[4] != canonical or row[3] != digest:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return payload["message"]

    def _validate_claim_message_locked(self, session_id: str, row: tuple) -> dict:
        from app.agent.claim_sources import _canonical
        run = self._select_run_locked(row[7])
        if (run is None or run.session_id != session_id or row[1] != session_id
                or row[3] != run.turn_number
                or row[8] != self._request_id_for_run_locked(run.run_id)):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        message = self._claim_request_message_locked(run)
        payload = self._claim_json(row[9])
        if row[4] == DiagnosticMessageRole.USER.value:
            if (row[2] != run.turn_number * 2 - 1
                    or row[5] != DiagnosticMessageKind.INPUT.value
                    or row[6] != message or payload != {}):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
        elif row[4] == DiagnosticMessageRole.ASSISTANT.value:
            response = self._decoded_claim_run_locked(run)["response"]
            kind = (DiagnosticMessageKind.FOLLOW_UP.value if response["status"] == "AWAITING_USER"
                    else DiagnosticMessageKind.RESULT.value)
            if (row[2] != run.turn_number * 2 or row[5] != kind
                    or row[6] != response["reply"] or _canonical(payload) != _canonical(response)):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
        else:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return payload

    def list_messages(self, session_id: str) -> DiagnosticMessagesResponse:
        """返回按 sequence_no 排序的服务端权威消息；空会话返回空列表。"""
        self._validate_session_id(session_id)
        try:
            with self._registry_lock:
                session = self._registry.execute(
                    "SELECT 1 FROM diagnostic_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise SessionNotFoundError(SESSION_NOT_FOUND)
                profile = self._claim_profile_locked(session_id)
                rows = self._registry.execute(
                    """
                    SELECT message_id, session_id, sequence_no, turn_number, role,
                           kind, content, run_id, request_id, payload_json, created_at
                    FROM diagnostic_messages
                    WHERE session_id = ? ORDER BY sequence_no ASC
                    """,
                    (session_id,),
                ).fetchall()
                active_payloads = (
                    [self._validate_claim_message_locked(session_id, row) for row in rows]
                    if profile == ACTIVE_CLAIM_EVIDENCE_PROFILE else None)
            items: list[DiagnosticMessageResponse] = []
            for index, row in enumerate(rows):
                payload = (active_payloads[index] if active_payloads is not None
                           else json.loads(str(row[9])))
                if not isinstance(payload, dict):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                items.append(
                    DiagnosticMessageResponse(
                        message_id=row[0],
                        session_id=row[1],
                        sequence_no=row[2],
                        turn_number=row[3],
                        role=DiagnosticMessageRole(row[4]),
                        kind=DiagnosticMessageKind(row[5]),
                        content=row[6],
                        run_id=row[7],
                        request_id=row[8],
                        payload=payload,
                        created_at=_parse_time(row[10]),
                    )
                )
            return DiagnosticMessagesResponse(session_id=session_id, items=items)
        except (InvalidSessionIdError, SessionNotFoundError):
            raise
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    @staticmethod
    def _draft_response_from_row(
        row: sqlite3.Row | tuple,
    ) -> DiagnosticDraftTicketResponse:
        try:
            diagnosis = json.loads(str(row[8]))
            solution_steps = json.loads(str(row[9]))
            if not isinstance(diagnosis, list) or not all(
                isinstance(item, str) for item in diagnosis
            ):
                raise ValueError("invalid diagnosis")
            if not isinstance(solution_steps, list) or not all(
                isinstance(item, str) for item in solution_steps
            ):
                raise ValueError("invalid solution steps")
            return DiagnosticDraftTicketResponse(
                draft_id=row[0],
                session_id=row[1],
                source_run_id=row[2],
                version=row[3],
                status=DiagnosticDraftStatus(row[4]),
                device_model=row[5],
                fault_category=row[6],
                fault_description=row[7],
                diagnosis=diagnosis,
                solution_steps=solution_steps,
                risk_level=row[10],
                need_human=bool(row[11]),
                external_ticket_id=row[12],
                projection_note=row[13],
                created_at=_parse_time(row[14]),
                updated_at=_parse_time(row[15]),
            )
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def _select_draft_locked(self, session_id: str) -> sqlite3.Row | tuple | None:
        return self._registry.execute(
            """
            SELECT draft_id, session_id, source_run_id, version, status,
                   device_model, fault_category, fault_description,
                   diagnosis_json, solution_steps_json, risk_level, need_human,
                   external_ticket_id, projection_note, created_at, updated_at
            FROM diagnostic_draft_tickets WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

    def _validate_claim_draft_locked(self, session_id: str, row: tuple) -> None:
        from app.agent.claim_sources import _canonical
        run = self._select_run_locked(row[2])
        if run is None or run.session_id != session_id or row[1] != session_id:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        decoded = self._decoded_claim_run_locked(run)
        response, authority = decoded["response"], decoded["authority"]
        diagnosis = response["diagnosis"]
        expected_status = ("COLLECTING" if response["status"] == "AWAITING_USER" else "READY")
        if (row[4] != expected_status or row[5] != authority["device_model"]
                or row[6] != authority["fault_category"] or row[7] != authority["fault_description"]
                or _canonical(self._claim_json(row[8])) != _canonical(diagnosis["causes"])
                or _canonical(self._claim_json(row[9])) != _canonical(diagnosis["steps"])
                or row[10] != diagnosis["risk_level"]
                or type(row[11]) is not int or row[11] != int(diagnosis["need_human"])
                or row[12] is not None or row[13] is not None):
            raise SessionStoreError(SANITIZED_STORE_ERROR)

    def get_draft_ticket(self, session_id: str) -> DiagnosticDraftTicketResponse:
        """返回会话唯一草稿；空白会话用 404 明确区分读取失败。"""
        self._validate_session_id(session_id)
        try:
            with self._registry_lock:
                session = self._registry.execute(
                    "SELECT 1 FROM diagnostic_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise SessionNotFoundError(SESSION_NOT_FOUND)
                profile = self._claim_profile_locked(session_id)
                row = self._select_draft_locked(session_id)
                if row is not None and profile == ACTIVE_CLAIM_EVIDENCE_PROFILE:
                    self._validate_claim_draft_locked(session_id, row)
            if row is None:
                raise DraftNotFoundError(DRAFT_NOT_FOUND)
            return self._draft_response_from_row(row)
        except (
            InvalidSessionIdError,
            SessionNotFoundError,
            DraftNotFoundError,
        ):
            raise
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def get_similar_history(
        self, session_id: str, *, limit: int = 3
    ) -> DiagnosticSimilarHistoryResponse:
        """读取相似历史；当前草稿与其外部工单在排序/截断前排除。"""
        self._validate_session_id(session_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("invalid history limit")
        try:
            with self._registry_lock:
                session = self._registry.execute(
                    "SELECT 1 FROM diagnostic_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise SessionNotFoundError(SESSION_NOT_FOUND)
                profile = self._claim_profile_locked(session_id)
                row = self._select_draft_locked(session_id)
                if row is not None and profile == ACTIVE_CLAIM_EVIDENCE_PROFILE:
                    self._validate_claim_draft_locked(session_id, row)
            if row is None:
                return DiagnosticSimilarHistoryResponse(session_id=session_id)
            draft = self._draft_response_from_row(row)
            provider = self._history_provider
            candidates = (
                tuple(
                    provider(
                        draft.device_model,
                        draft.fault_category,
                        exclude_draft_id=draft.draft_id,
                        exclude_ticket_id=draft.external_ticket_id,
                        limit=limit,
                    )
                )
                if provider is not None
                else ()
            )
            items = select_similar_history(
                candidates,
                exclude_draft_id=draft.draft_id,
                exclude_ticket_id=draft.external_ticket_id,
                limit=limit,
            )
            return DiagnosticSimilarHistoryResponse(
                session_id=session_id,
                current_draft_id=draft.draft_id,
                items=items,
            )
        except (InvalidSessionIdError, SessionNotFoundError):
            raise
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    @staticmethod
    def _draft_projection_payload(
        draft: DiagnosticDraftTicketResponse,
    ) -> tuple[str, str, str]:
        payload = {
            "draft_id": draft.draft_id,
            "session_id": draft.session_id,
            "version": draft.version,
            "device_model": draft.device_model,
            "fault_category": draft.fault_category,
            "fault_description": draft.fault_description,
            "diagnosis": draft.diagnosis,
            "solution_steps": draft.solution_steps,
            "risk_level": draft.risk_level,
            "need_human": draft.need_human,
        }
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        idempotency_key = hashlib.sha256(
            f"robotops-draft-projection-v1:{draft.draft_id}:{draft.version}".encode(
                "utf-8"
            )
        ).hexdigest()
        return payload_json, payload_sha256, idempotency_key

    def _recover_projection_finalize_failure(
        self,
        *,
        session_id: str,
        draft_id: str,
        version: int,
        idempotency_key: str,
    ) -> None:
        """尽力把本地提交失败遗留的投影抢占恢复为可重试状态。"""
        try:
            with self._registry_transaction():
                try:
                    self._registry.rollback()
                except Exception:
                    return
                self._registry.execute("BEGIN IMMEDIATE")
                projection = self._registry.execute(
                    """
                    SELECT status, idempotency_key
                    FROM diagnostic_draft_projections
                    WHERE draft_id = ? AND version = ?
                    """,
                    (draft_id, version),
                ).fetchone()
                draft = self._registry.execute(
                    """
                    SELECT status
                    FROM diagnostic_draft_tickets
                    WHERE session_id = ? AND draft_id = ? AND version = ?
                    """,
                    (session_id, draft_id, version),
                ).fetchone()
                if (
                    projection is None
                    or draft is None
                    or not secrets.compare_digest(
                        str(projection[1]), idempotency_key
                    )
                ):
                    self._registry.rollback()
                    return
                projection_status = str(projection[0])
                draft_status = str(draft[0])
                if projection_status in {"FAILED", "PROJECTED"}:
                    self._registry.commit()
                    return
                if (
                    projection_status != "IN_PROGRESS"
                    or draft_status != DiagnosticDraftStatus.PROJECTION_FAILED.value
                ):
                    self._registry.rollback()
                    return
                now = _now_text()
                projection_changed = self._registry.execute(
                    """
                    UPDATE diagnostic_draft_projections
                    SET status = 'FAILED', last_error_code = ?, updated_at = ?
                    WHERE draft_id = ? AND version = ?
                      AND idempotency_key = ? AND status = 'IN_PROGRESS'
                    """,
                    (
                        "DRAFT_PROJECTION_RESULT_PERSISTENCE_FAILED",
                        now,
                        draft_id,
                        version,
                        idempotency_key,
                    ),
                ).rowcount
                draft_changed = self._registry.execute(
                    """
                    UPDATE diagnostic_draft_tickets
                    SET projection_note = ?, updated_at = ?
                    WHERE session_id = ? AND draft_id = ? AND version = ?
                      AND status = 'PROJECTION_FAILED'
                    """,
                    (
                        "外部工单已接受请求，但本地结果保存失败；可安全重试",
                        now,
                        session_id,
                        draft_id,
                        version,
                    ),
                ).rowcount
                if (projection_changed, draft_changed) != (1, 1):
                    self._registry.rollback()
                    return
                self._registry.commit()
        except Exception:
            try:
                with self._registry_lock:
                    self._registry.rollback()
            except Exception:
                pass

    def retry_draft_projection(
        self, session_id: str, draft_id: str, *, version: int
    ) -> DiagnosticDraftTicketResponse:
        """CAS 抢占并重试同一草稿版本；外部调用不持有 SQLite 锁。"""
        self._validate_session_id(session_id)
        self._validate_draft_id(draft_id)
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise DraftConflictError(DRAFT_CONFLICT)
        projector = self._draft_projector
        if projector is None:
            raise SessionStoreError(SANITIZED_STORE_ERROR)

        # 1) 持久化稳定 payload/idempotency key，并以 CAS 抢占一次尝试。
        try:
            with self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                session = self._registry.execute(
                    "SELECT 1 FROM diagnostic_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if session is None:
                    self._registry.rollback()
                    raise SessionNotFoundError(SESSION_NOT_FOUND)
                if self._claim_profile_locked(session_id) == ACTIVE_CLAIM_EVIDENCE_PROFILE:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                row = self._select_draft_locked(session_id)
                if row is None:
                    self._registry.rollback()
                    raise DraftNotFoundError(DRAFT_NOT_FOUND)
                draft = self._draft_response_from_row(row)
                if draft.draft_id != draft_id or draft.version != version:
                    self._registry.rollback()
                    raise DraftConflictError(DRAFT_CONFLICT)
                if draft.status is DiagnosticDraftStatus.PROJECTED:
                    self._registry.commit()
                    return draft
                if draft.status is not DiagnosticDraftStatus.PROJECTION_FAILED:
                    self._registry.rollback()
                    raise DraftConflictError(DRAFT_CONFLICT)
                payload_json, payload_sha256, idempotency_key = (
                    self._draft_projection_payload(draft)
                )
                projection = self._registry.execute(
                    """
                    SELECT payload_json, payload_sha256, idempotency_key, status
                    FROM diagnostic_draft_projections
                    WHERE draft_id = ? AND version = ?
                    """,
                    (draft_id, version),
                ).fetchone()
                now = _now_text()
                if projection is None:
                    self._registry.execute(
                        """
                        INSERT INTO diagnostic_draft_projections (
                            draft_id, version, payload_json, payload_sha256,
                            idempotency_key, status, attempt_count, last_error_code,
                            external_ticket_id, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 'FAILED', 0,
                                  'PREVIOUS_PROJECTION_FAILED', NULL, ?, ?)
                        """,
                        (
                            draft_id,
                            version,
                            payload_json,
                            payload_sha256,
                            idempotency_key,
                            now,
                            now,
                        ),
                    )
                    projection_status = "FAILED"
                else:
                    if (
                        str(projection[0]) != payload_json
                        or not secrets.compare_digest(
                            str(projection[1]), payload_sha256
                        )
                        or not secrets.compare_digest(
                            str(projection[2]), idempotency_key
                        )
                    ):
                        self._registry.rollback()
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    projection_status = str(projection[3])
                if projection_status == "PROJECTED":
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                if projection_status != "FAILED":
                    self._registry.rollback()
                    raise DraftConflictError(DRAFT_CONFLICT)
                claimed = self._registry.execute(
                    """
                    UPDATE diagnostic_draft_projections
                    SET status = 'IN_PROGRESS', attempt_count = attempt_count + 1,
                        last_error_code = NULL, updated_at = ?
                    WHERE draft_id = ? AND version = ? AND status = 'FAILED'
                    """,
                    (now, draft_id, version),
                ).rowcount
                if claimed != 1:
                    self._registry.rollback()
                    raise DraftConflictError(DRAFT_CONFLICT)
                self._registry.commit()
        except (
            InvalidSessionIdError,
            InvalidDraftIdError,
            SessionNotFoundError,
            DraftNotFoundError,
            DraftConflictError,
        ):
            raise
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

        # 2) 不持有 registry lock/transaction 调用外部投影器；重试键已持久化。
        try:
            outcome = projector(draft, idempotency_key)
            external_ticket_id = outcome.get("external_ticket_id")
            projection_note = outcome.get("projection_note") or "外部工单投影成功"
            if (
                not isinstance(external_ticket_id, str)
                or not external_ticket_id.strip()
                or not isinstance(projection_note, str)
            ):
                raise ValueError("invalid draft projection result")
        except Exception as projection_exc:
            try:
                with self._registry_transaction():
                    self._registry.execute("BEGIN IMMEDIATE")
                    now = _now_text()
                    projection_changed = self._registry.execute(
                        """
                        UPDATE diagnostic_draft_projections
                        SET status = 'FAILED', last_error_code = ?, updated_at = ?
                        WHERE draft_id = ? AND version = ?
                          AND idempotency_key = ? AND status = 'IN_PROGRESS'
                        """,
                        (
                            "DRAFT_PROJECTION_FAILED",
                            now,
                            draft_id,
                            version,
                            idempotency_key,
                        ),
                    ).rowcount
                    draft_changed = self._registry.execute(
                        """
                        UPDATE diagnostic_draft_tickets
                        SET projection_note = ?, updated_at = ?
                        WHERE session_id = ? AND draft_id = ? AND version = ?
                          AND status = 'PROJECTION_FAILED'
                        """,
                        (
                            "外部工单投影失败，可重试",
                            now,
                            session_id,
                            draft_id,
                            version,
                        ),
                    ).rowcount
                    if (projection_changed, draft_changed) != (1, 1):
                        self._registry.rollback()
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    self._registry.commit()
            except SessionStoreError:
                raise
            except Exception as exc:
                raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
            raise SessionStoreError(SANITIZED_STORE_ERROR) from projection_exc

        # 3) 新事务精确提交投影结果与草稿状态；失败后可用同一键安全重试。
        try:
            with self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                now = _now_text()
                projection_changed = self._registry.execute(
                    """
                    UPDATE diagnostic_draft_projections
                    SET status = 'PROJECTED', external_ticket_id = ?,
                        last_error_code = NULL, updated_at = ?
                    WHERE draft_id = ? AND version = ?
                      AND idempotency_key = ? AND status = 'IN_PROGRESS'
                    """,
                    (
                        external_ticket_id.strip(),
                        now,
                        draft_id,
                        version,
                        idempotency_key,
                    ),
                ).rowcount
                draft_changed = self._registry.execute(
                    """
                    UPDATE diagnostic_draft_tickets
                    SET status = 'PROJECTED', external_ticket_id = ?,
                        projection_note = ?, updated_at = ?
                    WHERE session_id = ? AND draft_id = ? AND version = ?
                      AND status = 'PROJECTION_FAILED'
                    """,
                    (
                        external_ticket_id.strip(),
                        projection_note,
                        now,
                        session_id,
                        draft_id,
                        version,
                    ),
                ).rowcount
                if (projection_changed, draft_changed) != (1, 1):
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                updated = self._select_draft_locked(session_id)
                if updated is None:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                response = self._draft_response_from_row(updated)
                self._registry.commit()
                return response
        except Exception as exc:
            self._recover_projection_finalize_failure(
                session_id=session_id,
                draft_id=draft_id,
                version=version,
                idempotency_key=idempotency_key,
            )
            if isinstance(exc, SessionStoreError):
                raise
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def _track_future(self, run_id: str, future: Future[Any]) -> None:
        with self._worker_lock:
            self._futures.add(future)

        def discard(done: Future[Any]) -> None:
            worker_failed = False
            try:
                worker_failed = done.exception() is not None
            except Exception:
                worker_failed = True
            with self._worker_lock:
                self._futures.discard(done)
            if worker_failed:
                self._fail_run(run_id, "RUN_WORKER_FAILED")

        future.add_done_callback(discard)

    def _schedule(
        self,
        run_id: str,
        control: RunControl,
        function: Callable[..., None],
        *args: Any,
    ) -> None:
        with self._worker_lock:
            executor = self._executor
            if executor is None or self._active_controls.get(run_id) is not control:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            future = executor.submit(function, *args, control)
        self._track_future(run_id, future)

    def submit_run(
        self,
        session_id: str,
        message: str,
        request_id: str | None = None,
    ) -> tuple[DiagnosticRunResponse, bool]:
        """幂等登记逻辑 turn；返回权威 run 与是否首次创建。"""
        self._validate_session_id(session_id)
        effective_request_id = request_id or self._new_token()
        self._validate_request_id(effective_request_id)
        try:
            payload_json, payload_sha256 = self._request_payload(message)
        except Exception:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from None
        now = _now_text()
        run_id = ""
        control: RunControl | None = None
        control_registered = False
        try:
            with self._committed_run_observations() as committed, self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                row = self._registry.execute(
                    """
                    SELECT session_id, thread_id, status, created_at, updated_at, turn_count
                    FROM diagnostic_sessions WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if row is None:
                    self._registry.rollback()
                    raise SessionNotFoundError(SESSION_NOT_FOUND)
                record = self._row_to_record(row)
                existing_request = self._select_request_locked(
                    session_id, effective_request_id
                )
                if existing_request is not None:
                    if (
                        existing_request[2] != "RUN"
                        or not secrets.compare_digest(
                            str(existing_request[3]), payload_sha256
                        )
                        or str(existing_request[4]) != payload_json
                    ):
                        self._registry.rollback()
                        raise RunConflictError(RUN_CONFLICT)
                    existing_run = self._select_run_locked(str(existing_request[5]))
                    if existing_run is None or existing_run.session_id != session_id:
                        self._registry.rollback()
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    response = self._run_response_locked(record.status, existing_run)
                    self._registry.commit()
                    return response, False
                if record.status not in _WRITABLE_STATUSES:
                    self._registry.rollback()
                    raise SessionConflictError(SESSION_CONFLICT)
                if record.claim_evidence_profile == ACTIVE_CLAIM_EVIDENCE_PROFILE:
                    if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", effective_request_id) is None:
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    self._prepare_state(self._load_previous_state(record), message,
                                        claim_profile=record.claim_evidence_profile)
                run_id = self._new_token()
                control = RunControl()
                turn_number = record.turn_count + 1
                self._registry.execute(
                    """
                    INSERT INTO diagnostic_runs (
                        run_id, session_id, turn_number, status,
                        previous_session_status, base_turn_count, created_at, updated_at,
                        tool_iterations, max_tool_iterations
                    ) VALUES (?, ?, ?, 'QUEUED', ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        run_id,
                        session_id,
                        turn_number,
                        record.status,
                        record.turn_count,
                        now,
                        now,
                        MAX_TOOL_ITERATIONS,
                    ),
                )
                self._registry.execute(
                    """
                    INSERT INTO diagnostic_requests (
                        session_id, request_id, operation, payload_sha256, payload_json,
                        run_id, attempt_count, created_at, updated_at
                    ) VALUES (?, ?, 'RUN', ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        session_id,
                        effective_request_id,
                        payload_sha256,
                        payload_json,
                        run_id,
                        now,
                        now,
                    ),
                )
                self._append_message_locked(
                    session_id=session_id,
                    sequence_no=(turn_number * 2) - 1,
                    turn_number=turn_number,
                    role=DiagnosticMessageRole.USER,
                    kind=DiagnosticMessageKind.INPUT,
                    content=message,
                    run_id=run_id,
                    request_id=effective_request_id,
                    payload=None,
                    created_at=now,
                )
                changed = self._registry.execute(
                    """
                    UPDATE diagnostic_sessions
                    SET status = 'RUNNING', active_run_id = ?, updated_at = ?
                    WHERE session_id = ? AND status = ? AND active_run_id IS NULL
                    """,
                    (run_id, now, session_id, record.status),
                ).rowcount
                if changed != 1:
                    self._registry.rollback()
                    raise SessionConflictError(SESSION_CONFLICT)
                self._append_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="RUN_QUEUED",
                    from_status=None,
                    to_status=DiagnosticRunStatus.QUEUED.value,
                    dedupe_key=f"run:{run_id}:queued",
                    created_at=now,
                )
                self._append_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="SESSION_RUNNING",
                    from_status=record.status,
                    to_status=DiagnosticSessionStatus.RUNNING.value,
                    dedupe_key=f"run:{run_id}:session-running",
                    created_at=now,
                )
                self._register_control(run_id, control)
                control_registered = True
                self._registry.commit()
                committed.append((run_id, "RUN_QUEUED", "robotops.persistence.run_state"))
        except (
            InvalidSessionIdError,
            SessionNotFoundError,
            SessionConflictError,
            RunConflictError,
        ):
            if control_registered and control is not None:
                self._unregister_control(run_id, control)
            raise
        except sqlite3.IntegrityError as exc:
            if control_registered and control is not None:
                self._unregister_control(run_id, control)
            raise SessionConflictError(SESSION_CONFLICT) from exc
        except Exception as exc:
            if control_registered and control is not None:
                self._unregister_control(run_id, control)
            if isinstance(exc, SessionStoreError):
                raise
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

        try:
            if control is None:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            self._schedule(
                run_id,
                control,
                self._execute_initial_run,
                run_id,
                record,
                message,
            )
        except Exception as exc:
            if control is not None:
                self._unregister_control(run_id, control)
            self._fail_run(run_id, "INITIAL_RUN_SCHEDULING_FAILED")
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        return self.get_run(session_id, run_id), True

    def retry_run(
        self, session_id: str, run_id: str
    ) -> tuple[DiagnosticRunResponse, bool]:
        """只重试图尚未开始的调度失败；复用原 run/request/消息。"""
        self._validate_session_id(session_id)
        self._validate_run_id(run_id)
        now = _now_text()
        control: RunControl | None = None
        control_registered = False
        try:
            with self._committed_run_observations() as committed, self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                session_row = self._registry.execute(
                    """
                    SELECT session_id, thread_id, status, created_at, updated_at,
                           turn_count, active_run_id
                    FROM diagnostic_sessions WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if session_row is None:
                    self._registry.rollback()
                    raise SessionNotFoundError(SESSION_NOT_FOUND)
                self._claim_profile_locked(session_id)
                run = self._select_run_locked(run_id)
                if run is None:
                    self._registry.rollback()
                    raise RunNotFoundError(RUN_NOT_FOUND)
                if run.session_id != session_id:
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                if run.status != DiagnosticRunStatus.FAILED.value:
                    response = self._run_response_locked(str(session_row[2]), run)
                    self._registry.commit()
                    if run.status in _ACTIVE_RUN_STATUSES or run.status in {
                        DiagnosticRunStatus.COMPLETED.value,
                        DiagnosticRunStatus.CANCELLED.value,
                    }:
                        return response, False
                    raise RunConflictError(RUN_CONFLICT)
                if run.error_code not in _RETRYABLE_RUN_ERRORS:
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                request = self._registry.execute(
                    """
                    SELECT request_id, operation, payload_json, attempt_count
                    FROM diagnostic_requests WHERE run_id = ? AND session_id = ?
                    """,
                    (run_id, session_id),
                ).fetchone()
                if request is None or request[1] != "RUN":
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                try:
                    payload = json.loads(str(request[2]))
                except Exception as exc:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
                message = payload.get("message") if isinstance(payload, dict) else None
                if not isinstance(message, str) or not message:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                if (
                    session_row[2] != DiagnosticSessionStatus.FAILED.value
                    or session_row[6] is not None
                    or session_row[5] != run.base_turn_count
                ):
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                attempt_count = int(request[3]) + 1
                request_changed = self._registry.execute(
                    """
                    UPDATE diagnostic_requests
                    SET attempt_count = ?, updated_at = ?
                    WHERE run_id = ? AND session_id = ? AND attempt_count = ?
                    """,
                    (attempt_count, now, run_id, session_id, request[3]),
                ).rowcount
                run_changed = self._registry.execute(
                    """
                    UPDATE diagnostic_runs
                    SET status = 'QUEUED', updated_at = ?, result_json = NULL,
                        error_code = NULL, stop_reason = NULL, current_interrupt_id = NULL
                    WHERE run_id = ? AND session_id = ? AND status = 'FAILED'
                    """,
                    (now, run_id, session_id),
                ).rowcount
                session_changed = self._registry.execute(
                    """
                    UPDATE diagnostic_sessions
                    SET status = 'RUNNING', active_run_id = ?, updated_at = ?
                    WHERE session_id = ? AND status = 'FAILED' AND active_run_id IS NULL
                    """,
                    (run_id, now, session_id),
                ).rowcount
                if (request_changed, run_changed, session_changed) != (1, 1, 1):
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                self._append_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="RUN_RETRY_QUEUED",
                    from_status=DiagnosticRunStatus.FAILED.value,
                    to_status=DiagnosticRunStatus.QUEUED.value,
                    reason_code="RETRYABLE_SCHEDULING_FAILURE",
                    dedupe_key=f"run:{run_id}:retry:{attempt_count}:queued",
                    created_at=now,
                )
                self._append_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="SESSION_RETRY_RUNNING",
                    from_status=DiagnosticSessionStatus.FAILED.value,
                    to_status=DiagnosticSessionStatus.RUNNING.value,
                    reason_code="RETRYABLE_SCHEDULING_FAILURE",
                    dedupe_key=f"run:{run_id}:retry:{attempt_count}:session-running",
                    created_at=now,
                )
                record = _SessionRecord(
                    session_id=str(session_row[0]),
                    thread_id=str(session_row[1]),
                    status=run.previous_session_status,
                    created_at=str(session_row[3]),
                    updated_at=str(session_row[4]),
                    turn_count=run.base_turn_count,
                    claim_evidence_profile=self._claim_profile_locked(session_id),
                )
                control = RunControl()
                self._register_control(run_id, control)
                control_registered = True
                self._registry.commit()
                committed.append((run_id, "RUN_RETRY_QUEUED", "robotops.persistence.run_state"))
        except (
            InvalidSessionIdError,
            InvalidRunIdError,
            SessionNotFoundError,
            RunNotFoundError,
            RunConflictError,
        ):
            if control_registered and control is not None:
                self._unregister_control(run_id, control)
            raise
        except Exception as exc:
            if control_registered and control is not None:
                self._unregister_control(run_id, control)
            if isinstance(exc, SessionStoreError):
                raise
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        try:
            if control is None:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            self._schedule(
                run_id,
                control,
                self._execute_initial_run,
                run_id,
                record,
                message,
            )
        except Exception as exc:
            if control is not None:
                self._unregister_control(run_id, control)
            self._fail_run(run_id, "INITIAL_RUN_SCHEDULING_FAILED")
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        return self.get_run(session_id, run_id), True

    def _transition_run_to_running(self, run_id: str) -> bool:
        now = _now_text()
        try:
            with self._committed_run_observations() as committed, self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                run = self._select_run_locked(run_id)
                if run is None:
                    self._registry.rollback()
                    return False
                if run.status == DiagnosticRunStatus.CANCELLED.value:
                    self._registry.commit()
                    return False
                if run.status != DiagnosticRunStatus.QUEUED.value:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                changed = self._registry.execute(
                    """
                    UPDATE diagnostic_runs SET status = 'RUNNING', updated_at = ?
                    WHERE run_id = ? AND status = 'QUEUED'
                    """,
                    (now, run_id),
                ).rowcount
                if changed != 1:
                    self._registry.rollback()
                    return False
                self._append_event(
                    session_id=run.session_id,
                    run_id=run_id,
                    event_type="RUN_RUNNING",
                    from_status=DiagnosticRunStatus.QUEUED.value,
                    to_status=DiagnosticRunStatus.RUNNING.value,
                    dedupe_key=(
                        f"run:{run_id}:running{self._run_attempt_suffix_locked(run_id)}"
                    ),
                    created_at=now,
                )
                self._registry.commit()
                committed.append((run_id, "RUN_RUNNING", "robotops.persistence.run_state"))
                return True
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    @staticmethod
    def _execution_fields(state: Mapping[str, Any]) -> tuple[int, int, str | None]:
        count = state.get("tool_iterations", 0)
        limit = state.get("max_tool_iterations", MAX_TOOL_ITERATIONS)
        reason = state.get("execution_stop_reason", "")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or count > MAX_TOOL_ITERATIONS
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit != MAX_TOOL_ITERATIONS
            or not isinstance(reason, str)
            or STOP_REASON_PATTERN.fullmatch(reason) is None
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return count, limit, reason or None

    def _validate_agent_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        allow_active: bool,
        expected_profile: str | None = None,
    ) -> dict[str, Any]:
        """在服务发布/恢复边界验证完整 Agent checkpoint。

        `allow_active` 只对真实 interrupt/恢复点开放；普通终态不能把
        `decide`/`execute`/`handoff` 冒充为已完成结果。
        """
        self._execution_fields(state)
        if state.get("agent_tool_loop_enabled") is not self._enable_agent_tool_loop:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if state.get("ticket_mode") != "draft":
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        checkpoint_profile = self._authorized_tool_capability_profile(
            state.get("tool_capability_profile")
        )
        normalized_expected_profile = (
            self._authorized_tool_capability_profile(expected_profile)
            if expected_profile is not None
            else None
        )
        if (
            normalized_expected_profile is not None
            and checkpoint_profile != normalized_expected_profile
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        try:
            agent_fields = tool_loop.validate_checkpoint_progress(state)
        except (tool_loop.PlanContractError, tool_loop.ToolLoopError) as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        if (
            not self._enable_agent_tool_loop
            and agent_fields["agent_route"] in {"decide", "execute"}
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if not allow_active and agent_fields["agent_route"] not in {"", "done"}:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if (
            state.get("risk_level") not in {"", "低", "中", "高"}
            or state.get("urgency") not in {"", "低", "中", "高"}
            or not isinstance(state.get("need_human"), bool)
            or not isinstance(state.get("risk_check_failed"), bool)
            or not isinstance(state.get("llm_failed"), bool)
            or (
                (
                    state["risk_check_failed"]
                    or state["llm_failed"]
                    or state["risk_level"] == "高"
                )
                and (
                    state["need_human"] is not True
                    or state["risk_level"] != "高"
                    or state["urgency"] != "高"
                )
            )
            or (state["urgency"] == "高" and state["need_human"] is not True)
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        effect_runtime = state.get("effect_runtime", {})
        if not isinstance(effect_runtime, dict):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if effect_runtime:
            if set(effect_runtime) != _EFFECT_RUNTIME_FIELDS:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            status = effect_runtime.get("status")
            result_sha256 = effect_runtime.get("result_sha256")
            verification_sha256 = effect_runtime.get("verification_sha256")
            if (
                type(effect_runtime.get("schema_version")) is not int
                or effect_runtime["schema_version"] != 1
                or not self._valid_digest(effect_runtime.get("effect_key"))
                or not self._valid_digest(effect_runtime.get("binding_sha256"))
                or status not in _EFFECT_ATTEMPT_TERMINAL_STATUSES
                or type(effect_runtime.get("attempt_count")) is not int
                or effect_runtime["attempt_count"] != 1
                or (
                    result_sha256 is not None
                    and not self._valid_digest(result_sha256)
                )
                or (
                    verification_sha256 is not None
                    and not self._valid_digest(verification_sha256)
                )
            ):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            policy_status = agent_fields["effect_policy_status"]
            if status in {"APPLIED", "VERIFIED"}:
                valid_runtime = (
                    policy_status == "PROPOSED"
                    and result_sha256 is not None
                    and (
                        (status == "APPLIED" and verification_sha256 is None)
                        or (status == "VERIFIED" and verification_sha256 is not None)
                    )
                )
            elif status == "EXECUTION_FAILED":
                valid_runtime = (
                    policy_status == "ERROR"
                    and result_sha256 is None
                    and verification_sha256 is None
                )
            elif status == "VERIFICATION_FAILED":
                valid_runtime = (
                    policy_status == "ERROR"
                    and result_sha256 is not None
                    and verification_sha256 is not None
                )
            else:
                valid_runtime = (
                    policy_status == "ERROR"
                    and not (
                        result_sha256 is None and verification_sha256 is not None
                    )
                )
            if not valid_runtime:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
        if not self._enable_agent_tool_loop and effect_runtime:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        return {**agent_fields, "effect_runtime": dict(effect_runtime)}

    @staticmethod
    def _extract_interrupt(
        result: Mapping[str, Any],
    ) -> tuple[str, str, str] | None:
        if "__interrupt__" not in result:
            return None
        raw = result["__interrupt__"]
        if not isinstance(raw, (tuple, list)) or len(raw) != 1:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        item = raw[0]
        interrupt_id = getattr(item, "id", None)
        payload = getattr(item, "value", None)
        if not isinstance(interrupt_id, str) or not interrupt_id:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if not isinstance(payload, dict):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        try:
            payload_json = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        return interrupt_id, payload_json, digest

    def _validate_async_effect_terminal(
        self, *, run_id: str, state: Mapping[str, Any]
    ) -> None:
        policy_status = state.get("effect_policy_status", "NOT_STARTED")
        proposal = state.get("effect_proposal", {})
        proposal_sha256 = state.get("effect_proposal_sha256", "")
        state_runtime = state.get("effect_runtime", {})
        if (
            policy_status
            not in {"NOT_STARTED", "NOT_APPLICABLE", "PROPOSED", "DENIED", "ERROR"}
            or not isinstance(proposal, dict)
            or not isinstance(proposal_sha256, str)
            or not isinstance(state_runtime, dict)
        ):
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        with self._registry_lock:
            run = self._select_run_locked(run_id)
            if run is None:
                raise RunPersistenceError(SANITIZED_STORE_ERROR)
            if self._claim_profile_locked(run.session_id) == ACTIVE_CLAIM_EVIDENCE_PROFILE:
                if (policy_status != "NOT_STARTED" or proposal or proposal_sha256 or state_runtime):
                    raise RunPersistenceError(SANITIZED_STORE_ERROR)
                self._validate_claim_effect_terminal_locked(run, state)
                return
            approval_ids = self._registry.execute(
                "SELECT approval_id FROM diagnostic_effect_approvals "
                "WHERE run_id = ? ORDER BY approval_id",
                (run_id,),
            ).fetchall()
            if not approval_ids:
                if proposal or proposal_sha256 or state_runtime or policy_status == "PROPOSED":
                    raise RunPersistenceError(SANITIZED_STORE_ERROR)
                if (
                    self._effect_mode == EFFECT_MODE_SYNTHETIC_RESTART
                    and self._enable_agent_tool_loop
                    and not (
                        state.get("route") == "ask"
                        and bool(state.get("follow_up_questions"))
                    )
                    and policy_status not in {"NOT_APPLICABLE", "DENIED", "ERROR"}
                ):
                    raise RunPersistenceError(SANITIZED_STORE_ERROR)
                return
            if len(approval_ids) != 1:
                raise RunPersistenceError(SANITIZED_STORE_ERROR)
            parent = self._effect_parent_locked(str(approval_ids[0][0]))
            if parent["session_id"] != run.session_id or parent["run_id"] != run_id:
                raise RunPersistenceError(SANITIZED_STORE_ERROR)
            approval, _public = self._validate_typed_approval_locked(
                parent=parent,
                run=run,
                require_checkpoint=False,
            )
            ledger = self._select_effect_ledger_locked(str(approval["effect_key"]))
            if ledger is None:
                raise RunPersistenceError(SANITIZED_STORE_ERROR)
            ledger_runtime, _arguments, _result, _verification = (
                self._validated_effect_ledger_locked(ledger)
            )
            if (
                approval["status"] != "CONSUMED"
                or parent["status"] != "CONSUMED"
                or approval["decision"] != parent["decision"]
            ):
                raise RunPersistenceError(SANITIZED_STORE_ERROR)
            if approval["decision"] == ResumeDecision.REJECT.value:
                valid = (
                    policy_status == "DENIED"
                    and not proposal
                    and not proposal_sha256
                    and not state_runtime
                    and ledger_runtime["status"] == "PREPARED"
                    and ledger_runtime["attempt_count"] == 0
                )
            elif approval["decision"] == ResumeDecision.APPROVE.value:
                if ledger_runtime["status"] == "VERIFIED":
                    valid = (
                        policy_status == "PROPOSED"
                        and effect_capabilities.canonical_json(proposal)
                        == approval["proposal_json"]
                        and proposal_sha256 == approval["proposal_sha256"]
                        and state_runtime == ledger_runtime
                    )
                elif ledger_runtime["status"] in {
                    "EXECUTION_FAILED",
                    "VERIFICATION_FAILED",
                    "UNKNOWN",
                }:
                    valid = (
                        policy_status == "ERROR"
                        and not proposal
                        and not proposal_sha256
                        and state_runtime == ledger_runtime
                    )
                else:
                    valid = False
            else:
                valid = False
            if not valid:
                raise RunPersistenceError(SANITIZED_STORE_ERROR)

    def _validate_claim_effect_terminal_locked(
        self, run: _RunRecord, state: Mapping[str, Any],
    ) -> dict | None:
        """Bind graph completion to exactly one already consumed V2 ledger."""
        contract = self._execution_contract_locked(run.session_id)
        self._validate_claim_effect_state_domain(state, run.session_id)
        ids = self._registry.execute(
            "SELECT approval_id FROM diagnostic_claim_effect_approvals WHERE run_id=?",
            (run.run_id,),
        ).fetchall()
        intent = state.get("claim_effect_intent", {})
        runtime = state.get("claim_effect_runtime", {})
        decision = state.get("claim_effect_decision", "")
        if contract != CLAIM_EXECUTION_CONTRACT:
            if ids or intent or runtime or decision:
                raise RunPersistenceError(SANITIZED_STORE_ERROR)
            return None
        if not ids:
            if intent or runtime or decision:
                raise RunPersistenceError(SANITIZED_STORE_ERROR)
            return None
        if len(ids) != 1:
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        approval_id = str(ids[0][0])
        checked = self._validate_claim_effect_binding_locked(
            approval_id, require_checkpoint=False,
        )
        approval, ledger = checked["approval"], checked["ledger"]
        parent = self._effect_parent_locked(approval_id)
        expected_intent = {
            "approval_kind": "EFFECT_V2",
            "intent_profile": "robotops_claim_effect_intent_v2",
            "proposal": checked["material"]["proposal"],
            "verification_sha256": checked["material"]["proof"]["verification"]["verification_sha256"],
        }
        if (parent["status"] != "CONSUMED" or approval["status"] != "CONSUMED"
                or intent != expected_intent or decision != approval["decision"]
                or decision not in {"APPROVE", "REJECT"}
                or state.get("claim_verification") != checked["material"]["proof"]["verification"]):
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        if decision == "REJECT":
            valid = ledger["status"] == "PREPARED" and ledger["attempt_count"] == 0 and runtime == {}
        else:
            valid = (ledger["status"] in {
                "VERIFIED", "EXECUTION_FAILED", "VERIFICATION_FAILED", "UNKNOWN",
            } and runtime == checked["runtime"])
        if not valid:
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        if (decision == "REJECT" or ledger["status"] != "VERIFIED") and (
            state.get("risk_level") != "高" or state.get("urgency") != "高"
            or state.get("need_human") is not True
        ):
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        return self._claim_effect_response_locked(run, terminal=True).model_dump(mode="json")

    def _prepare_authoritative_claim_result(
        self, *, session_id: str, turn_number: int, previous: Mapping[str, Any],
        source: Mapping[str, Any], raw_result: Mapping[str, Any],
        run_id: str | None = None,
    ) -> tuple[dict, dict]:
        """Merge independent service facts before signing or publishing a claim."""
        from app.agent.claim_sources import CONTEXT_FIELDS, _canonical, _json_copy, MAX_SOURCE_BYTES
        if self._claim_profile_locked(session_id) != ACTIVE_CLAIM_EVIDENCE_PROFILE:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        try:
            query = source["user_query"]
            if (type(query) is not str or not query or raw_result.get("user_query") != query
                    or raw_result.get("claim_evidence_profile") != ACTIVE_CLAIM_EVIDENCE_PROFILE
                    or source.get("claim_evidence_profile") != ACTIVE_CLAIM_EVIDENCE_PROFILE):
                raise ValueError()
            for name in ("ticket_mode", "tool_capability_profile", "agent_tool_loop_enabled"):
                if raw_result.get(name) != source.get(name):
                    raise ValueError()
            seed = _json_copy(raw_result["claim_context_seed"], maximum=MAX_SOURCE_BYTES)
            expected_seed = source.get("claim_context_seed") or {
                name: source[name] for name in CONTEXT_FIELDS
            }
            if _canonical(seed) != _canonical(expected_seed):
                raise ValueError()
            result = merge_authoritative_facts(previous, raw_result, cumulative_query=query)
            result = {**result, **merge_sticky_safety(previous, result)}
            # Tool failure is another sticky safety fact, not model-controlled permission.
            if previous.get("tool_loop_failed") is True:
                result["tool_loop_failed"] = True
                result["need_human"] = True
            authority = {"session_id": session_id, "run_id": run_id, "request_id": None,
                         "turn_number": turn_number,
                         **{name: result[name] for name in CONTEXT_FIELDS}}
            authority["user_query"] = query
            if run_id is not None:
                with self._registry_lock:
                    run = self._select_run_locked(run_id)
                    if run is None or run.session_id != session_id or run.turn_number != turn_number:
                        raise ValueError()
                    authority["request_id"] = self._request_id_for_run_locked(run_id)
            return result, authority
        except Exception:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from None

    def _materialize_claim_result(
        self, *, session_id: str, turn_number: int, previous: Mapping[str, Any],
        source: Mapping[str, Any], raw_result: Mapping[str, Any], config: dict,
        run_id: str | None = None,
    ) -> tuple[DiagnosticTurnResponse, dict]:
        """Publish only after independently merging service facts and safety."""
        from app.services import claim_sessions
        try:
            if "__interrupt__" in raw_result:
                raise ValueError()
            result, authority = self._prepare_authoritative_claim_result(
                session_id=session_id, turn_number=turn_number, previous=previous,
                source=source, raw_result=raw_result, run_id=run_id,
            )
            query = authority["user_query"]
            published = claim_sessions.publish_claim_turn(
                result, authority=authority, follow_up_rounds=source["follow_up_rounds"],
            )
            projection = published["projection"]
            conversation = self._conversation(source)
            conversation.append({"role": "assistant", "content": projection["reply"]})
            persisted = {**result, **published["state_update"],
                         "user_query": query, "conversation": conversation}
            claim_sessions.validate_claim_terminal_state(persisted, expected_authority=authority)
            self._validate_agent_checkpoint_state(persisted, allow_active=False)
            if run_id is not None:
                self._validate_async_effect_terminal(run_id=run_id, state=persisted)
            response = DiagnosticTurnResponse(
                session_id=session_id, status=projection["status"], turn_number=turn_number,
                reply=projection["reply"], updated_at=_parse_time(_now_text()),
                references=projection["references"], follow_up=projection["follow_up"],
                diagnosis={**projection["diagnosis"], "claim_evidence": published["publication"]},
                nodes=[name for name in _infer_nodes(result) if name not in {"ticket", "answer"}]
                      + ["claim_verify"], ticket_id=None, ticket_note=None,
            )
            claim_sessions.validate_claim_response(response.model_dump(mode="json"),
                                                   expected_authority=authority)
        except Exception:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from None
        try:
            self._graph.update_state(config, persisted)
        except Exception:
            raise RunCheckpointError(SANITIZED_STORE_ERROR) from None
        return response, persisted

    def _materialize_async_result(
        self,
        *,
        run_id: str | None = None,
        session_id: str,
        turn_number: int,
        previous: Mapping[str, Any],
        source: Mapping[str, Any],
        raw_result: Mapping[str, Any],
        config: dict,
    ) -> tuple[DiagnosticTurnResponse, dict]:
        if self._claim_profile_locked(session_id) == ACTIVE_CLAIM_EVIDENCE_PROFILE:
            return self._materialize_claim_result(
                run_id=run_id, session_id=session_id, turn_number=turn_number,
                previous=previous, source=source, raw_result=raw_result, config=config,
            )
        self._validate_claim_checkpoint_profile(raw_result, DISABLED_CLAIM_EVIDENCE_PROFILE, terminal=True)
        if "__interrupt__" in raw_result:
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        result = {key: value for key, value in raw_result.items() if key != "__interrupt__"}
        cumulative_query = result.get("user_query") or source.get("user_query")
        if not isinstance(cumulative_query, str) or not cumulative_query:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        result = merge_authoritative_facts(
            previous,
            result,
            cumulative_query=cumulative_query,
        )
        result = {**result, **merge_sticky_safety(previous, result)}
        # `/diagnostic-sessions` 永远由服务端掌握草稿模式；自定义图或恢复载荷
        # 不能伪造外部工单成功/失败，从而越过审批后的真实投影边界。
        result["agent_tool_loop_enabled"] = self._enable_agent_tool_loop
        source_profile = source.get(
            "tool_capability_profile", tool_loop.CORE_TOOL_PROFILE
        )
        try:
            tool_loop.registry_for_profile(source_profile)
        except tool_loop.ToolLoopError as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        result["tool_capability_profile"] = source_profile
        result.setdefault(
            "tool_result_evidence", source.get("tool_result_evidence", {})
        )
        result["ticket_mode"] = "draft"
        result["ticket_id"] = ""
        result["ticket_note"] = ""
        questions = result.get("follow_up_questions") or []
        if not isinstance(questions, list):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        rounds = result.get("follow_up_rounds", source.get("follow_up_rounds", 0))
        if (
            not isinstance(rounds, int)
            or isinstance(rounds, bool)
            or not 0 <= rounds <= MAX_FOLLOW_UP_ROUNDS
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        asked_for_more = result.get("route") == "ask" and bool(questions)
        limit_reached = asked_for_more and rounds >= MAX_FOLLOW_UP_ROUNDS
        awaiting = asked_for_more and not limit_reached
        if limit_reached:
            result["need_human"] = True
            questions = []
        status = (
            DiagnosticSessionStatus.AWAITING_USER
            if awaiting
            else DiagnosticSessionStatus.COMPLETED
        )
        reply = (
            FOLLOW_UP_LIMIT_REPLY
            if limit_reached
            else (
                _follow_up_reply(questions)
                if awaiting
                else result.get("final_answer")
                or "诊断工作流未生成有效回答，请联系人工支持。"
            )
        )
        conversation_source = result.get("conversation", source.get("conversation", []))
        conversation = self._conversation({"conversation": conversation_source})
        conversation.append({"role": "assistant", "content": reply})
        persisted = {
            **result,
            "user_query": cumulative_query,
            "conversation": conversation,
            "follow_up_rounds": rounds + (1 if awaiting else 0),
        }
        persisted.setdefault("tool_iterations", 0)
        persisted.setdefault("max_tool_iterations", MAX_TOOL_ITERATIONS)
        persisted.setdefault("execution_stop_reason", "")
        self._validate_agent_checkpoint_state(persisted, allow_active=False)
        if run_id is not None:
            self._validate_async_effect_terminal(run_id=run_id, state=persisted)
        elif (
            persisted.get("effect_policy_status") == "PROPOSED"
            or bool(persisted.get("effect_proposal"))
            or bool(persisted.get("effect_proposal_sha256"))
            or bool(persisted.get("effect_runtime"))
        ):
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        try:
            self._graph.update_state(config, persisted)
        except Exception as exc:
            raise RunCheckpointError(SANITIZED_STORE_ERROR) from exc
        now = _now_text()
        response = DiagnosticTurnResponse(
            session_id=session_id,
            status=status,
            turn_number=turn_number,
            reply=reply,
            updated_at=_parse_time(now),
            references=result.get("references") or [],
            nodes=_infer_nodes(result),
            follow_up=questions if awaiting else [],
            diagnosis=_diagnosis(result),
            ticket_id=result.get("ticket_id") or None,
            ticket_note=result.get("ticket_note") or None,
        )
        return response, persisted

    def _set_cancelled_locked(
        self, run: _RunRecord, *, from_status: str, now: str, reason_code: str
    ) -> tuple[str, ...]:
        self._claim_profile_locked(run.session_id)
        expected_session_status = (
            DiagnosticSessionStatus.CANCEL_REQUESTED.value
            if from_status == DiagnosticRunStatus.CANCEL_REQUESTED.value
            else DiagnosticSessionStatus.INTERRUPTED.value
            if from_status == DiagnosticRunStatus.INTERRUPTED.value
            else DiagnosticSessionStatus.RUNNING.value
        )
        session = self._registry.execute(
            "SELECT status, active_run_id FROM diagnostic_sessions WHERE session_id = ?",
            (run.session_id,),
        ).fetchone()
        if session != (expected_session_status, run.run_id):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        pending_interrupts = self._registry.execute(
            """
            SELECT approval_id, interrupt_id, payload_json, payload_sha256
            FROM diagnostic_interrupts
            WHERE run_id = ? AND status = 'PENDING'
            """,
            (run.run_id,),
        ).fetchall()
        typed_now_ms: int | None = None
        newly_terminal_approvals: list[str] = []
        for approval_id, interrupt_id, payload_json, payload_sha256 in pending_interrupts:
            parent = {
                "approval_id": str(approval_id),
                "session_id": run.session_id,
                "run_id": run.run_id,
                "interrupt_id": str(interrupt_id),
                "payload_json": str(payload_json),
                "payload_sha256": str(payload_sha256),
                "status": "PENDING",
                "decision": None,
                "consumed_at": None,
            }
            try:
                parsed_payload = json.loads(str(payload_json))
            except Exception as exc:
                raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
            family = self._effect_family_locked(parent, run)
            if family == "EFFECT_V2":
                checked = self._validate_claim_effect_binding_locked(
                    str(approval_id), require_checkpoint=run.status == "INTERRUPTED",
                )
                approval = checked["approval"]
                if approval["status"] == "PENDING":
                    if typed_now_ms is None:
                        typed_now_ms = self._typed_now_ms()
                    terminal_capability_status = (
                        "EXPIRED" if typed_now_ms >= int(approval["expires_at"]) else "CANCELLED"
                    )
                    changed = self._registry.execute(
                        "UPDATE diagnostic_claim_effect_approvals "
                        "SET status=?, consumed_at=? "
                        "WHERE approval_id=? AND status='PENDING' AND decision IS NULL",
                        (terminal_capability_status, typed_now_ms, approval_id),
                    ).rowcount
                    if changed != 1:
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    newly_terminal_approvals.append(terminal_capability_status)
                elif approval["status"] != "EXPIRED":
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
            elif family == "EFFECT_V1":
                approval, _public = self._validate_typed_approval_locked(
                    parent=parent,
                    run=run,
                    require_checkpoint=True,
                )
                if approval["status"] == "PENDING":
                    if typed_now_ms is None:
                        typed_now_ms = self._typed_now_ms()
                    terminal_capability_status = (
                        "EXPIRED"
                        if typed_now_ms >= int(approval["expires_at"])
                        else "CANCELLED"
                    )
                    changed = self._registry.execute(
                        """
                        UPDATE diagnostic_effect_approvals
                        SET status = ?, consumed_at = ?
                        WHERE approval_id = ? AND status = 'PENDING'
                          AND decision IS NULL
                        """,
                        (terminal_capability_status, typed_now_ms, approval_id),
                    ).rowcount
                    if changed != 1:
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    newly_terminal_approvals.append(terminal_capability_status)
                elif approval["status"] != "EXPIRED":
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
        orphaned_pending_typed = self._registry.execute(
            """
            SELECT COUNT(*)
            FROM diagnostic_effect_approvals AS approvals
            LEFT JOIN diagnostic_interrupts AS interrupts
              ON interrupts.approval_id = approvals.approval_id
            WHERE approvals.run_id = ? AND approvals.status = 'PENDING'
              AND (interrupts.approval_id IS NULL OR interrupts.status != 'PENDING')
            """,
            (run.run_id,),
        ).fetchone()
        if orphaned_pending_typed is None or int(orphaned_pending_typed[0]) != 0:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        orphaned_pending_claim = self._registry.execute(
            "SELECT COUNT(*) FROM diagnostic_claim_effect_approvals AS approvals "
            "LEFT JOIN diagnostic_interrupts AS interrupts "
            "ON interrupts.approval_id=approvals.approval_id "
            "WHERE approvals.run_id=? AND approvals.status='PENDING' "
            "AND (interrupts.approval_id IS NULL OR interrupts.status!='PENDING')",
            (run.run_id,),
        ).fetchone()
        if orphaned_pending_claim is None or int(orphaned_pending_claim[0]) != 0:
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        run_changed = self._registry.execute(
            """
            UPDATE diagnostic_runs
            SET status = 'CANCELLED', updated_at = ?, current_interrupt_id = NULL,
                stop_reason = ?
            WHERE run_id = ? AND status = ?
            """,
            (now, reason_code, run.run_id, from_status),
        ).rowcount
        self._registry.execute(
            """
            UPDATE diagnostic_interrupts
            SET status = 'CANCELLED', consumed_at = ?
            WHERE run_id = ? AND status = 'PENDING'
            """,
            (now, run.run_id),
        )
        session_changed = self._registry.execute(
            """
            UPDATE diagnostic_sessions
            SET status = 'CANCELLED', active_run_id = NULL, updated_at = ?
            WHERE session_id = ? AND active_run_id = ? AND status = ?
            """,
            (now, run.session_id, run.run_id, expected_session_status),
        ).rowcount
        if (run_changed, session_changed) != (1, 1):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        self._append_event(
            session_id=run.session_id,
            run_id=run.run_id,
            event_type="RUN_CANCELLED",
            from_status=from_status,
            to_status=DiagnosticRunStatus.CANCELLED.value,
            reason_code=reason_code,
            dedupe_key=f"run:{run.run_id}:cancelled",
            created_at=now,
        )
        self._append_event(
            session_id=run.session_id,
            run_id=run.run_id,
            event_type="SESSION_CANCELLED",
            from_status=expected_session_status,
            to_status=DiagnosticSessionStatus.CANCELLED.value,
            reason_code=reason_code,
            dedupe_key=f"run:{run.run_id}:session-cancelled",
            created_at=now,
        )
        return tuple(newly_terminal_approvals)

    def _set_failed_locked(
        self,
        run: _RunRecord,
        *,
        from_status: str,
        now: str,
        error_code: str,
    ) -> None:
        self._claim_profile_locked(run.session_id)
        session = self._registry.execute(
            "SELECT status, active_run_id FROM diagnostic_sessions WHERE session_id = ?",
            (run.session_id,),
        ).fetchone()
        if (
            session is None
            or session[1] != run.run_id
            or session[0]
            not in {
                DiagnosticSessionStatus.RUNNING.value,
                DiagnosticSessionStatus.CANCEL_REQUESTED.value,
            }
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        session_from_status = str(session[0])
        run_changed = self._registry.execute(
            """
            UPDATE diagnostic_runs
            SET status = 'FAILED', updated_at = ?, result_json = NULL,
                error_code = ?, current_interrupt_id = NULL
            WHERE run_id = ? AND status = ?
            """,
            (now, error_code, run.run_id, from_status),
        ).rowcount
        self._registry.execute(
            """
            UPDATE diagnostic_interrupts
            SET status = 'CANCELLED', consumed_at = ?
            WHERE run_id = ? AND status = 'PENDING'
            """,
            (now, run.run_id),
        )
        session_changed = self._registry.execute(
            """
            UPDATE diagnostic_sessions
            SET status = 'FAILED', active_run_id = NULL, updated_at = ?
            WHERE session_id = ? AND active_run_id = ? AND status = ?
            """,
            (now, run.session_id, run.run_id, session_from_status),
        ).rowcount
        if (run_changed, session_changed) != (1, 1):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        self._append_event(
            session_id=run.session_id,
            run_id=run.run_id,
            event_type="RUN_FAILED",
            from_status=from_status,
            to_status=DiagnosticRunStatus.FAILED.value,
            reason_code=error_code,
            dedupe_key=(
                f"run:{run.run_id}:failed{self._run_attempt_suffix_locked(run.run_id)}"
            ),
            created_at=now,
        )
        self._append_event(
            session_id=run.session_id,
            run_id=run.run_id,
            event_type="SESSION_FAILED",
            from_status=session_from_status,
            to_status=DiagnosticSessionStatus.FAILED.value,
            reason_code=error_code,
            dedupe_key=(
                f"run:{run.run_id}:session-failed"
                f"{self._run_attempt_suffix_locked(run.run_id)}"
            ),
            created_at=now,
        )

    def _commit_interrupt(
        self,
        run_id: str,
        interrupt_data: tuple[str, str, str],
        result: Mapping[str, Any],
        *,
        expected_profile: str,
        previous: Mapping[str, Any] | None = None,
        source: Mapping[str, Any] | None = None,
    ) -> str:
        interrupt_id, payload_json, digest = interrupt_data
        approval_id = self._new_token()
        nonce = self._new_token()
        while nonce == approval_id:
            nonce = self._new_token()
        approval_event_id = _event_token_digest(approval_id)
        agent_fields = self._validate_agent_checkpoint_state(
            result,
            allow_active=True,
            expected_profile=expected_profile,
        )
        try:
            interrupt_request = json.loads(payload_json)
        except Exception as exc:
            raise RunPersistenceError(SANITIZED_STORE_ERROR) from exc
        is_typed = effect_capabilities.is_effect_interrupt_request(interrupt_request)
        is_claim_typed = (type(interrupt_request) is dict
                          and interrupt_request.get("approval_kind") == "EFFECT_V2")
        if is_claim_typed:
            approval_id = secrets.token_hex(16)
            approval_event_id = _event_token_digest(approval_id)
        if (type(interrupt_request) is dict
                and str(interrupt_request.get("approval_kind", "")).startswith("EFFECT_")
                and not is_typed and not is_claim_typed):
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        if bool(agent_fields.get("effect_proposal")) != is_typed:
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        if is_typed and self._effect_mode != EFFECT_MODE_SYNTHETIC_RESTART:
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        if is_typed and agent_fields.get("effect_runtime") != {}:
            raise RunPersistenceError(SANITIZED_STORE_ERROR)
        count, limit, reason = self._execution_fields(result)
        now = _now_text()
        try:
            with self._committed_run_observations() as committed, self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                run = self._select_run_locked(run_id)
                if run is None:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                if run.status == DiagnosticRunStatus.CANCEL_REQUESTED.value:
                    self._set_cancelled_locked(
                        run,
                        from_status=run.status,
                        now=now,
                        reason_code=run.stop_reason or "USER_CANCELLED",
                    )
                    self._registry.commit()
                    committed.append((run_id, "RUN_CANCELLED", "robotops.persistence.run_state"))
                    return DiagnosticRunStatus.CANCELLED.value
                if run.status != DiagnosticRunStatus.RUNNING.value:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                material: effect_capabilities.EffectCapabilityMaterial | None = None
                claim_records = None
                claim_profile = self._claim_profile_locked(run.session_id)
                self._validate_claim_checkpoint_profile(result, claim_profile, terminal=False)
                self._validate_claim_effect_state_domain(result, run.session_id)
                if is_typed and self._execution_contract_locked(run.session_id) != LEGACY_EXECUTION_CONTRACT:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                if is_typed:
                    material = effect_capabilities.build_capability_material(
                        session_id=run.session_id,
                        run_id=run_id,
                        interrupt_id=interrupt_id,
                        approval_id=approval_id,
                        nonce=nonce,
                        issued_at_ms=self._typed_now_ms(),
                        state=result,
                        interrupt_request=interrupt_request,
                    )
                    payload_json = effect_capabilities.canonical_json(
                        material.public_payload
                    )
                    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
                elif is_claim_typed:
                    from app.agent.claim_effect_graph import build_claim_effect_intent
                    from app.services import effect_capabilities_v2, effect_records_v2
                    if (self._execution_contract_locked(run.session_id) != CLAIM_EXECUTION_CONTRACT
                            or previous is None or source is None
                            or result.get("claim_effect_runtime", {}) != {}
                            or result.get("claim_effect_decision", "") != ""):
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    merged, authority = self._prepare_authoritative_claim_result(
                        session_id=run.session_id, turn_number=run.turn_number,
                        previous=previous, source=source, raw_result=result, run_id=run_id,
                    )
                    safety = {name: merged[name] for name in (
                        "risk_level", "urgency", "need_human", "risk_check_failed",
                        "llm_failed", "tool_loop_failed",
                    )}
                    intent = build_claim_effect_intent(
                        authority=authority, verification=merged["claim_verification"], safety=safety)
                    if (not intent or intent != interrupt_request
                            or intent != merged.get("claim_effect_intent")):
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    # The framework checkpoint must contain the same independently
                    # merged facts before they become an executable approval.
                    with safe_span("robotops.checkpoint.restore") as span:
                        set_safe_run_correlation(span, run.run_id)
                        checkpoint = self._typed_checkpoint_locked(run)
                        for name in (*authority.keys(), *safety.keys()):
                            if name in merged and checkpoint.get(name) != merged[name]:
                                raise SessionStoreError(SANITIZED_STORE_ERROR)
                        set_safe_outcome(span, "COMPLETED")
                    material_v2 = effect_capabilities_v2.build_capability_material(
                        authority=authority, execution_contract=CLAIM_EXECUTION_CONTRACT,
                        verification=merged["claim_verification"], safety=safety,
                        interrupt_id=interrupt_id, approval_id=approval_id, nonce=nonce,
                        issued_at_ms=self._typed_now_ms(),
                    )
                    claim_records = effect_records_v2.build_pending_records(
                        material_v2, expected_authority=authority,
                        execution_contract=CLAIM_EXECUTION_CONTRACT, interrupt_id=interrupt_id,
                        approval_id=approval_id, expected_binding_sha256=material_v2["binding_sha256"],
                    )
                    payload_json = effect_capabilities.canonical_json({
                        **material_v2["public_payload"], "status": "PENDING",
                        "decision": None, "can_decide": True,
                    })
                    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
                elif result.get("claim_effect_intent"):
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                interrupt_inserted = self._registry.execute(
                    """
                    INSERT INTO diagnostic_interrupts (
                        approval_id, session_id, run_id, interrupt_id,
                        payload_json, payload_sha256, status, decision,
                        created_at, consumed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', NULL, ?, NULL)
                    """,
                    (
                        approval_id,
                        run.session_id,
                        run_id,
                        interrupt_id,
                        payload_json,
                        digest,
                        now,
                    ),
                ).rowcount
                typed_inserted = 1
                ledger_inserted = 1
                if claim_records is not None:
                    from app.services.effect_records_v2 import APPROVAL_COLUMNS, LEDGER_COLUMNS
                    for table, columns, record in (
                        ("diagnostic_claim_effect_approvals", APPROVAL_COLUMNS, claim_records["approval"]),
                        ("diagnostic_claim_effect_ledger", LEDGER_COLUMNS, claim_records["ledger"]),
                    ):
                        inserted = self._registry.execute(
                            "INSERT INTO " + table + " (" + ",".join(columns) + ") VALUES ("
                            + ",".join("?" for _ in columns) + ")",
                            tuple(record[name] for name in columns),
                        ).rowcount
                        if inserted != 1:
                            raise SessionStoreError(SANITIZED_STORE_ERROR)
                if material is not None:
                    typed_inserted = self._registry.execute(
                        """
                        INSERT INTO diagnostic_effect_approvals (
                            approval_id, session_id, run_id, interrupt_id,
                            approval_kind, effect_key, effect_type, source,
                            proposal_json, proposal_sha256, arguments_json,
                            arguments_sha256, risk_summary_json,
                            risk_summary_sha256, verifier_summary_json,
                            verifier_summary_sha256, verifier_result_sha256,
                            binding_json, binding_sha256, nonce_digest,
                            issued_at, expires_at, status, decision, consumed_at
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, 'PENDING', NULL, NULL
                        )
                        """,
                        (
                            material.approval_id,
                            material.session_id,
                            material.run_id,
                            material.interrupt_id,
                            material.approval_kind,
                            material.effect_key,
                            material.effect_type,
                            material.source,
                            material.proposal_json,
                            material.proposal_sha256,
                            material.arguments_json,
                            material.arguments_sha256,
                            material.risk_summary_json,
                            material.risk_summary_sha256,
                            material.verifier_summary_json,
                            material.verifier_summary_sha256,
                            material.verifier_result_sha256,
                            material.binding_json,
                            material.binding_sha256,
                            material.nonce_digest,
                            material.issued_at,
                            material.expires_at,
                        ),
                    ).rowcount
                    ledger_inserted = self._registry.execute(
                        """
                        INSERT INTO diagnostic_effect_ledger (
                            effect_key, approval_id, binding_sha256, effect_type,
                            source, arguments_json, arguments_sha256, status,
                            attempt_count, result_json, result_sha256,
                            verification_json, verification_sha256,
                            created_at, updated_at
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, 'PREPARED', 0,
                            NULL, NULL, NULL, NULL, ?, ?
                        )
                        """,
                        (
                            material.effect_key,
                            material.approval_id,
                            material.binding_sha256,
                            material.effect_type,
                            material.source,
                            material.arguments_json,
                            material.arguments_sha256,
                            material.issued_at,
                            material.issued_at,
                        ),
                    ).rowcount
                run_changed = self._registry.execute(
                    """
                    UPDATE diagnostic_runs
                    SET status = 'INTERRUPTED', updated_at = ?,
                        tool_iterations = ?, max_tool_iterations = ?, stop_reason = ?,
                        current_interrupt_id = ?
                    WHERE run_id = ? AND status = 'RUNNING'
                    """,
                    (now, count, limit, reason, interrupt_id, run_id),
                ).rowcount
                if (
                    interrupt_inserted,
                    typed_inserted,
                    ledger_inserted,
                    run_changed,
                ) != (1, 1, 1, 1):
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                session_changed = self._registry.execute(
                    """
                    UPDATE diagnostic_sessions
                    SET status = 'INTERRUPTED', updated_at = ?
                    WHERE session_id = ? AND active_run_id = ? AND status = 'RUNNING'
                    """,
                    (now, run.session_id, run_id),
                ).rowcount
                if session_changed != 1:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                self._append_event(
                    session_id=run.session_id,
                    run_id=run_id,
                    event_type="RUN_INTERRUPTED",
                    from_status=DiagnosticRunStatus.RUNNING.value,
                    to_status=DiagnosticRunStatus.INTERRUPTED.value,
                    dedupe_key=f"run:{run_id}:interrupt:{approval_event_id}",
                    created_at=now,
                )
                self._append_event(
                    session_id=run.session_id,
                    run_id=run_id,
                    event_type="SESSION_INTERRUPTED",
                    from_status=DiagnosticSessionStatus.RUNNING.value,
                    to_status=DiagnosticSessionStatus.INTERRUPTED.value,
                    dedupe_key=f"run:{run_id}:session-interrupt:{approval_event_id}",
                    created_at=now,
                )
                self._registry.commit()
                committed.append((run_id, "RUN_INTERRUPTED", "robotops.persistence.run_state"))
                if material is not None or claim_records is not None:
                    # Only a committed pending effect approval is observed here.
                    # This marker does not measure the transaction's latency.
                    with safe_span("robotops.approval.issued") as span:
                        set_safe_outcome(span, "COMPLETED")
                return DiagnosticRunStatus.INTERRUPTED.value
        except RunPersistenceError:
            raise
        except Exception as exc:
            raise RunPersistenceError(SANITIZED_STORE_ERROR) from exc

    def _upsert_draft_locked(
        self,
        run: _RunRecord,
        response: DiagnosticTurnResponse,
        state: Mapping[str, Any],
        now: str,
    ) -> None:
        """在 run 完成事务内创建或更新同一 draft_id。"""
        diagnosis = response.diagnosis.get("causes", [])
        solution_steps = response.diagnosis.get("steps", [])
        if (
            not isinstance(diagnosis, list)
            or not all(isinstance(item, str) for item in diagnosis)
            or not isinstance(solution_steps, list)
            or not all(isinstance(item, str) for item in solution_steps)
        ):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        fault_description = state.get("fault_description", "")
        if not isinstance(fault_description, str):
            raise SessionStoreError(SANITIZED_STORE_ERROR)
        if response.status is DiagnosticSessionStatus.AWAITING_USER:
            draft_status = DiagnosticDraftStatus.COLLECTING.value
        else:
            draft_status = DiagnosticDraftStatus.READY.value
        existing = self._registry.execute(
            """
            SELECT draft_id, version, created_at
            FROM diagnostic_draft_tickets WHERE session_id = ?
            """,
            (run.session_id,),
        ).fetchone()
        diagnosis_json = json.dumps(
            diagnosis, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        steps_json = json.dumps(
            solution_steps, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        if existing is None:
            inserted = self._registry.execute(
                """
                INSERT INTO diagnostic_draft_tickets (
                    draft_id, session_id, source_run_id, version, status,
                    device_model, fault_category, fault_description,
                    diagnosis_json, solution_steps_json, risk_level, need_human,
                    external_ticket_id, projection_note, created_at, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._new_token(),
                    run.session_id,
                    run.run_id,
                    draft_status,
                    str(response.diagnosis.get("device_model") or ""),
                    str(response.diagnosis.get("fault_category") or ""),
                    fault_description,
                    diagnosis_json,
                    steps_json,
                    str(response.diagnosis.get("risk_level") or ""),
                    int(response.diagnosis.get("need_human") is True),
                    None,
                    None,
                    now,
                    now,
                ),
            ).rowcount
            if inserted != 1:
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            return
        changed = self._registry.execute(
            """
            UPDATE diagnostic_draft_tickets
            SET source_run_id = ?, version = ?, status = ?, device_model = ?,
                fault_category = ?, fault_description = ?, diagnosis_json = ?,
                solution_steps_json = ?, risk_level = ?, need_human = ?,
                external_ticket_id = ?, projection_note = ?, updated_at = ?
            WHERE session_id = ? AND draft_id = ? AND version = ?
            """,
            (
                run.run_id,
                int(existing[1]) + 1,
                draft_status,
                str(response.diagnosis.get("device_model") or ""),
                str(response.diagnosis.get("fault_category") or ""),
                fault_description,
                diagnosis_json,
                steps_json,
                str(response.diagnosis.get("risk_level") or ""),
                int(response.diagnosis.get("need_human") is True),
                None,
                None,
                now,
                run.session_id,
                existing[0],
                existing[1],
            ),
        ).rowcount
        if changed != 1:
            raise SessionStoreError(SANITIZED_STORE_ERROR)

    def _observe_completed_commit(
        self,
        run_id: str,
        response: DiagnosticTurnResponse,
        state: Mapping[str, Any],
    ) -> str:
        with safe_span("robotops.persistence.finalize") as span:
            outcome = self._commit_completed(run_id, response, state)
            if outcome == DiagnosticRunStatus.COMPLETED.value:
                set_safe_persisted_event(span, "RUN_COMPLETED")
            elif outcome == DiagnosticRunStatus.CANCELLED.value:
                set_safe_persisted_event(span, "RUN_CANCELLED")
            set_safe_outcome(span, outcome)
            return outcome

    def _commit_completed(
        self,
        run_id: str,
        response: DiagnosticTurnResponse,
        state: Mapping[str, Any],
    ) -> str:
        result_json = response.model_dump_json()
        self._validate_agent_checkpoint_state(state, allow_active=False)
        count, limit, reason = self._execution_fields(state)
        now = response.updated_at.isoformat().replace("+00:00", "Z")
        try:
            with self._committed_run_observations() as committed, self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                run = self._select_run_locked(run_id)
                if run is None:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                if run.status == DiagnosticRunStatus.CANCEL_REQUESTED.value:
                    self._set_cancelled_locked(
                        run,
                        from_status=run.status,
                        now=now,
                        reason_code=run.stop_reason or "USER_CANCELLED",
                    )
                    self._registry.commit()
                    committed.append((run_id, "RUN_CANCELLED", "robotops.persistence.run_state"))
                    return DiagnosticRunStatus.CANCELLED.value
                if run.status != DiagnosticRunStatus.RUNNING.value:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                request_id = self._request_id_for_run_locked(run_id)
                claim_profile = self._claim_profile_locked(run.session_id)
                self._validate_claim_effect_state_domain(state, run.session_id)
                if claim_profile == ACTIVE_CLAIM_EVIDENCE_PROFILE:
                    from app.agent.claim_sources import CONTEXT_FIELDS
                    from app.services.claim_sessions import (
                        encode_claim_run_result, validate_claim_terminal_state,
                    )
                    message = self._claim_request_message_locked(run)
                    conversation = self._conversation(state)
                    inputs = [item["content"] for item in conversation if item["role"] == "user"]
                    if (not inputs or inputs[-1] != message
                            or "\n".join(inputs) != state.get("user_query")):
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    authority = {"session_id": run.session_id, "run_id": run_id,
                                 "request_id": request_id, "turn_number": run.turn_number,
                                 **{key: state.get(key) for key in CONTEXT_FIELDS}}
                    validate_claim_terminal_state(dict(state), expected_authority=authority)
                    if self._execution_contract_locked(run.session_id) == CLAIM_EXECUTION_CONTRACT:
                        from app.services.claim_effect_results import encode_claim_effect_run_result
                        action = self._validate_claim_effect_terminal_locked(run, state)
                        result_json = encode_claim_effect_run_result(
                            response.model_dump(mode="json"), authority=authority, action=action,
                        )
                    else:
                        result_json = encode_claim_run_result(
                            response.model_dump(mode="json"), authority=authority)
                else:
                    self._validate_claim_checkpoint_profile(state, claim_profile, terminal=True)
                self._append_message_locked(
                    session_id=run.session_id,
                    sequence_no=run.turn_number * 2,
                    turn_number=run.turn_number,
                    role=DiagnosticMessageRole.ASSISTANT,
                    kind=(
                        DiagnosticMessageKind.FOLLOW_UP
                        if response.status is DiagnosticSessionStatus.AWAITING_USER
                        else DiagnosticMessageKind.RESULT
                    ),
                    content=response.reply,
                    run_id=run_id,
                    request_id=request_id,
                    payload=response.model_dump(mode="json"),
                    created_at=now,
                )
                self._upsert_draft_locked(run, response, state, now)
                changed = self._registry.execute(
                    """
                    UPDATE diagnostic_runs
                    SET status = 'COMPLETED', updated_at = ?, result_json = ?,
                        tool_iterations = ?, max_tool_iterations = ?, stop_reason = ?,
                        current_interrupt_id = NULL
                    WHERE run_id = ? AND status = 'RUNNING'
                    """,
                    (now, result_json, count, limit, reason, run_id),
                ).rowcount
                if changed != 1:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                session_changed = self._registry.execute(
                    """
                    UPDATE diagnostic_sessions
                    SET status = ?, updated_at = ?, turn_count = ?, active_run_id = NULL
                    WHERE session_id = ? AND status = 'RUNNING' AND active_run_id = ?
                    """,
                    (
                        response.status.value,
                        now,
                        run.turn_number,
                        run.session_id,
                        run_id,
                    ),
                ).rowcount
                if session_changed != 1:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                self._append_event(
                    session_id=run.session_id,
                    run_id=run_id,
                    event_type="RUN_COMPLETED",
                    from_status=DiagnosticRunStatus.RUNNING.value,
                    to_status=DiagnosticRunStatus.COMPLETED.value,
                    reason_code=reason,
                    dedupe_key=f"run:{run_id}:completed",
                    created_at=now,
                )
                self._append_event(
                    session_id=run.session_id,
                    run_id=run_id,
                    event_type="SESSION_STABLE",
                    from_status=DiagnosticSessionStatus.RUNNING.value,
                    to_status=response.status.value,
                    reason_code=reason,
                    dedupe_key=f"run:{run_id}:session-stable",
                    created_at=now,
                )
                self._registry.commit()
                committed.append((run_id, "RUN_COMPLETED", "robotops.persistence.run_state"))
                committed.append((run_id, "RUN_COMPLETED", "robotops.ticket.draft_committed"))
                return DiagnosticRunStatus.COMPLETED.value
        except RunPersistenceError:
            raise
        except Exception as exc:
            raise RunPersistenceError(SANITIZED_STORE_ERROR) from exc

    def _finalize_cancelled(self, run_id: str) -> None:
        now = _now_text()
        try:
            with self._committed_run_observations() as committed, self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                run = self._select_run_locked(run_id)
                if run is None:
                    self._registry.rollback()
                    return
                if run.status == DiagnosticRunStatus.CANCELLED.value:
                    self._registry.commit()
                    return
                if run.status != DiagnosticRunStatus.CANCEL_REQUESTED.value:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                self._set_cancelled_locked(
                    run,
                    from_status=run.status,
                    now=now,
                    reason_code=run.stop_reason or "USER_CANCELLED",
                )
                self._registry.commit()
                committed.append((run_id, "RUN_CANCELLED", "robotops.persistence.run_state"))
        except SessionStoreError:
            raise
        except Exception as exc:
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

    def _fail_run(self, run_id: str, error_code: str) -> None:
        now = _now_text()
        try:
            with self._committed_run_observations() as committed, self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                run = self._select_run_locked(run_id)
                if run is None or run.status in {
                    DiagnosticRunStatus.COMPLETED.value,
                    DiagnosticRunStatus.CANCELLED.value,
                    DiagnosticRunStatus.FAILED.value,
                }:
                    self._registry.commit()
                    return
                if run.status == DiagnosticRunStatus.CANCEL_REQUESTED.value:
                    self._set_cancelled_locked(
                        run,
                        from_status=run.status,
                        now=now,
                        reason_code=run.stop_reason or "USER_CANCELLED",
                    )
                    self._registry.commit()
                    committed.append((run_id, "RUN_CANCELLED", "robotops.persistence.run_state"))
                    return
                previous_status = run.status
                self._set_failed_locked(
                    run,
                    from_status=previous_status,
                    now=now,
                    error_code=error_code,
                )
                self._registry.commit()
                committed.append((run_id, "RUN_FAILED", "robotops.persistence.run_state"))
        except Exception:
            pass

    def _execute_initial_run(
        self,
        run_id: str,
        record: _SessionRecord,
        message: str,
        control: RunControl,
    ) -> None:
        with safe_span("robotops.run.initial") as span:
            set_safe_run_correlation(span, run_id)
            outcome = self._execute_initial_run_body(run_id, record, message, control)
            set_safe_outcome(span, outcome)

    def _observed_run_outcome(self, run_id: str) -> str:
        try:
            with self._registry_lock:
                row = self._registry.execute(
                    "SELECT status FROM diagnostic_runs WHERE run_id = ?", (run_id,),
                ).fetchone()
            return str(row[0]) if row is not None else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def _execute_initial_run_body(
        self,
        run_id: str,
        record: _SessionRecord,
        message: str,
        control: RunControl,
    ) -> str:
        try:
            if not self._transition_run_to_running(run_id):
                return self._observed_run_outcome(run_id)
            try:
                previous = self._load_previous_state(record)
                prepared = self._prepare_state(previous, message,
                                               claim_profile=record.claim_evidence_profile)
            except SessionStoreError as exc:
                raise RunCheckpointError(SANITIZED_STORE_ERROR) from exc
            config = self._config(record.thread_id)
            result = self._invoke_async_graph(
                prepared,
                config=config,
                session_id=record.session_id,
                run_id=run_id,
                approval_id=None,
                control=control,
            )
            if not isinstance(result, Mapping):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            interrupt_data = self._extract_interrupt(result)
            if interrupt_data is not None:
                return self._commit_interrupt(
                    run_id,
                    interrupt_data,
                    result,
                    expected_profile=prepared["tool_capability_profile"],
                    previous=previous, source=prepared,
                )
            response, persisted = self._materialize_async_result(
                run_id=run_id,
                session_id=record.session_id,
                turn_number=record.turn_count + 1,
                previous=previous,
                source=prepared,
                raw_result=result,
                config=config,
            )
            return self._observe_completed_commit(run_id, response, persisted)
        except GraphRecursionError:
            self._fail_run(run_id, "EXECUTION_LIMIT_EXCEEDED")
        except GraphDrained:
            self._finalize_cancelled(run_id)
        except RunCheckpointError:
            self._fail_run(run_id, "RUN_CHECKPOINT_FAILED")
        except RunPersistenceError:
            self._fail_run(run_id, "RUN_PERSISTENCE_FAILED")
        except Exception:
            self._fail_run(run_id, "RUN_EXECUTION_FAILED")
        finally:
            self._unregister_control(run_id, control)
        return self._observed_run_outcome(run_id)

    def resume_run(
        self,
        session_id: str,
        run_id: str,
        *,
        interrupt_id: str,
        approval_id: str,
        decision: ResumeDecision,
    ) -> tuple[DiagnosticRunResponse, bool]:
        """以服务端持久绑定恢复同一逻辑 run；相同重放只读回原状态。"""
        self._validate_session_id(session_id)
        self._validate_run_id(run_id)
        if not isinstance(interrupt_id, str) or not interrupt_id:
            raise RunConflictError(RUN_CONFLICT)
        if not isinstance(approval_id, str) or not SESSION_ID_PATTERN.fullmatch(
            approval_id
        ):
            raise RunConflictError(RUN_CONFLICT)
        if not isinstance(decision, ResumeDecision):
            raise RunConflictError(RUN_CONFLICT)
        control = RunControl()
        control_registered = False
        expired_committed = False
        try:
            with self._committed_run_observations() as committed, self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                session_status, run = self._bound_run_locked(session_id, run_id)
                row = self._registry.execute(
                    """
                    SELECT session_id, run_id, interrupt_id, payload_json,
                           payload_sha256, status, decision, consumed_at
                    FROM diagnostic_interrupts WHERE approval_id = ?
                    """,
                    (approval_id,),
                ).fetchone()
                if row is None or (row[0], row[1], row[2]) != (
                    session_id,
                    run_id,
                    interrupt_id,
                ):
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                payload_json = str(row[3])
                if not secrets.compare_digest(
                    hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                    str(row[4]),
                ):
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                try:
                    parent_payload = json.loads(payload_json)
                except Exception as exc:
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
                if not isinstance(parent_payload, dict):
                    self._registry.rollback()
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                stored_status = str(row[5])
                stored_decision = row[6]
                parent = {
                    "approval_id": approval_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "interrupt_id": interrupt_id,
                    "payload_json": payload_json,
                    "payload_sha256": str(row[4]),
                    "status": stored_status,
                    "decision": stored_decision,
                    "consumed_at": row[7],
                }
                effect_family = self._effect_family_locked(parent, run)
                typed_marker = effect_family == "EFFECT_V1"
                claim_marker = effect_family == "EFFECT_V2"
                typed_approval = None
                claim_validated = None
                if typed_marker:
                    typed_approval, _public = self._validate_typed_approval_locked(
                        parent=parent,
                        run=run,
                        require_checkpoint=stored_decision is None,
                    )
                elif claim_marker:
                    claim_validated = self._validate_claim_effect_binding_locked(
                        approval_id,
                        require_checkpoint=(stored_status == "PENDING" and stored_decision is None),
                    )
                if stored_decision is not None:
                    if stored_decision != decision.value:
                        self._registry.rollback()
                        raise RunConflictError(RUN_CONFLICT)
                    if claim_marker and (
                        claim_validated is None
                        or claim_validated["approval"]["status"] != "CONSUMED"
                        or claim_validated["approval"]["decision"] != decision.value
                    ):
                        self._registry.rollback()
                        raise SessionStoreError(SANITIZED_STORE_ERROR)
                    if typed_marker:
                        if (
                            typed_approval is None
                            or typed_approval["status"] != "CONSUMED"
                            or typed_approval["decision"] != decision.value
                        ):
                            self._registry.rollback()
                            raise SessionStoreError(SANITIZED_STORE_ERROR)
                        self._registry.commit()
                        return self.get_run(session_id, run_id), False
                if run.status in {
                    DiagnosticRunStatus.CANCEL_REQUESTED.value,
                    DiagnosticRunStatus.CANCELLED.value,
                } or session_status in {
                    DiagnosticSessionStatus.CANCEL_REQUESTED.value,
                    DiagnosticSessionStatus.CANCELLED.value,
                }:
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                if stored_decision is not None:
                    self._registry.commit()
                    return self.get_run(session_id, run_id), False
                if (
                    stored_status != "PENDING"
                    or run.status != DiagnosticRunStatus.INTERRUPTED.value
                    or session_status != DiagnosticSessionStatus.INTERRUPTED.value
                    or run.current_interrupt_id != interrupt_id
                ):
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                now = _now_text()
                typed_consumed = 1
                if claim_marker:
                    if (claim_validated is None
                            or claim_validated["approval"]["status"] != "PENDING"):
                        self._registry.rollback()
                        raise RunConflictError(RUN_CONFLICT)
                    claim_approval = claim_validated["approval"]
                    now_ms = self._typed_now_ms()
                    if now_ms < int(claim_approval["issued_at"]):
                        self._registry.rollback()
                        raise RunConflictError(RUN_CONFLICT)
                    if now_ms >= int(claim_approval["expires_at"]):
                        expired = self._registry.execute(
                            "UPDATE diagnostic_claim_effect_approvals "
                            "SET status='EXPIRED', consumed_at=? "
                            "WHERE approval_id=? AND status='PENDING' AND decision IS NULL",
                            (now_ms, approval_id),
                        ).rowcount
                        if expired != 1:
                            self._registry.rollback()
                            raise SessionStoreError(SANITIZED_STORE_ERROR)
                        self._registry.commit()
                        expired_committed = True
                        raise RunConflictError(RUN_CONFLICT)
                    if (decision is ResumeDecision.APPROVE
                            and (self._effect_mode != EFFECT_MODE_SYNTHETIC_RESTART
                                 or self._synthetic_effect_adapter is None)):
                        self._registry.rollback()
                        raise RunConflictError(RUN_CONFLICT)
                    typed_consumed = self._registry.execute(
                        "UPDATE diagnostic_claim_effect_approvals "
                        "SET status='CONSUMED', decision=?, consumed_at=? "
                        "WHERE approval_id=? AND status='PENDING' AND decision IS NULL",
                        (decision.value, now_ms, approval_id),
                    ).rowcount
                if typed_marker:
                    if typed_approval is None or typed_approval["status"] != "PENDING":
                        self._registry.rollback()
                        raise RunConflictError(RUN_CONFLICT)
                    now_ms = self._typed_now_ms()
                    if now_ms < int(typed_approval["issued_at"]):
                        self._registry.rollback()
                        raise RunConflictError(RUN_CONFLICT)
                    if now_ms >= int(typed_approval["expires_at"]):
                        expired = self._registry.execute(
                            """
                            UPDATE diagnostic_effect_approvals
                            SET status = 'EXPIRED', consumed_at = ?
                            WHERE approval_id = ? AND status = 'PENDING'
                              AND decision IS NULL
                            """,
                            (now_ms, approval_id),
                        ).rowcount
                        if expired != 1:
                            self._registry.rollback()
                            raise SessionStoreError(SANITIZED_STORE_ERROR)
                        self._registry.commit()
                        expired_committed = True
                        raise RunConflictError(RUN_CONFLICT)
                    if (
                        decision is ResumeDecision.APPROVE
                        and self._effect_mode != EFFECT_MODE_SYNTHETIC_RESTART
                    ):
                        self._registry.rollback()
                        raise RunConflictError(RUN_CONFLICT)
                    typed_consumed = self._registry.execute(
                        """
                        UPDATE diagnostic_effect_approvals
                        SET status = 'CONSUMED', decision = ?, consumed_at = ?
                        WHERE approval_id = ? AND status = 'PENDING'
                          AND decision IS NULL
                        """,
                        (decision.value, now_ms, approval_id),
                    ).rowcount
                approval_event_id = _event_token_digest(approval_id)
                consumed = self._registry.execute(
                    """
                    UPDATE diagnostic_interrupts
                    SET status = 'CONSUMED', decision = ?, consumed_at = ?
                    WHERE approval_id = ? AND status = 'PENDING' AND decision IS NULL
                    """,
                    (decision.value, now, approval_id),
                ).rowcount
                claimed = self._registry.execute(
                    """
                    UPDATE diagnostic_runs
                    SET status = 'RUNNING', updated_at = ?, current_interrupt_id = NULL
                    WHERE run_id = ? AND status = 'INTERRUPTED'
                    """,
                    (now, run_id),
                ).rowcount
                session_claimed = self._registry.execute(
                    """
                    UPDATE diagnostic_sessions SET status = 'RUNNING', updated_at = ?
                    WHERE session_id = ? AND status = 'INTERRUPTED'
                      AND active_run_id = ?
                    """,
                    (now, session_id, run_id),
                ).rowcount
                if (typed_consumed, consumed, claimed, session_claimed) != (
                    1,
                    1,
                    1,
                    1,
                ):
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                self._append_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="RUN_RESUMED",
                    from_status=DiagnosticRunStatus.INTERRUPTED.value,
                    to_status=DiagnosticRunStatus.RUNNING.value,
                    reason_code=decision.value,
                    dedupe_key=f"run:{run_id}:resume:{approval_event_id}",
                    created_at=now,
                )
                self._append_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="SESSION_RUNNING",
                    from_status=DiagnosticSessionStatus.INTERRUPTED.value,
                    to_status=DiagnosticSessionStatus.RUNNING.value,
                    reason_code=decision.value,
                    dedupe_key=f"run:{run_id}:session-resume:{approval_event_id}",
                    created_at=now,
                )
                self._register_control(run_id, control)
                control_registered = True
                self._registry.commit()
                committed.append((run_id, "RUN_RESUMED", "robotops.persistence.run_state"))
        except RunConflictError:
            if expired_committed:
                # The SQLite transaction is committed and its lock has exited.
                # This marker never measures the transaction or the rejected request.
                with safe_span("robotops.approval.expired") as span:
                    set_safe_outcome(span, "COMPLETED")
            if control_registered:
                self._unregister_control(run_id, control)
            raise
        except (
            InvalidSessionIdError,
            InvalidRunIdError,
            SessionNotFoundError,
            RunNotFoundError,
            SessionStoreError,
        ):
            if control_registered:
                self._unregister_control(run_id, control)
            raise
        except Exception as exc:
            if control_registered:
                self._unregister_control(run_id, control)
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

        if typed_marker or claim_marker:
            # A post-commit marker: it does not measure the transaction's latency.
            with safe_span("robotops.approval.consumed") as span:
                set_safe_approval_decision(span, decision.value)
                set_safe_outcome(span, "COMPLETED")

        try:
            self._schedule(
                run_id,
                control,
                self._execute_resume_run,
                run_id,
                interrupt_id,
                approval_id,
                decision,
            )
        except Exception as exc:
            self._unregister_control(run_id, control)
            self._fail_run(run_id, "RESUME_SCHEDULING_FAILED")
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc
        return self.get_run(session_id, run_id), True

    def _execute_resume_run(
        self,
        run_id: str,
        interrupt_id: str,
        approval_id: str,
        decision: ResumeDecision,
        control: RunControl,
    ) -> None:
        with safe_span("robotops.run.resume") as span:
            set_safe_run_correlation(span, run_id)
            outcome = self._execute_resume_run_body(
                run_id, interrupt_id, approval_id, decision, control,
            )
            set_safe_outcome(span, outcome)

    def _execute_resume_run_body(
        self,
        run_id: str,
        interrupt_id: str,
        approval_id: str,
        decision: ResumeDecision,
        control: RunControl,
    ) -> str:
        try:
            with self._committed_run_observations() as committed, self._registry_lock:
                run = self._select_run_locked(run_id)
                if run is None:
                    return "UNKNOWN"
                row = self._registry.execute(
                    "SELECT thread_id FROM diagnostic_sessions WHERE session_id = ?",
                    (run.session_id,),
                ).fetchone()
                if row is None:
                    raise SessionStoreError(SANITIZED_STORE_ERROR)
                self._claim_profile_locked(run.session_id)
                thread_id = str(row[0])
                if run.status == DiagnosticRunStatus.CANCEL_REQUESTED.value:
                    # This outer read lock is still owned: defer the committed
                    # fact until that lock exits, without changing cancellation.
                    self._finalize_cancelled(run_id)
                    return "CANCELLED"
                if run.status != DiagnosticRunStatus.RUNNING.value:
                    return run.status
            config = self._config(thread_id)
            with safe_span("robotops.checkpoint.restore") as span:
                try:
                    snapshot = self._graph.get_state(config)
                except Exception as exc:
                    raise RunCheckpointError(SANITIZED_STORE_ERROR) from exc
                if not isinstance(snapshot.values, Mapping) or not snapshot.values:
                    raise RunCheckpointError(SANITIZED_STORE_ERROR)
                previous = dict(snapshot.values)
                try:
                    resume_update = self._authoritative_resume_update(
                        previous, claim_profile=self._claim_profile_locked(run.session_id),
                    )
                except SessionStoreError as exc:
                    raise RunCheckpointError(SANITIZED_STORE_ERROR) from exc
                previous.update(resume_update)
                set_safe_outcome(span, "COMPLETED")
            result = self._invoke_async_graph(
                Command(
                    update=resume_update,
                    resume={interrupt_id: decision.value},
                ),
                config=config,
                session_id=run.session_id,
                run_id=run_id,
                approval_id=approval_id,
                control=control,
            )
            if not isinstance(result, Mapping):
                raise SessionStoreError(SANITIZED_STORE_ERROR)
            interrupt_data = self._extract_interrupt(result)
            if interrupt_data is not None:
                return self._commit_interrupt(
                    run_id,
                    interrupt_data,
                    result,
                    expected_profile=previous["tool_capability_profile"],
                    previous=previous, source=previous,
                )
            response, persisted = self._materialize_async_result(
                run_id=run_id,
                session_id=run.session_id,
                turn_number=run.turn_number,
                previous=previous,
                source=previous,
                raw_result=result,
                config=config,
            )
            return self._observe_completed_commit(run_id, response, persisted)
        except GraphRecursionError:
            self._fail_run(run_id, "EXECUTION_LIMIT_EXCEEDED")
        except GraphDrained:
            self._finalize_cancelled(run_id)
        except RunCheckpointError:
            self._fail_run(run_id, "RUN_CHECKPOINT_FAILED")
        except RunPersistenceError:
            self._fail_run(run_id, "RUN_PERSISTENCE_FAILED")
        except Exception:
            self._fail_run(run_id, "RUN_EXECUTION_FAILED")
        finally:
            self._unregister_control(run_id, control)
        return self._observed_run_outcome(run_id)

    def cancel_run(
        self, session_id: str, run_id: str
    ) -> DiagnosticRunResponse:
        """精确取消一个 run；重复请求不新增事件也不重复发信号。"""
        self._validate_session_id(session_id)
        self._validate_run_id(run_id)
        should_signal = False
        approval_terminal_transitions: tuple[str, ...] = ()
        committed: list[tuple[str, str, str]] = []
        try:
            with self._registry_transaction():
                self._registry.execute("BEGIN IMMEDIATE")
                session_status, run = self._bound_run_locked(session_id, run_id)
                if run.status == DiagnosticRunStatus.CANCELLED.value:
                    self._registry.commit()
                    return self._run_response_locked(session_status, run)
                if run.status in {
                    DiagnosticRunStatus.COMPLETED.value,
                    DiagnosticRunStatus.FAILED.value,
                }:
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                if run.status == DiagnosticRunStatus.CANCEL_REQUESTED.value:
                    self._registry.commit()
                    return self._run_response_locked(session_status, run)
                now = _now_text()
                if run.status in {
                    DiagnosticRunStatus.QUEUED.value,
                    DiagnosticRunStatus.INTERRUPTED.value,
                }:
                    approval_terminal_transitions = self._set_cancelled_locked(
                        run,
                        from_status=run.status,
                        now=now,
                        reason_code="USER_CANCELLED",
                    )
                elif run.status == DiagnosticRunStatus.RUNNING.value:
                    run_changed = self._registry.execute(
                        """
                        UPDATE diagnostic_runs
                        SET status = 'CANCEL_REQUESTED', updated_at = ?,
                            stop_reason = 'USER_CANCELLED'
                        WHERE run_id = ? AND status = 'RUNNING'
                        """,
                        (now, run_id),
                    ).rowcount
                    session_changed = self._registry.execute(
                        """
                        UPDATE diagnostic_sessions
                        SET status = 'CANCEL_REQUESTED', updated_at = ?
                        WHERE session_id = ? AND status = 'RUNNING'
                          AND active_run_id = ?
                        """,
                        (now, session_id, run_id),
                    ).rowcount
                    if (run_changed, session_changed) != (1, 1):
                        self._registry.rollback()
                        raise RunConflictError(RUN_CONFLICT)
                    self._append_event(
                        session_id=session_id,
                        run_id=run_id,
                        event_type="RUN_CANCEL_REQUESTED",
                        from_status=DiagnosticRunStatus.RUNNING.value,
                        to_status=DiagnosticRunStatus.CANCEL_REQUESTED.value,
                        reason_code="USER_CANCELLED",
                        dedupe_key=f"run:{run_id}:cancel-requested",
                        created_at=now,
                    )
                    self._append_event(
                        session_id=session_id,
                        run_id=run_id,
                        event_type="SESSION_CANCEL_REQUESTED",
                        from_status=DiagnosticSessionStatus.RUNNING.value,
                        to_status=DiagnosticSessionStatus.CANCEL_REQUESTED.value,
                        reason_code="USER_CANCELLED",
                        dedupe_key=f"run:{run_id}:session-cancel-requested",
                        created_at=now,
                    )
                    should_signal = True
                else:
                    self._registry.rollback()
                    raise RunConflictError(RUN_CONFLICT)
                self._registry.commit()
                committed.append((
                    run_id,
                    "RUN_CANCEL_REQUESTED" if run.status == DiagnosticRunStatus.RUNNING.value
                    else "RUN_CANCELLED",
                    "robotops.persistence.run_state",
                ))
        except (
            InvalidSessionIdError,
            InvalidRunIdError,
            SessionNotFoundError,
            RunNotFoundError,
            RunConflictError,
        ):
            raise
        except Exception as exc:
            if isinstance(exc, SessionStoreError):
                raise
            raise SessionStoreError(SANITIZED_STORE_ERROR) from exc

        # The cancellation transaction has committed and released its lock.
        # The marker describes only newly terminal typed approval rows.
        for approval_status in approval_terminal_transitions:
            if approval_status == "EXPIRED":
                span_name = "robotops.approval.expired"
            elif approval_status == "CANCELLED":
                span_name = "robotops.approval.cancelled"
            else:
                continue
            with safe_span(span_name) as span:
                set_safe_outcome(span, "COMPLETED")
        if should_signal:
            control = self._process_control(run_id)
            if control is not None:
                control.request_drain("user_cancelled")
        for observed_run_id, event_type, name in committed:
            self._observe_committed_run(observed_run_id, event_type, name=name)
        return self.get_run(session_id, run_id)
