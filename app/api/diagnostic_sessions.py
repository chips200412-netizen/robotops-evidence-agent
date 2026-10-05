"""MOD-019～021：服务端权威诊断会话、run、消息与草稿 API。"""

from __future__ import annotations

import threading

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Response, status

from app.agent.tool_loop import SYNTHETIC_EVIDENCE_TOOL_PROFILE
from app.core.config import settings
from app.schemas.diagnostic_sessions import (
    CancelDiagnosticRunRequest,
    CreateDiagnosticSessionRequest,
    CreateDiagnosticSessionResponse,
    DiagnosticDraftTicketResponse,
    DiagnosticMessagesResponse,
    DiagnosticRunRequest,
    DiagnosticRunResponse,
    DiagnosticRunStatus,
    DiagnosticSessionResponse,
    DiagnosticSimilarHistoryResponse,
    DiagnosticTurnRequest,
    DiagnosticTurnResponse,
    ReopenDiagnosticSessionRequest,
    RetryDiagnosticDraftProjectionRequest,
    RetryDiagnosticRunRequest,
    ResumeDiagnosticRunRequest,
)
from app.services.diagnostic_sessions import (
    DRAFT_NOT_FOUND,
    DRAFT_CONFLICT,
    REQUEST_NOT_FOUND,
    RUN_CONFLICT,
    RUN_NOT_FOUND,
    SANITIZED_STORE_ERROR,
    SESSION_CONFLICT,
    SESSION_NOT_FOUND,
    ACTIVE_CLAIM_EVIDENCE_PROFILE,
    CLAIM_EXECUTION_CONTRACT,
    EFFECT_MODE_SYNTHETIC_RESTART,
    DiagnosticSessionService,
    DraftConflictError,
    DraftNotFoundError,
    InvalidDraftIdError,
    InvalidRunIdError,
    InvalidSessionIdError,
    RunConflictError,
    RunNotFoundError,
    RequestNotFoundError,
    SessionConflictError,
    SessionNotFoundError,
    SessionStoreError,
)


router = APIRouter(prefix="/diagnostic-sessions", tags=["diagnostic-sessions"])
_default_service: DiagnosticSessionService | None = None
_default_service_lock = threading.Lock()


def get_diagnostic_session_service() -> DiagnosticSessionService:
    """首次访问新接口时才打开 SQLite，不影响旧 `/chat` 的导入行为。"""
    global _default_service
    if _default_service is None:
        with _default_service_lock:
            if _default_service is None:
                try:
                    _default_service = DiagnosticSessionService(
                        settings.diagnostic_session_db_path,
                        enable_agent_tool_loop=True,
                        tool_capability_profile=SYNTHETIC_EVIDENCE_TOOL_PROFILE,
                        effect_mode=EFFECT_MODE_SYNTHETIC_RESTART,
                        claim_evidence_profile=ACTIVE_CLAIM_EVIDENCE_PROFILE,
                        execution_contract=CLAIM_EXECUTION_CONTRACT,
                    )
                except SessionStoreError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=SANITIZED_STORE_ERROR,
                    ) from exc
    return _default_service


def close_default_diagnostic_session_service() -> None:
    """应用关闭时释放默认 SQLite 连接，并允许同进程干净重建。"""
    global _default_service
    with _default_service_lock:
        service = _default_service
        _default_service = None
    if service is not None:
        service.close()


@router.post(
    "",
    response_model=CreateDiagnosticSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_diagnostic_session(
    payload: CreateDiagnosticSessionRequest | None = Body(default=None),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> CreateDiagnosticSessionResponse:
    try:
        return service.create_session(payload.request_id if payload else None)
    except SessionStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=SANITIZED_STORE_ERROR,
        ) from exc


@router.post("/{session_id}/turns", response_model=DiagnosticTurnResponse)
def submit_diagnostic_turn(
    payload: DiagnosticTurnRequest,
    session_id: str = Path(
        min_length=32,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]{32}$",
    ),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticTurnResponse:
    try:
        return service.submit_turn(session_id, payload.message, payload.request_id)
    except InvalidSessionIdError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY) from exc
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=SESSION_NOT_FOUND
        ) from exc
    except SessionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=SESSION_CONFLICT
        ) from exc
    except SessionStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=SANITIZED_STORE_ERROR,
        ) from exc


def _session_path():
    return Path(
        min_length=32,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]{32}$",
    )


def _run_path():
    return Path(
        min_length=32,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]{32}$",
    )


def _request_path():
    return Path(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._~-]+$",
    )


def _draft_path():
    return Path(
        min_length=32,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]{32}$",
    )


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (InvalidSessionIdError, InvalidRunIdError, InvalidDraftIdError)):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
    if isinstance(exc, SessionNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=SESSION_NOT_FOUND)
    if isinstance(exc, RunNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=RUN_NOT_FOUND)
    if isinstance(exc, RequestNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=REQUEST_NOT_FOUND
        )
    if isinstance(exc, DraftNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=DRAFT_NOT_FOUND)
    if isinstance(exc, DraftConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=DRAFT_CONFLICT)
    if isinstance(exc, (SessionConflictError, RunConflictError)):
        detail = RUN_CONFLICT if isinstance(exc, RunConflictError) else SESSION_CONFLICT
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=SANITIZED_STORE_ERROR,
    )


_PUBLIC_ERRORS = (
    InvalidSessionIdError,
    InvalidRunIdError,
    InvalidDraftIdError,
    SessionNotFoundError,
    RunNotFoundError,
    RequestNotFoundError,
    DraftNotFoundError,
    DraftConflictError,
    SessionConflictError,
    RunConflictError,
    SessionStoreError,
)


def _mark_run_response_no_store(response: Response) -> None:
    """Run envelope may contain bearer approval/effect data and must not cache."""
    response.headers["Cache-Control"] = "no-store"


@router.get("/{session_id}", response_model=DiagnosticSessionResponse)
def get_diagnostic_session(
    session_id: str = _session_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticSessionResponse:
    try:
        return service.get_session(session_id)
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc


@router.post(
    "/{session_id}/runs",
    response_model=DiagnosticRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_diagnostic_run(
    payload: DiagnosticRunRequest,
    response: Response,
    session_id: str = _session_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticRunResponse:
    try:
        result, created = service.submit_run(
            session_id, payload.message, payload.request_id
        )
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc
    response.headers["Location"] = (
        f"/diagnostic-sessions/{session_id}/runs/{result.run_id}"
    )
    response.status_code = (
        status.HTTP_202_ACCEPTED
        if created
        or result.status
        in {
            DiagnosticRunStatus.QUEUED,
            DiagnosticRunStatus.RUNNING,
            DiagnosticRunStatus.INTERRUPTED,
            DiagnosticRunStatus.CANCEL_REQUESTED,
        }
        else status.HTTP_200_OK
    )
    _mark_run_response_no_store(response)
    return result


@router.get(
    "/{session_id}/requests/{request_id}", response_model=DiagnosticRunResponse
)
def get_diagnostic_request(
    response: Response,
    session_id: str = _session_path(),
    request_id: str = _request_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticRunResponse:
    try:
        result = service.get_run_by_request(session_id, request_id)
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc
    _mark_run_response_no_store(response)
    return result


@router.get("/{session_id}/messages", response_model=DiagnosticMessagesResponse)
def get_diagnostic_messages(
    session_id: str = _session_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticMessagesResponse:
    try:
        return service.list_messages(session_id)
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc


@router.get(
    "/{session_id}/draft-ticket", response_model=DiagnosticDraftTicketResponse
)
def get_diagnostic_draft_ticket(
    session_id: str = _session_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticDraftTicketResponse:
    try:
        return service.get_draft_ticket(session_id)
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc


@router.get(
    "/{session_id}/similar-history", response_model=DiagnosticSimilarHistoryResponse
)
def get_diagnostic_similar_history(
    session_id: str = _session_path(),
    limit: int = Query(default=3, ge=1, le=20),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticSimilarHistoryResponse:
    try:
        return service.get_similar_history(session_id, limit=limit)
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc


@router.post(
    "/{session_id}/draft-ticket/{draft_id}/retry",
    response_model=DiagnosticDraftTicketResponse,
)
def retry_diagnostic_draft_projection(
    payload: RetryDiagnosticDraftProjectionRequest,
    session_id: str = _session_path(),
    draft_id: str = _draft_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticDraftTicketResponse:
    try:
        return service.retry_draft_projection(
            session_id, draft_id, version=payload.version
        )
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc


@router.get(
    "/{session_id}/runs/{run_id}", response_model=DiagnosticRunResponse
)
def get_diagnostic_run(
    response: Response,
    session_id: str = _session_path(),
    run_id: str = _run_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticRunResponse:
    try:
        result = service.get_run(session_id, run_id)
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc
    _mark_run_response_no_store(response)
    return result


@router.post(
    "/{session_id}/runs/{run_id}/retry", response_model=DiagnosticRunResponse
)
def retry_diagnostic_run(
    response: Response,
    payload: RetryDiagnosticRunRequest | None = Body(default=None),
    session_id: str = _session_path(),
    run_id: str = _run_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticRunResponse:
    del payload
    try:
        result, scheduled = service.retry_run(session_id, run_id)
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc
    response.status_code = (
        status.HTTP_202_ACCEPTED
        if scheduled
        or result.status
        in {
            DiagnosticRunStatus.QUEUED,
            DiagnosticRunStatus.RUNNING,
            DiagnosticRunStatus.INTERRUPTED,
            DiagnosticRunStatus.CANCEL_REQUESTED,
        }
        else status.HTTP_200_OK
    )
    response.headers["Location"] = (
        f"/diagnostic-sessions/{session_id}/runs/{result.run_id}"
    )
    _mark_run_response_no_store(response)
    return result


@router.post(
    "/{session_id}/runs/{run_id}/resume", response_model=DiagnosticRunResponse
)
def resume_diagnostic_run(
    payload: ResumeDiagnosticRunRequest,
    response: Response,
    session_id: str = _session_path(),
    run_id: str = _run_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticRunResponse:
    try:
        result, accepted = service.resume_run(
            session_id,
            run_id,
            interrupt_id=payload.interrupt_id,
            approval_id=payload.approval_id,
            decision=payload.decision,
        )
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc
    response.status_code = (
        status.HTTP_202_ACCEPTED if accepted else status.HTTP_200_OK
    )
    _mark_run_response_no_store(response)
    return result


@router.post(
    "/{session_id}/runs/{run_id}/cancel", response_model=DiagnosticRunResponse
)
def cancel_diagnostic_run(
    response: Response,
    payload: CancelDiagnosticRunRequest | None = Body(default=None),
    session_id: str = _session_path(),
    run_id: str = _run_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticRunResponse:
    del payload
    try:
        result = service.cancel_run(session_id, run_id)
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc
    response.status_code = (
        status.HTTP_202_ACCEPTED
        if result.status is DiagnosticRunStatus.CANCEL_REQUESTED
        else status.HTTP_200_OK
    )
    _mark_run_response_no_store(response)
    return result


@router.post(
    "/{session_id}/reopen",
    response_model=DiagnosticSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def reopen_diagnostic_session(
    response: Response,
    payload: ReopenDiagnosticSessionRequest | None = Body(default=None),
    session_id: str = _session_path(),
    service: DiagnosticSessionService = Depends(get_diagnostic_session_service),
) -> DiagnosticSessionResponse:
    del payload
    try:
        result, created = service.reopen_session(session_id)
    except _PUBLIC_ERRORS as exc:
        raise _http_error(exc) from exc
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    response.headers["Location"] = f"/diagnostic-sessions/{result.session_id}"
    return result
