from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.diagnostic_sessions import (
    close_default_diagnostic_session_service,
    router as diagnostic_sessions_router,
)
from app.api.devices import router as devices_router
from app.api.documents import router as documents_router
from app.api.tickets import router as tickets_router
from app.observability.local_otlp import start_local_otlp, stop_local_otlp

@asynccontextmanager
async def lifespan(_app: FastAPI):
    local_provider = start_local_otlp()
    try:
        yield
    finally:
        try:
            close_default_diagnostic_session_service()
        finally:
            stop_local_otlp(local_provider)


app = FastAPI(
    title="RobotOps Evidence Agent API",
    description="工业机器人故障诊断与安全处置平台（LangGraph + RAG）",
    version="0.6.0",
    lifespan=lifespan,
)

_EVIDENCE_POST_PATHS = {
    "/documents/evidence-claims",
    "/documents/evidence-search",
    "/documents/evidence-rerank",
}


@app.exception_handler(RequestValidationError)
async def scoped_request_validation_error(request: Request, exc: RequestValidationError):
    """Do not echo rejected evidence endpoint input, including invalid Unicode."""
    if request.method == "POST" and request.scope.get("path") in _EVIDENCE_POST_PATHS:
        safe_errors = []
        allowed_fields = (
            {"query", "doc_type", "draft"}
            if request.scope.get("path") == "/documents/evidence-claims"
            else {"query", "doc_type", "top_k"}
        )
        for error in exc.errors():
            raw_location = list(error.get("loc", ("body",)))
            if len(raw_location) > 1 and raw_location[1] not in allowed_fields:
                location = ["body", "unknown_field"]
            else:
                location = raw_location[:2]
            safe_errors.append(
                {
                    "type": str(error.get("type", "validation_error")),
                    "loc": location,
                    "msg": str(error.get("msg", "Invalid request")),
                }
            )
        return JSONResponse(status_code=422, content={"detail": safe_errors})
    return await request_validation_exception_handler(request, exc)


@app.middleware("http")
async def evidence_post_no_store(request, call_next):
    """Protect evidence response bodies even when validation precedes the handler."""
    response = await call_next(request)
    if request.method == "POST" and request.scope.get("path") in _EVIDENCE_POST_PATHS:
        response.headers["Cache-Control"] = "no-store"
    return response

app.include_router(chat_router)
app.include_router(diagnostic_sessions_router)
app.include_router(documents_router)
app.include_router(tickets_router)
app.include_router(devices_router)


@app.get("/health")
def health() -> dict:
    """健康检查：服务可用性探针。"""
    return {"status": "ok"}
