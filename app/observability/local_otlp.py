"""Opt-in loopback OTLP bridge for explicit, content-free RobotOps spans.

No exporter is constructed until the API lifespan reads the exact local opt-in.
This is an assembly seam, not proof that Phoenix accepted any spans.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import SpanKind

from app.observability import tracing


_SWITCH = "ROBOTOPS_LOCAL_OTLP"
_ENDPOINT = "127.0.0.1:4317"
_SCOPE = "robotops.diagnostic"
_SERVICE_NAME = "robotops-diagnostic-local"
_CORRELATION = re.compile(r"[0-9a-f]{32}\Z")
_ENUMS = {
    "robotops.outcome": tracing._OUTCOMES,
    "robotops.tool_name": tracing._READONLY_TOOL_NAMES,
    "robotops.retrieval_wait_result": frozenset({"READY", "TIMEOUT", "BUSY", "FAILED"}),
    "robotops.draft_kind": tracing._DRAFT_KINDS,
    "robotops.verification_status": tracing._VERIFICATION_STATUSES,
    "robotops.process_verification_status": tracing._PROCESS_VERIFICATION_STATUSES,
    "robotops.persisted_event": tracing._PERSISTED_EVENTS,
    "robotops.approval_decision": tracing._APPROVAL_DECISIONS,
    "robotops.usage_status": frozenset({"KNOWN", "UNKNOWN"}),
    "robotops.cache_status": frozenset({"KNOWN", "UNKNOWN"}),
    "robotops.usage_source": frozenset({"PROVIDER"}),
    "robotops.cache_source": frozenset({"PROVIDER"}),
}
_BOOLEANS = frozenset({"robotops.handoff_required", "robotops.readback_matched"})
_INTEGERS = frozenset({
    "robotops.input_tokens", "robotops.output_tokens", "robotops.total_tokens",
    "robotops.cache_hit_tokens", "robotops.cache_miss_tokens", "robotops.attempt",
})
_ATTRIBUTES = frozenset(_ENUMS) | _BOOLEANS | _INTEGERS | {"robotops.run_correlation"}


class _ExplicitLocalProvider(TracerProvider):
    """Only the RobotOps opt-in controls this private provider (SDK 1.44.0)."""

    def __init__(self) -> None:
        super().__init__(
            sampler=ALWAYS_ON,
            resource=Resource({"service.name": _SERVICE_NAME}),
            shutdown_on_exit=False,
            span_limits=SpanLimits(
                max_attributes=64, max_events=32, max_links=32,
                max_span_attributes=64, max_event_attributes=32,
                max_link_attributes=32, max_attribute_length=1024,
                max_span_attribute_length=1024,
            ),
            meter_provider=NoOpMeterProvider(),
        )
        # Pinned SDK reads OTEL_SDK_DISABLED even for a private provider.
        # The explicit RobotOps opt-in wins here; global OTel remains untouched.
        self._disabled = False


def _allow_attribute(name: object, value: object) -> bool:
    if type(name) is not str or name not in _ATTRIBUTES:
        return False
    if name in _ENUMS:
        return type(value) is str and value in _ENUMS[name]
    if name in _BOOLEANS:
        return type(value) is bool
    if name in _INTEGERS:
        return type(value) is int and 0 <= value <= (2**63 - 1)
    return type(value) is str and _CORRELATION.fullmatch(value) is not None


def _allow_span(span: object) -> bool:
    """Drop an entire span on any unexpected field; never redact in place."""
    try:
        attributes = span.attributes
        resource = span.resource
        resource_attributes = resource.attributes
        scope = span.instrumentation_scope
        return (
            span.name in tracing._SPAN_NAMES
            and scope.name == _SCOPE
            and scope.version == ""
            and scope.schema_url == ""
            and len(scope.attributes) == 0
            and span.kind is SpanKind.INTERNAL
            and span.context.trace_state.to_header() == ""
            and (span.parent is None or span.parent.trace_state.to_header() == "")
            and len(span.events) == 0
            and len(span.links) == 0
            and span.status.description in (None, "")
            and resource.schema_url == ""
            and set(resource_attributes) == {"service.name"}
            and resource_attributes["service.name"] == _SERVICE_NAME
            and all(_allow_attribute(key, value) for key, value in attributes.items())
        )
    except Exception:
        return False


class _AllowlistedExporter(SpanExporter):
    """Only explicitly authored RobotOps spans reach the OTLP delegate."""

    def __init__(self, delegate: SpanExporter) -> None:
        self._delegate = delegate

    def export(self, spans: Sequence) -> SpanExportResult:
        allowed = tuple(span for span in spans if _allow_span(span))
        if not allowed:
            return SpanExportResult.SUCCESS
        try:
            return self._delegate.export(allowed)
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self, timeout_millis: float = 30_000) -> None:
        try:
            self._delegate.shutdown(timeout_millis=timeout_millis)
        except Exception:
            pass

    def force_flush(self, timeout_millis: float = 30_000) -> bool:
        try:
            return bool(self._delegate.force_flush(timeout_millis=timeout_millis))
        except Exception:
            return False


class _LoopbackGrpcExporter(SpanExporter):
    """Fixed OTLP/gRPC transport; no SDK environment-derived configuration."""

    def __init__(self) -> None:
        import grpc
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import TraceServiceStub

        self._grpc = grpc
        self._channel = grpc.insecure_channel(
            _ENDPOINT, options=(("grpc.enable_http_proxy", 0),),
        )
        try:
            self._stub = TraceServiceStub(self._channel)
        except Exception:
            self._channel.close()
            raise

    def export(self, spans: Sequence) -> SpanExportResult:
        from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

        try:
            self._stub.Export(
                encode_spans(spans), timeout=2.0,
                metadata=(("x-robotops-local", "1"),),
                compression=self._grpc.Compression.NoCompression,
            )
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self, timeout_millis: float = 30_000) -> None:
        try:
            self._channel.close()
        except Exception:
            pass

    def force_flush(self, timeout_millis: float = 30_000) -> bool:
        return True


def _make_exporter() -> SpanExporter:
    return _LoopbackGrpcExporter()


def start_local_otlp() -> TracerProvider | None:
    """Install an app-private provider only after exact process-local opt-in."""
    setting = os.environ.get(_SWITCH, "0")
    if setting == "0":
        return None
    if setting != "1":
        raise RuntimeError("LOCAL_OTLP_CONFIG_INVALID")
    try:
        delegate = _make_exporter()
    except Exception:
        raise RuntimeError("LOCAL_OTLP_UNAVAILABLE") from None
    provider: TracerProvider | None = None
    processor: BatchSpanProcessor | None = None
    processor_attached = False
    try:
        provider = _ExplicitLocalProvider()
        processor = BatchSpanProcessor(
            _AllowlistedExporter(delegate),
            max_queue_size=512,
            max_export_batch_size=64,
            schedule_delay_millis=1000,
            export_timeout_millis=2000,
            meter_provider=NoOpMeterProvider(),
        )
        provider.add_span_processor(processor)
        processor_attached = True
        tracing._install_private_provider(provider)
        return provider
    except Exception:
        if provider is not None:
            try:
                provider.shutdown()
            except Exception:
                pass
        if processor is not None and not processor_attached:
            try:
                processor.shutdown()
            except Exception:
                pass
        try:
            delegate.shutdown()
        except Exception:
            pass
        raise RuntimeError("LOCAL_OTLP_UNAVAILABLE") from None


def stop_local_otlp(provider: TracerProvider | None) -> None:
    if provider is None:
        return
    tracing._clear_private_provider(provider)
    try:
        provider.shutdown()
    except Exception:
        # A broken exporter must not stop API cleanup or alter diagnosis state.
        pass
