"""OpenTelemetry setup for non-dapr-agents framework runners.

Provides shared utilities so every framework runner can emit traces
to an OTLP collector (Tempo / Grafana Alloy / etc.) without each
runner duplicating boilerplate.

Key design decisions
--------------------
* Uses the **standard OTEL SDK** (not ``DaprAgentsOtel``) so that
  framework-native instrumentation (Strands, ADK, CrewAI, …) can layer
  on top of—or instead of—the global ``TracerProvider``.
* **No-op when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset** so local-dev
  "just works" without a collector.
* Uses the **gRPC OTLP exporter** — matching the collector endpoint
  (``:4317``) that dapr-agents already export to successfully.
* Accepts an optional ``AgentObservabilityConfig`` so that callers
  can pass resolved config (explicit > env > statestore) instead of
  relying solely on env vars.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dapr_agents.agents.configs import AgentObservabilityConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OTEL_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
_OTEL_SERVICE_ENV = "OTEL_SERVICE_NAME"


def _resolve_endpoint(
    config: Optional[AgentObservabilityConfig] = None,
) -> Optional[str]:
    """Return the OTLP gRPC endpoint from config or env, or *None*."""
    raw: Optional[str] = None

    if config is not None:
        # When config says disabled, return None immediately
        if config.enabled is False:
            return None
        raw = config.endpoint

    if not raw:
        raw = os.environ.get(_OTEL_ENDPOINT_ENV)

    if not raw:
        return None

    endpoint = raw.rstrip("/")
    for suffix in ("/v1/traces", "/v1/metrics", "/v1/logs"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break

    return endpoint


def _resolve_headers(
    config: Optional[AgentObservabilityConfig] = None,
) -> Optional[dict[str, str]]:
    """Extract OTLP headers from config, or None."""
    if config is not None and config.headers:
        return dict(config.headers)
    return None


def _get_otlp_endpoint() -> Optional[str]:
    """Return the OTLP gRPC endpoint, or *None* when not configured."""
    return _resolve_endpoint()


def _make_span_processor(
    config: Optional[AgentObservabilityConfig] = None,
) -> Any:
    """Create a BatchSpanProcessor targeting the OTLP gRPC endpoint.

    Returns None when the endpoint is not configured.
    """
    endpoint = _resolve_endpoint(config)
    if endpoint is None:
        return None

    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    headers = _resolve_headers(config)
    return BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint, insecure=True, headers=headers)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tracer(name: str = "diagrid.agent") -> Any:
    """Return an OTEL ``Tracer`` from the global provider (or a no-op)."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:
        return None


def setup_telemetry(
    service_name: str | None = None,
    config: Optional[AgentObservabilityConfig] = None,
) -> Any:
    """Create a global ``TracerProvider`` + ``LoggerProvider`` via OTLP/gRPC.

    This is the right call for frameworks that rely on
    ``trace.get_tracer()`` at module level (Google ADK, LangGraph).

    When ``config`` is provided, its endpoint/service_name/headers are
    used with higher priority than env vars.

    Returns the TracerProvider so callers can hand it to framework-specific
    bridges (e.g. ``OtelTracingProcessor`` for OpenAI Agents SDK), or
    *None* when no endpoint is configured.
    """
    endpoint = _resolve_endpoint(config)
    if endpoint is None:
        logger.debug("OTEL endpoint not set — telemetry disabled")
        return None

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    svc_from_config = config.service_name if config else None
    svc = (
        svc_from_config
        or os.environ.get(_OTEL_SERVICE_ENV)
        or service_name
        or "unknown-service"
    )
    resource = Resource.create({"service.name": svc})

    # --- Traces ---
    processor = _make_span_processor(config)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    logger.info("OTEL TracerProvider set (service=%s, endpoint=%s)", svc, endpoint)

    # --- Logs ---
    _setup_logging(resource, endpoint, config)

    return provider


def _setup_logging(
    resource: Any,
    endpoint: str,
    config: Optional[AgentObservabilityConfig] = None,
) -> None:
    """Set up OTLP log export so Python logs flow to the collector."""
    try:
        from opentelemetry import _logs
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter,
        )

        headers = _resolve_headers(config)
        log_provider = LoggerProvider(resource=resource)
        log_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(endpoint=endpoint, insecure=True, headers=headers)
            )
        )
        _logs.set_logger_provider(log_provider)

        # Bridge Python stdlib logging → OTEL
        from opentelemetry.sdk._logs import LoggingHandler

        handler = LoggingHandler(logger_provider=log_provider)
        logging.getLogger().addHandler(handler)
        logger.info("OTEL LoggerProvider set")
    except Exception:
        logger.debug("OTEL logging setup failed", exc_info=True)


def patch_crewai_telemetry(
    config: Optional[AgentObservabilityConfig] = None,
) -> None:
    """Monkey-patch CrewAI's ``Telemetry.set_tracer`` to also export to our collector.

    CrewAI's ``EventListener`` lazily creates a ``Telemetry`` singleton and
    calls ``set_tracer()`` on the first crew run, replacing whatever global
    ``TracerProvider`` was active. By patching ``set_tracer`` we inject our
    OTLP span processor into CrewAI's own provider the moment it initialises
    — regardless of timing.
    """
    endpoint = _resolve_endpoint(config)
    if endpoint is None:
        logger.debug("OTEL endpoint not set — skipping CrewAI patch")
        return

    try:
        from crewai.telemetry import Telemetry
    except ImportError:
        logger.debug("crewai not installed — skipping patch")
        return

    original_set_tracer = Telemetry.set_tracer
    # Capture config in closure for the patched function
    _config = config

    def _patched_set_tracer(self: Any) -> None:
        original_set_tracer(self)
        # After CrewAI sets its provider, attach our exporter
        if getattr(self, "ready", False) and getattr(self, "provider", None):
            processor = _make_span_processor(_config)
            if processor is not None:
                self.provider.add_span_processor(processor)
                logger.info("Injected OTLP exporter into CrewAI TracerProvider")

    Telemetry.set_tracer = _patched_set_tracer  # type: ignore[assignment]
    logger.info("CrewAI Telemetry.set_tracer patched")


def instrument_grpc(
    config: Optional[AgentObservabilityConfig] = None,
) -> None:
    """Patch gRPC client channels so Dapr sidecar calls emit spans."""
    endpoint = _resolve_endpoint(config)
    if endpoint is None and not os.environ.get(_OTEL_ENDPOINT_ENV):
        return

    try:
        from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient

        GrpcInstrumentorClient().instrument()
        logger.info("gRPC client OTEL instrumentation enabled")
    except Exception:
        logger.debug("gRPC OTEL instrumentation not available", exc_info=True)


# ---------------------------------------------------------------------------
# OpenAI Agents SDK bridge
# ---------------------------------------------------------------------------


class OtelTracingProcessor:
    """Bridge OpenAI Agents SDK tracing to OpenTelemetry spans.

    The Agents SDK defines its own ``TracingProcessor`` interface with
    ``on_trace_start / on_span_start / on_span_end / on_trace_end``
    callbacks. This class implements that interface and forwards events
    as standard OTEL spans, so they show up in Tempo alongside every
    other framework's traces.
    """

    def __init__(self, provider: Any) -> None:
        from opentelemetry.sdk.trace import TracerProvider

        if not isinstance(provider, TracerProvider):
            raise TypeError(f"Expected TracerProvider, got {type(provider)}")
        self._tracer = provider.get_tracer("openai-agents-bridge")
        self._spans: dict[str, Any] = {}

    # -- TracingProcessor interface ------------------------------------------

    def on_trace_start(self, trace: object) -> None:
        """Called when a new Agents SDK trace begins."""
        trace_id = getattr(trace, "trace_id", None) or id(trace)
        name = getattr(trace, "name", "agent-trace")
        span = self._tracer.start_span(name)
        self._spans[str(trace_id)] = span

    def on_trace_end(self, trace: object) -> None:
        trace_id = str(getattr(trace, "trace_id", None) or id(trace))
        span = self._spans.pop(trace_id, None)
        if span:
            span.end()

    def on_span_start(self, span: object) -> None:
        span_id = getattr(span, "span_id", None) or id(span)
        span_data = getattr(span, "span_data", None)
        if span_data is not None:
            name = type(span_data).__name__
        else:
            name = str(getattr(span, "name", "agent-span"))
        otel_span = self._tracer.start_span(name)
        self._spans[str(span_id)] = otel_span

    def on_span_end(self, span: object) -> None:
        span_id = str(getattr(span, "span_id", None) or id(span))
        otel_span = self._spans.pop(span_id, None)
        if otel_span:
            otel_span.end()

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        self._spans.clear()
