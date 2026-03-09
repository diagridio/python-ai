"""Observability configuration resolution with 3-tier precedence.

Reuses ``AgentObservabilityConfig`` from ``dapr_agents.agents.configs``
so both dapr-agents and ai-python share the same config model.
"""

from __future__ import annotations

import logging
from typing import Optional

from dapr_agents.agents.configs import (
    AgentObservabilityConfig,
    AgentLoggingExporter,
    AgentTracingExporter,
)

logger = logging.getLogger(__name__)


def resolve_observability_config(
    explicit: Optional[AgentObservabilityConfig] = None,
    runtime_conf: Optional[dict[str, str]] = None,
) -> AgentObservabilityConfig:
    """Resolve OTEL config with 3-tier precedence.

    1. ``explicit`` (highest) — passed programmatically
    2. Standard OTEL env vars via ``AgentObservabilityConfig.from_env()``
    3. ``runtime_conf`` from agent-runtime statestore (lowest)

    Args:
        explicit: Programmatic override config.
        runtime_conf: Key-value config loaded from agent-runtime statestore.

    Returns:
        Merged ``AgentObservabilityConfig``.
    """
    # Tier 3: statestore config (lowest priority)
    base = (
        _config_from_runtime(runtime_conf)
        if runtime_conf
        else AgentObservabilityConfig()
    )

    # Tier 2: env vars
    env_config = AgentObservabilityConfig.from_env()
    base = _merge(base, env_config)

    # Tier 1: explicit (highest priority)
    if explicit is not None:
        base = _merge(base, explicit)

    return base


def _config_from_runtime(
    runtime_conf: dict[str, str],
) -> AgentObservabilityConfig:
    """Build an ``AgentObservabilityConfig`` from agent-runtime statestore data."""
    try:
        sdk_disabled = runtime_conf.get("OTEL_SDK_DISABLED", "true").lower()
        enabled = sdk_disabled != "true"

        endpoint = runtime_conf.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None
        service_name = runtime_conf.get("OTEL_SERVICE_NAME") or None
        auth_token = runtime_conf.get("OTEL_EXPORTER_OTLP_HEADERS") or None

        logging_enabled_str = runtime_conf.get("OTEL_LOGGING_ENABLED", "false")
        logging_enabled = logging_enabled_str.lower() == "true"

        tracing_enabled_str = runtime_conf.get("OTEL_TRACING_ENABLED", "false")
        tracing_enabled = tracing_enabled_str.lower() == "true"

        logging_exporter: Optional[AgentLoggingExporter] = None
        if log_exp_str := runtime_conf.get("OTEL_LOGS_EXPORTER"):
            try:
                logging_exporter = AgentLoggingExporter(log_exp_str)
            except (ValueError, KeyError):
                logging_exporter = AgentLoggingExporter.CONSOLE

        tracing_exporter: Optional[AgentTracingExporter] = None
        if trace_exp_str := runtime_conf.get("OTEL_TRACES_EXPORTER"):
            try:
                tracing_exporter = AgentTracingExporter(trace_exp_str)
            except (ValueError, KeyError):
                tracing_exporter = AgentTracingExporter.CONSOLE

        return AgentObservabilityConfig(
            enabled=enabled,
            auth_token=auth_token,
            endpoint=endpoint,
            service_name=service_name,
            logging_enabled=logging_enabled,
            logging_exporter=logging_exporter,
            tracing_enabled=tracing_enabled,
            tracing_exporter=tracing_exporter,
        )
    except Exception as exc:
        logger.debug("Failed to parse runtime observability config: %s", exc)
        return AgentObservabilityConfig()


def _merge(
    base: AgentObservabilityConfig,
    override: AgentObservabilityConfig,
) -> AgentObservabilityConfig:
    """Merge two configs; override values win when not None."""
    merged_headers = {**base.headers, **override.headers}
    return AgentObservabilityConfig(
        enabled=override.enabled if override.enabled is not None else base.enabled,
        headers=merged_headers,
        auth_token=override.auth_token
        if override.auth_token is not None
        else base.auth_token,
        endpoint=override.endpoint if override.endpoint is not None else base.endpoint,
        service_name=override.service_name
        if override.service_name is not None
        else base.service_name,
        logging_enabled=override.logging_enabled
        if override.logging_enabled is not None
        else base.logging_enabled,
        logging_exporter=override.logging_exporter
        if override.logging_exporter is not None
        else base.logging_exporter,
        tracing_enabled=override.tracing_enabled
        if override.tracing_enabled is not None
        else base.tracing_enabled,
        tracing_exporter=override.tracing_exporter
        if override.tracing_exporter is not None
        else base.tracing_exporter,
    )
