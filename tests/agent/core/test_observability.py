"""Tests for observability configuration resolution."""

import os
from unittest import TestCase, mock

from dapr_agents.agents.configs import (
    AgentObservabilityConfig,
    AgentTracingExporter,
    AgentLoggingExporter,
)

from diagrid.agent.core.observability import (
    resolve_observability_config,
    _config_from_runtime,
    _merge,
)


class TestResolveObservabilityConfig(TestCase):
    """Test 3-tier precedence: explicit > env > runtime."""

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_empty_returns_defaults(self):
        config = resolve_observability_config()
        self.assertIsNone(config.enabled)
        self.assertIsNone(config.endpoint)

    @mock.patch.dict(
        os.environ,
        {
            "OTEL_SDK_DISABLED": "false",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://env:4317",
            "OTEL_SERVICE_NAME": "env-svc",
        },
        clear=True,
    )
    def test_env_vars_used(self):
        config = resolve_observability_config()
        self.assertTrue(config.enabled)
        self.assertEqual(config.endpoint, "http://env:4317")
        self.assertEqual(config.service_name, "env-svc")

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_runtime_conf_used(self):
        runtime = {
            "OTEL_SDK_DISABLED": "false",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://runtime:4317",
            "OTEL_SERVICE_NAME": "runtime-svc",
        }
        config = resolve_observability_config(runtime_conf=runtime)
        self.assertTrue(config.enabled)
        self.assertEqual(config.endpoint, "http://runtime:4317")
        self.assertEqual(config.service_name, "runtime-svc")

    @mock.patch.dict(
        os.environ,
        {
            "OTEL_SERVICE_NAME": "env-svc",
        },
        clear=True,
    )
    def test_env_overrides_runtime(self):
        runtime = {
            "OTEL_SERVICE_NAME": "runtime-svc",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://runtime:4317",
        }
        config = resolve_observability_config(runtime_conf=runtime)
        self.assertEqual(config.service_name, "env-svc")
        # Endpoint comes from runtime (env didn't set it)
        self.assertEqual(config.endpoint, "http://runtime:4317")

    @mock.patch.dict(
        os.environ,
        {
            "OTEL_SERVICE_NAME": "env-svc",
        },
        clear=True,
    )
    def test_explicit_overrides_env(self):
        explicit = AgentObservabilityConfig(
            service_name="explicit-svc",
        )
        config = resolve_observability_config(explicit=explicit)
        self.assertEqual(config.service_name, "explicit-svc")

    @mock.patch.dict(
        os.environ,
        {
            "OTEL_SDK_DISABLED": "false",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://env:4317",
        },
        clear=True,
    )
    def test_full_precedence(self):
        runtime = {
            "OTEL_SDK_DISABLED": "false",
            "OTEL_SERVICE_NAME": "runtime-svc",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://runtime:4317",
            "OTEL_TRACES_EXPORTER": "console",
        }
        explicit = AgentObservabilityConfig(
            service_name="explicit-svc",
        )
        config = resolve_observability_config(explicit=explicit, runtime_conf=runtime)
        # explicit wins for service_name
        self.assertEqual(config.service_name, "explicit-svc")
        # env wins for endpoint (overrides runtime)
        self.assertEqual(config.endpoint, "http://env:4317")
        # runtime tracing_exporter used (not set in env or explicit)
        self.assertEqual(config.tracing_exporter, AgentTracingExporter.CONSOLE)


class TestConfigFromRuntime(TestCase):
    def test_all_fields(self):
        runtime = {
            "OTEL_SDK_DISABLED": "false",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://rt:4317",
            "OTEL_SERVICE_NAME": "rt-svc",
            "OTEL_EXPORTER_OTLP_HEADERS": "token123",
            "OTEL_LOGGING_ENABLED": "true",
            "OTEL_TRACING_ENABLED": "true",
            "OTEL_LOGS_EXPORTER": "otlp_grpc",
            "OTEL_TRACES_EXPORTER": "zipkin",
        }
        config = _config_from_runtime(runtime)
        self.assertTrue(config.enabled)
        self.assertEqual(config.endpoint, "http://rt:4317")
        self.assertEqual(config.service_name, "rt-svc")
        self.assertEqual(config.auth_token, "token123")
        self.assertTrue(config.logging_enabled)
        self.assertTrue(config.tracing_enabled)
        self.assertEqual(config.logging_exporter, AgentLoggingExporter.OTLP_GRPC)
        self.assertEqual(config.tracing_exporter, AgentTracingExporter.ZIPKIN)

    def test_disabled_when_sdk_disabled_true(self):
        runtime = {"OTEL_SDK_DISABLED": "true"}
        config = _config_from_runtime(runtime)
        self.assertFalse(config.enabled)

    def test_invalid_exporter_defaults_console(self):
        runtime = {
            "OTEL_TRACES_EXPORTER": "not_a_real_exporter",
            "OTEL_LOGS_EXPORTER": "also_invalid",
        }
        config = _config_from_runtime(runtime)
        self.assertEqual(config.tracing_exporter, AgentTracingExporter.CONSOLE)
        self.assertEqual(config.logging_exporter, AgentLoggingExporter.CONSOLE)


class TestMerge(TestCase):
    def test_override_wins(self):
        base = AgentObservabilityConfig(service_name="base")
        override = AgentObservabilityConfig(service_name="override")
        merged = _merge(base, override)
        self.assertEqual(merged.service_name, "override")

    def test_none_doesnt_override(self):
        base = AgentObservabilityConfig(
            service_name="base", endpoint="http://base:4317"
        )
        override = AgentObservabilityConfig(service_name="override")
        merged = _merge(base, override)
        self.assertEqual(merged.service_name, "override")
        self.assertEqual(merged.endpoint, "http://base:4317")

    def test_headers_merged(self):
        base = AgentObservabilityConfig(headers={"X-Base": "b"})
        override = AgentObservabilityConfig(headers={"X-Override": "o"})
        merged = _merge(base, override)
        self.assertEqual(merged.headers, {"X-Base": "b", "X-Override": "o"})

    def test_headers_override_wins(self):
        base = AgentObservabilityConfig(headers={"Auth": "base-token"})
        override = AgentObservabilityConfig(headers={"Auth": "override-token"})
        merged = _merge(base, override)
        self.assertEqual(merged.headers["Auth"], "override-token")
