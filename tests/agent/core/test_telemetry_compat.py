# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Pin the contract diagrid expects from upstream OpenTelemetry + CrewAI.

These tests are companion to ``test_cross_framework_imports.py`` and exist
to fail loudly when:

* CrewAI renames or removes the ``Telemetry.set_tracer`` method that
  ``patch_crewai_telemetry`` depends on.
* OpenTelemetry's API surface that ``setup_telemetry`` relies on shifts
  (``trace.set_tracer_provider``, ``TracerProvider``, OTLP gRPC exporter).
* The endpoint-resolution precedence between ``AgentObservabilityConfig``
  and ``OTEL_EXPORTER_OTLP_ENDPOINT`` regresses.

A failure here = a future Dependabot bump that would silently break the
`override-dependencies` block in ``pyproject.toml``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from diagrid.agent.core import telemetry


# CrewAI assertions run in a subprocess because ``tests/agent/crewai/__init__.py``
# is a deliberate test stub that shadows the real ``crewai`` package when
# imported in-process. The subprocess uses only the project root on PYTHONPATH
# so it sees the installed ``crewai`` distribution.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _run_in_clean_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a fresh interpreter that does NOT see tests/ shadow paths.

    Inherits the parent environment and overrides only ``PYTHONPATH`` so
    that ``diagrid`` resolves to the workspace package while
    ``tests/agent/<pkg>`` shadow stubs stay invisible. Inheriting is
    required on Windows where ``import asyncio`` fails (WinError 10106 —
    Winsock cannot initialise) without ``SYSTEMROOT`` / ``WINDIR`` /
    ``TEMP`` / ``APPDATA`` in the env.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        env=env,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Endpoint resolution precedence
# ---------------------------------------------------------------------------


@mock.patch.dict(os.environ, {}, clear=True)
def test_resolve_endpoint_returns_none_when_unset() -> None:
    assert telemetry._resolve_endpoint() is None
    assert telemetry._get_otlp_endpoint() is None


@mock.patch.dict(
    os.environ,
    {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317"},
    clear=True,
)
def test_resolve_endpoint_uses_env_var() -> None:
    assert telemetry._resolve_endpoint() == "http://collector:4317"


@mock.patch.dict(
    os.environ,
    {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317/v1/traces"},
    clear=True,
)
def test_resolve_endpoint_strips_signal_suffix() -> None:
    """gRPC OTLP wants the bare host:port, not the `/v1/traces` HTTP path."""
    assert telemetry._resolve_endpoint() == "http://collector:4317"


@mock.patch.dict(
    os.environ,
    {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://env:4317/"},
    clear=True,
)
def test_resolve_endpoint_strips_trailing_slash() -> None:
    assert telemetry._resolve_endpoint() == "http://env:4317"


# ---------------------------------------------------------------------------
# setup_telemetry contract — the function frameworks rely on
# ---------------------------------------------------------------------------


@mock.patch.dict(os.environ, {}, clear=True)
def test_setup_telemetry_returns_none_without_endpoint() -> None:
    assert telemetry.setup_telemetry("svc") is None


@mock.patch.dict(
    os.environ,
    {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317"},
    clear=True,
)
def test_setup_telemetry_returns_provider_when_configured() -> None:
    """Verify a real OTEL TracerProvider is returned and registered globally.

    This pins the OTEL SDK shape we depend on:
    ``TracerProvider`` exists, ``trace.set_tracer_provider`` accepts it,
    and the OTLP gRPC exporter can be constructed.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    provider = telemetry.setup_telemetry("compat-test")
    assert provider is not None, "setup_telemetry should return a provider"
    assert isinstance(provider, TracerProvider)
    # The global tracer must now resolve through our provider.
    assert trace.get_tracer_provider() is provider


@mock.patch.dict(
    os.environ,
    {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317"},
    clear=True,
)
def test_get_tracer_returns_usable_tracer_after_setup() -> None:
    telemetry.setup_telemetry("compat-test")
    tracer = telemetry.get_tracer("compat-test")
    assert tracer is not None
    # `start_as_current_span` is the surface every adapter relies on.
    with tracer.start_as_current_span("smoke") as span:
        # The span object should expose the standard OTEL methods.
        assert hasattr(span, "set_attribute")
        assert hasattr(span, "end")


# ---------------------------------------------------------------------------
# CrewAI monkey-patch contract — the most fragile touchpoint
# ---------------------------------------------------------------------------


def test_crewai_telemetry_set_tracer_method_still_exists() -> None:
    """Pin the symbol that ``patch_crewai_telemetry`` depends on.

    If CrewAI ever renames, removes, or restructures
    ``crewai.telemetry.Telemetry.set_tracer``, this test fails *before*
    a CrewAI bump merges and breaks tracing in production.
    """
    code = (
        "from crewai.telemetry import Telemetry\n"
        "assert hasattr(Telemetry, 'set_tracer'), "
        "'crewai.telemetry.Telemetry.set_tracer is gone'\n"
        "assert callable(getattr(Telemetry, 'set_tracer'))\n"
    )
    result = _run_in_clean_subprocess(code)
    assert result.returncode == 0, (
        f"CrewAI Telemetry.set_tracer probe failed: {result.stderr}"
    )


def test_patch_crewai_telemetry_is_noop_without_endpoint() -> None:
    """Without an OTLP endpoint configured, the patch must not run."""
    code = (
        "import os\n"
        "os.environ.pop('OTEL_EXPORTER_OTLP_ENDPOINT', None)\n"
        "from crewai.telemetry import Telemetry\n"
        "from diagrid.agent.core import telemetry\n"
        "original = Telemetry.set_tracer\n"
        "telemetry.patch_crewai_telemetry()\n"
        "assert Telemetry.set_tracer is original, "
        "'patch_crewai_telemetry should not modify Telemetry when OTEL is not configured'\n"
    )
    result = _run_in_clean_subprocess(code)
    assert result.returncode == 0, f"crewai noop test failed: {result.stderr}"


def test_patch_crewai_telemetry_replaces_set_tracer_when_configured() -> None:
    code = (
        "import os\n"
        "os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://collector:4317'\n"
        "from crewai.telemetry import Telemetry\n"
        "from diagrid.agent.core import telemetry\n"
        "original = Telemetry.set_tracer\n"
        "telemetry.patch_crewai_telemetry()\n"
        "assert Telemetry.set_tracer is not original, "
        "'patch_crewai_telemetry was a no-op despite OTEL being configured'\n"
    )
    result = _run_in_clean_subprocess(code)
    assert result.returncode == 0, f"crewai patch test failed: {result.stderr}"


# ---------------------------------------------------------------------------
# OTLP exporter import contract — the override-dependencies surface
# ---------------------------------------------------------------------------


def test_otlp_grpc_exporter_importable() -> None:
    """The override-dependencies block in pyproject.toml exists to keep
    these imports working across crewai / dapr-agents version skew. Pin
    them so a misaligned override surfaces immediately."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: F401
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: F401


@pytest.mark.parametrize(
    "submodule",
    [
        "opentelemetry.trace",
        "opentelemetry.sdk.trace",
        "opentelemetry.sdk.trace.export",
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    ],
)
def test_pinned_opentelemetry_packages_importable(submodule: str) -> None:
    """If the override-dependencies block in pyproject.toml ever drops one
    of these, the import will fail and dependabot PRs will turn red on
    deps-check.yaml instead of nightly e2e-ollama.yaml."""
    __import__(submodule)
