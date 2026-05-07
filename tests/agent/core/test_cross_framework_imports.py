# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Detect dependency conflicts between agent-framework adapters at PR time.

These tests are the **fast, preventive layer** for AI-552: a Dependabot
upgrade that would break ``diagrid.agent.<framework>`` import (e.g. an
OpenTelemetry version pin tightening, a CrewAI internal API rename,
LangChain core breaking change) fails this suite in seconds, instead of
waiting 15 minutes for ``e2e-ollama.yaml`` to flake on the same root cause.

The tests run each import in a **fresh Python subprocess**. Two reasons:

1. ``tests/agent/<framework>/__init__.py`` shadow paths (used by other
   suites to stub third-party libs) would otherwise hijack ``import
   pydantic_ai`` / ``import crewai`` etc. when this test runs in-process.
2. Module-level monkey-patching (notably ``crewai.telemetry``) leaves
   global state behind. Subprocess isolation guarantees each adapter is
   tested against the real installed package, not against state shaped
   by a previous import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


# Project root, used as the only entry on PYTHONPATH so ``diagrid`` resolves
# to the workspace package, but ``tests/agent/<pkg>`` shadow paths are not
# visible to the subprocess.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Public adapter modules that must always be importable when the project is
# installed with ``--extra all``. Each entry maps the importable module path
# to a runner symbol the adapter is expected to expose. The symbol assertion
# guards against a partially-broken import that would silently succeed
# (e.g. ImportError swallowed inside the package's ``__init__``).
_ADAPTERS: tuple[tuple[str, str], ...] = (
    ("diagrid.agent.langgraph", "DaprWorkflowGraphRunner"),
    ("diagrid.agent.crewai", "DaprWorkflowAgentRunner"),
    ("diagrid.agent.adk", "DaprWorkflowAgentRunner"),
    ("diagrid.agent.strands", "DaprWorkflowAgentRunner"),
    ("diagrid.agent.openai_agents", "DaprWorkflowAgentRunner"),
    ("diagrid.agent.pydantic_ai", "DaprWorkflowAgentRunner"),
    ("diagrid.agent.deepagents", "DaprWorkflowDeepAgentRunner"),
)


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter rooted at the project.

    Inherits the parent environment and overrides only ``PYTHONPATH`` to
    point at the project root. This keeps Windows-essential vars
    (``SYSTEMROOT``, ``WINDIR``, ``TEMP``, ``APPDATA``) — without them
    ``import asyncio`` fails with ``WinError 10106`` because Winsock
    cannot initialise.
    """
    env = os.environ.copy()
    # Only the project root: ensures ``diagrid`` resolves to the workspace
    # package while ``tests/agent/<pkg>`` shadow stubs are not visible.
    env["PYTHONPATH"] = str(_PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        env=env,
        timeout=60,
    )


@pytest.mark.parametrize(("module", "symbol"), _ADAPTERS, ids=lambda v: v)
def test_adapter_imports_in_isolation(module: str, symbol: str) -> None:
    """Each adapter must import cleanly in a fresh interpreter.

    Catches: a Dependabot bump that breaks one adapter's transitive deps,
    even when the other adapters still resolve.
    """
    code = (
        "import importlib\n"
        f"mod = importlib.import_module({module!r})\n"
        f"assert hasattr(mod, {symbol!r}), "
        f"'missing symbol ' + {symbol!r} + ' on ' + {module!r}\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, (
        f"failed to import {module}: {result.stderr}\nstdout: {result.stdout}"
    )


def test_all_adapters_import_in_one_process() -> None:
    """All adapters must coexist in the same interpreter.

    Catches: module-level conflicts (CrewAI's ``Telemetry.set_tracer``
    monkey-patch vs. another adapter's tracer setup; OpenTelemetry version
    skew between two adapters; LangChain singleton state).
    """
    imports = "\n".join(
        f"import {mod}; assert hasattr({mod}, {sym!r})" for mod, sym in _ADAPTERS
    )
    code = (
        f"{imports}\n"
        "from diagrid.agent.core import telemetry\n"
        "assert callable(telemetry.get_tracer)\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, (
        f"co-import failed: {result.stderr}\nstdout: {result.stdout}"
    )


def test_otel_setup_after_all_adapters_imported() -> None:
    """OTLP setup must remain functional after every adapter has loaded.

    Catches: a TracerProvider being clobbered by a late-importing adapter
    (the original symptom that drove the explicit ``override-dependencies``
    block in ``pyproject.toml``).
    """
    imports = "\n".join(f"import {mod}" for mod, _ in _ADAPTERS)
    code = (
        "import os\n"
        # No OTEL endpoint => setup_telemetry returns None and no exporter
        # is wired. We still expect get_tracer() to work (no-op tracer).
        "os.environ.pop('OTEL_EXPORTER_OTLP_ENDPOINT', None)\n"
        "os.environ.pop('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT', None)\n"
        f"{imports}\n"
        "from diagrid.agent.core import telemetry\n"
        "result = telemetry.setup_telemetry('test-svc')\n"
        "assert result is None, f'expected no-op when no endpoint, got {result!r}'\n"
        "tracer = telemetry.get_tracer('cross-fw')\n"
        "with tracer.start_as_current_span('smoke'):\n"
        "    pass\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, (
        f"OTEL setup after imports failed: {result.stderr}\nstdout: {result.stdout}"
    )
