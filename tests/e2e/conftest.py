"""E2E test configuration and fixtures."""

import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Generator

from dapr.conf import settings as _dapr_settings

import pytest

logger = logging.getLogger(__name__)

_DAPR_GRPC_PORT = 50001
_DAPR_HTTP_PORT = 3500
_RESOURCES_DIR = str(Path(__file__).parent / "resources")

# Prevent DaprClient from blocking 60s per instantiation when the sidecar's
# HTTP health endpoint is slow to respond.  The _dapr_sidecar fixture already
# verifies full readiness before yielding; this is a safety net.
# NOTE: dapr.conf.Settings caches env vars at import time, so os.environ
# changes after import are ignored.  Patch the settings object directly.


_dapr_settings.DAPR_HEALTH_TIMEOUT = 5


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid pyproject.toml changes."""
    config.addinivalue_line(
        "markers", "ollama: marks tests requiring Ollama LLM backend"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring Dapr sidecar (no LLM needed)",
    )
    config.addinivalue_line(
        "markers",
        "chaos: marks tests requiring Chaos Mesh on a K8s cluster",
    )


def _is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _is_dapr_healthy(http_port: int, timeout: float = 2.0) -> bool:
    """Check if the Dapr sidecar HTTP health endpoint responds."""
    url = f"http://127.0.0.1:{http_port}/v1.0/healthz/outbound"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


_APP_ID = "e2e-test"


def _stop_sidecar(dapr_bin: str) -> None:
    """Stop any running sidecar for the e2e app-id."""
    subprocess.run(
        [dapr_bin, "stop", "--app-id", _APP_ID],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    # Give the placement service time to deregister.
    time.sleep(1)


def _start_sidecar(
    dapr_bin: str, grpc_port: int, http_port: int
) -> subprocess.Popen[bytes] | None:
    """Start a fresh Dapr sidecar and wait for readiness."""
    # Stop any stale sidecar first so the ports are free.
    if _is_port_open("127.0.0.1", grpc_port):
        _stop_sidecar(dapr_bin)

    logger.info("Starting Dapr sidecar (grpc=%d, http=%d)", grpc_port, http_port)
    proc = subprocess.Popen(
        [
            dapr_bin,
            "run",
            "--app-id",
            _APP_ID,
            "--dapr-grpc-port",
            str(grpc_port),
            "--dapr-http-port",
            str(http_port),
            "--resources-path",
            _RESOURCES_DIR,
            "--log-level",
            "warn",
            "--",
            "tail",
            "-f",
            "/dev/null",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    for i in range(30):
        if _is_port_open("127.0.0.1", grpc_port) and _is_dapr_healthy(http_port):
            logger.info("Dapr sidecar ready after %ds", i + 1)
            return proc
        time.sleep(1)

    proc.terminate()
    stderr = proc.stderr.read().decode() if proc.stderr else ""
    logger.error("Dapr sidecar failed to start: %s", stderr[:500])
    proc.wait(timeout=5)
    return None


@pytest.fixture(scope="module", autouse=True)
def _dapr_sidecar() -> Generator[None, None, None]:
    """Start a fresh Dapr sidecar for each test module.

    Module scope ensures the sidecar is restarted between test files,
    preventing stale placement/scheduler state after crash-recovery
    tests (which spawn and destroy many short-lived sidecars).

    Skips startup if:
    - OLLAMA_ENDPOINT and DAPR_E2E are both unset
    - The ``dapr`` CLI is not installed
    """
    if not os.environ.get("OLLAMA_ENDPOINT") and not os.environ.get("DAPR_E2E"):
        yield
        return

    dapr_bin = shutil.which("dapr")
    if dapr_bin is None:
        logger.warning("dapr CLI not found — tests will skip")
        yield
        return

    grpc_port = int(os.environ.get("DAPR_GRPC_PORT", str(_DAPR_GRPC_PORT)))
    http_port = int(os.environ.get("DAPR_HTTP_PORT", str(_DAPR_HTTP_PORT)))

    proc = _start_sidecar(dapr_bin, grpc_port, http_port)

    yield

    if proc is not None:
        logger.info("Stopping Dapr sidecar")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture(scope="module")
def ollama_endpoint() -> str:
    """Ollama-compatible OpenAI API endpoint URL."""
    return os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1")


@pytest.fixture(scope="module")
def ollama_model() -> str:
    """Ollama model name to use for tests."""
    return os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")


def _import_unshadowed(package_name: str, symbol: str) -> object:
    """Import a symbol from an installed package, bypassing test shadow paths.

    pytest adds ``tests/agent/`` to ``sys.path`` during collection, causing
    packages like ``crewai`` or ``pydantic_ai`` to resolve to the local test
    stubs (``tests/agent/<pkg>/__init__.py``) instead of the installed one.

    This helper temporarily removes non-site-packages shadow paths that
    contain a directory matching the top-level package, purges cached modules,
    imports the requested symbol, then restores the paths.
    """
    top_level = package_name.split(".")[0]

    shadow_paths = [
        p
        for p in sys.path
        if os.path.isdir(os.path.join(p, top_level)) and "site-packages" not in p
    ]
    for p in shadow_paths:
        sys.path.remove(p)

    for key in list(sys.modules):
        if key == top_level or key.startswith(f"{top_level}."):
            del sys.modules[key]

    try:
        mod = __import__(package_name, fromlist=[symbol])
        return getattr(mod, symbol)
    finally:
        sys.path.extend(shadow_paths)


def import_pydantic_ai_agent() -> type:
    """Import ``Agent`` from the installed ``pydantic_ai`` package."""
    return _import_unshadowed("pydantic_ai", "Agent")  # type: ignore[return-value]


def import_crewai() -> tuple[type, type, object]:
    """Import ``Agent``, ``Task``, and ``crewai.tools.tool`` from the installed ``crewai``.

    All imports happen in a single unshadow window so that the classes
    share the same module identity (required for pydantic isinstance checks).

    Returns:
        (Agent, Task, tool) tuple.
    """
    top_level = "crewai"
    shadow_paths = [
        p
        for p in sys.path
        if os.path.isdir(os.path.join(p, top_level)) and "site-packages" not in p
    ]
    for p in shadow_paths:
        sys.path.remove(p)
    for key in list(sys.modules):
        if key == top_level or key.startswith(f"{top_level}."):
            del sys.modules[key]
    try:
        from crewai import Agent, Task
        from crewai.tools import tool
    finally:
        sys.path.extend(shadow_paths)
    return Agent, Task, tool


def import_deepagents() -> object:
    """Import ``create_deep_agent`` from the installed ``deepagents`` package."""
    return _import_unshadowed("deepagents", "create_deep_agent")


# ---------------------------------------------------------------------------
# Dapr registration cleanup
# ---------------------------------------------------------------------------

_DAPR_FN_ATTRS = (
    "_workflow_registered",
    "_activity_registered",
    "_dapr_alternate_name",
)


def _clear_fn_registration(*fns: object) -> None:
    """Remove Dapr registration attributes from the given functions."""
    for fn in fns:
        for attr in _DAPR_FN_ATTRS:
            fn.__dict__.pop(attr, None)


def clear_dapr_registration() -> None:
    """Clear Dapr registration state on LangGraph workflow functions."""
    from diagrid.agent.langgraph.workflow import (
        agent_workflow,
        evaluate_condition_activity,
        execute_node_activity,
    )

    _clear_fn_registration(
        agent_workflow, execute_node_activity, evaluate_condition_activity
    )


def clear_agent_registration(framework_workflow_module: str) -> None:
    """Clear Dapr registration state on an agent framework's workflow functions.

    Agent frameworks (CrewAI, PydanticAI, OpenAI Agents, ADK) each define
    module-level ``agent_workflow``, ``call_llm_activity``, and
    ``execute_tool_activity`` functions that accumulate Dapr registration
    attributes. This helper clears them so the next test can re-register.

    Args:
        framework_workflow_module: Dotted module path, e.g.
            ``"diagrid.agent.crewai.workflow"``.
    """
    import importlib

    mod = importlib.import_module(framework_workflow_module)
    fns = [
        getattr(mod, name, None)
        for name in ("agent_workflow", "call_llm_activity", "execute_tool_activity")
    ]
    _clear_fn_registration(*(fn for fn in fns if fn is not None))


@pytest.fixture(scope="module")
def chaos_enabled() -> bool:
    """True when the ``CHAOS_ENABLED`` env var is set."""
    return bool(os.environ.get("CHAOS_ENABLED"))


@pytest.fixture(scope="module")
def catalyst_operator_mode() -> bool:
    """True when running against a Catalyst-backed cluster."""
    return bool(os.environ.get("DAPR_HTTP_ENDPOINT"))


@pytest.fixture(scope="module")
def target_namespace() -> str:
    """Namespace for E2E tests (default: catalyst-agents)."""
    return os.environ.get("E2E_NAMESPACE", "catalyst-agents")


@pytest.fixture(scope="module")
def ollama_litellm_model(ollama_model: str) -> str:
    """LiteLLM-prefixed model name for CrewAI/LiteLLM routing."""
    return f"openai/{ollama_model}"


@pytest.fixture(autouse=True)
def skip_without_prerequisites(
    request: pytest.FixtureRequest,
    _dapr_sidecar: None,
) -> None:
    """Skip tests when required infrastructure is unavailable.

    - ``ollama`` marker: skip if OLLAMA_ENDPOINT is unset or Dapr is down.
    - ``integration`` marker (without ``ollama``): skip if Dapr is down.
    """
    has_ollama_marker = request.node.get_closest_marker("ollama") is not None
    has_integration_marker = request.node.get_closest_marker("integration") is not None
    grpc_port = int(os.environ.get("DAPR_GRPC_PORT", str(_DAPR_GRPC_PORT)))
    http_port = int(os.environ.get("DAPR_HTTP_PORT", str(_DAPR_HTTP_PORT)))

    if has_ollama_marker:
        if not os.environ.get("OLLAMA_ENDPOINT"):
            pytest.skip("OLLAMA_ENDPOINT not set")
        if not _is_port_open("127.0.0.1", grpc_port):
            pytest.skip(f"Dapr sidecar not reachable on port {grpc_port}")
        if not _is_dapr_healthy(http_port):
            pytest.skip(f"Dapr sidecar HTTP health check failed on port {http_port}")
    elif has_integration_marker:
        if not _is_port_open("127.0.0.1", grpc_port):
            pytest.skip(f"Dapr sidecar not reachable on port {grpc_port}")
        if not _is_dapr_healthy(http_port):
            pytest.skip(f"Dapr sidecar HTTP health check failed on port {http_port}")
