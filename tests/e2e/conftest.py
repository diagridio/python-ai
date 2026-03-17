"""E2E test configuration and fixtures."""

import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from typing import Generator

import pytest

logger = logging.getLogger(__name__)

_DAPR_GRPC_PORT = 50001
_DAPR_HTTP_PORT = 3500


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid pyproject.toml changes."""
    config.addinivalue_line(
        "markers", "ollama: marks tests requiring Ollama LLM backend"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring Dapr sidecar (no LLM needed)",
    )


def _is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


@pytest.fixture(scope="session", autouse=True)
def _dapr_sidecar() -> Generator[None, None, None]:
    """Auto-start a Dapr sidecar for the e2e test session.

    Skips startup if:
    - OLLAMA_ENDPOINT is not set (tests will skip anyway)
    - A sidecar is already listening (e.g. pytest is wrapped with ``dapr run``)
    - The ``dapr`` CLI is not installed
    """
    if not os.environ.get("OLLAMA_ENDPOINT") and not os.environ.get("DAPR_E2E"):
        yield
        return

    grpc_port = int(os.environ.get("DAPR_GRPC_PORT", str(_DAPR_GRPC_PORT)))
    http_port = int(os.environ.get("DAPR_HTTP_PORT", str(_DAPR_HTTP_PORT)))

    if _is_port_open("127.0.0.1", grpc_port):
        logger.info("Dapr sidecar already running on port %d", grpc_port)
        yield
        return

    dapr_bin = shutil.which("dapr")
    if dapr_bin is None:
        logger.warning("dapr CLI not found — tests will skip")
        yield
        return

    logger.info("Starting Dapr sidecar (grpc=%d, http=%d)", grpc_port, http_port)
    proc = subprocess.Popen(
        [
            dapr_bin,
            "run",
            "--app-id",
            "e2e-test",
            "--dapr-grpc-port",
            str(grpc_port),
            "--dapr-http-port",
            str(http_port),
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

    ready = False
    for i in range(30):
        if _is_port_open("127.0.0.1", grpc_port):
            ready = True
            logger.info("Dapr sidecar ready after %ds", i + 1)
            break
        time.sleep(1)

    if not ready:
        proc.terminate()
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        logger.error("Dapr sidecar failed to start: %s", stderr[:500])
        proc.wait(timeout=5)
        yield
        return

    yield

    logger.info("Stopping Dapr sidecar")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="session")
def ollama_endpoint() -> str:
    """Ollama-compatible OpenAI API endpoint URL."""
    return os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1")


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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

    if has_ollama_marker:
        if not os.environ.get("OLLAMA_ENDPOINT"):
            pytest.skip("OLLAMA_ENDPOINT not set")
        if not _is_port_open("127.0.0.1", grpc_port):
            pytest.skip(f"Dapr sidecar not reachable on port {grpc_port}")
    elif has_integration_marker:
        if not _is_port_open("127.0.0.1", grpc_port):
            pytest.skip(f"Dapr sidecar not reachable on port {grpc_port}")
