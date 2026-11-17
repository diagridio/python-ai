# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for CLI main entry point — env/api flag handling."""

from __future__ import annotations

import socket
import time
import urllib.parse
from typing import Any

import click
import pytest
from click.testing import CliRunner

from diagrid.cli.main import cli
from diagrid.core import analytics
from diagrid.core.config.constants import PROD_API_URL, STAGING_API_URL


def _invoke_noop(*args: str) -> "click.testing.Result":
    """Invoke the CLI with a no-op subcommand to inspect ctx.obj."""

    @cli.command(name="_noop", hidden=True)
    @click.pass_context
    def _noop(ctx: click.Context) -> None:
        # Print api_url so the test can inspect it
        api_url = (ctx.obj or {}).get("api_url", "NOT_SET")
        click.echo(f"api_url={api_url}")

    runner = CliRunner()
    result = runner.invoke(cli, [*args, "_noop"])
    # Remove the ephemeral command after each test to keep state clean
    cli.commands.pop("_noop", None)
    return result


def test_cli_env_staging_sets_api_url() -> None:
    """--env staging sets api_url to STAGING_API_URL in ctx.obj."""
    result = _invoke_noop("--env", "staging")
    assert result.exit_code == 0, result.output
    assert f"api_url={STAGING_API_URL}" in result.output


def test_cli_env_prod_sets_api_url() -> None:
    """--env prod sets api_url to PROD_API_URL in ctx.obj."""
    result = _invoke_noop("--env", "prod")
    assert result.exit_code == 0, result.output
    assert f"api_url={PROD_API_URL}" in result.output


def test_cli_api_override_sets_api_url() -> None:
    """--api sets a custom api_url in ctx.obj."""
    result = _invoke_noop("--api", "https://custom.example.com")
    assert result.exit_code == 0, result.output
    assert "api_url=https://custom.example.com" in result.output


def test_cli_no_env_sets_none() -> None:
    """No env/api flags → api_url is None in ctx.obj."""
    result = _invoke_noop()
    assert result.exit_code == 0, result.output
    assert "api_url=None" in result.output


# --- Usage reporting (Reo only, blocking, once per process) -----------------

REO_TEST_ENDPOINT = "https://example.invalid/data"

_GATE_ENV_VARS = (
    *analytics.OPT_OUT_ENV_VARS,
    *analytics._CI_TRUTHY_ENV_VARS,
    *analytics._CI_PRESENCE_ENV_VARS,
    *analytics._DAPR_ENDPOINT_ENV_VARS,
    "DAPR_API_TOKEN",
    "KUBERNETES_SERVICE_HOST",
)


def clear_all_gate_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _GATE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def record_requests(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the URL of every request instead of sending it."""
    recorded: list[str] = []

    def fake_urlopen(request: Any, timeout: float = 0) -> _Response:
        recorded.append(request.full_url)
        return _Response()

    monkeypatch.setattr(analytics.urllib.request, "urlopen", fake_urlopen)
    return recorded


def invoke_cli(args: list[str]) -> "click.testing.Result":
    return CliRunner().invoke(cli, args)


def test_cli_reports_to_reo_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "REO_ENDPOINT", REO_TEST_ENDPOINT)
    # Scarf stays configured, so only the explicit sink list keeps it out.
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", "https://scarf.example.invalid/x")
    recorded = record_requests(monkeypatch)
    clear_all_gate_vars(monkeypatch)
    invoke_cli(["--help"])
    assert len(recorded) == 1
    assert recorded[0].startswith(REO_TEST_ENDPOINT + "?")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(recorded[0]).query)
    assert query["product"] == ["diagrid-cli"]
    assert query["activity_type"] == ["pypi_download"]
    assert "kind" not in query
    assert "framework" not in query
    assert not any("scarf" in url for url in recorded)


@pytest.mark.parametrize("args", [["--version"], ["--help"], ["deploy", "--help"]])
def test_cli_reports_on_eager_flags(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    monkeypatch.setattr(analytics, "REO_ENDPOINT", REO_TEST_ENDPOINT)
    recorded = record_requests(monkeypatch)
    clear_all_gate_vars(monkeypatch)
    result = invoke_cli(args)
    assert result.exit_code == 0, result.output
    assert len(recorded) == 1


def test_cli_send_is_blocking_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    # A one-shot CLI exits before a daemon thread finishes, so the send blocks,
    # and must therefore be bounded. The server accepts and never replies.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        monkeypatch.setattr(analytics, "REO_ENDPOINT", f"http://127.0.0.1:{port}/data")
        monkeypatch.setattr(analytics, "USAGE_TIMEOUT_SECONDS", 1.0)
        clear_all_gate_vars(monkeypatch)

        start = time.monotonic()
        analytics.report_usage_blocking("diagrid-cli", sinks=(analytics.REO_SINK,))
        elapsed = time.monotonic() - start
    finally:
        server.close()
    assert elapsed < 3.0
    assert elapsed >= 0.5  # it really waited on the socket, not skipped the send


def test_cli_exits_zero_when_telemetry_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # A port that was just closed refuses connections immediately.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    monkeypatch.setattr(analytics, "REO_ENDPOINT", f"http://127.0.0.1:{port}/data")
    clear_all_gate_vars(monkeypatch)
    result = invoke_cli(["--help"])
    assert result.exit_code == 0
    assert "telemetry" not in result.output.lower()
    assert "diagrid.io" not in result.output


def test_cli_exits_zero_when_reporter_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(analytics, "report_usage_blocking", boom)
    result = invoke_cli(["--help"])
    assert result.exit_code == 0


def test_cli_sends_nothing_when_opted_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "REO_ENDPOINT", REO_TEST_ENDPOINT)
    recorded = record_requests(monkeypatch)
    clear_all_gate_vars(monkeypatch)
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    invoke_cli(["--help"])
    assert recorded == []


def test_cli_reports_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "REO_ENDPOINT", REO_TEST_ENDPOINT)
    recorded = record_requests(monkeypatch)
    clear_all_gate_vars(monkeypatch)
    invoke_cli(["--help"])
    invoke_cli(["--version"])
    assert len(recorded) == 1
