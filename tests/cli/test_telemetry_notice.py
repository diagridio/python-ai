# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The one-time telemetry notice ``diagridpy`` prints before it first reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from diagrid.cli import telemetry_notice
from diagrid.cli.main import cli
from diagrid.core import analytics

REO_TEST_ENDPOINT = "https://example.invalid/data"

_GATE_ENV_VARS = (
    *analytics.OPT_OUT_ENV_VARS,
    *analytics._CI_TRUTHY_ENV_VARS,
    *analytics._CI_PRESENCE_ENV_VARS,
    *analytics._DAPR_ENDPOINT_ENV_VARS,
    "DAPR_API_TOKEN",
    "KUBERNETES_SERVICE_HOST",
)


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Open the Reo gate and record each request instead of sending it."""
    for name in _GATE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(analytics, "REO_ENDPOINT", REO_TEST_ENDPOINT)
    urls: list[str] = []

    def fake_urlopen(request: Any, timeout: float = 0) -> _Response:
        urls.append(request.full_url)
        return _Response()

    monkeypatch.setattr(analytics.urllib.request, "urlopen", fake_urlopen)
    return urls


def test_first_reporting_run_prints_notice_to_stderr(recorded: list[str]) -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert telemetry_notice.NOTICE in result.stderr
    assert telemetry_notice.NOTICE not in result.stdout
    assert telemetry_notice.marker_path().exists()
    assert len(recorded) == 1


def test_second_run_prints_no_notice_but_still_reports(
    monkeypatch: pytest.MonkeyPatch, recorded: list[str]
) -> None:
    CliRunner().invoke(cli, ["--help"])
    monkeypatch.setattr(analytics, "_reported_packages", set())
    second = CliRunner().invoke(cli, ["--help"])
    assert second.exit_code == 0, second.output
    assert telemetry_notice.NOTICE not in second.stderr
    assert len(recorded) == 2


def test_notice_is_shown_before_the_event_is_sent(
    monkeypatch: pytest.MonkeyPatch, recorded: list[str]
) -> None:
    order: list[str] = []
    real_echo = click.echo

    def tracking_echo(message: Any = None, *args: Any, **kwargs: Any) -> None:
        if message == telemetry_notice.NOTICE:
            order.append("notice")
        real_echo(message, *args, **kwargs)

    def tracking_urlopen(request: Any, timeout: float = 0) -> _Response:
        order.append("send")
        return _Response()

    monkeypatch.setattr(telemetry_notice.click, "echo", tracking_echo)
    monkeypatch.setattr(analytics.urllib.request, "urlopen", tracking_urlopen)
    CliRunner().invoke(cli, ["--help"])
    assert order == ["notice", "send"]


@pytest.mark.parametrize(
    ("name", "value"),
    [("DO_NOT_TRACK", "1"), ("DIAGRID_NO_ANALYTICS", "1"), ("CI", "true")],
)
def test_no_notice_and_no_marker_when_nothing_is_sent(
    monkeypatch: pytest.MonkeyPatch, recorded: list[str], name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert telemetry_notice.NOTICE not in result.stderr
    assert not telemetry_notice.marker_path().exists()
    assert recorded == []


def test_no_notice_and_no_marker_without_an_endpoint(
    monkeypatch: pytest.MonkeyPatch, recorded: list[str]
) -> None:
    monkeypatch.setattr(analytics, "REO_ENDPOINT", "")
    result = CliRunner().invoke(cli, ["--help"])
    assert telemetry_notice.NOTICE not in result.stderr
    assert not telemetry_notice.marker_path().exists()


def test_no_notice_for_shell_completion(
    monkeypatch: pytest.MonkeyPatch, recorded: list[str]
) -> None:
    monkeypatch.setenv("_DIAGRIDPY_COMPLETE", "zsh_source")
    CliRunner().invoke(cli, [], prog_name="diagridpy")
    assert not telemetry_notice.marker_path().exists()


def test_unwritable_diagrid_home_does_not_break_the_command(
    monkeypatch: pytest.MonkeyPatch, recorded: list[str], tmp_path: Path
) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    monkeypatch.setenv("DIAGRID_HOME", str(blocker / "sub"))
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert "diagridpy" in result.stdout
    assert len(recorded) == 1


def test_marker_path_follows_diagrid_home_at_call_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DIAGRID_HOME", str(tmp_path / "a"))
    assert telemetry_notice.marker_path() == tmp_path / "a" / "telemetry-notice-shown"


def test_marker_path_defaults_to_dot_diagrid_in_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DIAGRID_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake-home")
    assert (
        telemetry_notice.marker_path()
        == tmp_path / "fake-home" / ".diagrid" / "telemetry-notice-shown"
    )


def test_cli_tests_never_touch_the_real_home(
    monkeypatch: pytest.MonkeyPatch, recorded: list[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake-home")
    CliRunner().invoke(cli, ["--help"])
    assert not (tmp_path / "fake-home" / ".diagrid").exists()
