# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the anonymous usage reporter in ``diagrid.core.analytics``."""

from __future__ import annotations

import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, NamedTuple
from unittest.mock import patch

import pytest

from diagrid.core import analytics

ENDPOINT = "https://example.invalid/python-ai"

_ENV_TO_CLEAR = (
    *analytics.OPT_OUT_ENV_VARS,
    *analytics._CI_TRUTHY_ENV_VARS,
    *analytics._CI_PRESENCE_ENV_VARS,
    *analytics._DAPR_ENDPOINT_ENV_VARS,
    "DAPR_API_TOKEN",
)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts unreported, off CI, and with no opt-out or Dapr variable set.

    The root ``tests/conftest.py`` blanks ``USAGE_ENDPOINT`` for the whole
    suite. Tests that need a live endpoint set it explicitly.
    """
    monkeypatch.setattr(analytics, "_reported_packages", set())
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)


class Recorded(NamedTuple):
    url: str
    sink: str
    package: str


class _SyncThread:
    """Runs the target inline on ``start``, so tests never sleep or join."""

    def __init__(
        self,
        target: Callable[..., Any],
        args: Any = (),
        kwargs: Any = None,
        **_: Any,
    ) -> None:
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self) -> None:
        self._target(*self._args, **self._kwargs)


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def record_requests(monkeypatch: pytest.MonkeyPatch) -> list[Recorded]:
    """Make sends synchronous and record each request instead of sending it.

    A recorded request carries the sink name from the ``X-Test-Sink`` header
    (empty for the real Scarf sink) and the ``package`` query parameter.
    """
    recorded: list[Recorded] = []

    def fake_urlopen(request: Any, timeout: float = 0) -> _Response:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)
        recorded.append(
            Recorded(
                request.full_url,
                request.get_header("X-test-sink") or "",
                query.get("package", [""])[0],
            )
        )
        return _Response()

    monkeypatch.setattr(analytics.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(analytics.threading, "Thread", _SyncThread)
    return recorded


def clear_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in analytics.OPT_OUT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_scarf_url_is_the_documented_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analytics, "USAGE_ENDPOINT", "https://diagrid.gateway.scarf.sh/python-ai"
    )
    recorded = record_requests(monkeypatch)
    clear_opt_out(monkeypatch)
    # ``version`` is the installed version of the package, so pin it. Every
    # host-derived default is overridden, so the URL is the same on every
    # machine, OS and Python version.
    monkeypatch.setattr(analytics, "package_version", lambda _package: "0.5.3")
    analytics.report_usage(
        "diagrid",
        core_version="0.5.3",
        os="linux",
        arch="x86_64",
        python_version="3.12.1",
        target="dapr",
        ci="false",
        kind="workflow",
        framework="langgraph",
        framework_version="0.2.0",
    )
    assert len(recorded) == 1
    assert recorded[0].url == (
        "https://diagrid.gateway.scarf.sh/python-ai"
        "?package=diagrid&version=0.5.3&core_version=0.5.3&os=linux&arch=x86_64"
        "&python_version=3.12.1&target=dapr&ci=false&kind=workflow"
        "&framework=langgraph&framework_version=0.2.0"
    )


def test_does_nothing_while_endpoint_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", "")
    with patch.object(analytics.threading, "Thread") as thread:
        analytics.report_usage("diagrid")
    thread.assert_not_called()
    # Not marked as reported either, so a later call with an endpoint still fires.
    assert analytics._reported_packages == set()


def test_enabled_when_no_opt_out_set() -> None:
    assert not analytics.usage_reporting_disabled()


@pytest.mark.parametrize("name", analytics.OPT_OUT_ENV_VARS)
def test_each_opt_out_variable_disables(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "1")
    assert analytics.usage_reporting_disabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_falsy_values_do_not_disable(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("DO_NOT_TRACK", value)
    assert not analytics.usage_reporting_disabled()


@pytest.mark.parametrize("value", ["1", "TRUE", " yes ", "On"])
def test_truthy_values_are_case_and_space_insensitive(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("DO_NOT_TRACK", value)
    assert analytics.usage_reporting_disabled()


def test_no_event_sent_when_opted_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    with patch.object(analytics.threading, "Thread") as thread:
        analytics.report_usage("diagrid")
    thread.assert_not_called()


def test_event_sent_once_per_package_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    with patch.object(analytics.threading, "Thread") as thread:
        analytics.report_usage("diagrid", kind="agent")
        analytics.report_usage("diagrid", kind="agent")
        analytics.report_usage("diagrid-cli")
    # One thread per distinct package, never a second for the same package.
    assert thread.call_count == 2
    first, second = (call.kwargs for call in thread.call_args_list)
    assert first["args"] == ("diagrid", {"kind": "agent"})
    assert second["args"] == ("diagrid-cli", {})
    assert first["daemon"] is True and second["daemon"] is True
    assert thread.return_value.start.call_count == 2


def test_send_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    with patch.object(
        analytics.urllib.request, "urlopen", side_effect=OSError("network unreachable")
    ):
        # Must not raise: blocked egress is a normal condition.
        analytics._send_event("diagrid", {})


def test_send_uses_the_short_timeout_and_computes_the_version_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    with (
        patch.object(analytics.urllib.request, "urlopen") as urlopen,
        patch.object(analytics, "package_version", return_value="1.2.3") as version,
    ):
        analytics._send_event("diagrid", {"kind": "agent"})
    _, kwargs = urlopen.call_args
    assert kwargs["timeout"] == analytics.USAGE_TIMEOUT_SECONDS
    request = urlopen.call_args.args[0]
    assert request.get_header("User-agent") == "diagrid/1.2.3"
    assert "version=1.2.3" in request.full_url
    # Once for the package itself, once for diagrid-core.
    assert [call.args[0] for call in version.call_args_list] == [
        "diagrid",
        "diagrid-core",
    ]


def test_report_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    with patch.object(
        analytics, "usage_reporting_disabled", side_effect=RuntimeError("boom")
    ):
        analytics.report_usage("diagrid")


def test_url_contains_expected_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    url = analytics.build_url("diagrid", "0.5.0", kind="agent", framework="langgraph")
    assert url.startswith(ENDPOINT + "?")
    for fragment in (
        "package=diagrid",
        "version=0.5.0",
        "core_version=",
        "os=",
        "arch=",
        "python_version=",
        "target=dapr",
        "ci=false",
        "kind=agent",
        "framework=langgraph",
    ):
        assert fragment in url


def test_caller_dimensions_override_defaults_and_empty_ones_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    url = analytics.build_url(
        "diagrid", "0.5.0", target="catalyst", framework="", kind=None
    )
    assert "target=catalyst" in url
    assert "framework=" not in url
    assert "kind=" not in url


def test_dimension_values_are_trimmed_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    url = analytics.build_url("diagrid", "0.5.0", framework="  " + "x" * 200 + "  ")
    assert "framework=" + "x" * analytics._DIMENSION_MAX_LEN + "&" in url + "&"
    assert "x" * (analytics._DIMENSION_MAX_LEN + 1) not in url


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"CI": "true"}, True),
        ({"CI": "0"}, False),
        ({"GITHUB_ACTIONS": "true"}, True),
        ({"TF_BUILD": "True"}, True),
        ({"BUILDKITE": "true"}, True),
        ({"JENKINS_URL": "https://ci.example.invalid/"}, True),
    ],
)
def test_ci_detection(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: bool
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert analytics.running_in_ci() is expected
    url = analytics.build_url("diagrid", "0.5.0")
    assert ("ci=true" in url) is expected


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "dapr"),
        ({"DAPR_HTTP_ENDPOINT": "http://localhost:3500"}, "dapr"),
        (
            {"DAPR_GRPC_ENDPOINT": "https://grpc-prj1.api.cloud.diagrid.io:443"},
            "catalyst",
        ),
        ({"DAPR_HTTP_ENDPOINT": "https://http-prj1.api.cloud.diagrid.io"}, "catalyst"),
        ({"DAPR_GRPC_ENDPOINT": "https://notdiagrid.io:443"}, "dapr"),
        ({"DAPR_API_TOKEN": "diagrid://abc"}, "catalyst"),
        ({"DAPR_API_TOKEN": "   "}, "dapr"),
    ],
)
def test_target_detection(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: str
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert analytics.detect_target() == expected


def test_unknown_package_version_does_not_raise() -> None:
    assert analytics.package_version("diagrid-does-not-exist") == "unknown"


def _fake_sink(
    name: str,
    *,
    allowed: Callable[[], bool] = lambda: True,
    raises: bool = False,
) -> analytics.Sink:
    def build_request(package: str, dimensions: Any) -> urllib.request.Request:
        if raises:
            raise RuntimeError("boom")
        return urllib.request.Request(
            f"https://example.invalid/{name}?package={package}",
            headers={"X-Test-Sink": name},
        )

    return analytics.Sink(name=name, allowed=allowed, build_request=build_request)


def sink_recording_as(name: str) -> analytics.Sink:
    return _fake_sink(name)


def sink_never_allowed_as(name: str) -> analytics.Sink:
    return _fake_sink(name, allowed=lambda: False)


def sink_that_raises_as(name: str) -> analytics.Sink:
    return _fake_sink(name, raises=True)


def test_report_usage_sends_to_each_allowed_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = record_requests(monkeypatch)
    sinks = (sink_recording_as("a"), sink_recording_as("b"))
    analytics.report_usage("diagrid", sinks=sinks)
    assert {r.sink for r in recorded} == {"a", "b"}


def test_report_usage_skips_a_sink_whose_gate_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = record_requests(monkeypatch)
    sinks = (sink_recording_as("a"), sink_never_allowed_as("b"))
    analytics.report_usage("diagrid", sinks=sinks)
    assert {r.sink for r in recorded} == {"a"}


def test_dedupe_is_per_sink_per_package(monkeypatch: pytest.MonkeyPatch) -> None:
    # One event per package per process, per destination. A closed gate on one
    # sink must not consume the other sink's single send.
    recorded = record_requests(monkeypatch)
    sinks = (sink_recording_as("a"), sink_recording_as("b"))
    analytics.report_usage("diagrid", sinks=sinks)
    analytics.report_usage("diagrid", sinks=sinks)
    analytics.report_usage("diagrid-cli", sinks=sinks)
    assert sorted((r.sink, r.package) for r in recorded) == [
        ("a", "diagrid"),
        ("a", "diagrid-cli"),
        ("b", "diagrid"),
        ("b", "diagrid-cli"),
    ]


def test_a_closed_gate_does_not_consume_another_sinks_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = record_requests(monkeypatch)
    analytics.report_usage("diagrid", sinks=(sink_never_allowed_as("a"),))
    analytics.report_usage("diagrid", sinks=(sink_recording_as("b"),))
    assert [(r.sink, r.package) for r in recorded] == [("b", "diagrid")]


def test_a_raising_sink_does_not_stop_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = record_requests(monkeypatch)
    sinks = (sink_that_raises_as("a"), sink_recording_as("b"))
    analytics.report_usage("diagrid", sinks=sinks)
    assert {r.sink for r in recorded} == {"b"}


def test_a_sink_whose_gate_raises_does_not_stop_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode() -> bool:
        raise RuntimeError("boom")

    recorded = record_requests(monkeypatch)
    sinks = (_fake_sink("a", allowed=explode), sink_recording_as("b"))
    analytics.report_usage("diagrid", sinks=sinks)
    assert {r.sink for r in recorded} == {"b"}


def test_scarf_is_the_default_destination() -> None:
    assert analytics.USAGE_SINKS == (analytics.SCARF_SINK,)
    assert analytics.SCARF_SINK.name == "scarf"


def clear_all_gate_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every variable the Reo gate reads, so a gate-open test is not vacuous."""
    for name in (*_ENV_TO_CLEAR, "KUBERNETES_SERVICE_HOST"):
        monkeypatch.delenv(name, raising=False)


def test_reo_gate_open_on_a_clean_developer_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_all_gate_vars(monkeypatch)
    assert analytics.reo_allowed() is True


@pytest.mark.parametrize(
    "var", ["DO_NOT_TRACK", "SCARF_NO_ANALYTICS", "DIAGRID_NO_ANALYTICS"]
)
def test_reo_gate_closed_on_opt_out(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    clear_all_gate_vars(monkeypatch)
    monkeypatch.setenv(var, "1")
    assert analytics.reo_allowed() is False


def test_reo_gate_closed_in_kubernetes(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_all_gate_vars(monkeypatch)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert analytics.reo_allowed() is False


@pytest.mark.parametrize(
    "var,value",
    [
        ("DAPR_API_TOKEN", "t"),
        ("DAPR_GRPC_ENDPOINT", "https://grpc.abc.diagrid.io:443"),
        ("DAPR_HTTP_ENDPOINT", "https://http.abc.diagrid.io"),
    ],
)
def test_reo_gate_closed_when_pointed_at_catalyst(
    monkeypatch: pytest.MonkeyPatch, var: str, value: str
) -> None:
    clear_all_gate_vars(monkeypatch)
    monkeypatch.setenv(var, value)
    assert analytics.reo_allowed() is False


@pytest.mark.parametrize(
    "var",
    [
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "TRAVIS",
        "TF_BUILD",
        "BUILDKITE",
        "JENKINS_URL",
    ],
)
def test_reo_gate_closed_in_ci(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    clear_all_gate_vars(monkeypatch)
    monkeypatch.setenv(var, "true")
    assert analytics.reo_allowed() is False


def test_reo_gate_fails_closed_on_malformed_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # DAPR_GRPC_ENDPOINT is user-supplied. If detect_target() raises on it, the gate
    # must suppress the event rather than let the exception reach the runner.
    clear_all_gate_vars(monkeypatch)
    monkeypatch.setenv("DAPR_GRPC_ENDPOINT", "http://[:::::")
    # Prove the input is not vacuous: it really makes detect_target() raise.
    with pytest.raises(ValueError):
        analytics.detect_target()
    assert analytics.reo_allowed() is False


def test_reo_gate_fails_closed_when_a_check_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_all_gate_vars(monkeypatch)

    def boom() -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(analytics, "detect_target", boom)
    assert analytics.reo_allowed() is False


def test_reo_gate_is_not_vacuous_under_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guards the Review Focus trap.
    monkeypatch.setenv("CI", "true")
    assert analytics.reo_allowed() is False
    clear_all_gate_vars(monkeypatch)
    assert analytics.reo_allowed() is True
