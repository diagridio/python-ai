# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the anonymous usage reporter in ``diagrid.core.analytics``."""

from __future__ import annotations

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
