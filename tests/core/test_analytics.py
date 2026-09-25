# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the anonymous usage reporter in ``diagrid.core.analytics``."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from diagrid.core import analytics

ENDPOINT = "https://example.invalid/python-ai"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts unreported and with no opt-out variable set."""
    monkeypatch.setattr(analytics, "_reported", False)
    for name in analytics.OPT_OUT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_does_nothing_while_endpoint_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", "")
    with patch.object(analytics.threading, "Thread") as thread:
        analytics.report_usage("diagrid")
    thread.assert_not_called()
    # Not marked as reported either, so a later call with an endpoint still fires.
    assert analytics._reported is False


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


def test_event_sent_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    with patch.object(analytics.threading, "Thread") as thread:
        analytics.report_usage("diagrid")
        analytics.report_usage("diagrid-cli")
        analytics.report_usage("diagrid")
    assert thread.call_count == 1
    thread.return_value.start.assert_called_once()
    _, kwargs = thread.call_args
    assert kwargs["daemon"] is True
    assert kwargs["args"] == ("diagrid",)


def test_send_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    with patch.object(
        analytics.urllib.request, "urlopen", side_effect=OSError("network unreachable")
    ):
        # Must not raise: blocked egress is a normal condition.
        analytics._send_event("diagrid")


def test_send_uses_the_short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    with patch.object(analytics.urllib.request, "urlopen") as urlopen:
        analytics._send_event("diagrid")
    _, kwargs = urlopen.call_args
    assert kwargs["timeout"] == analytics.USAGE_TIMEOUT_SECONDS


def test_report_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    with patch.object(
        analytics, "usage_reporting_disabled", side_effect=RuntimeError("boom")
    ):
        analytics.report_usage("diagrid")


def test_url_contains_expected_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytics, "USAGE_ENDPOINT", ENDPOINT)
    url = analytics.build_url("diagrid-core")
    assert url.startswith(ENDPOINT + "?")
    for fragment in (
        "package=diagrid-core",
        "version=",
        "os=",
        "arch=",
        "python_version=",
    ):
        assert fragment in url


def test_unknown_package_version_does_not_raise() -> None:
    assert analytics.package_version("diagrid-does-not-exist") == "unknown"
