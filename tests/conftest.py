# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import pytest


@pytest.fixture(autouse=True)
def _no_usage_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the test suite out of the production usage analytics.

    Every runner construction calls ``diagrid.core.analytics.report_usage``.
    With a live endpoint the first such call in a pytest run would start a real
    daemon thread against the Scarf gateway, so every CI job and every local run
    would count as usage. Blanking the endpoint makes the reporter a no-op, and
    resetting the per-process guard gives every test a clean slate. Tests that
    exercise the reporter itself set the endpoint explicitly.
    """
    monkeypatch.setattr("diagrid.core.analytics.USAGE_ENDPOINT", "")
    monkeypatch.setattr("diagrid.core.analytics._reported_packages", set())
