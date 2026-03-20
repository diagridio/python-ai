# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""E2E test fixtures — chaos markers and Catalyst operator detection."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for E2E tests."""
    config.addinivalue_line(
        "markers",
        "chaos: marks tests requiring Chaos Mesh on a K8s cluster",
    )


@pytest.fixture(scope="module")
def chaos_enabled() -> bool:
    """True when the ``CHAOS_ENABLED`` env var is set."""
    return bool(os.environ.get("CHAOS_ENABLED"))


@pytest.fixture(scope="module")
def catalyst_operator_mode() -> bool:
    """True when running against a Catalyst-backed cluster.

    The operator webhook injects ``DAPR_HTTP_ENDPOINT`` into pods.
    When this env var is set in the test runner environment it means
    the Catalyst operator is providing the Dapr runtime.
    """
    return bool(os.environ.get("DAPR_HTTP_ENDPOINT"))


@pytest.fixture(scope="module")
def target_namespace() -> str:
    """Namespace for E2E tests (default: catalyst-agents)."""
    return os.environ.get("E2E_NAMESPACE", "catalyst-agents")
