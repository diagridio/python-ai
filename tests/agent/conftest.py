# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _skip_dapr_health_check():
    """Prevent DaprClient from blocking 60s on a missing sidecar.

    DaprClient.__init__ calls DaprHealth.wait_for_sidecar() which polls
    the Dapr healthz endpoint in a retry loop (default DAPR_HEALTH_TIMEOUT=60s).
    In unit tests there is no sidecar, so every DaprClient instantiation that
    is not otherwise mocked would block for a full minute.
    """
    with mock.patch("dapr.clients.health.DaprHealth.wait_for_sidecar"):
        yield
