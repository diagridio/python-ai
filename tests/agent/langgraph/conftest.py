# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from typing import Iterator

import pytest

from diagrid.agent.langgraph import workflow


@pytest.fixture(autouse=True, scope="session")
def _inline_when_all() -> Iterator[None]:
    """Run ``when_all`` inline for the whole session, not per test or per thread.

    These tests drive ``agent_workflow`` inline, where ``when_all`` must
    return the activities' results as a list, and many patch it per run. A
    patch saves and restores the module attribute, so one that ends while
    another thread's run is still going (a paused run of an earlier test
    whose wait timed out, say) put the real ``when_all`` back under it, and
    that run failed with ``'dict' object has no attribute '_parent'``. With
    the inline one in place for the session, every restore restores it.
    """
    real = workflow.when_all
    workflow.when_all = list  # type: ignore[assignment]
    try:
        yield
    finally:
        workflow.when_all = real  # type: ignore[assignment]
