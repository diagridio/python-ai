# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A stream whose oldest log batches expired goes on with what is still there.

A run that waited on ``interrupt()`` longer than the log's batch TTL has lost
the chunks of its first steps, but everything after the resume is fresh: a
reconnect says ``stream_purged`` and then streams the rest, live.
"""

import operator
import time
from typing import Any

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import Annotated, TypedDict

from ..core.log_store_fake import FakeLogStore
from .log_helpers import log_runner
from .streaming_helpers import serve_app
from .test_reconnect_helpers import frames
from .test_runner import _RunnerTestCase


class _S(TypedDict, total=False):
    log: Annotated[list, operator.add]


def _graph() -> Any:
    def first(s: _S) -> dict:
        return {"log": ["first"]}

    def ask(s: _S) -> dict:
        return {"log": [f"ask:{interrupt('go?')}"]}

    def after(s: _S) -> dict:
        return {"log": ["after"]}

    g = StateGraph(_S)
    g.add_node("first", first)
    g.add_node("ask", ask)
    g.add_node("after", after)
    g.add_edge(START, "first")
    g.add_edge("first", "ask")
    g.add_edge("ask", "after")
    g.add_edge("after", END)
    return g.compile()


def _expire_batches(store: FakeLogStore, step: int, unit: str) -> None:
    """What the batch TTL does: the batches go, the (longer-lived) head stays."""
    for key in store.keys_like(f":{step}:{unit}:"):
        if not key.endswith(":head"):
            store.delete(key)


class TestPurgedThenLive(_RunnerTestCase):
    def test_a_reconnect_after_a_long_interrupt_streams_the_rest(self):
        store = FakeLogStore()
        runner = log_runner(_graph(), "purge-live", store=store)
        http = TestClient(serve_app(runner))
        got = frames(http.post("/agent/run", json={"log": [], "stream": True}).text)
        self.assertEqual(got[-1][1], "workflow_interrupted")
        wf = got[-1][2]["workflow_id"]
        _expire_batches(store, 0, "first")
        http.post(f"/agent/run/{wf}/resume", json={"resume": "yes"})
        deadline = time.monotonic() + 10
        while wf not in runner._workflow_client.done and time.monotonic() < deadline:
            time.sleep(0.01)
        again = frames(http.get(f"/agent/run/{wf}/stream").text)
        names = [e for _, e, _ in again]
        self.assertIn("stream_purged", names)
        updates = [list(d)[0] for _, e, d in again if e == "updates"]
        self.assertEqual(updates, ["ask", "after"])
        self.assertEqual(names[-1], "workflow_completed")
        self.assertEqual(names.count("stream_purged"), 1)
