# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A reconnect sends each ``values`` chunk once, whatever SSE id it resumes from.

Every activity of a step offers the step's ``values`` chunk and the log keeps
each unit's copy, so the cursor carries the last step whose ``values`` the
client has; the final ``values`` chunk has its own id.
"""

import operator
import time
from typing import Any

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from .log_helpers import log_runner
from .streaming_helpers import serve_app
from .test_reconnect_helpers import frames
from .test_runner import _RunnerTestCase


class _S(TypedDict, total=False):
    log: Annotated[list, operator.add]


def _fan_out() -> Any:
    def a(s: _S) -> dict:
        return {"log": ["a"]}

    def b(s: _S) -> dict:
        return {"log": ["b"]}

    def c(s: _S) -> dict:
        return {"log": ["c"]}

    g = StateGraph(_S)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_node("c", c)
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", "c")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    return g.compile()


def _wait_done(runner: Any, wf: str) -> None:
    deadline = time.monotonic() + 10
    while wf not in runner._workflow_client.done and time.monotonic() < deadline:
        time.sleep(0.01)


class TestReconnectValues(_RunnerTestCase):
    def test_reconnect_every_cut_values_parallel(self):
        runner = log_runner(_fan_out(), "rv-fan")
        http = TestClient(serve_app(runner))
        resp = http.post(
            "/agent/run",
            json={"log": ["x"], "stream": True, "stream_mode": ["values", "updates"]},
        )
        full = frames(resp.text)
        self.assertEqual([e for _, e, _ in full].count("values"), 3)
        wf = full[0][2]["workflow_id"]
        _wait_done(runner, wf)
        bad = []
        for cut in range(1, len(full) - 1):
            got = frames(
                http.get(
                    f"/agent/run/{wf}/stream",
                    params={"stream_mode": "values,updates"},
                    headers={"Last-Event-ID": full[cut][0]},
                ).text
            )
            rest = [(e, d) for _, e, d in got[1:]]
            want = [(e, d) for _, e, d in full[cut + 1 :]]
            if rest != want:
                bad.append((cut, full[cut][1], [e for e, _ in rest]))
        self.assertEqual(bad, [])

    def test_every_values_chunk_has_its_own_id(self):
        runner = log_runner(_fan_out(), "rv-ids")
        http = TestClient(serve_app(runner))
        full = frames(
            http.post(
                "/agent/run",
                json={"log": ["x"], "stream": True, "stream_mode": ["values"]},
            ).text
        )
        ids = [i for i, e, _ in full if e == "values"]
        self.assertEqual(len(ids), len(set(ids)))
