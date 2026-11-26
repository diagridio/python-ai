# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""What a streamed run's custom status costs: its size, and how often it is read.

The status is rewritten every step and read by every stream, and Dapr returns
it only with the run's input and output (``fetch_payloads``), so it holds the
last steps' units only, and a stream reads it when it needs to find new units,
not on every poll.
"""

import operator
import threading
import time
from types import SimpleNamespace
from typing import Any, List
from unittest import mock

from fastapi.testclient import TestClient
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import activity_stream
from diagrid.agent.langgraph.activity_stream import STATUS_EXECUTED_FROM, run_status
from diagrid.agent.langgraph.stream_follow import units_of
from diagrid.agent.langgraph.streaming import STREAM_RUN_KEY

from .log_helpers import log_runner
from .streaming_helpers import serve_app
from .test_reconnect_helpers import frames
from .test_runner import _RunnerTestCase


class _S(TypedDict, total=False):
    log: Annotated[list, operator.add]
    n: int


def _run(steps: int) -> Any:
    return SimpleNamespace(
        config={STREAM_RUN_KEY: True},
        executed=[[f"node{i}"] for i in range(steps)],
        branches=[[i, f"node{i}:branch:r"] for i in range(steps)],
        attempts={f"{i}:node{i}": f"att{i}" for i in range(steps)},
        interrupt=None,
        interrupt_seq=0,
        resumed=None,
    )


class TestStatusSize(_RunnerTestCase):
    def test_the_status_holds_the_last_steps_only(self):
        with mock.patch.object(activity_stream, "STATUS_STEP_WINDOW", 10):
            status = run_status(_run(500))
        self.assertEqual(status[STATUS_EXECUTED_FROM], 490)
        self.assertEqual(len(status["executed"]), 10)
        self.assertEqual(len(status["branches"]), 10)
        self.assertEqual(len(status["attempts"]), 10)
        units = units_of(status)
        self.assertEqual(units[0], (490, "node490"))
        self.assertEqual(units[-1], (499, "node499:branch:r"))

    def test_a_short_run_keeps_every_step(self):
        status = run_status(_run(5))
        self.assertEqual(status.get(STATUS_EXECUTED_FROM, 0), 0)
        self.assertEqual(units_of(status)[0], (0, "node0"))


def _long_node_graph() -> Any:
    def talk(s: _S) -> dict:
        write = get_stream_writer()
        for i in range(40):
            write({"tick": i})
            time.sleep(0.02)
        return {"log": ["talk"]}

    g = StateGraph(_S)
    g.add_node("talk", talk)
    g.add_edge(START, "talk")
    g.add_edge("talk", END)
    return g.compile()


class TestStatusReads(_RunnerTestCase):
    def test_a_stream_does_not_read_the_status_on_every_poll(self):
        from diagrid.agent.langgraph import runner as runner_module

        calls: List[float] = []
        real = runner_module.custom_status

        def counting(client: Any, wf: str) -> Any:
            calls.append(time.monotonic())
            return real(client, wf)

        runner = log_runner(_long_node_graph(), "status-reads")
        http = TestClient(serve_app(runner))
        with mock.patch.object(runner_module, "custom_status", counting):
            got = frames(
                http.post(
                    "/agent/run",
                    json={"log": [], "stream": True, "stream_mode": ["custom"]},
                ).text
            )
        self.assertEqual(len([e for _, e, _ in got if e == "custom"]), 40)
        # ~0.8 s of streaming at a 10 ms read interval: about 80 polls
        self.assertLess(len(calls), 30, len(calls))


def _loop_then_ask() -> Any:
    def step(s: _S) -> dict:
        return {"log": [f"s{s.get('n', 0)}"], "n": s.get("n", 0) + 1}

    def ask(s: _S) -> dict:
        return {"log": [f"ask:{interrupt('go?')}"]}

    def route(s: _S) -> str:
        return "step" if s.get("n", 0) < 6 else "ask"

    g = StateGraph(_S)
    g.add_node("step", step)
    g.add_node("ask", ask)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", route, ["step", "ask"])
    g.add_edge("ask", END)
    return g.compile()


class TestReconnectBehindTheWindow(_RunnerTestCase):
    def test_a_reader_behind_the_window_still_gets_every_step_in_order(self):
        with mock.patch.object(activity_stream, "STATUS_STEP_WINDOW", 3):
            runner = log_runner(_loop_then_ask(), "status-window")
            http = TestClient(serve_app(runner))
            got = frames(
                http.post(
                    "/agent/run",
                    json={"log": [], "stream": True, "stream_mode": ["updates"]},
                ).text
            )
            self.assertEqual(got[-1][1], "workflow_interrupted")
            wf = got[-1][2]["workflow_id"]
            out: List[Any] = []

            def reconnect() -> None:
                out.extend(
                    frames(
                        http.get(
                            f"/agent/run/{wf}/stream", params={"stream_mode": "updates"}
                        ).text
                    )
                )

            reader = threading.Thread(target=reconnect)
            reader.start()
            time.sleep(0.3)  # the reader is following the paused run
            http.post(f"/agent/run/{wf}/resume", json={"resume": "yes"})
            reader.join(timeout=30)
        updates = [list(d)[0] for _, e, d in out if e == "updates"]
        self.assertEqual(updates[:6], ["step"] * 6)
        self.assertEqual(updates[-1], "ask")
        self.assertEqual(out[-1][1], "workflow_completed")
