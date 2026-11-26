# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``interrupt()`` in a node: the run pauses durably and resumes with ``Command(resume=...)``."""

import json
import operator
import time
import unittest
from typing import Any

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, interrupt
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.models import ChannelState, ExecuteNodeInput
from diagrid.agent.langgraph.streaming import STREAM_RUN_KEY
from diagrid.agent.langgraph.workflow import execute_node_activity

from .log_helpers import LogClient, log_runner
from .streaming_helpers import _collect, _Ctx, serve_app
from .test_reconnect_helpers import frames
from .test_runner import _RunnerTestCase


class _State(TypedDict, total=False):
    answer: str
    log: Annotated[list, operator.add]


def _approval_graph(asks: int = 1) -> Any:
    def ask(state: _State) -> dict:
        answers = [interrupt({"question": f"approve #{i}?"}) for i in range(asks)]
        return {"answer": "+".join(str(a) for a in answers), "log": ["ask"]}

    def after(state: _State) -> dict:
        return {"log": [f"after:{state['answer']}"]}

    graph = StateGraph(_State)
    graph.add_node("ask", ask)
    graph.add_node("after", after)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", "after")
    graph.add_edge("after", END)
    return graph.compile()


def _wait(predicate: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)


def _custom(runner: Any, wf: str) -> dict:
    raw = runner._workflow_client.custom.get(wf)
    return json.loads(raw) if raw else {}


INPUT = {"log": ["start"]}


class TestActivity(_RunnerTestCase):
    def _payload(self, resume: Any = None) -> dict:
        return ExecuteNodeInput(
            node_name="ask",
            channel_state=ChannelState(values={"log": []}, versions={"log": 1}),
            config={STREAM_RUN_KEY: True},
            step=0,
            resume=resume,
        ).to_dict()

    def test_a_node_that_interrupts_returns_its_interrupt_not_an_error(self):
        log_runner(_approval_graph(), "act-int")  # registers the graph's nodes
        out = execute_node_activity(_Ctx("wf-act"), self._payload())
        self.assertIsNone(out["error"])
        self.assertEqual(out["writes"], [])
        self.assertEqual(len(out["interrupts"]), 1)
        self.assertEqual(out["interrupts"][0]["value"], {"question": "approve #0?"})
        self.assertTrue(out["interrupts"][0]["id"])

    def test_the_resume_value_is_what_interrupt_returns(self):
        log_runner(_approval_graph(), "act-res")
        out = execute_node_activity(_Ctx("wf-act2"), self._payload(resume=["yes"]))
        self.assertNotIn("interrupts", out)
        writes = {w["channel"]: w["value"] for w in out["writes"]}
        self.assertEqual(writes["answer"], "yes")
        self.assertNotIn("__resume__", writes)


class TestInterruptOverHttp(_RunnerTestCase):
    def _start(self, name: str, asks: int = 1) -> Any:
        runner = log_runner(_approval_graph(asks), name)
        http = TestClient(serve_app(runner))
        resp = http.post(
            "/agent/run", json={**INPUT, "stream": True, "thread_id": "t-1"}
        )
        return runner, http, frames(resp.text)

    def test_the_stream_shows_the_interrupt_and_ends_while_the_run_waits(self):
        runner, http, got = self._start("h-int")
        names = [e for _, e, _ in got]
        self.assertEqual(names[-1], "workflow_interrupted")
        final = got[-1][2]
        wf = final["workflow_id"]
        self.assertEqual(final["interrupts"][0]["value"], {"question": "approve #0?"})
        updates = [d for _, e, d in got if e == "updates"]
        self.assertIn("__interrupt__", updates[-1])
        # durably waiting: the workflow is still running, not done
        self.assertNotIn(wf, runner._workflow_client.done)
        status = http.get(f"/agent/run/{wf}").json()
        self.assertIn("RUNNING", status["status"])
        self.assertTrue(status["interrupted"])
        self.assertEqual(status["interrupts"][0]["value"], {"question": "approve #0?"})

    def test_resuming_continues_the_same_run_and_streams_the_rest(self):
        runner, http, got = self._start("h-res")
        wf = got[-1][2]["workflow_id"]
        resp = http.post(
            f"/agent/run/{wf}/resume", json={"resume": "yes", "stream": True}
        )
        self.assertEqual(resp.status_code, 200)
        rest = frames(resp.text)
        self.assertEqual(rest[0][1], "workflow_started")
        self.assertEqual(rest[0][2]["workflow_id"], wf)
        self.assertEqual(rest[0][2]["thread_id"], "t-1")
        updates = [d for _, e, d in rest if e == "updates"]
        self.assertEqual([list(u) for u in updates], [["ask"], ["after"]])
        self.assertEqual(rest[-1][1], "workflow_completed")
        self.assertEqual(rest[-1][2]["output"]["answer"], "yes")
        self.assertEqual(rest[-1][2]["output"]["log"], ["start", "ask", "after:yes"])

    def test_a_plain_resume_returns_the_result(self):
        runner, http, got = self._start("h-plain")
        wf = got[-1][2]["workflow_id"]
        body = http.post(f"/agent/run/{wf}/resume", json={"resume": "ok"}).json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["output"]["answer"], "ok")

    def test_each_interrupt_waits_for_its_own_resume(self):
        runner, http, got = self._start("h-two", asks=2)
        wf = got[-1][2]["workflow_id"]
        first = frames(
            http.post(
                f"/agent/run/{wf}/resume", json={"resume": "a", "stream": True}
            ).text
        )
        self.assertEqual(first[-1][1], "workflow_interrupted")
        self.assertEqual(
            first[-1][2]["interrupts"][0]["value"], {"question": "approve #1?"}
        )
        second = frames(
            http.post(
                f"/agent/run/{wf}/resume", json={"resume": "b", "stream": True}
            ).text
        )
        self.assertEqual(second[-1][2]["output"]["answer"], "a+b")

    def test_resume_is_refused_when_it_cannot_apply(self):
        runner, http, got = self._start("h-bad")
        wf = got[-1][2]["workflow_id"]
        self.assertEqual(
            http.post(f"/agent/run/{wf}/resume", json={"value": 1}).status_code, 422
        )
        self.assertEqual(
            http.post("/agent/run/nope/resume", json={"resume": 1}).status_code, 404
        )
        http.post(f"/agent/run/{wf}/resume", json={"resume": "x"})
        _wait(lambda: wf in runner._workflow_client.done)
        self.assertEqual(
            http.post(f"/agent/run/{wf}/resume", json={"resume": 1}).status_code, 409
        )


class TestInterruptInPython(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_astream_with_a_command_resumes_the_run(self):
        runner = log_runner(_approval_graph(), "py-res")
        first = await _collect(
            runner.astream(
                INPUT, workflow_id="wf-py", stream_mode="updates", poll_interval=0.01
            )
        )
        (interrupted,) = first[-1]["__interrupt__"]
        self.assertIsInstance(interrupted, Interrupt)
        self.assertEqual(interrupted.value, {"question": "approve #0?"})
        rest = await _collect(
            runner.astream(
                Command(resume="go"),
                workflow_id="wf-py",
                stream_mode="updates",
                poll_interval=0.01,
            )
        )
        self.assertEqual(rest[-1], {"after": {"log": ["after:go"]}})

    async def test_a_command_without_a_run_to_resume_is_refused(self):
        runner = log_runner(_approval_graph(), "py-bad")
        with self.assertRaises(ValueError):
            await _collect(runner.astream(Command(resume="go")))


class TestInvoke(_RunnerTestCase):
    def test_invoke_returns_the_interrupt_instead_of_waiting_forever(self):
        runner = log_runner(_approval_graph(), "inv")
        out = runner.invoke(INPUT, workflow_id="wf-inv", poll_interval=0.01, timeout=10)
        (interrupted,) = out["__interrupt__"]
        self.assertEqual(interrupted.value, {"question": "approve #0?"})


class TestInProcessResume(_RunnerTestCase):
    def test_resume_streams_in_process_too(self):
        # no store: the in-process transport
        runner = DaprWorkflowGraphRunner(graph=_approval_graph(), name="ip-int")
        runner._started = True
        runner._workflow_client = LogClient(runner)  # type: ignore[assignment]
        http = TestClient(serve_app(runner))
        got = frames(http.post("/agent/run", json={**INPUT, "stream": True}).text)
        self.assertEqual(got[-1][1], "workflow_interrupted")
        wf = got[-1][2]["workflow_id"]
        rest = frames(
            http.post(
                f"/agent/run/{wf}/resume", json={"resume": "y", "stream": True}
            ).text
        )
        self.assertEqual(rest[-1][1], "workflow_completed")
        self.assertIn("after", [list(d)[0] for _, e, d in rest if e == "updates"])
