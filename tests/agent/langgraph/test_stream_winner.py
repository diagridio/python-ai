# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The stream follows the attempt of a node whose output the workflow used.

Dapr may run an activity twice (a completion that got lost, a redelivery) and
the workflow uses one of the results. The stream must show that attempt's
chunks, so what a client saw and the run's state agree.
"""

import json
import operator
import threading
from typing import Any
from unittest import mock

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph.models import GraphWorkflowInput

from .log_helpers import LogClient, StatusCtx, log_runner
from .streaming_helpers import serve_app
from .test_reconnect_helpers import frames
from .test_runner import _RunnerTestCase


class _S(TypedDict, total=False):
    log: Annotated[list, operator.add]


def _graph() -> Any:
    runs = {"n": 0}
    lock = threading.Lock()

    def flaky(s: _S) -> dict:
        with lock:
            runs["n"] += 1
            n = runs["n"]
        return {"log": [f"attempt-{n}"]}

    def after(s: _S) -> dict:
        return {"log": ["after"]}

    g = StateGraph(_S)
    g.add_node("flaky", flaky)
    g.add_node("after", after)
    g.add_edge(START, "flaky")
    g.add_edge("flaky", "after")
    g.add_edge("after", END)
    return g.compile()


class _RedeliverCtx(StatusCtx):
    """Runs ``flaky``'s activity twice and hands the workflow the second result."""

    def call_activity(self, activity: Any, *, input: Any, retry_policy: Any = None):
        out = super().call_activity(activity, input=input, retry_policy=retry_policy)
        if isinstance(input, dict) and input.get("node_name") == "flaky":
            out = super().call_activity(
                activity, input=input, retry_policy=retry_policy
            )
        return out


class _RedeliverClient(LogClient):
    def _work(self, input: Any, instance_id: str) -> None:
        from diagrid.agent.langgraph.workflow import agent_workflow

        wf_input = GraphWorkflowInput.from_dict(json.loads(json.dumps(input)))
        ctx = _RedeliverCtx(self._activity_runner, instance_id, self)
        with mock.patch("diagrid.agent.langgraph.workflow.when_all", side_effect=list):
            workflow = agent_workflow(ctx, json.loads(json.dumps(wf_input.to_dict())))
            result = None
            try:
                while True:
                    result = workflow.send(result)
            except StopIteration as stop:
                self.done[instance_id] = stop.value


def _flaky_updates(got: Any) -> list:
    return [d["flaky"]["log"] for _, e, d in got if e == "updates" and "flaky" in d]


class TestWinningAttempt(_RunnerTestCase):
    def _run(self, name: str) -> Any:
        runner = log_runner(_graph(), name)
        runner._workflow_client = _RedeliverClient(runner)  # type: ignore[assignment]
        http = TestClient(serve_app(runner))
        got = frames(
            http.post(
                "/agent/run",
                json={"log": [], "stream": True, "stream_mode": ["updates"]},
            ).text
        )
        return http, got

    def test_a_live_stream_ends_on_the_attempt_the_run_used(self):
        _, got = self._run("winner-live")
        output = got[-1][2]["output"]["log"]
        self.assertEqual(output, ["attempt-2", "after"])
        self.assertEqual(_flaky_updates(got)[-1], ["attempt-2"])
        if len(_flaky_updates(got)) > 1:  # the first attempt was seen: reset it
            self.assertIn("node_reset", [e for _, e, _ in got])

    def test_a_replay_shows_only_the_attempt_the_run_used(self):
        http, got = self._run("winner-replay")
        wf = got[0][2]["workflow_id"]
        again = frames(
            http.get(f"/agent/run/{wf}/stream", params={"stream_mode": "updates"}).text
        )
        self.assertEqual(_flaky_updates(again), [["attempt-2"]])
        self.assertNotIn("node_reset", [e for _, e, _ in again])
