# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``stream_mode="tasks"`` and ``"debug"``, and the refusal of ``"checkpoints"``.

A node's activity sends a task's start event when it starts the node and its
result next to the node's ``updates`` chunk. ``debug`` wraps the same events
with the step, a timestamp and their type. Without a checkpointer that is all
native ``debug`` holds, which is the only case compared here.
"""

import asyncio
import unittest
from collections import Counter
from typing import Any, Dict, List, Tuple

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from diagrid.agent.core.workflow.stream import StreamSinkRegistry
from diagrid.agent.langgraph.workflow import (
    clear_registries,
    execute_node_activity,
    register_node,
)

from .golden_harness import (
    _both,
    _canonical,
    _FakeDaprTest,
    _golden_graph,
    _Out,
)
from .streaming_helpers import (
    INPUT,
    _chat_graph,
    _collect,
    _Ctx,
    _drain,
    _payload,
    serve_app,
)
from .test_runner import _RunnerTestCase
from .test_streaming_golden import _fan_out_graph, _router_graph


def _stable(items: List[Tuple[str, Any]], keep_ids: bool = True) -> List:
    """Task ids numbered by first appearance (or dropped), timestamps blanked."""
    ids: Dict[str, str] = {}

    def fix(payload: Dict[str, Any]) -> Dict[str, Any]:
        task_id = ids.setdefault(payload["id"], f"task-{len(ids)}")
        return {**payload, "id": task_id if keep_ids else "<id>"}

    out: List[Tuple[str, Any]] = []
    for mode, payload in items:
        if mode == "tasks":
            out.append((mode, fix(payload)))
        elif mode == "debug":
            debug = {**payload, "timestamp": "<ts>", "payload": fix(payload["payload"])}
            out.append((mode, debug))
        else:
            out.append((mode, payload))
    return out


class TestTasksMatchNative(_FakeDaprTest):
    async def test_a_sequential_graph(self):
        native, ours = await _both(_golden_graph(), "tasks-seq", INPUT, ["tasks"])
        self.assertTrue(native)
        self._assert_same_sequence(_stable(native), _stable(ours))

    async def test_the_result_follows_the_nodes_update(self):
        native, ours = await _both(
            _golden_graph(), "tasks-updates", INPUT, ["updates", "tasks", "custom"]
        )
        self._assert_same_sequence(_stable(native), _stable(ours))

    async def test_a_node_with_a_router(self):
        native, ours = await _both(
            _router_graph(), "tasks-router", {"log": []}, ["updates", "tasks", "custom"]
        )
        self._assert_same_sequence(_stable(native), _stable(ours))

    async def test_a_parallel_step(self):
        native, ours = await _both(
            _fan_out_graph(), "tasks-fan", {"log": []}, ["tasks", "updates"]
        )
        self.assertEqual(
            _canonical(_stable(ours, keep_ids=False)),
            _canonical(_stable(native, keep_ids=False)),
        )

    async def test_send_tasks(self):
        graph = StateGraph(_Out)
        graph.add_node("work", lambda arg: {"out": [arg["i"]]})
        graph.add_conditional_edges(
            START, lambda state: [Send("work", {"i": i}) for i in (1, 2)], ["work"]
        )
        graph.add_edge("work", END)
        native, ours = await _both(
            graph.compile(), "tasks-send", {"out": []}, ["tasks"]
        )
        self.assertEqual(
            _canonical(_stable(ours, keep_ids=False)),
            _canonical(_stable(native, keep_ids=False)),
        )

    async def test_a_tasks_start_and_result_share_one_id(self):
        _, ours = await _both(_fan_out_graph(), "tasks-ids", {"log": []}, ["tasks"])
        counts = Counter(payload["id"] for _, payload in ours)
        self.assertEqual(len(counts), 4)
        self.assertEqual(set(counts.values()), {2})


class TestDebugMatchesNative(_FakeDaprTest):
    async def test_without_a_checkpointer_it_is_the_task_events(self):
        native, ours = await _both(_golden_graph(), "debug-seq", INPUT, ["debug"])
        self.assertEqual(
            {p["type"] for _, p in native}, {"task", "task_result"}
        )  # no checkpoint events without a checkpointer
        self._assert_same_sequence(_stable(native), _stable(ours))

    async def test_next_to_tasks_and_updates(self):
        modes = ["updates", "tasks", "debug"]
        native, ours = await _both(_router_graph(), "debug-all", {"log": []}, modes)
        self._assert_same_sequence(_stable(native), _stable(ours))


class TestAFailedAttempt(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_registries()
        self.addCleanup(clear_registries)

    async def test_it_sends_its_start_but_no_result(self):
        def boom(state: Any) -> dict:
            raise RuntimeError("nope")

        register_node("boom", boom)
        registry = StreamSinkRegistry()
        sink = registry.open("wf-boom")
        with self.assertRaises(RuntimeError):
            await asyncio.to_thread(
                execute_node_activity,
                _Ctx("wf-boom"),
                _payload("boom"),
                stream_registry=registry,
            )
        sink.close()
        events = [e for e in await _drain(sink) if hasattr(e, "mode")]
        tasks = [e.data for e in events if e.mode == "tasks"]
        self.assertEqual(
            [sorted(t) for t in tasks], [["id", "input", "name", "triggers"]]
        )
        debug = [e.data["type"] for e in events if e.mode == "debug"]
        self.assertEqual(debug, ["task"])


CHECKPOINTS_REASON = "workflow history"


class TestCheckpointsAreRefused(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_astream_says_why(self):
        runner = _chat_graph(name="no-checkpoints")
        runner._started = True
        for mode in ("checkpoints", ["messages", "checkpoints"]):
            with self.subTest(stream_mode=mode):
                with self.assertRaises(ValueError) as caught:
                    await _collect(runner.astream(INPUT, stream_mode=mode))
                self.assertIn("checkpoints", str(caught.exception))
                self.assertIn(CHECKPOINTS_REASON, str(caught.exception))

    def test_a_streaming_request_gets_a_422_that_says_why(self):
        from fastapi.testclient import TestClient

        runner = _chat_graph(name="no-checkpoints-http")
        runner._started = True
        client = TestClient(serve_app(runner))
        resp = client.post(
            "/agent/run",
            json={"task": "hi", "stream": True, "stream_mode": ["checkpoints"]},
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn(CHECKPOINTS_REASON, resp.json()["detail"])
