# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A streamed run alongside other things going on: a client that stays or leaves
and the result that is published either way, runs that overlap, and attempts
that overlap."""

import asyncio
import gc
import json
import threading
import unittest
import weakref
from contextlib import aclosing
from typing import Any, Dict, List
from unittest import mock

from fastapi.testclient import TestClient
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph

from diagrid.agent.core.workflow.stream import (
    InProcessSink,
    StreamClosed,
    StreamEvent,
    StreamSinkRegistry,
)
from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.workflow import (
    clear_registries,
    execute_node_activity,
    register_node,
)

from .streaming_helpers import (
    ENDINGS,
    INPUT,
    STARTED,
    _chat_graph,
    _collect,
    _completed,
    _Ctx,
    _fake_run_async,
    _payload,
    _ScriptedCase,
    serve_app,
)
from .test_runner import _RunnerTestCase

PUBLISH_WAIT = 5


class _PublishCase(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    def _setup(self, name: str):
        """A runner whose run waits on ``self.release`` before it completes."""
        self.release = asyncio.Event()
        self.published: List[Dict[str, Any]] = []
        self.topics: List[str] = []
        runner = _chat_graph(name=name)
        runner._started = True

        async def run_async(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            yield {
                "type": "workflow_started",
                "workflow_id": workflow_id,
                "thread_id": thread_id,
            }
            runner.stream_registry.get(workflow_id).put(
                StreamEvent(workflow_id, 0, "chat", "a", 0, (), "custom", "first")
            )
            await self.release.wait()
            yield {
                "type": "workflow_completed",
                "workflow_id": workflow_id,
                "output": {"x": 1},
                "steps": 1,
                "status": "completed",
            }

        patch = mock.patch.object(runner, "run_async", run_async)
        patch.start()
        self.addCleanup(patch.stop)

        fake = mock.Mock()
        fake.publish.side_effect = lambda topic, data: (
            self.topics.append(topic),
            self.published.append(data),
        )
        with mock.patch("diagrid.agent.core.pubsub.DaprPubSub", return_value=fake):
            app = serve_app(runner, pubsub_name="ps", publish_topic="out")
        self.route = next(
            r
            for r in app.routes
            if getattr(r, "path", "") == "/agent/run" and "POST" in r.methods
        )
        return runner, app

    async def _published(self, count: int = 1) -> None:
        async def wait() -> None:
            while len(self.published) < count:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait(), PUBLISH_WAIT)
        await asyncio.sleep(0.05)  # a second publish would show up now


class TestDisconnect(_PublishCase):
    async def test_client_leaves_at_the_terminal_frame(self):
        runner, _ = self._setup("pub-end")
        self.release.set()
        response = await self.route.endpoint({"task": "hi", "stream": True})
        body = response.body_iterator
        async for frame in body:
            if "event: workflow_completed" in frame:
                break  # the client goes away while the generator is suspended here
        await body.aclose()
        await self._published()
        self.assertEqual(len(self.published), 1)
        self.assertEqual(self.published[0]["type"], "workflow_completed")

    async def test_client_leaves_mid_run_and_the_run_still_publishes(self):
        runner, _ = self._setup("pub-mid")
        response = await self.route.endpoint({"task": "hi", "stream": True})
        body = response.body_iterator
        self.assertIn("event: workflow_started", await body.__anext__())
        self.assertIn("event: custom", await body.__anext__())
        self.assertEqual(len(runner.stream_registry._sinks), 1)  # the run's own
        await body.aclose()  # what the server does when the client hangs up
        # The stream is closed at once, the run is still going.
        self.assertEqual(self.published, [])
        self.assertEqual(runner.stream_registry._sinks, {})
        self.release.set()
        await self._published()
        self.assertEqual(len(self.published), 1)
        result = self.published[0]
        self.assertEqual(result["type"], "workflow_completed")
        self.assertEqual(result["output"], {"x": 1})
        self.assertTrue(result["instance_id"].startswith("graph-"))

    async def test_a_client_that_stays_causes_exactly_one_publish(self):
        runner, _ = self._setup("pub-stay")
        self.release.set()
        response = await self.route.endpoint({"task": "hi", "stream": True})
        frames = [frame async for frame in response.body_iterator]
        self.assertIn("event: workflow_completed", frames[-1])
        await self._published()
        self.assertEqual(len(self.published), 1)
        self.assertEqual(self.topics, ["out"])
        self.assertEqual(self.published[0]["type"], "workflow_completed")

    async def test_a_real_asgi_disconnect_mid_run_still_publishes(self):
        runner, app = self._setup("pub-asgi")
        body = json.dumps({"task": "hi", "stream": True}).encode()
        first_frame = asyncio.Event()
        sent: List[Dict[str, Any]] = []
        delivered = False

        async def receive() -> Dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await first_frame.wait()
            return {"type": "http.disconnect"}

        async def send(message: Dict[str, Any]) -> None:
            sent.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                first_frame.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/agent/run",
            "raw_path": b"/agent/run",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("t", 1),
            "server": ("t", 80),
            "scheme": "http",
        }
        await asyncio.wait_for(app(scope, receive, send), PUBLISH_WAIT)
        self.assertEqual(self.published, [])
        self.release.set()
        await self._published()
        self.assertEqual(len(self.published), 1)
        self.assertEqual(self.published[0]["type"], "workflow_completed")


class TestDetachedPumpIsKeptAlive(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_a_pump_nobody_else_references_survives_garbage_collection(self):
        runner = _chat_graph(name="gc-pump")
        runner._started = True
        gate: List["weakref.ref[asyncio.Event]"] = []

        async def run_async(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            # Only the pump's own frames reference this event, and the test only
            # a weak reference: whatever keeps the pump alive is what keeps it.
            event = asyncio.Event()
            gate.append(weakref.ref(event))
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            await event.wait()
            yield {"type": "workflow_completed", "workflow_id": workflow_id}

        published: List[Any] = []
        fake = mock.Mock()
        fake.publish.side_effect = lambda topic, data: published.append(data)
        with (
            mock.patch.object(runner, "run_async", run_async),
            mock.patch("diagrid.agent.core.pubsub.DaprPubSub", return_value=fake),
        ):
            app = serve_app(runner, pubsub_name="ps", publish_topic="out")
        route = next(
            r
            for r in app.routes
            if getattr(r, "path", "") == "/agent/run" and "POST" in r.methods
        )
        with mock.patch.object(runner, "run_async", run_async):
            response = await route.endpoint({"task": "hi", "stream": True})
            body = response.body_iterator
            await body.__anext__()  # workflow_started
            await body.aclose()  # the client leaves; the pump carries on alone
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0)
        event = gate[0]()
        self.assertIsNotNone(event, "the detached pump was garbage collected")
        event.set()
        for _ in range(100):
            if published:
                break
            await asyncio.sleep(0.01)
        self.assertEqual([p["type"] for p in published], ["workflow_completed"])


class TestPublishedPayloadsMatch(_ScriptedCase):
    def _publish(self, runner, body: Dict[str, Any]) -> Dict[str, Any]:
        published: List[Any] = []
        fake = mock.Mock()
        fake.publish.side_effect = lambda topic, data: published.append(data)
        with mock.patch("diagrid.agent.core.pubsub.DaprPubSub", return_value=fake):
            client = self._client(runner, pubsub_name="ps", publish_topic="out")
        client.post("/agent/run", json=body)
        self.assertEqual(len(published), 1)
        return published[0]

    async def test_every_failure_ending_publishes_the_same_result_streamed_or_not(self):
        for name, row in ENDINGS.items():
            with self.subTest(ending=name):
                self.setUp()
                runner = self._runner(row["events"], f"pub-{name}")
                body = {"task": "hi", "thread_id": "t-pub"}
                plain = self._publish(runner, body)
                streamed = self._publish(runner, {**body, "stream": True})
                self.assertEqual(streamed, plain)
                if "error" in row:  # a workflow_failed final carries its detail
                    self.assertEqual(plain["error"], row["error"])
                    self.assertEqual(plain["workflow_id"], "wf-fixed")
                    self.assertEqual(plain["thread_id"], "t-pub")

    async def test_success_publishes_the_same_shape_streamed_or_not(self):
        runner = self._runner([STARTED, _completed("completed")], "pub-ok")
        body = {"task": "hi", "thread_id": "t-pub"}
        plain = self._publish(runner, body)
        streamed = self._publish(runner, {**body, "stream": True})
        self.assertEqual(streamed, plain)


class TestPublishLeavesTheLoop(_RunnerTestCase, unittest.TestCase):
    def _post(self, body: Dict[str, Any]) -> Dict[str, List[int]]:
        """Thread idents of the event loop and of each publish call."""
        runner = _chat_graph(name="offloop-pub")
        runner._started = True
        idents: Dict[str, List[int]] = {"loop": [], "publish": []}

        async def run_async(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            idents["loop"].append(threading.get_ident())
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            yield {
                "type": "workflow_completed",
                "workflow_id": workflow_id,
                "output": {"x": 1},
                "steps": 0,
                "status": "completed",
            }

        fake = mock.Mock()
        fake.publish.side_effect = lambda topic, data: idents["publish"].append(
            threading.get_ident()
        )
        with (
            mock.patch.object(runner, "run_async", run_async),
            mock.patch("diagrid.agent.core.pubsub.DaprPubSub", return_value=fake),
        ):
            app = serve_app(runner, pubsub_name="ps", publish_topic="out")
            TestClient(app).post("/agent/run", json=body)
        return idents

    def _assert_off_the_loop(self, body: Dict[str, Any]) -> None:
        idents = self._post(body)
        self.assertEqual(len(idents["publish"]), 1)
        self.assertNotIn(idents["publish"][0], idents["loop"])

    def test_a_run_publishes_off_the_loop_streamed_or_not(self):
        for name, body in (
            ("plain", {"task": "hi"}),
            ("streamed", {"task": "hi", "stream": True}),
        ):
            with self.subTest(run=name):
                self.setUp()
                self._assert_off_the_loop(body)


WAIT = 10


class TestTwoRunsAtOnce(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_runs_each_get_only_their_own_chunks(self):
        both_inside = threading.Barrier(2, timeout=WAIT)
        reached_barrier: List[str] = []

        def node(state: MessagesState) -> dict:
            last = state["messages"][-1]
            who = last["content"] if isinstance(last, dict) else last.content
            writer = get_stream_writer()
            writer(f"{who}-1")
            # Hold both activities inside their node at the same moment; a
            # sequential run would time out here.
            both_inside.wait()
            reached_barrier.append(who)
            writer(f"{who}-2")
            return {"messages": [{"role": "ai", "content": f"done {who}"}]}

        graph = StateGraph(MessagesState)
        graph.add_node("n", node)
        graph.add_edge(START, "n")
        graph.add_edge("n", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="concurrent")
        runner._started = True

        async def one(who: str) -> List[Any]:
            return await _collect(
                runner.astream(
                    {"messages": [{"role": "user", "content": who}]},
                    workflow_id=f"wf-{who}",
                    stream_mode="custom",
                )
            )

        with (
            mock.patch.object(runner, "run_async", _fake_run_async(runner)),
            mock.patch("diagrid.agent.langgraph.workflow.when_all", side_effect=list),
        ):
            a, b = await asyncio.gather(one("A"), one("B"))

        self.assertCountEqual(reached_barrier, ["A", "B"])
        self.assertEqual(a, ["A-1", "A-2"])
        self.assertEqual(b, ["B-1", "B-2"])
        self.assertIsNone(runner.stream_registry.get("wf-A"))
        self.assertIsNone(runner.stream_registry.get("wf-B"))


TASK_MODES = ("tasks", "debug")


class TestRetryWhileTheOldAttemptIsStillRunning(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_registries()
        self.addCleanup(clear_registries)

    async def test_late_chunks_of_the_old_attempt_are_dropped(self):
        a_streamed = threading.Event()
        b_finished = threading.Event()
        calls: List[str] = []
        lock = threading.Lock()

        def node(state) -> dict:
            with lock:
                attempt = "A" if not calls else "B"
                calls.append(attempt)
            writer = get_stream_writer()
            if attempt == "A":
                writer("A-early")
                a_streamed.set()
                self.assertTrue(b_finished.wait(WAIT))
                writer("A-late")  # the engine already moved on to attempt B
                return {"x": "from-A"}
            self.assertTrue(a_streamed.wait(WAIT))
            writer("B-only")
            return {"x": "from-B"}

        register_node("n", node)
        registry = StreamSinkRegistry()
        sink = registry.open("wf-retry")

        def run(ctx_id: int) -> Any:
            return execute_node_activity(
                _Ctx("wf-retry", ctx_id),
                _payload("n", step=3, streamed=False),
                stream_registry=registry,
            )

        async def attempt_b() -> Any:
            out = await asyncio.to_thread(run, 2)
            b_finished.set()
            return out

        a_task = asyncio.create_task(asyncio.to_thread(run, 1))
        self.assertTrue(await asyncio.to_thread(a_streamed.wait, WAIT))
        out_b = await attempt_b()
        out_a = await a_task  # A finishes last, with its own (stale) result

        sink.close({"type": "workflow_completed"})
        items: List[Any] = []
        while True:
            item = await sink.get()
            if isinstance(item, StreamClosed):
                break
            items.append(item)

        # a task's start and result events are not what this is about
        shown = [(e.mode, e.data) for e in items if e.mode not in TASK_MODES]
        self.assertEqual(
            shown,
            [
                ("custom", "A-early"),
                ("node_reset", {"node": "n", "step": 4}),
                ("custom", "B-only"),
                ("updates", {"n": {"x": "from-B"}}),
            ],
        )
        # Dapr keeps only one attempt's result; the sink must never show a
        # chunk the surviving attempt did not produce.
        self.assertEqual(out_b["writes"][0]["value"], "from-B")
        self.assertEqual(out_a["writes"][0]["value"], "from-A")

    async def test_late_chunks_are_dropped_while_the_new_attempt_is_running(self):
        """A writes after B started but before B finished: B must not be mixed
        with A's text."""
        b_streamed = threading.Event()
        a_wrote_late = threading.Event()
        a_streamed = threading.Event()
        calls: List[str] = []
        lock = threading.Lock()

        def node(state) -> dict:
            with lock:
                attempt = "A" if not calls else "B"
                calls.append(attempt)
            writer = get_stream_writer()
            if attempt == "A":
                writer("A-early")
                a_streamed.set()
                self.assertTrue(b_streamed.wait(WAIT))
                writer("A-late")
                a_wrote_late.set()
                return {"x": "from-A"}
            self.assertTrue(a_streamed.wait(WAIT))
            writer("B-1")
            b_streamed.set()
            self.assertTrue(a_wrote_late.wait(WAIT))
            writer("B-2")
            return {"x": "from-B"}

        register_node("n", node)
        registry = StreamSinkRegistry()
        sink = registry.open("wf-retry2")

        def run(task_id: int) -> Any:
            return execute_node_activity(
                _Ctx("wf-retry2", task_id),
                _payload("n", streamed=False),
                stream_registry=registry,
            )

        a_task = asyncio.create_task(asyncio.to_thread(run, 1))
        self.assertTrue(await asyncio.to_thread(a_streamed.wait, WAIT))
        await asyncio.to_thread(run, 2)
        await a_task

        sink.close({"type": "workflow_completed"})
        shown: List[Any] = []
        while True:
            item = await sink.get()
            if isinstance(item, StreamClosed):
                break
            if item.mode not in TASK_MODES:
                shown.append((item.mode, item.data))
        self.assertEqual(
            shown,
            [
                ("custom", "A-early"),
                ("node_reset", {"node": "n", "step": 1}),
                ("custom", "B-1"),
                ("custom", "B-2"),
                ("updates", {"n": {"x": "from-B"}}),
            ],
        )


class TestClientGoesAway(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    def _runner(self, name: str):
        runner = DaprWorkflowGraphRunner(graph=_slow_graph(), name=name)
        runner._started = True
        return runner

    async def test_abandoned_astream_closes_the_sink_and_stops_the_run(self):
        runner = self._runner("gone-astream")
        run_cancelled = asyncio.Event()

        async def endless(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            sink = runner.stream_registry.get(workflow_id)
            sink.put(StreamEvent(workflow_id, 0, "n", "a", 0, (), "custom", "first"))
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            try:
                await asyncio.sleep(3600)
                yield {}  # pragma: no cover
            except asyncio.CancelledError:
                run_cancelled.set()
                raise

        with mock.patch.object(runner, "run_async", endless):
            stream = runner.astream(INPUT, workflow_id="wf-gone", stream_mode="custom")
            self.assertEqual(await stream.__anext__(), "first")
            self.assertIsNotNone(runner.stream_registry.get("wf-gone"))
            await stream.aclose()
        self.assertIsNone(runner.stream_registry.get("wf-gone"))
        self.assertTrue(run_cancelled.is_set())

    async def _endless_run(self, runner):
        async def endless(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            sink = runner.stream_registry.get(workflow_id)
            sink.put(StreamEvent(workflow_id, 0, "n", "a", 0, (), "custom", "first"))
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            await asyncio.sleep(3600)
            yield {}  # pragma: no cover

        return endless

    async def test_aclosing_releases_the_stream_on_break(self):
        runner = self._runner("gone-aclosing")
        with mock.patch.object(runner, "run_async", await self._endless_run(runner)):
            async with aclosing(
                runner.astream(INPUT, workflow_id="wf-ac", stream_mode="custom")
            ) as stream:
                async for _ in stream:
                    break
            self.assertIsNone(runner.stream_registry.get("wf-ac"))

    async def test_bare_break_keeps_the_stream_open_until_it_is_closed(self):
        # What the astream docstring says: break alone does not close the
        # stream while the generator is still referenced.
        runner = self._runner("gone-break")
        with mock.patch.object(runner, "run_async", await self._endless_run(runner)):
            stream = runner.astream(INPUT, workflow_id="wf-br", stream_mode="custom")
            async for _ in stream:
                break
            await asyncio.sleep(0.05)
            self.assertIsNotNone(runner.stream_registry.get("wf-br"))
            await stream.aclose()
            self.assertIsNone(runner.stream_registry.get("wf-br"))

    async def test_late_chunks_after_the_client_left_are_ignored(self):
        runner = self._runner("gone-late")
        registry = runner.stream_registry
        captured: List[InProcessSink] = []
        original = registry.open

        def spy(workflow_id):
            sink = original(workflow_id)
            captured.append(sink)
            return sink

        async def endless(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            registry.get(workflow_id).put(
                StreamEvent(workflow_id, 0, "n", "a", 0, (), "custom", "first")
            )
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            await asyncio.sleep(3600)
            yield {}  # pragma: no cover

        with (
            mock.patch.object(registry, "open", spy),
            mock.patch.object(runner, "run_async", endless),
        ):
            stream = runner.astream(INPUT, workflow_id="wf-late", stream_mode="custom")
            await stream.__anext__()
            await stream.aclose()
        (sink,) = captured
        # The activity is still running on its thread and keeps writing.
        sink.put(StreamEvent("wf-late", 0, "n", "a", 1, (), "custom", "late"))
        await asyncio.sleep(0)
        # Only the close marker is delivered: the late chunk was refused.
        self.assertIsInstance(await sink.get(), StreamClosed)
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(sink.get(), 0.1)


def _slow_graph() -> Any:
    graph = StateGraph(MessagesState)
    graph.add_node("n", lambda state: {})
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return graph.compile()
