# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The stream sink: its bounded buffer, many emitting threads, a retried node and
the registry that hands each run its sink."""

import asyncio
import threading
import unittest
from typing import Any, List
from unittest import mock

from langgraph.config import get_stream_writer

from diagrid.agent.core.workflow import stream as stream_module
from diagrid.agent.core.workflow.stream import (
    NODE_RESET,
    InProcessSink,
    StreamClosed,
    StreamEvent,
    StreamSinkRegistry,
    default_registry,
    get_stream_sink,
)
from diagrid.agent.langgraph.models import ChannelState, ExecuteNodeInput
from diagrid.agent.langgraph.streaming import GraphRunError, NodeStreamer
from diagrid.agent.langgraph.workflow import (
    clear_registries,
    execute_node_activity,
    register_node,
)

from .streaming_helpers import (
    INPUT,
    TOKENS,
    _chat_graph,
    _collect,
    _Ctx,
    _drain,
    _event,
    _fake_run_async,
    _payload,
    _StreamingModel,
    parse_sse,
    serve_app,
)
from .test_runner import _clear_dapr_registration, _RunnerTestCase

CAP = 3


class TestBufferCap(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patch = mock.patch.object(stream_module, "MAX_BUFFERED_EVENTS", CAP)
        patch.start()
        self.addCleanup(patch.stop)

    async def test_chunks_past_the_cap_are_dropped_and_counted(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())
        for seq in range(10):
            sink.put(_event("a1", seq))
        sink.close({"type": "workflow_completed"})
        items = await _drain(sink)
        kept = [i for i in items if isinstance(i, StreamEvent)]
        self.assertEqual([e.data for e in kept], ["a1-0", "a1-1", "a1-2"])
        self.assertEqual(items[-1].dropped, 7)

    async def test_updates_are_never_dropped(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())
        for seq in range(10):
            sink.put(_event("a1", seq))
        sink.put(_event("a1", 10, "updates"))
        sink.put(_event("a1", 11, "updates", node="other"))
        sink.close()
        items = await _drain(sink)
        updates = [
            i for i in items if isinstance(i, StreamEvent) and i.mode == "updates"
        ]
        self.assertEqual(len(updates), 2)
        self.assertEqual(items[-1].dropped, 7)

    async def test_node_reset_is_never_dropped(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())
        for seq in range(10):
            sink.put(_event("a1", seq))
        sink.put(_event("a2", 0))  # a retry starts: reset must reach the client
        sink.close()
        items = await _drain(sink)
        resets = [
            i for i in items if isinstance(i, StreamEvent) and i.mode == "node_reset"
        ]
        self.assertEqual(len(resets), 1)

    async def test_close_marker_survives_a_full_buffer(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())
        for seq in range(10):
            sink.put(_event("a1", seq))
        sink.close({"type": "workflow_completed", "x": 1})
        items = await _drain(sink)
        self.assertEqual(items[-1].final, {"type": "workflow_completed", "x": 1})

    async def test_no_drops_reports_zero(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())
        sink.put(_event("a1", 0))
        sink.close()
        self.assertEqual((await _drain(sink))[-1].dropped, 0)

    async def test_drops_from_activity_threads_are_counted(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())

        def produce(attempt: str) -> None:
            for seq in range(20):
                sink.put(_event(attempt, seq, node=attempt))

        threads = [threading.Thread(target=produce, args=(a,)) for a in ("x", "y")]
        for t in threads:
            t.start()
        for t in threads:
            await asyncio.to_thread(t.join)
        sink.close()
        items = await _drain(sink)
        kept = [i for i in items if isinstance(i, StreamEvent)]
        self.assertEqual(len(kept) + items[-1].dropped, 40)
        self.assertEqual(len(kept), CAP)


class TestDroppedIsReported(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        patch = mock.patch.object(stream_module, "MAX_BUFFERED_EVENTS", CAP)
        patch.start()
        self.addCleanup(patch.stop)

    def _runner(self, name: str, final_type: str = "workflow_completed"):
        runner = _chat_graph(name=name)
        runner._started = True

        async def flood(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            sink = runner.stream_registry.get(workflow_id)
            for seq in range(10):
                sink.put(_event("a1", seq))
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            yield {"type": final_type, "workflow_id": workflow_id, "error": "boom"}

        patch = mock.patch.object(runner, "run_async", flood)
        patch.start()
        self.addCleanup(patch.stop)
        return runner

    async def test_sse_final_event_reports_dropped(self):
        from fastapi.testclient import TestClient

        runner = self._runner("bp-sse")
        resp = TestClient(serve_app(runner)).post(
            "/agent/run", json={"task": "hi", "stream": True}
        )
        name, data = parse_sse(resp.text)[-1]
        self.assertEqual(name, "workflow_completed")
        self.assertEqual(data["dropped"], 7)

    async def test_astream_yields_a_dropped_event(self):
        runner = self._runner("bp-astream")
        items = await _collect(runner.astream(INPUT, stream_mode=["messages"]))
        self.assertEqual(items[-1], ("stream_dropped", {"dropped": 7}))

    async def test_a_failed_run_still_yields_the_dropped_event_before_it_raises(self):
        runner = self._runner("bp-failed", final_type="workflow_failed")
        items: List[Any] = []
        with self.assertRaises(GraphRunError):
            async for item in runner.astream(INPUT, stream_mode=["messages"]):
                items.append(item)
        self.assertEqual(items[-1], ("stream_dropped", {"dropped": 7}))

    async def test_final_event_has_no_dropped_key_without_drops(self):
        from fastapi.testclient import TestClient

        runner = _chat_graph(name="bp-none")
        runner._started = True

        async def quiet(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            yield {"type": "workflow_completed", "workflow_id": workflow_id}

        with mock.patch.object(runner, "run_async", quiet):
            resp = TestClient(serve_app(runner)).post(
                "/agent/run", json={"task": "hi", "stream": True}
            )
        self.assertNotIn("dropped", parse_sse(resp.text)[-1][1])


THREADS = 8


PER_THREAD = 500


class TestConcurrentEmitters(unittest.IsolatedAsyncioTestCase):
    async def test_every_chunk_of_a_multithreaded_node_is_delivered(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())
        streamer = NodeStreamer(sink, workflow_id="wf", step=0, node="n", task_id=1)
        start = threading.Barrier(THREADS)

        def work(worker: int) -> None:
            start.wait()
            for i in range(PER_THREAD):
                streamer.writer((worker, i))

        threads = [threading.Thread(target=work, args=(w,)) for w in range(THREADS)]
        for t in threads:
            t.start()
        await asyncio.to_thread(lambda: [t.join() for t in threads])
        sink.close({"type": "workflow_completed"})
        items = await _drain(sink)

        events = [i for i in items if isinstance(i, StreamEvent)]
        self.assertEqual(len(events), THREADS * PER_THREAD)
        self.assertEqual(items[-1].dropped, 0)
        # Arrival order is the order seq was assigned, so a consumer sees
        # each thread's chunks in the order that thread wrote them.
        self.assertEqual([e.seq for e in events], sorted(e.seq for e in events))
        for worker in range(THREADS):
            mine = [e.data[1] for e in events if e.data[0] == worker]
            self.assertEqual(mine, list(range(PER_THREAD)))

    async def test_sink_keeps_chunks_that_arrive_out_of_seq_order(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())
        for seq in (6, 5, 7):
            sink.put(StreamEvent("wf", 0, "n", "a1", seq, (), "custom", seq))
        sink.close()
        items = await _drain(sink)
        got = [i.data for i in items if isinstance(i, StreamEvent)]
        self.assertEqual(got, [6, 5, 7])
        self.assertEqual(items[-1].dropped, 0)


async def _reset_after_retry(**streamer_args: Any) -> Any:
    sink = InProcessSink("wf", asyncio.get_running_loop())
    first = NodeStreamer(sink, workflow_id="wf", step=1, task_id=1, **streamer_args)
    first._emit((), "custom", "partial")
    retry = NodeStreamer(sink, workflow_id="wf", step=1, task_id=1, **streamer_args)
    retry._emit((), "custom", "full")
    await asyncio.sleep(0)
    items: List[Any] = []
    while not sink._queue.empty():
        items.append(sink._queue.get_nowait())
    (reset,) = [i for i in items if i.mode == NODE_RESET]
    return reset, retry


class TestResetNode(unittest.IsolatedAsyncioTestCase):
    async def test_a_send_task_is_reset_as_its_node_with_its_path(self):
        path = ("__pregel_push", 1, False)
        reset, retry = await _reset_after_retry(
            node="work", sink_node="work#1", path=path
        )
        self.assertEqual(reset.data["node"], retry.metadata["langgraph_node"])
        self.assertEqual(reset.data["node"], "work")
        self.assertEqual(reset.data["step"], 2)
        self.assertEqual(tuple(reset.data["path"]), retry.metadata["langgraph_path"])

    async def test_a_conditional_edge_is_reset_as_its_source_node(self):
        reset, retry = await _reset_after_retry(
            node="a", sink_node="a:branch:a_cond_condition", branch="a_cond_condition"
        )
        self.assertEqual(reset.data["node"], "a")
        self.assertEqual(reset.data["step"], 2)
        # the sink's key is internal; the edge's name is what tells the reset
        # apart from a retry of the node itself
        self.assertEqual(reset.data["branch"], "a_cond_condition")
        self.assertNotIn(":branch:", str(reset.data))

    async def test_a_send_tasks_conditional_edge_names_its_task_and_edge(self):
        path = ("__pregel_push", 1, False)
        reset, retry = await _reset_after_retry(
            node="work",
            sink_node="work#1:branch:route",
            path=path,
            branch="route",
        )
        self.assertEqual(reset.data["node"], "work")
        self.assertEqual(tuple(reset.data["path"]), path)
        self.assertEqual(reset.data["branch"], "route")

    async def test_a_plain_node_reset_is_unchanged(self):
        reset, _ = await _reset_after_retry(node="chat")
        self.assertEqual(reset.data, {"node": "chat", "step": 2})


class TestRetryDedupe(unittest.IsolatedAsyncioTestCase):
    async def _drain(self, sink: InProcessSink) -> List[StreamEvent]:
        sink.close()
        return [i for i in await _drain(sink) if isinstance(i, StreamEvent)]

    async def test_new_attempt_resets_and_old_attempt_is_dropped(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())
        sink.put(_event("a1", 0))
        sink.put(_event("a2", 0))  # retry starts
        sink.put(_event("a1", 1))  # late chunk from the dead attempt
        sink.put(_event("a2", 1))
        got = await self._drain(sink)
        self.assertEqual(
            [(e.mode, e.data) for e in got],
            [
                ("messages", "a1-0"),
                ("node_reset", {"node": "n", "step": 1}),
                ("messages", "a2-0"),
                ("messages", "a2-1"),
            ],
        )

    async def test_chunks_after_accepted_updates_are_dropped(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())
        sink.put(_event("a1", 0, "updates"))
        sink.put(_event("a2", 0))
        got = await self._drain(sink)
        self.assertEqual([e.attempt_id for e in got], ["a1"])

    async def test_plain_callable_streams_messages(self):
        model = _StreamingModel()

        def plain_node(state):
            return {"messages": [model.invoke("hi")]}

        from diagrid.agent.langgraph.workflow import clear_registries, register_node

        clear_registries()
        self.addCleanup(clear_registries)
        register_node("plain", plain_node)
        sink = default_registry.open("wf-plain")
        self.addCleanup(default_registry.close, "wf-plain")
        payload = ExecuteNodeInput(
            node_name="plain", channel_state=ChannelState(), step=0
        ).to_dict()
        await asyncio.to_thread(execute_node_activity, _Ctx("wf-plain"), payload)
        got = await self._drain(sink)
        texts = [e.data[0].content for e in got if e.mode == "messages"]
        self.assertEqual([t for t in texts if t], TOKENS)

    async def test_retried_activity_emits_reset(self):
        calls = {"n": 0}

        def flaky(state):
            calls["n"] += 1
            get_stream_writer()(f"try-{calls['n']}")
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return {"x": 1}

        from diagrid.agent.langgraph.workflow import clear_registries, register_node

        clear_registries()
        self.addCleanup(clear_registries)
        register_node("flaky", flaky)
        sink = default_registry.open("wf-retry")
        self.addCleanup(default_registry.close, "wf-retry")
        payload = ExecuteNodeInput(
            node_name="flaky", channel_state=ChannelState(), step=2
        ).to_dict()
        with self.assertRaises(RuntimeError):
            await asyncio.to_thread(execute_node_activity, _Ctx("wf-retry"), payload)
        await asyncio.to_thread(execute_node_activity, _Ctx("wf-retry"), payload)
        got = await self._drain(sink)
        self.assertEqual(
            [(e.mode, e.data) for e in got if e.mode in ("custom", "node_reset")],
            [
                ("custom", "try-1"),
                ("node_reset", {"node": "flaky", "step": 3}),
                ("custom", "try-2"),
            ],
        )
        # the retry's update closes the node; only its task result follows it
        self.assertEqual([e.mode for e in got][-3:], ["updates", "tasks", "debug"])


class TestStreamSinkRegistry(unittest.IsolatedAsyncioTestCase):
    async def test_registries_do_not_share_sinks(self):
        first, second = StreamSinkRegistry(), StreamSinkRegistry()
        sink = first.open("wf")
        self.assertIs(first.get("wf"), sink)
        self.assertIsNone(second.get("wf"))
        second.open("wf")  # the same id is fine in another registry

    async def test_get_stream_sink_uses_the_default_registry(self):
        sink = default_registry.open("wf-default")
        self.addCleanup(default_registry.close, "wf-default")
        self.assertIs(default_registry.get("wf-default"), sink)
        self.assertIs(get_stream_sink("wf-default"), sink)

    async def test_get_stream_sink_takes_an_explicit_registry(self):
        registry = StreamSinkRegistry()
        sink = registry.open("wf-x")
        self.assertIs(get_stream_sink("wf-x", registry=registry), sink)
        self.assertIsNone(get_stream_sink("wf-x"))

    async def test_close_marker_is_public(self):
        registry = StreamSinkRegistry()
        sink = registry.open("wf")
        sink.close(final={"type": "workflow_failed"})
        await asyncio.sleep(0)
        item = await sink.get()
        self.assertIsInstance(item, StreamClosed)
        self.assertEqual(item.final, {"type": "workflow_failed"})


class TestSinkOnAClosedLoop(unittest.TestCase):
    """An activity still writing after the serving loop is gone must not fail."""

    def _sink(self) -> InProcessSink:
        loop = asyncio.new_event_loop()
        sink = InProcessSink("wf-gone", loop)
        loop.close()
        return sink

    def test_put_and_close_do_not_raise(self):
        sink = self._sink()
        sink.put(StreamEvent("wf-gone", 0, "n", "a1", 0, (), "custom", "x"))
        sink.put(StreamEvent("wf-gone", 0, "n", "a1", 1, (), "updates", {"n": None}))
        sink.close({"type": "workflow_completed"})

    def test_later_puts_are_no_ops(self):
        sink = self._sink()
        sink.put(StreamEvent("wf-gone", 0, "n", "a1", 0, (), "custom", "x"))
        with mock.patch.object(sink, "_enqueue") as enqueue:
            sink.put(StreamEvent("wf-gone", 0, "n", "a1", 1, (), "custom", "y"))
            sink.close()
        enqueue.assert_not_called()


class TestRunnerOwnsRegistry(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    def test_each_runner_gets_its_own_registry_by_default(self):
        first = _chat_graph(name="reg-a")
        self._reset()
        second = _chat_graph(name="reg-b")
        self.assertIsInstance(first.stream_registry, StreamSinkRegistry)
        self.assertIsNot(first.stream_registry, second.stream_registry)
        self.assertIsNot(first.stream_registry, default_registry)

    def test_registry_can_be_injected(self):
        registry = StreamSinkRegistry()
        runner = _chat_graph(name="reg-inj", stream_registry=registry)
        self.assertIs(runner.stream_registry, registry)

    def _reset(self):
        # DaprWorkflowGraphRunner registers module-level functions with dapr.
        _clear_dapr_registration()

    async def test_astream_streams_through_the_runners_registry(self):
        registry = StreamSinkRegistry()
        runner = _chat_graph(name="reg-run", stream_registry=registry)
        runner._started = True
        seen = []
        original = registry.open

        def spy(workflow_id):
            seen.append(workflow_id)
            return original(workflow_id)

        with (
            mock.patch.object(registry, "open", spy),
            mock.patch.object(runner, "run_async", _fake_run_async(runner)),
        ):
            chunks = await _collect(
                runner.astream(INPUT, workflow_id="wf-own", stream_mode="messages")
            )
        self.assertEqual(seen, ["wf-own"])
        texts = [c[0].content for c in chunks]
        self.assertEqual([t for t in texts if t], TOKENS)
        self.assertIsNone(registry.get("wf-own"))

    async def test_a_runner_never_reads_another_runners_sinks(self):
        mine = _chat_graph(name="reg-mine")
        other_registry = StreamSinkRegistry()
        other_sink = other_registry.open("wf-iso")
        mine._started = True
        with mock.patch.object(mine, "run_async", _fake_run_async(mine)):
            await _collect(mine.astream(INPUT, workflow_id="wf-iso"))
        other_sink.close()
        await asyncio.sleep(0)
        # Nothing from the run above was delivered to the other registry.
        self.assertIsInstance(await other_sink.get(), StreamClosed)


LOGGER = "diagrid.agent.core.workflow.stream"


class TestRegistryRemembersClosedRuns(unittest.IsolatedAsyncioTestCase):
    async def test_closed_ids_are_remembered_and_reopened_ones_forgotten(self):
        registry = StreamSinkRegistry()
        self.assertFalse(registry.was_closed("wf"))
        sink = registry.open("wf")
        self.assertFalse(registry.was_closed("wf"))
        registry.close("wf", sink)
        self.assertTrue(registry.was_closed("wf"))
        registry.open("wf")
        self.assertFalse(registry.was_closed("wf"))

    async def test_closing_an_unknown_id_remembers_nothing(self):
        registry = StreamSinkRegistry()
        registry.close("never-opened")
        self.assertFalse(registry.was_closed("never-opened"))


# The registry's two bounded memories: (remember an id, is it still remembered?).
BOUNDED_MEMORIES = {
    "closed-runs": (
        lambda registry, wf: registry.close(wf, registry.open(wf)),
        lambda registry, wf: registry.was_closed(wf),
    ),
    "first-report": (
        lambda registry, wf: registry.first_report(wf),
        lambda registry, wf: not registry.first_report(wf),
    ),
}


class TestMemoriesAreBounded(unittest.IsolatedAsyncioTestCase):
    async def test_only_the_most_recent_runs_are_remembered(self):
        limit = stream_module._MAX_REPORTED_RUNS
        for name, (remember, remembered) in BOUNDED_MEMORIES.items():
            with self.subTest(memory=name):
                registry = StreamSinkRegistry()
                for i in range(limit + 1):
                    remember(registry, f"wf-{i}")
                # wf-0 is the oldest of limit + 1: forgotten, the newest is not
                self.assertFalse(remembered(registry, "wf-0"))
                self.assertTrue(remembered(registry, f"wf-{limit}"))

    async def test_a_run_is_reported_first_only_once(self):
        registry = StreamSinkRegistry()
        self.assertTrue(registry.first_report("wf-0"))
        self.assertFalse(registry.first_report("wf-0"))


class TestLateActivityOfAClosedStream(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_registries()
        self.addCleanup(clear_registries)
        register_node("n", lambda state: {"x": 1})

    async def test_no_replica_warning_when_this_process_closed_the_stream(self):
        registry = StreamSinkRegistry()
        registry.close("wf-gone", registry.open("wf-gone"))
        await asyncio.sleep(0)
        with self.assertLogs(LOGGER, level="DEBUG") as logs:
            out = execute_node_activity(
                _Ctx("wf-gone"), _payload(), stream_registry=registry
            )
        self.assertEqual(out["writes"][0]["value"], 1)
        self.assertEqual([r.levelname for r in logs.records], ["DEBUG"])
        self.assertIn("closed", logs.records[0].getMessage())
        self.assertNotIn("replica", logs.records[0].getMessage())

    async def test_a_run_this_process_never_opened_still_warns(self):
        registry = StreamSinkRegistry()
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            execute_node_activity(_Ctx("wf-else"), _payload(), stream_registry=registry)
        self.assertIn("another process", logs.records[0].getMessage())


class TestStreamRegistry(unittest.IsolatedAsyncioTestCase):
    async def test_reused_workflow_id_is_rejected_and_first_sink_kept(self):
        first = default_registry.open("same")
        self.addCleanup(default_registry.close, "same")
        with self.assertRaises(ValueError):
            default_registry.open("same")
        self.assertIs(get_stream_sink("same"), first)

    async def test_close_with_stale_sink_keeps_current(self):
        first = default_registry.open("id1")
        default_registry.close("id1", first)
        second = default_registry.open("id1")
        self.addCleanup(default_registry.close, "id1")
        default_registry.close("id1", first)  # stale handle
        self.assertIs(get_stream_sink("id1"), second)
