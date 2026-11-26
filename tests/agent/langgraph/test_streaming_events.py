# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""How a streamed run ends and what its control events say: failures, a stream
that did not reach this process, and how ``astream`` routes the control events."""

import asyncio
import json
import re
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence
from unittest import mock

from dapr.ext.workflow import WorkflowStatus
from langchain_core.messages import AIMessageChunk
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from diagrid.agent.core.workflow import stream
from diagrid.agent.core.workflow.stream import (
    CONTROL_MODES,
    NODE_RESET,
    STREAM_UNAVAILABLE,
    UPDATES,
    WORKFLOW_STARTED,
    InProcessSink,
    StreamClosed,
    StreamEvent,
)
from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.models import ChannelState, GraphWorkflowOutput
from diagrid.agent.langgraph.stream_events import route_astream_item
from diagrid.agent.langgraph.streaming import (
    STREAM_RUN_KEY,
    GraphRunError,
    NodeStreamer,
    StreamControl,
    stream_unavailable_event,
)
from diagrid.agent.langgraph.workflow import (
    clear_registries,
    execute_node_activity,
    register_node,
)

from .streaming_helpers import (
    ENDINGS,
    FAILED,
    FAILED_ERROR,
    INPUT,
    STARTED,
    _chat_graph,
    _collect,
    _completed,
    _Ctx,
    _payload,
    _ScriptedCase,
    parse_sse,
    serve_app,
)
from .test_runner import _run, _RunnerTestCase


class TestEveryFailureEnding(_ScriptedCase):
    """A run that does not end in a result is a failure, in both transports."""

    async def test_astream_raises(self):
        for name, row in ENDINGS.items():
            with self.subTest(ending=name):
                self.setUp()
                runner = self._runner(row["events"], f"end-raise-{name}")
                with self.assertRaisesRegex(GraphRunError, re.escape(row["message"])):
                    await _collect(runner.astream(INPUT, stream_mode="updates"))

    async def test_sse_ends_with_workflow_failed(self):
        for name, row in ENDINGS.items():
            with self.subTest(ending=name):
                self.setUp()
                runner = self._runner(row["events"], f"end-sse-{name}")
                resp = self._client(runner).post(
                    "/agent/run", json={"task": "hi", "stream": True, "thread_id": "t9"}
                )
                event, data = parse_sse(resp.text)[-1]
                self.assertEqual(event, "workflow_failed")
                self.assertEqual(data["thread_id"], "t9")
                self.assertEqual(data["error"]["message"], row["message"])
                if "error_type" in row:
                    self.assertEqual(data["error"]["error_type"], row["error_type"])
                if "error" in row:
                    self.assertEqual(data["error"], row["error"])
                if row.get("workflow_id", "wf-fixed"):
                    self.assertEqual(data["workflow_id"], "wf-fixed")
                else:
                    self.assertTrue(data["workflow_id"])


class TestStatusErrorIsAFailure(_ScriptedCase):
    async def test_completed_status_still_completes(self):
        runner = self._runner([STARTED, _completed("completed")], "st-ok")
        resp = self._client(runner).post(
            "/agent/run", json={"task": "hi", "stream": True}
        )
        self.assertEqual(parse_sse(resp.text)[-1][0], "workflow_completed")

    async def test_real_poll_path_keeps_the_workflow_error(self):
        """``run_async`` itself must carry the output's error to the caller."""
        runner = _chat_graph(name="st-poll")
        runner._started = True
        output = GraphWorkflowOutput(
            output={},
            channel_state=ChannelState(),
            steps=0,
            status="error",
            error="Node 'chat' not found in registry",
        )
        client = mock.Mock()
        client.get_workflow_state.return_value = SimpleNamespace(
            runtime_status=WorkflowStatus.COMPLETED,
            serialized_output=json.dumps(output.to_dict()),
            serialized_custom_status=None,
            failure_details=None,
        )
        runner._workflow_client = client
        with self.assertRaisesRegex(RuntimeError, "not found in registry"):
            await _collect(
                runner.astream(INPUT, stream_mode="updates", poll_interval=0)
            )


class TestFailureDetailSurvives(_ScriptedCase):
    async def test_astream_exception_carries_the_final(self):
        runner = self._runner([STARTED, FAILED], "detail-b")
        with self.assertRaises(RuntimeError) as caught:
            await _collect(
                runner.astream(INPUT, thread_id="t-2", stream_mode="updates")
            )
        self.assertEqual(str(caught.exception), "activity blew up")
        self.assertEqual(caught.exception.final["error"], FAILED_ERROR)
        self.assertEqual(caught.exception.final["thread_id"], "t-2")

    async def test_unexpected_exception_becomes_a_failed_final(self):
        runner = _chat_graph(name="detail-c")
        runner._started = True

        async def broken(*a, **kw):
            raise ConnectionError("sidecar gone")
            yield  # pragma: no cover

        with mock.patch.object(runner, "run_async", broken):
            resp = self._client(runner).post(
                "/agent/run", json={"task": "hi", "stream": True}
            )
        name, data = parse_sse(resp.text)[-1]
        self.assertEqual(name, "workflow_failed")
        self.assertEqual(data["error"]["message"], "sidecar gone")
        self.assertEqual(data["error"]["error_type"], "ConnectionError")


LOGGER = "diagrid.agent.langgraph.workflow"


class TestActivityWithoutASink(unittest.TestCase):
    def setUp(self):
        clear_registries()
        self.addCleanup(clear_registries)
        register_node("n", lambda state: {"x": 1})

    def test_streamed_run_without_a_sink_warns_once_per_run(self):
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            first = execute_node_activity(_Ctx("wf-lost"), _payload(streamed=True))
            execute_node_activity(_Ctx("wf-lost"), _payload(streamed=True, step=1))
            execute_node_activity(_Ctx("wf-other"), _payload(streamed=True))
        warnings = [r for r in logs.records if "no stream sink" in r.getMessage()]
        self.assertEqual(len(warnings), 2)  # one per run, not per node
        self.assertIn("wf-lost", warnings[0].getMessage())
        self.assertIn("wf-other", warnings[1].getMessage())
        # The node itself is unaffected.
        self.assertEqual(first["writes"][0]["value"], 1)

    def test_a_run_that_is_not_streamed_stays_quiet(self):
        with self.assertNoLogs(LOGGER, level="WARNING"):
            execute_node_activity(_Ctx("wf-plain"), _payload(streamed=False))


class TestStreamedRunsAreMarked(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_the_workflow_config_says_the_run_is_streamed(self):
        runner = _chat_graph(name="mark")
        runner._started = True
        seen: Dict[str, Any] = {}

        async def spy(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            seen.update(config or {})
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            yield {"type": "workflow_completed", "workflow_id": workflow_id}

        with mock.patch.object(runner, "run_async", spy):
            await _collect(runner.astream(INPUT, config={"tags": ["t"]}))
        self.assertIs(seen[STREAM_RUN_KEY], True)
        self.assertEqual(seen["tags"], ["t"])  # the caller's config is kept


def _chunk(step: int, node: str, mode: str, attempt: str = "a1") -> StreamEvent:
    return StreamEvent("wf", step, node, attempt, 0, (), mode, {node: None})


class _FakeRun(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    """A runner whose ``run_async`` puts ``chunks`` in its sink, then completes
    with ``final`` (the keys of the ``workflow_completed`` event)."""

    def _runner(self, name: str, chunks: Sequence[StreamEvent] = (), **final: Any):
        runner = _chat_graph(name=name)
        runner._started = True

        async def run(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            sink = runner.stream_registry.get(workflow_id)
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            for chunk in chunks:
                sink.put(chunk)
            await asyncio.sleep(0)
            yield {
                "type": "workflow_completed",
                "workflow_id": workflow_id,
                "steps": len(final.get("executed", ())),
                "status": "completed",
                **final,
            }

        patch = mock.patch.object(runner, "run_async", run)
        patch.start()
        self.addCleanup(patch.stop)
        return runner


class TestStreamUnavailableEvent(_FakeRun):
    async def test_nodes_ran_but_no_chunk_arrived(self):
        runner = self._runner("un-none", steps=2)
        items = await _collect(
            runner.astream(INPUT, workflow_id="wf-u1", stream_mode=["updates"])
        )
        self.assertEqual(len(items), 1)
        name, data = items[0]
        self.assertEqual(name, "stream_unavailable")
        self.assertEqual(data["workflow_id"], "wf-u1")
        self.assertEqual(data["missing_steps"], [1, 2])
        self.assertIn("another process", data["reason"])

    async def test_only_some_steps_arrived(self):
        runner = self._runner("un-some", [_chunk(1, "n", "updates")], steps=2)
        items = await _collect(runner.astream(INPUT, stream_mode=["updates"]))
        self.assertEqual([m for m, _ in items], ["updates", "stream_unavailable"])
        self.assertEqual(items[-1][1]["missing_steps"], [1])

    async def test_sse_sends_it_before_the_final_event(self):
        from fastapi.testclient import TestClient

        runner = self._runner("un-sse", steps=1)
        resp = TestClient(serve_app(runner)).post(
            "/agent/run", json={"task": "hi", "stream": True}
        )
        names = [n for n, _ in parse_sse(resp.text)]
        self.assertEqual(
            names, ["workflow_started", "stream_unavailable", "workflow_completed"]
        )

    async def test_nothing_is_reported_when_no_node_ran(self):
        runner = self._runner("un-zero", steps=0)
        self.assertEqual(await _collect(runner.astream(INPUT)), [])

    async def test_a_failed_run_is_just_a_failure(self):
        # A node failure has steps and executed, and its node sent no chunk: it
        # must not be blamed on another replica. Single-mode astream hides
        # control events, so look at them through on_control and a list of modes.
        runner = self._runner("un-fail", executed=[["n"]], status="error", error="boom")
        controls: List[Any] = []
        with self.assertRaisesRegex(RuntimeError, "boom"):
            await _collect(
                runner.astream(INPUT, workflow_id="wf-f1", on_control=controls.append)
            )
        self.assertEqual([c.kind for c in controls], ["workflow_started"])

        seen: List[Any] = []
        with self.assertRaisesRegex(RuntimeError, "boom"):
            async for item in runner.astream(
                INPUT, workflow_id="wf-f2", stream_mode=["updates"]
            ):
                seen.append(item)
        self.assertEqual(seen, [])

    async def test_sse_of_a_failed_run_has_no_stream_unavailable(self):
        from fastapi.testclient import TestClient

        runner = self._runner(
            "un-fail-sse", executed=[["n"]], status="error", error="boom"
        )
        resp = TestClient(serve_app(runner)).post(
            "/agent/run", json={"task": "hi", "stream": True}
        )
        names = [n for n, _ in parse_sse(resp.text)]
        self.assertEqual(names, ["workflow_started", "workflow_failed"])


class TestDecision(unittest.TestCase):
    FINAL = {"type": "workflow_completed", "steps": 2, "executed": [["a"], ["b", "c"]]}

    def test_every_executed_node_streamed(self):
        seen = {(0, "a"), (1, "b"), (1, "c")}
        self.assertIsNone(stream_unavailable_event(self.FINAL, seen, "wf"))

    def test_one_node_of_a_parallel_step_ran_elsewhere(self):
        event = stream_unavailable_event(self.FINAL, {(0, "a"), (1, "b")}, "wf")
        assert event is not None
        self.assertEqual(event.data["missing_steps"], [2])
        self.assertEqual(event.data["missing_nodes"], [{"step": 2, "node": "c"}])

    def test_a_whole_step_missing_lists_all_its_nodes(self):
        event = stream_unavailable_event(self.FINAL, {(0, "a")}, "wf")
        assert event is not None
        self.assertEqual(event.data["missing_steps"], [2])
        self.assertEqual(
            event.data["missing_nodes"],
            [{"step": 2, "node": "b"}, {"step": 2, "node": "c"}],
        )

    def test_an_older_workflow_output_falls_back_to_steps(self):
        final = {"type": "workflow_completed", "steps": 2}
        event = stream_unavailable_event(final, {(0, "a")}, "wf")
        assert event is not None
        self.assertEqual(event.data["missing_steps"], [2])
        self.assertEqual(event.data["missing_nodes"], [])
        self.assertIsNone(stream_unavailable_event(final, {(0, "a"), (1, "z")}, "wf"))


class TestThroughAstream(_FakeRun):
    async def test_parallel_node_on_another_replica_is_reported(self):
        runner = self._runner(
            "un-par", [_chunk(0, "a", "updates")], executed=[["a", "b"]]
        )
        items = await _collect(runner.astream(INPUT, stream_mode=["updates"]))
        self.assertEqual([m for m, _ in items], ["updates", "stream_unavailable"])
        self.assertEqual(items[-1][1]["missing_nodes"], [{"step": 1, "node": "b"}])

    async def test_a_retry_that_finished_on_another_replica_is_reported(self):
        # Attempt 1 streamed some tokens here, then failed; attempt 2 ran
        # elsewhere, so no updates chunk for the node ever arrived.
        runner = self._runner(
            "un-retry", [_chunk(0, "a", "messages")], executed=[["a"]]
        )
        items = await _collect(runner.astream(INPUT, stream_mode=["messages"]))
        # The partial text reached the client; so does the warning about it.
        self.assertEqual([m for m, _ in items], ["messages", "stream_unavailable"])
        self.assertEqual(items[-1][1]["missing_nodes"], [{"step": 1, "node": "a"}])

    async def test_complete_stream_reports_nothing(self):
        runner = self._runner(
            "un-full",
            [_chunk(0, "a", "updates"), _chunk(0, "b", "updates")],
            executed=[["a", "b"]],
        )
        items = await _collect(runner.astream(INPUT, stream_mode=["updates"]))
        self.assertEqual([m for m, _ in items], ["updates", "updates"])

    async def test_executed_is_not_part_of_the_http_body_or_the_published_result(self):
        from fastapi.testclient import TestClient

        runner = _chat_graph(name="un-http")
        runner._started = True

        async def run(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            # like the real run_async, report it to a streamed run only
            extra = (
                {"executed": [["chat"]]} if (config or {}).get(STREAM_RUN_KEY) else {}
            )
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            yield {
                "type": "workflow_completed",
                "workflow_id": workflow_id,
                "steps": 1,
                "status": "completed",
                **extra,
            }

        published: List[Any] = []
        fake = mock.Mock()
        fake.publish.side_effect = lambda topic, data: published.append(data)
        with (
            mock.patch.object(runner, "run_async", run),
            mock.patch("diagrid.agent.core.pubsub.DaprPubSub", return_value=fake),
        ):
            client = TestClient(serve_app(runner, pubsub_name="ps", publish_topic="o"))
            plain = client.post("/agent/run", json={"task": "hi"}).json()
            sse = client.post("/agent/run", json={"task": "hi", "stream": True})
        self.assertNotIn("executed", parse_sse(sse.text)[-1][1])
        self.assertNotIn("executed", plain)
        self.assertEqual(len(published), 2)
        self.assertNotIn("executed", published[1])


class TestWorkflowReportsWhatRan(_RunnerTestCase):
    def test_output_lists_the_nodes_of_each_step(self):
        from operator import add

        class State(TypedDict, total=False):
            log: Annotated[list, add]

        graph = StateGraph(State)
        for name in ("s", "a", "b", "c"):
            graph.add_node(name, lambda state, n=name: {"log": [n]})
        graph.add_edge(START, "s")
        graph.add_edge("s", "a")
        graph.add_edge("s", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="executed")
        out = _run(runner, {"log": []})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.executed[0], ["s"])
        self.assertCountEqual(out.executed[1], ["a", "b"])
        self.assertEqual(out.executed[-1], ["c"])
        self.assertEqual(len(out.executed), out.steps)


class TestResetStepMatchesTheChunksItResets(unittest.IsolatedAsyncioTestCase):
    async def test_node_reset_step_is_the_langgraph_step_of_the_chunks(self):
        sink = InProcessSink("wf", asyncio.get_running_loop())
        # a loop graph: 'inc' runs in workflow steps 1, 2 and 3 (langgraph_step
        # 2, 3, 4); its second invocation is retried
        for step in (1, 2):
            first = NodeStreamer(
                sink, workflow_id="wf", step=step, node="inc", task_id=1
            )
            first._emit((), "custom", "partial")
        retry = NodeStreamer(sink, workflow_id="wf", step=2, node="inc", task_id=1)
        retry._emit((), "custom", "full")
        await asyncio.sleep(0)

        items = []
        while not sink._queue.empty():
            items.append(sink._queue.get_nowait())
        (reset,) = [i for i in items if i.mode == NODE_RESET]
        self.assertEqual(reset.data["node"], "inc")
        self.assertEqual(reset.data["step"], retry.metadata["langgraph_step"])
        self.assertEqual(reset.data["step"], 3)


def _control_event(mode: str, node: str) -> StreamEvent:
    return StreamEvent("wf", 0, node, "attempt", 0, (), mode, {})


class TestControlModes(unittest.IsolatedAsyncioTestCase):
    def test_control_modes_are_the_events_about_the_stream_itself(self):
        self.assertEqual(
            set(CONTROL_MODES), {WORKFLOW_STARTED, NODE_RESET, STREAM_UNAVAILABLE}
        )

    def test_updates_are_a_stream_mode_not_a_control_event(self):
        # the runner delivers control events whatever modes were asked for
        self.assertNotIn(UPDATES, CONTROL_MODES)

    async def test_a_full_buffer_keeps_control_events_and_updates_only(self):
        modes = [
            "messages",
            "custom",
            UPDATES,
            WORKFLOW_STARTED,
            NODE_RESET,
            STREAM_UNAVAILABLE,
        ]
        sink = InProcessSink("wf", asyncio.get_running_loop())
        with mock.patch.object(stream, "MAX_BUFFERED_EVENTS", 1):
            sink.put(_control_event("messages", "filler"))
            for mode in modes:
                sink.put(_control_event(mode, mode))
            await asyncio.sleep(0)
            sink.close()
            kept = []
            while True:
                item = await sink.get()
                if isinstance(item, StreamClosed):
                    break
                kept.append(item.mode)
        self.assertEqual(
            sorted(set(kept)), sorted({*CONTROL_MODES, UPDATES, "messages"})
        )
        # the filler is the only chunk of a droppable mode that got in
        self.assertEqual(kept.count("messages"), 1)
        self.assertNotIn("custom", kept)


RESET = StreamEvent("wf", 0, "chat", "a2", 0, (), NODE_RESET, {"node": "chat"})


CHUNK = StreamEvent("wf", 0, "chat", "a2", 1, (), "messages", ("msg", {"k": 1}))


STARTED_EVENT = StreamEvent(
    "wf", 0, "", "", 0, (), WORKFLOW_STARTED, {"workflow_id": "wf", "thread_id": "t"}
)


UNAVAILABLE = StreamEvent(
    "wf", 0, "", "", 0, (), STREAM_UNAVAILABLE, {"workflow_id": "wf", "reason": "r"}
)


DROPPED = {"type": "workflow_completed", "dropped": 3}


def _control(item: StreamEvent) -> StreamControl:
    return StreamControl(item.mode, item.data)


# How astream treats one item of the event stream. The columns are the way it
# was called: a single mode (the bare chunks of native astream) or a list of
# modes (``(mode, chunk)`` tuples), with or without ``on_control``.
SINGLE, SINGLE_CB, LIST, LIST_CB = (
    "single",
    "single+on_control",
    "list",
    "list+on_control",
)


# item: {call: (yielded, handed to on_control, warns the reset went unseen)}
ROUTES = {
    "chunk": (
        CHUNK,
        {
            SINGLE: ([("msg", {"k": 1})], [], False),
            SINGLE_CB: ([("msg", {"k": 1})], [], False),
            LIST: ([("messages", ("msg", {"k": 1}))], [], False),
            LIST_CB: ([("messages", ("msg", {"k": 1}))], [], False),
        },
    ),
    "node_reset": (
        RESET,
        {
            SINGLE: ([], [], True),
            SINGLE_CB: ([], [_control(RESET)], False),
            LIST: ([(NODE_RESET, {"node": "chat"})], [], False),
            LIST_CB: ([(NODE_RESET, {"node": "chat"})], [_control(RESET)], False),
        },
    ),
    "stream_unavailable": (
        UNAVAILABLE,
        {
            SINGLE: ([], [], False),
            SINGLE_CB: ([], [_control(UNAVAILABLE)], False),
            LIST: ([(STREAM_UNAVAILABLE, UNAVAILABLE.data)], [], False),
            LIST_CB: (
                [(STREAM_UNAVAILABLE, UNAVAILABLE.data)],
                [_control(UNAVAILABLE)],
                False,
            ),
        },
    ),
    # only informs on_control: native astream has no such mode
    "workflow_started": (
        STARTED_EVENT,
        {
            SINGLE: ([], [], False),
            SINGLE_CB: ([], [_control(STARTED_EVENT)], False),
            LIST: ([], [], False),
            LIST_CB: ([], [_control(STARTED_EVENT)], False),
        },
    ),
    "dropped": (
        DROPPED,
        {
            SINGLE: ([], [], False),
            SINGLE_CB: ([], [StreamControl("stream_dropped", {"dropped": 3})], False),
            LIST: ([("stream_dropped", {"dropped": 3})], [], False),
            LIST_CB: (
                [("stream_dropped", {"dropped": 3})],
                [StreamControl("stream_dropped", {"dropped": 3})],
                False,
            ),
        },
    ),
}


class TestRouteAstreamItem(unittest.TestCase):
    def test_every_item_in_every_call_style(self):
        for name, (item, calls) in ROUTES.items():
            for call, (yielded, handed, warns) in calls.items():
                with self.subTest(item=name, call=call):
                    got: List[StreamControl] = []
                    on_control = got.append if call.endswith("on_control") else None
                    is_list = call.startswith("list")
                    if warns:
                        with self.assertLogs(EVENTS_LOGGER, level="WARNING") as logs:
                            out, error = route_astream_item(
                                item, ["messages"], is_list, on_control
                            )
                        # said once, with the node and the way to handle it
                        (warning,) = logs.records
                        self.assertIn("on_control", warning.getMessage())
                        self.assertIn("'chat'", warning.getMessage())
                    else:
                        with self.assertNoLogs(EVENTS_LOGGER, level="WARNING"):
                            out, error = route_astream_item(
                                item, ["messages"], is_list, on_control
                            )
                    self.assertEqual(out, yielded)
                    self.assertEqual(got, handed)
                    self.assertIsNone(error)

    def test_a_chunk_of_a_mode_nobody_asked_for_is_dropped(self):
        for is_list in (False, True):
            with self.subTest(is_list=is_list):
                out, _ = route_astream_item(CHUNK, ["custom"], is_list, None)
                self.assertEqual(out, [])

    def test_a_failed_final_comes_back_as_the_error_after_what_came_with_it(self):
        final = {"type": "workflow_failed", "dropped": 2, "error": {"message": "x"}}
        out, error = route_astream_item(final, ["messages"], True, None)
        self.assertEqual(out, [("stream_dropped", {"dropped": 2})])
        self.assertIsInstance(error, GraphRunError)


EVENTS_LOGGER = "diagrid.agent.langgraph.stream_events"


def _token(attempt: str, text: str, seq: int) -> StreamEvent:
    return StreamEvent(
        "wf", 0, "n", attempt, seq, (), "messages", (AIMessageChunk(content=text), {})
    )


class TestRetryInSingleMode(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    def _runner(self, name: str):
        runner = _chat_graph(name=name)
        runner._started = True

        async def run(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            sink = runner.stream_registry.get(workflow_id)
            for event in (
                _token("A", "Hel", 0),
                _token("A", "lo ", 1),
                _token("B", "Hello", 0),  # Dapr retried the activity
                _token("B", " world", 1),
            ):
                sink.put(event)
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            await asyncio.sleep(0)
            yield {"type": "workflow_completed", "workflow_id": workflow_id}

        patch = mock.patch.object(runner, "run_async", run)
        patch.start()
        self.addCleanup(patch.stop)
        return runner

    async def test_a_consumer_who_cannot_see_the_reset_is_warned(self):
        runner = self._runner("retry-single")
        with self.assertLogs(EVENTS_LOGGER, level="WARNING") as logs:
            chunks = await _collect(runner.astream(INPUT, stream_mode="messages"))
        # the attempt that was abandoned is still in what was yielded ...
        self.assertEqual("".join(c[0].content for c in chunks), "Hello Hello world")
        # ... and that is said, once per reset, with the way to handle it
        (warning,) = [r for r in logs.records if "node_reset" in r.getMessage()]
        self.assertIn("on_control", warning.getMessage())
        self.assertIn("'n'", warning.getMessage())
