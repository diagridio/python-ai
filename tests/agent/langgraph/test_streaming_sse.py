# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The SSE side of ``POST /agent/run``: the request that switches it on, what a
client needs to survive a real network, how values are put on the wire, and
what happens when a chunk cannot be encoded or the run cannot be scheduled."""

import asyncio
import datetime
import decimal
import unittest
import uuid
from typing import Any, Dict, List, Tuple
from unittest import mock

from dapr.ext.workflow import WorkflowStatus
from fastapi.testclient import TestClient
from langchain_core.load import load
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from diagrid.agent.core.workflow import stream as stream_module
from diagrid.agent.core.workflow.stream import (
    StreamEvent,
    format_sse,
)
from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.streaming import normalize_final, to_wire

from .golden_harness import _golden_graph
from .streaming_helpers import (
    INPUT,
    _chat_graph,
    _collect,
    _completed_state,
    _fake_run_async,
    _PollCase,
    _state,
    parse_sse,
    serve_app,
)
from .test_runner import _RunnerTestCase


def _runner(name: str):
    runner = _chat_graph(name=name)
    runner._started = True
    patch = mock.patch.object(runner, "run_async", _fake_run_async(runner))
    patch.start()
    return runner, patch


class TestSseTransport(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_the_first_event_names_the_workflow(self):
        runner, patch = _runner("sse-first")
        self.addCleanup(patch.stop)
        resp = TestClient(serve_app(runner)).post(
            "/agent/run", json={"task": "hi", "stream": True, "thread_id": "t-9"}
        )
        events = parse_sse(resp.text)
        name, data = events[0]
        self.assertEqual(name, "workflow_started")
        self.assertEqual(data["thread_id"], "t-9")
        self.assertEqual(data["workflow_id"], events[-1][1]["workflow_id"])
        self.assertEqual([n for n, _ in events].count("workflow_started"), 1)

    async def test_response_headers_stop_proxies_buffering(self):
        runner, patch = _runner("sse-headers")
        self.addCleanup(patch.stop)
        resp = TestClient(serve_app(runner)).post(
            "/agent/run", json={"task": "hi", "stream": True}
        )
        self.assertEqual(resp.headers["cache-control"], "no-cache")
        self.assertEqual(resp.headers["x-accel-buffering"], "no")

    async def test_idle_stream_sends_comment_keepalives(self):
        runner = _chat_graph(name="sse-idle")
        runner._started = True
        release = asyncio.Event()

        async def silent(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            await release.wait()  # idle until the test has seen its keepalives
            yield {"type": "workflow_completed", "workflow_id": workflow_id}

        app = serve_app(runner)
        route = next(
            r
            for r in app.routes
            if getattr(r, "path", "") == "/agent/run" and "POST" in r.methods
        )
        frames: List[str] = []
        with (
            mock.patch.object(runner, "run_async", silent),
            mock.patch.object(stream_module, "SSE_KEEPALIVE_SECONDS", 0.01),
        ):
            response = await route.endpoint({"task": "hi", "stream": True})

            async def consume() -> None:
                async for frame in response.body_iterator:
                    frames.append(frame)
                    # No timing assertion: the run waits for as many keepalives
                    # as the test asks for, however slow the machine is.
                    if len([f for f in frames if f.startswith(":")]) == 2:
                        release.set()

            # the bound only turns "no keepalive ever came" into a failure
            await asyncio.wait_for(consume(), 30)
        comments = [f for f in frames if f.startswith(":")]
        self.assertGreaterEqual(len(comments), 2)
        self.assertEqual({c.strip() for c in comments}, {": keepalive"})
        # Comments carry no event and do not use up an event id.
        events = [f for f in frames if not f.startswith(":")]
        self.assertEqual([f.split("\n")[0] for f in events], ["id: 1", "id: 2"])
        self.assertEqual(parse_sse("".join(events))[-1][0], "workflow_completed")

    async def test_astream_reports_the_workflow_through_on_control_only(self):
        runner, patch = _runner("sse-astream")
        self.addCleanup(patch.stop)
        got: List[Any] = []
        items = await _collect(
            runner.astream(
                INPUT,
                workflow_id="wf-known",
                thread_id="t-3",
                stream_mode=["custom"],
                on_control=got.append,
            )
        )
        self.assertEqual([m for m, _ in items], ["custom"])
        (started,) = [c for c in got if c.kind == "workflow_started"]
        self.assertEqual(started.data["workflow_id"], "wf-known")
        self.assertEqual(started.data["thread_id"], "t-3")


class Answer(BaseModel):
    text: str
    score: float


class _State(TypedDict, total=False):
    result: Any
    when: Any


WHEN = datetime.datetime(2026, 1, 1)


class TestToWire(unittest.TestCase):
    def test_values_that_are_not_serializable_keep_their_data(self):
        update = {"n": {"result": Answer(text="hi", score=0.5), "when": WHEN}}
        wire = to_wire("updates", update)
        self.assertEqual(wire["n"]["result"], {"text": "hi", "score": 0.5})
        self.assertEqual(wire["n"]["when"], str(WHEN))
        # and what comes out is load()-able
        self.assertEqual(load(wire, allowed_objects="messages"), wire)

    def test_uuid_decimal_set_and_nested_models(self):
        ident = uuid.UUID(int=7)
        wire = to_wire(
            "updates",
            {
                "n": {
                    "id": ident,
                    "price": decimal.Decimal("1.5"),
                    "tags": {"a"},
                    "items": [Answer(text="x", score=1.0)],
                }
            },
        )
        self.assertEqual(wire["n"]["id"], str(ident))
        self.assertEqual(wire["n"]["price"], "1.5")
        self.assertEqual(wire["n"]["tags"], str({"a"}))
        self.assertEqual(wire["n"]["items"], [{"text": "x", "score": 1.0}])

    def test_messages_stay_loadable_objects(self):
        wire = to_wire("updates", {"n": {"messages": [AIMessage("hi")]}})
        back = load(wire, allowed_objects="messages")
        self.assertIsInstance(back["n"]["messages"][0], BaseMessage)

    def test_custom_payloads_get_the_same_treatment(self):
        wire = to_wire(
            "custom", {"m": AIMessage("hi"), "answer": Answer(text="a", score=2)}
        )
        self.assertEqual(wire["answer"], {"text": "a", "score": 2.0})
        self.assertIsInstance(load(wire["m"], allowed_objects="messages"), BaseMessage)

    def test_control_events_pass_through(self):
        self.assertEqual(to_wire("node_reset", {"node": "n"}), {"node": "n"})


class TestSseUpdates(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_update_matches_the_final_output(self):
        def node(state: _State) -> dict:
            get_stream_writer()({"answer": Answer(text="c", score=1.0)})
            return {"result": Answer(text="hi", score=0.5), "when": WHEN}

        graph = StateGraph(_State)
        graph.add_node("n", node)
        graph.add_edge(START, "n")
        graph.add_edge("n", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="wire")
        runner._started = True
        from fastapi.testclient import TestClient

        with mock.patch.object(runner, "run_async", _fake_run_async(runner)):
            resp = TestClient(serve_app(runner)).post(
                "/agent/run", json={"task": "hi", "stream": True}
            )
        events = parse_sse(resp.text)
        update = next(d for e, d in events if e == "updates")
        final = events[-1][1]

        self.assertEqual(load(update, allowed_objects="messages"), update)
        self.assertEqual(update["n"]["result"], final["output"]["result"])
        self.assertEqual(update["n"]["when"], final["output"]["when"])
        custom = next(d for e, d in events if e == "custom")
        self.assertEqual(custom, {"answer": {"text": "c", "score": 1.0}})


class _Bad:
    def to_dict(self):
        raise ValueError("cannot convert")


def _data_event(mode: str, data: Any, seq: int = 0) -> StreamEvent:
    return StreamEvent("wf-1", 0, "n", "a", seq, (), mode, data)


def _events(*items: Any):
    async def stream(graph_input, **kw):
        for item in items:
            if isinstance(item, BaseException):
                raise item
            yield item

    return stream


_STARTED = _data_event("workflow_started", {"workflow_id": "wf-1", "thread_id": "t"})


_DONE = normalize_final(
    {"type": "workflow_completed", "output": {"v": 1}},
    workflow_id="wf-1",
    thread_id="t",
)


class TestUnencodableChunk(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def _serve(self, *items: Any) -> List[Tuple[str, Any]]:
        runner = _chat_graph(name="enc")
        with mock.patch.object(runner, "_stream_events", _events(*items)):
            return await _collect(runner._serve_stream({"task": "hi"}, "sess"))

    async def test_the_stream_carries_on_and_the_loss_is_counted(self):
        got = await self._serve(
            _STARTED,
            _data_event("custom", "first", 1),
            _data_event("custom", _Bad(), 2),
            _data_event("custom", "after", 3),
            {**_DONE, "dropped": 1},
        )
        names = [n for n, _ in got]
        self.assertEqual(
            names,
            ["workflow_started", "custom", "custom", "custom", "workflow_completed"],
        )
        self.assertEqual(got[1][1], "first")
        self.assertIn("unserializable", got[2][1])
        self.assertEqual(got[3][1], "after")
        self.assertEqual(got[-1][1]["dropped"], 2)  # the sink's drop and ours

    async def test_a_cyclic_chunk_is_not_a_failed_run(self):
        loop: List[Any] = []
        loop.append(loop)
        got = await self._serve(_STARTED, _data_event("custom", loop, 1), _DONE)
        self.assertEqual(got[-1][0], "workflow_completed")
        self.assertEqual(got[-1][1]["dropped"], 1)


class TestStreamFailureAfterStart(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def _serve(self, *items: Any):
        runner = _chat_graph(name="enc-fail")
        with mock.patch.object(runner, "_stream_events", _events(*items)):
            return await _collect(runner._serve_stream({"task": "hi"}, "sess"))

    async def test_after_start_the_workflow_is_said_to_be_still_running(self):
        got = await self._serve(_STARTED, RuntimeError("queue broke"))
        name, data = got[-1]
        self.assertEqual(name, "workflow_failed")
        self.assertEqual(data["error"]["error_type"], "StreamError")
        self.assertTrue(data["workflow_still_running"])
        self.assertEqual(data["workflow_id"], "wf-1")

    async def test_before_start_it_is_a_plain_failure(self):
        got = await self._serve(RuntimeError("could not open"))
        name, data = got[-1]
        self.assertEqual(name, "workflow_failed")
        self.assertEqual(data["error"]["error_type"], "RuntimeError")
        self.assertNotIn("workflow_still_running", data)


class TestStartedIsFirst(_PollCase):
    async def test_a_chunk_written_while_the_workflow_is_scheduled_comes_second(self):
        runner, client = self._runner(
            [_state(WorkflowStatus.RUNNING), _completed_state()], "first-ev"
        )

        def schedule(*, workflow, input, instance_id):
            # a fast engine: the first node's activity has already run
            sink = runner.stream_registry.get(instance_id)
            sink.put(StreamEvent(instance_id, 0, "n", "a1", 0, (), "custom", "x"))

        client.schedule_new_workflow.side_effect = schedule
        order: List[str] = []
        async for item in runner.astream(
            INPUT,
            workflow_id="wf-fast",
            stream_mode=["custom"],
            poll_interval=0,
            on_control=lambda c: order.append(c.kind),
        ):
            order.append(item[0])
        self.assertEqual(order[:2], ["workflow_started", "custom"])

    async def test_the_sink_order_puts_it_first(self):
        runner, client = self._runner([_completed_state()], "first-ev-2")
        order: List[str] = []

        def schedule(*, workflow, input, instance_id):
            sink = runner.stream_registry.get(instance_id)
            sink.put(StreamEvent(instance_id, 0, "n", "a1", 0, (), "custom", "x"))

        client.schedule_new_workflow.side_effect = schedule
        agen = runner._stream_events(
            INPUT,
            config=None,
            thread_id="t",
            workflow_id="wf-order",
            poll_interval=0,
        )
        async for item in agen:
            order.append(item.mode if isinstance(item, StreamEvent) else item["type"])
        self.assertEqual(order[0], "workflow_started")
        self.assertEqual(order.count("workflow_started"), 1)
        self.assertEqual(order[1], "custom")


class TestSchedulingFailure(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_a_failed_schedule_is_a_failed_run_not_a_pump_crash(self):
        runner = _chat_graph(name="first-ev-fail")
        runner._started = True
        client = mock.Mock()
        client.schedule_new_workflow.side_effect = RuntimeError("no sidecar")
        runner._workflow_client = client
        items = [
            i
            async for i in runner._stream_events(
                INPUT, config=None, thread_id="t", workflow_id="w", poll_interval=0
            )
        ]
        self.assertEqual(items[-1]["type"], "workflow_failed")
        self.assertEqual(items[-1]["error"]["error_type"], "RuntimeError")


REQUEST = {
    "messages": [{"role": "user", "content": "hi"}],
    "stream": "a graph's own state key",
    "stream_mode": "also a state key",
}


class TestReservedKeys(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    def test_a_plain_request_keeps_its_stream_keys(self):
        runner = _chat_graph(name="rk-plain")
        self.assertEqual(runner._request_to_input(REQUEST), REQUEST)

    def test_a_streaming_request_drops_them(self):
        runner = _chat_graph(name="rk-stream")
        self.assertEqual(
            runner._request_to_input(REQUEST, streaming=True),
            {"messages": REQUEST["messages"]},
        )

    def test_same_rule_with_an_input_mapper(self):
        runner = _chat_graph(name="rk-mapper")
        runner._input_mapper = lambda request: dict(request)
        self.assertEqual(runner._request_to_input(REQUEST), REQUEST)
        self.assertEqual(
            runner._request_to_input(REQUEST, streaming=True),
            {"messages": REQUEST["messages"]},
        )

    def test_task_shorthand_still_works_when_streaming(self):
        runner = _chat_graph(name="rk-task")
        got = runner._request_to_input({"task": "hi", "stream": True}, streaming=True)
        self.assertEqual(got, {"messages": [{"role": "user", "content": "hi"}]})

    async def test_non_streaming_serve_run_passes_the_keys_to_the_graph(self):
        runner = _chat_graph(name="rk-run")
        runner._started = True
        seen: Dict[str, Any] = {}

        async def spy(input, **kw):
            seen.update(input)
            yield {"type": "workflow_completed"}

        with mock.patch.object(runner, "run_async", spy):
            await _collect(runner._serve_run(dict(REQUEST), "sess"))
        self.assertEqual(seen["stream"], REQUEST["stream"])

    async def test_streaming_serve_drops_them_with_or_without_a_mapper(self):
        for name, mapper in (("plain", None), ("mapper", lambda r: dict(r))):
            with self.subTest(input=name):
                self.setUp()
                runner = _chat_graph(name=f"rk-serve-{name}")
                runner._input_mapper = mapper
                seen: Dict[str, Any] = {}

                async def events(graph_input, **kw):
                    seen.update(graph_input)
                    yield {"type": "workflow_completed"}

                request = {**REQUEST, "stream": True, "stream_mode": "updates"}
                with mock.patch.object(runner, "_stream_events", events):
                    await _collect(runner._serve_stream(request, "sess"))
                self.assertEqual(list(seen), ["messages"])


MESSAGES = [{"role": "user", "content": "hi"}]


class _RouteCase(_RunnerTestCase, unittest.TestCase):
    """``POST /agent/run`` against a runner whose ``run_async`` records its input."""

    def _post(self, body: Dict[str, Any]):
        runner = _chat_graph(name="route")
        runner._started = True
        seen: List[Dict[str, Any]] = []

        async def spy(input, **kw):
            seen.append(dict(input))
            yield {"type": "workflow_completed", "output": {"ok": 1}}

        with mock.patch.object(runner, "run_async", spy):
            resp = TestClient(serve_app(runner)).post("/agent/run", json=body)
        return resp, seen


class TestStreamRouteDecision(_RouteCase):
    def test_a_stream_value_that_is_not_true_stays_a_plain_request(self):
        for value in ("false", "no", "x", {"a": 1}, [1], 0, False, None):
            with self.subTest(value=value):
                self.setUp()  # one runner (and workflow registration) per value
                resp, seen = self._post({"messages": MESSAGES, "stream": value})
                self.assertTrue(
                    resp.headers["content-type"].startswith("application/json"),
                    resp.headers["content-type"],
                )
                # a graph may have a state key of that name: it must arrive
                self.assertEqual(seen[0]["stream"], value)

    def test_true_streams(self):
        resp, _ = self._post({"messages": MESSAGES, "stream": True})
        self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))


class TestInvalidStreamRequest(_RouteCase):
    def test_an_unsupported_stream_mode_is_rejected_before_the_stream_starts(self):
        for mode in ("bogus", ["messages", "bogus"], 7):
            with self.subTest(stream_mode=mode):
                self.setUp()
                resp, started = self._post(
                    {"task": "hi", "stream": True, "stream_mode": mode}
                )
                self.assertEqual(resp.status_code, 422)
                self.assertIn("Unsupported stream_mode", resp.json()["detail"])
                self.assertEqual(started, [])  # no run was started

    def test_a_valid_stream_mode_streams(self):
        resp, started = self._post(
            {"task": "hi", "stream": True, "stream_mode": ["updates"]}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(started), 1)

    def test_a_plain_request_ignores_stream_mode(self):
        resp, started = self._post({"messages": MESSAGES, "stream_mode": "bogus"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(started[0]["stream_mode"], "bogus")


def _modes_sent(runner: Any, body: Dict[str, Any]) -> List[str]:
    resp = TestClient(serve_app(runner)).post("/agent/run", json=body)
    return [name for name, _ in parse_sse(resp.text)]


class TestStreamModeFiltersTheSse(_RunnerTestCase, unittest.TestCase):
    """The request's ``stream_mode`` decides which chunks an SSE client receives."""

    def _runner(self) -> Any:
        from diagrid.agent.langgraph import DaprWorkflowGraphRunner

        runner = DaprWorkflowGraphRunner(graph=_golden_graph(), name="route-filter")
        runner._started = True
        patch = mock.patch.object(runner, "run_async", _fake_run_async(runner))
        patch.start()
        self.addCleanup(patch.stop)
        return runner

    def test_the_default_sends_every_mode(self):
        names = set(_modes_sent(self._runner(), {"task": "hi", "stream": True}))
        self.assertTrue({"messages", "updates", "custom"} <= names)

    def test_a_single_mode_string_sends_only_that_mode(self):
        names = _modes_sent(
            self._runner(), {"task": "hi", "stream": True, "stream_mode": "updates"}
        )
        self.assertEqual(
            set(names), {"workflow_started", "updates", "workflow_completed"}
        )

    def test_a_list_sends_only_the_listed_modes(self):
        names = _modes_sent(
            self._runner(), {"task": "hi", "stream": True, "stream_mode": ["custom"]}
        )
        self.assertEqual(
            set(names), {"workflow_started", "custom", "workflow_completed"}
        )
        self.assertEqual(names[0], "workflow_started")
        self.assertEqual(names[-1], "workflow_completed")


class TestControlEventsPassTheModeFilter(_RunnerTestCase, unittest.TestCase):
    def test_node_reset_and_stream_unavailable_are_sent_whatever_the_modes(self):
        from diagrid.agent.core.workflow.stream import (
            NODE_RESET,
            STREAM_UNAVAILABLE,
            StreamEvent,
        )

        def event(mode: str, data: Any) -> StreamEvent:
            return StreamEvent(
                workflow_id="wf",
                step=0,
                node="chat",
                attempt_id="a",
                seq=0,
                ns=(),
                mode=mode,
                data=data,
            )

        runner = _chat_graph(name="route-control")
        runner._started = True

        async def events(*args: Any, **kw: Any):
            yield event("messages", ["x", {}])  # not asked for: filtered out
            yield event(NODE_RESET, {"node": "chat", "step": 1})
            yield event("updates", {"chat": {}})
            yield event(STREAM_UNAVAILABLE, {"workflow_id": "wf", "reason": "r"})
            yield {"type": "workflow_completed", "workflow_id": "wf"}

        with mock.patch.object(runner, "_stream_events", events):
            names = _modes_sent(
                runner, {"task": "hi", "stream": True, "stream_mode": ["updates"]}
            )
        self.assertEqual(
            names, [NODE_RESET, "updates", STREAM_UNAVAILABLE, "workflow_completed"]
        )


class TestSseFormat(unittest.TestCase):
    def test_frame(self):
        self.assertEqual(
            format_sse("messages", {"a": 1}, 3),
            'id: 3\nevent: messages\ndata: {"a": 1}\n\n',
        )

    def test_newlines_are_escaped_not_split(self):
        frame = format_sse("custom", "a\nb")
        self.assertEqual(frame, 'event: custom\ndata: "a\\nb"\n\n')
