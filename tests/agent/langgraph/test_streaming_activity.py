# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""What ``execute_node_activity`` streams, and what it must survive.

The streaming side of a node activity: its failures, a writer with nothing
listening, the ``messages`` metadata native ``astream`` puts there, and a
langgraph release that moves its private message handler."""

import asyncio
import inspect
import sys
import unittest
from typing import Any, Dict, List
from unittest import mock

from langgraph.graph import END, START, MessagesState, StateGraph

from diagrid.agent.core.workflow.stream import (
    StreamClosed,
    StreamEvent,
    StreamSinkRegistry,
    default_registry,
)
from diagrid.agent.langgraph.models import ChannelState, ExecuteNodeInput
from diagrid.agent.langgraph.streaming import (
    STREAM_RUN_KEY,
    NodeStreamer,
    _own_node_handler_class,
)
from diagrid.agent.langgraph.workflow import (
    clear_registries,
    execute_node_activity,
    register_node,
)

from .streaming_helpers import INPUT, TOKENS, _chat_graph, _Ctx, _IdCtx, _payload
from .test_runner import _run, _RunnerTestCase

# The two ways a node can be written: with and without a ``runtime`` argument.
NODE_KINDS = {"plain": False, "runtime": True}


class _RecordingStreamer(NodeStreamer):
    created: List["_RecordingStreamer"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.created.append(self)


class TestNodeThatRaises(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_registries()
        self.addCleanup(clear_registries)
        _RecordingStreamer.created = []
        patch = mock.patch(
            "diagrid.agent.langgraph.workflow.NodeStreamer", _RecordingStreamer
        )
        patch.start()
        self.addCleanup(patch.stop)

    async def _fail(self, node: Any, name: str = "boom") -> _RecordingStreamer:
        register_node(name, node)
        registry = StreamSinkRegistry()
        registry.open("wf-err")
        with self.assertRaises(RuntimeError):
            await asyncio.to_thread(
                execute_node_activity,
                _Ctx("wf-err"),
                _payload(name, streamed=False),
                stream_registry=registry,
            )
        (streamer,) = _RecordingStreamer.created
        return streamer

    async def test_the_message_handler_is_told_the_node_failed(self):
        def boom(state):
            raise RuntimeError("nope")

        streamer = await self._fail(boom)
        # StreamMessagesHandler keeps per-run state from on_chain_start until
        # the matching on_chain_end / on_chain_error; none may be left behind.
        self.assertEqual(streamer._handler.metadata, {})

    async def test_same_for_a_runnable_node(self):
        graph = StateGraph(MessagesState)

        def boom(state: MessagesState) -> dict:
            raise RuntimeError("nope")

        graph.add_node("boom", boom)
        graph.add_edge(START, "boom")
        graph.add_edge("boom", END)
        runnable = graph.compile().nodes["boom"].bound
        streamer = await self._fail(runnable)
        self.assertEqual(streamer._handler.metadata, {})

    async def test_success_also_leaves_no_state(self):
        register_node("fine", lambda state: {"x": 1})
        registry = StreamSinkRegistry()
        registry.open("wf-ok")
        await asyncio.to_thread(
            execute_node_activity,
            _Ctx("wf-ok"),
            _payload("fine", streamed=False),
            stream_registry=registry,
        )
        (streamer,) = _RecordingStreamer.created
        self.assertEqual(streamer._handler.metadata, {})


class TestWriterWithoutAStreamer(_RunnerTestCase):
    def _plain_run(self, with_runtime: bool) -> None:
        runner = _chat_graph(with_runtime=with_runtime, writes=True)
        out = _run(runner, INPUT, _IdCtx(runner, "wf-plain"))
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["messages"][-1]["content"], "".join(TOKENS))

    def _streamed_run_without_a_sink(self, with_runtime: bool) -> None:
        registry = StreamSinkRegistry()
        runner = _chat_graph(
            with_runtime=with_runtime, writes=True, stream_registry=registry
        )
        payload = ExecuteNodeInput(
            node_name="chat",
            channel_state=ChannelState(
                values=runner._serialize_input(INPUT), versions={}
            ),
            config={STREAM_RUN_KEY: True},
        ).to_dict()
        # the run's sink is in no registry this activity can see: a default one
        # that does hold it must not be what answers
        wf = f"wf-elsewhere-{with_runtime}"
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)

        async def hold() -> None:
            default_registry.open(wf)

        loop.run_until_complete(hold())
        self.addCleanup(default_registry.close, wf)
        with self.assertLogs("diagrid.agent.langgraph.workflow", level="WARNING"):
            out = execute_node_activity(_Ctx(wf), payload, stream_registry=registry)
        self.assertIsNone(out.get("error"))
        self.assertTrue(out["writes"])

    def test_a_node_in_a_run_that_is_not_streamed(self):
        for kind, with_runtime in NODE_KINDS.items():
            with self.subTest(node=kind):
                self.setUp()
                self._plain_run(with_runtime)

    def test_a_node_in_a_streamed_run_with_no_sink(self):
        for kind, with_runtime in NODE_KINDS.items():
            with self.subTest(node=kind):
                self.setUp()
                self._streamed_run_without_a_sink(with_runtime)


CONFIG = {
    STREAM_RUN_KEY: True,
    "thread_id": "t1",
    "metadata": {"foo": "bar"},
    "tags": ["x"],
    "configurable": {"user": "u-7", "__private": "no", "obj": {"a": 1}},
}


async def _messages_metadata(runner: Any, registry: StreamSinkRegistry) -> List[Dict]:
    sink = registry.open("wf-meta")
    payload = ExecuteNodeInput(
        node_name="chat",
        channel_state=ChannelState(values=runner._serialize_input(INPUT), versions={}),
        config=CONFIG,
        thread_id="t1",
        step=0,
    ).to_dict()
    out = await asyncio.to_thread(
        execute_node_activity, _Ctx("wf-meta"), payload, stream_registry=registry
    )
    assert not out.get("error"), out
    sink.close()
    items = []
    while True:
        item = await sink.get()
        if isinstance(item, StreamClosed):
            break
        if isinstance(item, StreamEvent) and item.mode == "messages":
            items.append(item.data[1])
    return items


class TestMessagesMetadata(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def _check(self, with_runtime: bool) -> None:
        registry = StreamSinkRegistry()
        runner = _chat_graph(with_runtime=with_runtime, stream_registry=registry)
        metadata = await _messages_metadata(runner, registry)
        self.assertTrue(metadata)
        for meta in metadata:
            self.assertEqual(meta["thread_id"], "t1")
            self.assertEqual(meta["foo"], "bar")
            # configurable values native copies into metadata: primitives only,
            # never the private ones.
            self.assertEqual(meta["user"], "u-7")
            self.assertNotIn("__private", meta)
            self.assertNotIn("obj", meta)
            self.assertEqual(meta["langgraph_node"], "chat")

    async def test_the_metadata_native_puts_there(self):
        for kind, with_runtime in NODE_KINDS.items():
            with self.subTest(node=kind):
                self.setUp()
                await self._check(with_runtime)


LOGGER = "diagrid.agent.langgraph.streaming"


class TestHandlerContract(unittest.TestCase):
    """CI notices a langgraph that changes what the streamer relies on."""

    def test_the_real_handler_is_built_the_way_the_streamer_builds_it(self):
        from langgraph.pregel._messages import StreamMessagesHandler

        params = list(inspect.signature(StreamMessagesHandler).parameters)
        self.assertEqual(params[:2], ["stream", "subgraphs"])
        streamer = NodeStreamer(
            mock.Mock(), workflow_id="wf", step=0, node="n", task_id=1
        )
        self.assertIsNotNone(streamer._messages_handler())
        self.assertEqual(len(streamer.callbacks()), 1)


class TestHandlerDrift(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_registries()
        self.addCleanup(clear_registries)
        register_node("n", lambda state: {"x": 1})
        NodeStreamer._handler_failure_logged = False
        self.addCleanup(setattr, NodeStreamer, "_handler_failure_logged", False)
        # the handler class is built once; build it again under each patch
        _own_node_handler_class.cache_clear()
        self.addCleanup(_own_node_handler_class.cache_clear)

    async def _run_streamed(self, wf: str) -> List[StreamEvent]:
        registry = StreamSinkRegistry()
        sink = registry.open(wf)
        out = await asyncio.to_thread(
            execute_node_activity, _Ctx(wf), _payload(), stream_registry=registry
        )
        self.assertIsNone(out.get("error"))
        self.assertEqual(out["writes"][0]["value"], 1)
        sink.close()
        items: List[Any] = []
        while True:
            item = await sink.get()
            if isinstance(item, StreamEvent):
                items.append(item)
            else:
                return items

    async def _assert_run_survives(self) -> None:
        with self.assertLogs(LOGGER, level="ERROR") as logs:
            items = await self._run_streamed("wf-drift-1")
            await self._run_streamed("wf-drift-2")
        # the node ran, its update still streams, and the cause is logged once
        self.assertEqual([i.mode for i in items], ["updates"])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("messages", logs.records[0].getMessage())

    async def test_a_moved_module(self):
        with mock.patch.dict(sys.modules, {"langgraph.pregel._messages": None}):
            await self._assert_run_survives()

    async def test_a_changed_signature(self):
        def renamed(stream: Any) -> Any:  # takes one argument, as if it changed
            raise AssertionError("not reached")

        with mock.patch("langgraph.pregel._messages.StreamMessagesHandler", renamed):
            await self._assert_run_survives()
