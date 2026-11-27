# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``astream`` and the streamed run it drives: which chunks arrive and in what
shape, the thread id of a run, and what is done with the LangGraph config the
caller hands in."""

import asyncio
import unittest
from typing import Any, Dict, List
from unittest import mock

from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.models import ChannelState, ExecuteNodeInput
from diagrid.agent.langgraph.streaming import GraphRunError, StreamControl
from diagrid.agent.langgraph.workflow import execute_node_activity

from .golden_harness import _chain, _FakeDaprTest, _runner
from .streaming_helpers import (
    INPUT,
    TOKENS,
    _chat_graph,
    _collect,
    _Ctx,
    _fake_run_async,
    _StreamingModel,
)
from .test_runner import _run, _RunnerTestCase


class TestAstream(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    def _runner(self, **kw) -> DaprWorkflowGraphRunner:
        runner = _chat_graph(**kw)
        runner._started = True
        self._patch = mock.patch.object(runner, "run_async", _fake_run_async(runner))
        self._patch.start()
        self.addCleanup(self._patch.stop)
        return runner

    async def test_messages_arrive_in_order_for_both_node_kinds(self):
        for with_runtime in (False, True):
            self.setUp()
            runner = self._runner(with_runtime=with_runtime, name=f"s-{with_runtime}")
            chunks = await _collect(
                runner.astream(INPUT, workflow_id="wf-a", stream_mode="messages")
            )
            # LangChain closes a stream with one empty chunk; only text counts.
            texts = [c[0].content for c in chunks]
            self.assertEqual([t for t in texts if t], TOKENS, with_runtime)
            meta = chunks[0][1]
            self.assertEqual(meta["langgraph_node"], "chat")
            self.assertEqual(meta["langgraph_step"], 1)  # as native Pregel
            self.assertIn("langgraph_checkpoint_ns", meta)

    async def test_updates_and_custom_with_mode_list(self):
        runner = self._runner()
        items = await _collect(
            runner.astream(INPUT, workflow_id="wf-b", stream_mode=["updates", "custom"])
        )
        self.assertEqual(items[0], ("custom", {"progress": 1}))
        mode, update = items[-1]
        self.assertEqual(mode, "updates")
        self.assertEqual(list(update), ["chat"])
        self.assertEqual(len(items), 2)  # no messages requested

    async def test_single_mode_yields_bare_chunks(self):
        runner = self._runner()
        items = await _collect(
            runner.astream(INPUT, workflow_id="wf-c", stream_mode="custom")
        )
        self.assertEqual(items, [{"progress": 1}])

    async def test_unsupported_mode_and_subgraphs(self):
        runner = self._runner()
        with self.assertRaises(ValueError):
            await _collect(runner.astream(INPUT, stream_mode="bogus"))
        with self.assertRaises(NotImplementedError):
            await _collect(runner.astream(INPUT, subgraphs=True))

    async def test_no_events_leak_across_runs(self):
        runner = self._runner()

        async def one(wf: str):
            return await _collect(
                runner.astream(INPUT, workflow_id=wf, stream_mode="messages")
            )

        # _run patches when_all per call; two threads doing it at once race,
        # so hold one patch open for the whole overlap.
        with mock.patch("diagrid.agent.langgraph.workflow.when_all", side_effect=list):
            a, b = await asyncio.gather(one("wf-1"), one("wf-2"))
        # Broadcasting every chunk to every open run would double these.
        for run in (a, b):
            texts = [c[0].content for c in run]
            self.assertEqual([t for t in texts if t], TOKENS)
        # The runner opens its sinks in its own registry, not the default one.
        self.assertIsNone(runner.stream_registry.get("wf-1"))
        self.assertIsNone(runner.stream_registry.get("wf-2"))

    async def test_node_returned_messages_stream_without_an_llm(self):
        def tool(state: MessagesState) -> dict:
            return {"messages": [AIMessage("from-node", id="m1")]}

        graph = StateGraph(MessagesState)
        graph.add_node("tool", tool)
        graph.add_edge(START, "tool")
        graph.add_edge("tool", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="node-msg")
        runner._started = True
        with mock.patch.object(runner, "run_async", _fake_run_async(runner)):
            chunks = await _collect(
                runner.astream(INPUT, workflow_id="wf-m", stream_mode="messages")
            )
        self.assertEqual([c[0].content for c in chunks], ["from-node"])
        self.assertEqual(chunks[0][1]["langgraph_node"], "tool")

    async def test_repeated_channel_writes_are_all_in_updates(self):
        def many(state: MessagesState) -> list:
            return [
                Command(update={"messages": [AIMessage("a", id="a")]}),
                Command(update={"messages": [AIMessage("b", id="b")]}),
            ]

        graph = StateGraph(MessagesState)
        graph.add_node("n", many)
        graph.add_edge(START, "n")
        graph.add_edge("n", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="multi")
        runner._started = True
        with mock.patch.object(runner, "run_async", _fake_run_async(runner)):
            items = await _collect(
                runner.astream(INPUT, workflow_id="wf-u", stream_mode="updates")
            )
        update = items[-1]["n"]
        self.assertEqual(len(update), 2)
        self.assertEqual([u["messages"][0].content for u in update], ["a", "b"])

    async def test_post_agent_run_with_stream_returns_sse(self):
        from fastapi.testclient import TestClient

        runner = self._runner()
        captured: dict = {}
        with mock.patch("uvicorn.run", lambda app, **kw: captured.update(app=app)):
            with (
                mock.patch.object(runner, "_setup_telemetry"),
                mock.patch.object(runner, "_setup_serve_defaults"),
            ):
                runner.serve()
        # No context manager: the lifespan (runtime start) is not run.
        client = TestClient(captured["app"])
        resp = client.post("/agent/run", json={"task": "hi", "stream": True})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("event: messages", resp.text)
        self.assertIn("event: workflow_completed", resp.text)

    async def test_async_only_node_streams_tokens(self):
        model = _StreamingModel()

        async def achat(state: MessagesState) -> dict:
            return {"messages": [await model.ainvoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("chat", achat)
        graph.add_edge(START, "chat")
        graph.add_edge("chat", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="async-only")
        runner._started = True
        with mock.patch.object(runner, "run_async", _fake_run_async(runner)):
            chunks = await _collect(
                runner.astream(INPUT, workflow_id="wf-async", stream_mode="messages")
            )
        texts = [c[0].content for c in chunks]
        self.assertEqual([t for t in texts if t], TOKENS)

    async def test_serve_stream_yields_event_pairs(self):
        runner = self._runner()
        pairs = await _collect(runner._serve_stream({"task": "hi"}, "sess"))
        names = [name for name, _ in pairs]
        self.assertEqual(names[-1], "workflow_completed")
        self.assertGreaterEqual(names.count("messages"), len(TOKENS))
        self.assertIn("updates", names)
        self.assertIn("custom", names)
        self.assertEqual(pairs[-1][1]["type"], "workflow_completed")

    async def test_sse_frames_are_numbered_and_terminated(self):
        from fastapi.testclient import TestClient

        runner = self._runner()
        captured: dict = {}
        with (
            mock.patch("uvicorn.run", lambda app, **kw: captured.update(app=app)),
            mock.patch.object(runner, "_setup_telemetry"),
            mock.patch.object(runner, "_setup_serve_defaults"),
        ):
            runner.serve()
        resp = TestClient(captured["app"]).post(
            "/agent/run", json={"task": "hi", "stream": True}
        )
        frames = resp.text.split("\n\n")[:-1]
        self.assertEqual(
            [f.split("\n")[0] for f in frames],
            [f"id: {n}" for n in range(1, len(frames) + 1)],
        )
        self.assertEqual(frames[-1].split("\n")[1], "event: workflow_completed")
        self.assertTrue(resp.text.endswith("\n\n"))


class TestStreamingOff(_RunnerTestCase):
    def test_no_sink_means_no_streaming_and_same_output(self):
        # The node calls get_stream_writer(), as a graph that streams does.
        runner = _chat_graph(name="off", writes=True)
        self.assertIsNone(runner.stream_registry.get("wf-test"))
        out = _run(runner, INPUT)
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["messages"][-1]["content"], "".join(TOKENS))


CFG = {"configurable": {"thread_id": "CFG-T"}}


class TestThreadId(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    def _runner(self, name: str):
        """A runner whose run_async runs the node as the workflow would."""
        runner = _chat_graph(name=name)
        runner._started = True
        self.seen: Dict[str, Any] = {}

        async def run_async(
            input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
        ):
            self.seen.update(thread_id=thread_id, config=dict(config or {}))
            yield {"type": "workflow_started", "workflow_id": workflow_id}
            payload = ExecuteNodeInput(
                node_name="chat",
                channel_state=ChannelState(
                    values=runner._serialize_input(input), versions={}
                ),
                config=config,
                thread_id=thread_id,
                step=0,
            ).to_dict()
            await asyncio.to_thread(
                execute_node_activity,
                _Ctx(workflow_id),
                payload,
                stream_registry=runner.stream_registry,
            )
            yield {
                "type": "workflow_completed",
                "workflow_id": workflow_id,
                "steps": 1,
                "status": "completed",
                "executed": [["chat"]],
            }

        patch = mock.patch.object(runner, "run_async", run_async)
        patch.start()
        self.addCleanup(patch.stop)
        return runner

    async def test_config_is_positional_and_gives_the_thread_id(self):
        runner = self._runner("tid-cfg")
        controls: List[StreamControl] = []
        items = await _collect(
            runner.astream(
                INPUT, CFG, stream_mode=["messages"], on_control=controls.append
            )
        )
        (started,) = [c for c in controls if c.kind == "workflow_started"]
        self.assertEqual(started.data["thread_id"], "CFG-T")
        self.assertEqual(self.seen["thread_id"], "CFG-T")
        self.assertEqual(
            {meta["thread_id"] for mode, (_, meta) in items if mode == "messages"},
            {"CFG-T"},
        )

    async def test_an_explicit_thread_id_wins_and_the_run_has_just_one(self):
        runner = self._runner("tid-arg")
        items = await _collect(
            runner.astream(INPUT, CFG, thread_id="ARG-T", stream_mode=["messages"])
        )
        self.assertEqual(self.seen["thread_id"], "ARG-T")
        self.assertEqual(self.seen["config"]["configurable"]["thread_id"], "ARG-T")
        self.assertEqual(
            {meta["thread_id"] for mode, (_, meta) in items if mode == "messages"},
            {"ARG-T"},
        )

    async def test_the_callers_config_is_not_modified(self):
        runner = self._runner("tid-keep")
        cfg = {"configurable": {"user": "u"}}
        await _collect(runner.astream(INPUT, cfg, thread_id="T"))
        self.assertEqual(cfg, {"configurable": {"user": "u"}})

    async def test_without_either_a_thread_id_is_generated(self):
        runner = self._runner("tid-gen")
        await _collect(runner.astream(INPUT, {"tags": ["t"]}))
        self.assertTrue(self.seen["thread_id"])
        self.assertEqual(
            self.seen["config"]["configurable"]["thread_id"], self.seen["thread_id"]
        )


class _Handler:
    """Stands in for a LangChain callback handler: not JSON."""


class TestConfigThatCannotCrossToTheWorkflow(_FakeDaprTest):
    def _astream(self, config: Any) -> Any:
        runner = _runner(_chain(1), "config-json")
        client = mock.Mock()
        runner._workflow_client = client
        return runner, client, runner.astream({"log": []}, config=config)

    async def test_callbacks_are_refused_before_a_run_starts(self):
        runner, client, stream = self._astream({"callbacks": [_Handler()]})
        with self.assertRaisesRegex(ValueError, "callbacks"):
            await asyncio.wait_for(_collect(stream), 5)
        client.schedule_new_workflow.assert_not_called()

    async def test_a_configurable_value_that_is_not_json_names_its_key(self):
        runner, client, stream = self._astream({"configurable": {"db": _Handler()}})
        with self.assertRaisesRegex(ValueError, "db"):
            await asyncio.wait_for(_collect(stream), 5)
        client.schedule_new_workflow.assert_not_called()

    async def test_json_config_is_fine(self):
        runner = _runner(_chain(1), "config-ok")
        items = await _collect(
            runner.astream(
                {"log": []},
                config={"tags": ["t"], "metadata": {"a": 1}},
                stream_mode=["updates"],
                poll_interval=0.01,
            )
        )
        self.assertEqual([m for m, _ in items], ["updates"])


class TestRecursionLimit(_FakeDaprTest):
    """``config["recursion_limit"]`` bounds the run as it bounds native LangGraph."""

    async def test_matches_native_at_every_limit(self):
        compiled = _chain(3)
        runner = _runner(compiled, "config-recursion")
        for limit in range(1, 6):
            with self.subTest(limit=limit):
                try:
                    compiled.invoke({"log": []}, {"recursion_limit": limit})
                    native_ok = True
                except GraphRecursionError:
                    native_ok = False
                seen: List[Any] = []
                stream = runner.astream(
                    {"log": []},
                    config={"recursion_limit": limit},
                    workflow_id=f"wf-rl-{limit}",
                    stream_mode=["updates"],
                    poll_interval=0.01,
                )
                try:
                    seen = await asyncio.wait_for(_collect(stream), 10)
                    ours_ok = True
                except GraphRunError:
                    ours_ok = False
                self.assertEqual(ours_ok, native_ok)
                if ours_ok:
                    self.assertEqual(len(seen), 3)

    async def test_a_run_cut_by_the_limit_raises_langgraphs_recursion_error(self):
        runner = _runner(_chain(4), "config-recursion-type")
        stream = runner.astream(
            {"log": []},
            config={"recursion_limit": 4},
            stream_mode=["updates"],
            poll_interval=0.01,
        )
        with self.assertRaises(GraphRecursionError) as ctx:
            await asyncio.wait_for(_collect(stream), 10)
        self.assertIsInstance(ctx.exception, GraphRunError)
        self.assertEqual(ctx.exception.final["status"], "max_steps_reached")

    async def test_a_bad_limit_is_refused(self):
        runner = _runner(_chain(1), "config-bad-limit")
        for bad in (0, -1, "5", 2.5, True):
            with self.subTest(limit=bad):
                with self.assertRaisesRegex(ValueError, "recursion_limit"):
                    await asyncio.wait_for(
                        _collect(
                            runner.astream({"log": []}, config={"recursion_limit": bad})
                        ),
                        5,
                    )
