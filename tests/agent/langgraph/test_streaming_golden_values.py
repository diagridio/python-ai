# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``stream_mode="values"``: the graph's state after each step, as native yields it.

The state after step s is what every activity of step s+1 is handed, so the
first of them to start sends it; the sink keeps the first and drops the rest.
The state after the last step comes from the run's final event.
"""

import asyncio
import operator
from typing import Any, List
from unittest import mock

from langchain_core.load import load
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from diagrid.agent.core.workflow.stream import (
    NODE_RESET,
    VALUES,
    StreamSinkRegistry,
)
from diagrid.agent.langgraph.models import ChannelState, ExecuteNodeInput
from diagrid.agent.langgraph.streaming import STREAM_RUN_KEY
from diagrid.agent.langgraph.workflow import (
    clear_registries,
    execute_node_activity,
    register_node,
)

from .golden_harness import (
    _both,
    _canonical,
    _chain,
    _FakeDaprTest,
    _golden_graph,
    _runner,
    _wire,
)
from .streaming_helpers import (
    INPUT,
    _collect,
    _Ctx,
    _drain,
    parse_sse,
    serve_app,
)
from .test_streaming_golden import _fan_out_graph, _router_loop_graph

VALUES_ONLY = ["values"]
WITH_CHUNKS = ["messages", "updates", "custom", "values"]


def _values(items: List) -> List[Any]:
    return [p for m, p in _wire(items) if m == "values"]


class _Log(TypedDict, total=False):
    log: Annotated[list, operator.add]


def _quiet_middle_graph() -> Any:
    """a -> quiet (writes nothing) -> b: no values chunk after quiet's step."""
    graph = StateGraph(_Log)
    graph.add_node("a", lambda state: {"log": ["a"]})
    graph.add_node("quiet", lambda state: None)
    graph.add_node("b", lambda state: {"log": ["b"]})
    graph.add_edge(START, "a")
    graph.add_edge("a", "quiet")
    graph.add_edge("quiet", "b")
    graph.add_edge("b", END)
    return graph.compile()


class TestValuesMatchNative(_FakeDaprTest):
    async def test_a_sequential_graph(self):
        native, ours = await _both(_golden_graph(), "values-seq", INPUT, VALUES_ONLY)
        self.assertTrue(_values(native))
        self._assert_same_sequence(native, ours)

    async def test_each_value_sits_between_its_steps_chunks(self):
        native, ours = await _both(_golden_graph(), "values-mixed", INPUT, WITH_CHUNKS)
        self._assert_same_sequence(native, ours)

    async def test_a_parallel_step(self):
        native, ours = await _both(
            _fan_out_graph(), "values-fan", {"log": []}, WITH_CHUNKS
        )
        self.assertEqual(_canonical(ours), _canonical(native))
        self.assertEqual(_values(ours), _values(native))

    async def test_a_router_loop(self):
        native, ours = await _both(
            _router_loop_graph(), "values-loop", {"log": []}, ["values", "updates"]
        )
        self._assert_same_sequence(native, ours)

    async def test_the_last_value_is_the_final_state(self):
        compiled = _chain(3)
        native, ours = await _both(compiled, "values-final", {"log": []}, VALUES_ONLY)
        self._assert_same_sequence(native, ours)
        self.assertEqual(ours[-1], ("values", compiled.invoke({"log": []})))

    async def test_a_step_that_writes_no_output_channel_sends_none(self):
        for name, compiled, graph_input in (
            ("values-quiet-middle", _quiet_middle_graph(), {"log": []}),
            # the golden graph's last node writes nothing: no final value
            ("values-quiet-last", _golden_graph(), INPUT),
        ):
            with self.subTest(case=name):
                native, ours = await _both(
                    compiled, name, graph_input, ["values", "updates"]
                )
                self._assert_same_sequence(native, ours)

    async def test_an_empty_input_sends_no_first_value(self):
        native, ours = await _both(_chain(2), "values-empty", {}, VALUES_ONLY)
        self.assertEqual(len(_values(native)), 2)
        self._assert_same_sequence(native, ours)


class _Count(TypedDict, total=False):
    n: int


def _endless_graph() -> Any:
    graph = StateGraph(_Count)
    graph.add_node("inc", lambda state: {"n": state.get("n", 0) + 1})
    graph.add_edge(START, "inc")
    graph.add_conditional_edges("inc", lambda s: "inc" if s["n"] < 99 else END)
    return graph.compile()


async def _until_error(stream: Any) -> List[Any]:
    items: List[Any] = []
    with mock.patch("diagrid.agent.langgraph.workflow.when_all", side_effect=list):
        try:
            async for item in stream:
                items.append(item)
        except GraphRecursionError:
            return items
    raise AssertionError("the run did not hit its step limit")


class TestValuesOfAFailedRun(_FakeDaprTest):
    async def test_a_step_limit_ends_with_the_last_steps_value_like_native(self):
        compiled = _endless_graph()
        config = {"recursion_limit": 3}
        native = await _until_error(
            compiled.astream({"n": 0}, config, stream_mode=VALUES_ONLY)
        )
        ours = await _until_error(
            _runner(compiled, "values-limit").astream(
                {"n": 0},
                config=config,
                workflow_id="wf-values-limit",
                stream_mode=VALUES_ONLY,
                poll_interval=0.01,
            )
        )
        self.assertEqual(native[-1], ("values", {"n": 3}))
        self.assertEqual(ours, native)


class TestValuesOfARetriedStep(_FakeDaprTest):
    def setUp(self):
        super().setUp()
        clear_registries()
        self.addCleanup(clear_registries)

    async def test_a_retry_does_not_send_the_value_again(self):
        attempts: List[int] = []

        def flaky(state: Any) -> dict:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("first attempt fails")
            return {"log": ["b"]}

        register_node("b", flaky)
        registry = StreamSinkRegistry()
        sink = registry.open("wf-retry")
        payload = ExecuteNodeInput(
            node_name="b",
            channel_state=ChannelState(
                values={"log": ["a"]}, versions={"log": 2}, updated_channels=["log"]
            ),
            config={STREAM_RUN_KEY: True},
            step=1,
            output_channels=["log"],
        ).to_dict()
        for _ in range(2):
            try:
                await asyncio.to_thread(
                    execute_node_activity,
                    _Ctx("wf-retry"),
                    payload,
                    stream_registry=registry,
                )
            except RuntimeError:
                pass
        sink.close()
        events = [e for e in await _drain(sink) if hasattr(e, "mode")]
        values = [e for e in events if e.mode == VALUES]
        self.assertEqual([e.data for e in values], [{"log": ["a"]}])
        self.assertEqual(events[0].mode, VALUES)
        # the node's own retry is still reported, and only once
        self.assertEqual([e.mode for e in events].count(NODE_RESET), 1)


class TestValuesOverSse(_FakeDaprTest):
    def test_they_load_back_as_the_native_values(self):
        from fastapi.testclient import TestClient

        compiled = _golden_graph()
        native = list(compiled.stream(INPUT, stream_mode="values"))
        runner = _runner(compiled, "values-sse")
        client = TestClient(serve_app(runner))
        resp = client.post(
            "/agent/run",
            json={**INPUT, "stream": True, "stream_mode": ["values"]},
        )
        events = parse_sse(resp.text)
        self.assertEqual(events[-1][0], "workflow_completed", events[-1])
        ours = [load(d, allowed_objects="messages") for e, d in events if e == "values"]
        self.assertEqual(
            _values([("values", v) for v in ours]),
            _values([("values", v) for v in native]),
        )
        self.assertNotIn("channel_state", events[-1][1])
