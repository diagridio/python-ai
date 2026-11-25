# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Graph shapes: the stream matches what native ``astream`` yields."""

import asyncio
import operator
import unittest
from typing import Any, Callable, Dict, List, Literal
from unittest import mock

from langchain_core.load import load
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.workflow import _has_async_entry

from .golden_harness import (
    ADD_INPUT,
    _both,
    _canonical,
    _Golden,
    _golden_graph,
    _Log,
    _ReactModel,
    _wire,
    add,
)
from .streaming_helpers import (
    INPUT,
    _fake_run_async,
    _StreamingModel,
    parse_sse,
    serve_app,
)
from .test_runner import _run, _RunnerTestCase


class TestMatchesNativeLangGraph(_Golden):
    async def test_same_sequence_of_mode_and_payload(self):
        native, ours = await _both(_golden_graph(), "golden", INPUT)
        self._assert_same_sequence(native, ours)

    async def test_step_numbers_follow_the_pregel_loop(self):
        native, ours = await _both(_golden_graph(), "golden-steps", INPUT)
        self.assertEqual(set(self._steps(ours)), set(self._steps(native)))
        first = next(p for m, p in ours if m == "messages")
        self.assertIsInstance(first[0], BaseMessage)


class TestSseCarriesLoadableObjects(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_messages_and_updates_load_back(self):
        from fastapi.testclient import TestClient

        runner = DaprWorkflowGraphRunner(graph=_golden_graph(), name="golden-sse")
        runner._started = True
        with mock.patch.object(runner, "run_async", _fake_run_async(runner)):
            client = TestClient(serve_app(runner))
            resp = client.post("/agent/run", json={"task": "hi", "stream": True})
        events = parse_sse(resp.text)

        messages = [
            load(d, allowed_objects="messages") for e, d in events if e == "messages"
        ]
        self.assertTrue(messages)
        chunk, metadata = messages[0]
        self.assertIsInstance(chunk, BaseMessage)
        self.assertEqual(metadata["langgraph_node"], "chat")

        updates = [
            load(d, allowed_objects="messages") for e, d in events if e == "updates"
        ]
        self.assertEqual([list(u) for u in updates], [["chat"], ["counter"], ["quiet"]])
        self.assertIsInstance(updates[0]["chat"]["messages"][0], BaseMessage)
        self.assertEqual(updates[1], {"counter": {"count": 1}})
        self.assertEqual(updates[2], {"quiet": None})


def _react_graph() -> Any:
    model = _ReactModel()

    def agent(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([add]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


class _Fan(TypedDict, total=False):
    log: Annotated[list, operator.add]
    messages: Annotated[list, add_messages]


def _fan_out_graph() -> Any:
    model = _ReactModel()

    def start(state: _Fan) -> dict:
        get_stream_writer()("start")
        return {"log": ["start"]}

    def left(state: _Fan) -> dict:
        get_stream_writer()("left")
        return {"log": ["left"]}

    def right(state: _Fan) -> dict:
        get_stream_writer()("right")
        return {"log": ["right"], "messages": [model.invoke([("user", "hi")])]}

    def join(state: _Fan) -> dict:
        get_stream_writer()("join")
        return {"log": ["join"]}

    graph = StateGraph(_Fan)
    for name, fn in (
        ("start", start),
        ("left", left),
        ("right", right),
        ("join", join),
    ):
        graph.add_node(name, fn)
    graph.add_edge(START, "start")
    graph.add_edge("start", "left")
    graph.add_edge("start", "right")
    graph.add_edge("left", "join")
    graph.add_edge("right", "join")
    graph.add_edge("join", END)
    return graph.compile()


class _Loop(TypedDict, total=False):
    n: int
    log: Annotated[list, operator.add]


def _router_loop_graph() -> Any:
    def inc(state: _Loop) -> dict:
        get_stream_writer()({"n": state.get("n", 0) + 1})
        return {"n": state.get("n", 0) + 1, "log": ["inc"]}

    def route(state: _Loop) -> Literal["inc", "__end__"]:
        return "inc" if state["n"] < 3 else END

    graph = StateGraph(_Loop)
    graph.add_node("inc", inc)
    graph.add_edge(START, "inc")
    graph.add_conditional_edges("inc", route)
    return graph.compile()


class TestReactLoop(_Golden):
    async def test_matches_native_with_thread_tags_and_metadata(self):
        native, ours = await _both(_react_graph(), "golden-react", ADD_INPUT)
        self._assert_same_sequence(native, ours)

    async def test_steps_and_the_tool_message_chunk(self):
        native, ours = await _both(_react_graph(), "golden-react-steps", ADD_INPUT)
        self.assertEqual(sorted(set(self._steps(ours))), [1, 2, 3])
        self.assertEqual(self._steps(ours), self._steps(native))
        tool_chunks = [
            i[1]
            for i in ours
            if i[0] == "messages" and isinstance(i[1][0], ToolMessage)
        ]
        self.assertEqual(len(tool_chunks), 1)
        message, metadata = tool_chunks[0]
        self.assertEqual(message.content, "5")
        self.assertEqual(metadata["langgraph_node"], "tools")
        self.assertEqual(metadata["langgraph_step"], 2)
        self.assertEqual(metadata["thread_id"], "t1")
        self.assertEqual(metadata["foo"], "bar")


class TestRouterLoop(_Golden):
    async def test_matches_native_and_numbers_repeated_nodes(self):
        native, ours = await _both(_router_loop_graph(), "golden-loop", {"log": []})
        self._assert_same_sequence(native, ours)
        self.assertEqual(
            [i[1] for i in ours if i[0] == "custom"], [{"n": 1}, {"n": 2}, {"n": 3}]
        )


class TestParallelStep(_Golden):
    async def test_same_chunks_with_the_parallel_nodes_unordered(self):
        native, ours = await _both(_fan_out_graph(), "golden-fan", {"log": []})

        # The two nodes of a step run at once: their chunks may interleave
        # differently from native, so compare them per node, not as a sequence.
        def by_node(items: List) -> Dict[str, List]:
            groups: Dict[str, List] = {}
            for mode, payload in _wire(items):
                node = (
                    payload[1]["langgraph_node"]
                    if mode == "messages"
                    else next(iter(payload))
                    if mode == "updates"
                    else None
                )
                groups.setdefault(str(node), []).append((mode, payload))
            return groups

        self.assertEqual(_canonical(ours), _canonical(native))
        # One updates chunk per node, as native emits.
        updates = [list(c) for m, c in ours if m == "updates"]
        self.assertEqual(
            sorted(u[0] for u in updates), ["join", "left", "right", "start"]
        )
        self.assertEqual(sorted(len(u) for u in updates), [1, 1, 1, 1])
        self.assertEqual(
            sorted(len(u) for m, u in native if m == "updates" for u in [list(u)]),
            [1, 1, 1, 1],
        )
        # Within a node the order is native's.
        self.assertEqual(by_node(ours)["right"], by_node(native)["right"])


def _router_graph(*, async_router: bool = False) -> Any:
    model = _StreamingModel()

    def first(state: _Log) -> dict:
        return {"log": ["first"]}

    def end_node(state: _Log) -> dict:
        return {"log": ["end"]}

    graph = StateGraph(_Log)
    graph.add_node("first", first)
    graph.add_node("end_node", end_node)
    graph.add_edge(START, "first")
    if async_router:

        async def arouter(state: _Log) -> str:
            get_stream_writer()("routing")
            await model.ainvoke("which way?")
            return "end_node"

        graph.add_conditional_edges("first", arouter, ["end_node"])
    else:

        def router(state: _Log) -> str:
            get_stream_writer()("routing")
            model.invoke("which way?")
            return "end_node"

        graph.add_conditional_edges("first", router, ["end_node"])
    graph.add_edge("end_node", END)
    return graph.compile()


class TestRouterThatStreams(_Golden):
    async def test_its_tokens_and_custom_event_reach_the_stream(self):
        native, ours = await _both(_router_graph(), "router-sync", {"log": []})
        router_tokens = [
            i
            for i in ours
            if i[0] == "messages"
            and isinstance(i[1][0], AIMessageChunk)
            and i[1][1]["langgraph_node"] == "first"
        ]
        self.assertTrue(router_tokens, "the router's tokens were lost")
        self.assertIn(("custom", "routing"), ours)
        # a router's chunks are the source node's in native LangGraph
        self.assertEqual(_canonical(ours), _canonical(native))

    async def test_an_async_router_streams_too(self):
        native, ours = await _both(
            _router_graph(async_router=True), "router-async", {"log": []}
        )
        self.assertIn(("custom", "routing"), ours)
        self.assertEqual(_canonical(ours), _canonical(native))

    async def test_it_does_not_look_like_a_missing_stream(self):
        # the router's chunks are not a node's updates: no stream_unavailable
        _, ours = await _both(_router_graph(), "router-no-unavailable", {"log": []})
        self.assertNotIn("stream_unavailable", [m for m, _ in ours])


def _start_router_graph() -> Any:
    model = _StreamingModel()

    def router(state: _Log) -> str:
        model.invoke("which way?")
        return "end_node"

    graph = StateGraph(_Log)
    graph.add_node("end_node", lambda state: {"log": ["end"]})
    graph.add_conditional_edges(START, router, ["end_node"])
    graph.add_edge("end_node", END)
    return graph.compile()


class TestStartRouter(_Golden):
    async def test_its_tokens_carry_the_start_trigger_like_native(self):
        native, ours = await _both(_start_router_graph(), "start-router", {"log": []})

        def metadata(items: List) -> List:
            return [
                {k: i[1][1][k] for k in ("langgraph_node", "langgraph_triggers")}
                for i in items
                if i[0] == "messages" and isinstance(i[1][0], AIMessageChunk)
            ]

        self.assertTrue(metadata(ours))
        self.assertEqual(metadata(ours), metadata(native))
        self.assertEqual(metadata(ours)[0]["langgraph_triggers"], ("__start__",))


def _node(name: str) -> Callable[[_Log], dict]:
    def run(state: _Log) -> dict:
        get_stream_writer()(name)
        return {"log": [name]}

    run.__name__ = name
    return run


def _build(nodes: List[str], wire: Callable[[StateGraph], None]) -> Any:
    graph = StateGraph(_Log)
    for name in nodes:
        graph.add_node(name, _node(name))
    wire(graph)
    return graph.compile()


def _two_starts(graph: StateGraph) -> None:
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    graph.add_edge("a", END)
    graph.add_edge("b", END)


def _conditional_start(graph: StateGraph) -> None:
    graph.add_conditional_edges(START, lambda state: "b", ["a", "b"])
    graph.add_edge("a", END)
    graph.add_edge("b", END)


def _conditional_start_and_edge(graph: StateGraph) -> None:
    graph.add_edge(START, "a")
    graph.add_conditional_edges(START, lambda state: ["b"], ["b"])
    graph.add_edge("a", END)
    graph.add_edge("b", END)


def _waiting(graph: StateGraph) -> None:
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    graph.add_edge(["a", "b"], "j")
    graph.add_edge("j", END)


def _waiting_uneven(graph: StateGraph) -> None:
    # the join's sources finish in different steps: a -> c, and b at once
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    graph.add_edge("a", "c")
    graph.add_edge(["c", "b"], "j")
    graph.add_edge("j", END)


def _end_in_one_branch(graph: StateGraph) -> None:
    # x reaches END in step 0 while y still has z to run
    graph.add_edge(START, "x")
    graph.add_edge(START, "y")
    graph.add_edge("x", END)
    graph.add_edge("y", "z")
    graph.add_edge("z", END)


def _conditional_end_in_one_branch(graph: StateGraph) -> None:
    graph.add_edge(START, "x")
    graph.add_edge(START, "y")
    graph.add_conditional_edges("x", lambda state: END, [END])
    graph.add_edge("y", "z")
    graph.add_edge("z", END)


CASES = {
    "two-starts": (["a", "b"], _two_starts),
    "conditional-start": (["a", "b"], _conditional_start),
    "conditional-and-direct-start": (["a", "b"], _conditional_start_and_edge),
    "waiting-edge": (["a", "b", "j"], _waiting),
    "waiting-uneven": (["a", "b", "c", "j"], _waiting_uneven),
    "end-in-one-branch": (["x", "y", "z"], _end_in_one_branch),
    "conditional-end-in-one-branch": (["x", "y", "z"], _conditional_end_in_one_branch),
}


class TestEntryShapes(_Golden):
    async def test_the_stream_has_native_chunks(self):
        for name, (nodes, wire) in CASES.items():
            with self.subTest(case=name):
                self.setUp()
                native, ours = await _both(_build(nodes, wire), f"entry-{name}", {})
                # nodes of one step run at once: compare the chunks, not their order
                self.assertEqual(_canonical(ours), _canonical(native))

    def test_the_final_state_is_native(self):
        for name, (nodes, wire) in CASES.items():
            with self.subTest(case=name):
                self.setUp()
                compiled = _build(nodes, wire)
                out = _run(
                    DaprWorkflowGraphRunner(graph=compiled, name=f"entry-out-{name}"),
                    {"log": []},
                )
                self.assertEqual(out.status, "completed", out.error)
                self.assertEqual(out.output["log"], compiled.invoke({"log": []})["log"])


class _Seen(TypedDict, total=False):
    log: Annotated[list, operator.add]
    seen: list


class TestUnevenJoinFiresAfterItsLastSource(_Golden):
    """add_edge(["c", "b"], "j") with b in step 1 and c in step 2: j runs in step 3."""

    def _graph(self) -> Any:
        model = _StreamingModel()

        def j(state: _Seen) -> dict:
            model.invoke("hi")
            return {"seen": list(state["log"]), "log": ["j"]}

        graph = StateGraph(_Seen)
        for name in ("a", "b", "c"):
            graph.add_node(name, _node(name))  # type: ignore[arg-type]
        graph.add_node("j", j)
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge("a", "c")
        graph.add_edge(["c", "b"], "j")
        graph.add_edge("j", END)
        return graph.compile()

    def test_j_runs_alone_in_the_step_after_c_and_sees_its_write(self):
        compiled = self._graph()
        out = _run(
            DaprWorkflowGraphRunner(graph=compiled, name="uneven-join-state"),
            {"log": []},
        )
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.executed, [["a", "b"], ["c"], ["j"]])
        self.assertEqual(out.output["seen"], ["a", "b", "c"])
        self.assertEqual(out.output["seen"], compiled.invoke({"log": []})["seen"])

    async def test_j_s_messages_carry_the_step_native_gives_them(self):
        native, ours = await _both(self._graph(), "uneven-join-steps", {"log": []})
        self.assertEqual(self._steps(ours), self._steps(native))
        self.assertEqual(set(self._steps(ours)), {3})


class TestStartConditionError(_Golden):
    def test_a_start_condition_that_cannot_route_is_an_error_not_a_silent_run(self):
        graph = StateGraph(_Log)
        graph.add_node("a", _node("a"))
        graph.add_conditional_edges(START, lambda state: "nowhere", {"x": "a"})
        graph.add_edge("a", END)
        out = _run(
            DaprWorkflowGraphRunner(graph=graph.compile(), name="entry-bad-route"),
            {"log": []},
        )
        self.assertEqual(out.status, "error")
        self.assertIn("nowhere", out.error or "")
        self.assertEqual(out.executed, [])


class TestTriggersMetadata(_Golden):
    async def test_a_nodes_messages_carry_all_its_triggers(self):
        model = _StreamingModel()

        def j(state: _Log) -> dict:
            model.invoke("hi")
            return {"log": ["j"]}

        graph = StateGraph(_Log)
        graph.add_node("a", _node("a"))
        graph.add_node("b", _node("b"))
        graph.add_node("j", j)
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge(["a", "b"], "j")
        graph.add_edge("j", END)
        compiled = graph.compile()
        native, ours = await _both(compiled, "entry-triggers", {})

        def triggers(items: List) -> List[Any]:
            return [
                list(i[1][1]["langgraph_triggers"])
                for i in items
                if i[0] == "messages" and isinstance(i[1][0], AIMessageChunk)
            ]

        self.assertEqual(triggers(ours)[0], ["branch:to:j", "join:a+b:j"])
        self.assertEqual(triggers(ours), triggers(native))
        self.assertEqual(_canonical(ours), _canonical(native))


class TestAnOlderGraphConfig(_Golden):
    def test_a_config_without_entry_points_starts_at_its_entry_point(self):
        compiled = _build(["a", "b"], _two_starts)
        runner = DaprWorkflowGraphRunner(graph=compiled, name="entry-legacy")
        config = runner.graph_config.to_dict()
        config.pop("entry_points")
        config.pop("waiting_edges")
        from diagrid.agent.langgraph.models import GraphConfig

        runner._graph_config = GraphConfig.from_dict(config)
        self.assertIsNone(runner._graph_config.entry_points)
        out = _run(runner, {"log": []})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["log"], ["a"])  # as before the field existed


def _two_edges_graph(source: Any = "a") -> Any:
    def r1(state: _Log) -> str:
        get_stream_writer()("router1")
        return "b"

    def r2(state: _Log) -> str:
        get_stream_writer()("router2")
        return "c"

    graph = StateGraph(_Log)
    for name in ("a", "b", "c"):
        graph.add_node(name, lambda s, n=name: {"log": [n]})
    graph.add_edge(START, "a")
    graph.add_conditional_edges(source, r1, ["b"])
    graph.add_conditional_edges(source, r2, ["c"])
    graph.add_edge("b", END)
    graph.add_edge("c", END)
    return graph.compile()


def _two_start_edges_graph() -> Any:
    def r1(state: _Log) -> str:
        get_stream_writer()("router1")
        return "b"

    def r2(state: _Log) -> str:
        get_stream_writer()("router2")
        return "c"

    graph = StateGraph(_Log)
    for name in ("b", "c"):
        graph.add_node(name, lambda s, n=name: {"log": [n]})
    graph.add_conditional_edges(START, r1, ["b"])
    graph.add_conditional_edges(START, r2, ["c"])
    graph.add_edge("b", END)
    graph.add_edge("c", END)
    return graph.compile()


class TestTwoConditionalEdges(_Golden):
    async def test_two_edges_of_one_node_are_not_a_retry(self):
        native, ours = await _both(_two_edges_graph(), "two-edges", {"log": []})
        self.assertNotIn("node_reset", [m for m, _ in ours])
        self.assertEqual(
            sorted(p for m, p in ours if m == "custom"),
            sorted(p for m, p in native if m == "custom"),
        )

    async def test_two_conditional_start_edges_are_not_a_retry(self):
        native, ours = await _both(
            _two_start_edges_graph(), "two-start-edges", {"log": []}
        )
        self.assertNotIn("node_reset", [m for m, _ in ours])
        self.assertEqual(
            sorted(p for m, p in ours if m == "custom"),
            sorted(p for m, p in native if m == "custom"),
        )


@tool("add")
async def async_add(a: int, b: int) -> int:
    """Add two numbers (async only)."""
    get_stream_writer()({"adding": [a, b]})
    return a + b


def _async_tool_graph() -> Any:
    model = _ReactModel()

    def agent(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([async_add]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


class TestAsyncOnlyTool(_Golden):
    def test_the_run_completes_with_the_tools_result(self):
        graph = StateGraph(MessagesState)
        graph.add_node("tools", ToolNode([async_add]))
        graph.add_edge(START, "tools")
        compiled = graph.compile()
        call = AIMessage(
            content="",
            tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "c1"}],
        )
        runner = DaprWorkflowGraphRunner(graph=compiled, name="async-tool-run")
        out = _run(runner, {"messages": [call]})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["messages"][-1]["content"], "5")

    async def test_matches_native_including_the_tools_custom_chunk(self):
        # a failing activity never completes the fake run: bound the wait
        native, ours = await asyncio.wait_for(
            _both(_async_tool_graph(), "golden-async-tool", ADD_INPUT), 20
        )
        self._assert_same_sequence(native, ours)
        self.assertIn(("custom", {"adding": [2, 3]}), [i for i in ours])


class TestWhichNodesUseTheAsyncEntry(_Golden):
    def test_a_sync_function_node_keeps_the_sync_entry(self):
        # add_node gives every sync function an ``afunc`` that only wraps it in
        # an executor: using it would run the node on a loop of its own.
        def plain(state: MessagesState) -> dict:
            return {}

        graph = StateGraph(MessagesState)
        graph.add_node("plain", plain)
        graph.add_node("tools", ToolNode([async_add]))
        graph.add_edge(START, "plain")
        compiled = graph.compile()
        self.assertFalse(_has_async_entry(compiled.nodes["plain"].bound))

    def test_a_tool_node_and_an_async_function_node_use_it(self):
        async def aplain(state: MessagesState) -> dict:
            return {}

        graph = StateGraph(MessagesState)
        graph.add_node("aplain", aplain)
        graph.add_node("tools", ToolNode([async_add]))
        graph.add_edge(START, "aplain")
        compiled = graph.compile()
        self.assertTrue(_has_async_entry(compiled.nodes["aplain"].bound))
        self.assertTrue(_has_async_entry(compiled.nodes["tools"].bound))
