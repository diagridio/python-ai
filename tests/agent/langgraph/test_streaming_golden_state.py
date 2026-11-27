# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""State, channels, ``Send``, config and ``Command(goto=...)``: the stream matches native ``astream``."""

import json
import operator
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from unittest import mock

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, Send
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.workflow import _result_to_raw_writes

from .golden_harness import (
    ADD_INPUT,
    _both,
    _Golden,
    _Log,
    _Out,
    _ReactModel,
    _wire,
    add,
)
from .streaming_helpers import (
    INPUT,
    _collect,
    _fake_run_async,
    _StreamingModel,
)
from .test_runner import _run, _RunnerTestCase


class _Writer(TypedDict, total=False):
    log: Annotated[list, operator.add]
    total: Annotated[int, operator.add]
    messages: Annotated[list, add_messages]


def _writer_graph() -> Any:
    def write(state: _Writer) -> dict:
        return {
            "messages": [
                {"role": "assistant", "content": "one"},
                ("assistant", "two"),
                AIMessage(content="three"),
            ],
            "total": 5,
        }

    def echo(state: _Writer) -> dict:
        return {"messages": state["messages"]}

    graph = StateGraph(_Writer)
    graph.add_node("write", write)
    graph.add_node("echo", echo)
    graph.add_edge(START, "write")
    graph.add_edge("write", "echo")
    graph.add_edge("echo", END)
    return graph.compile()


def _messages(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    return out["messages"]


class TestFirstWriteToAReducerChannel(_Golden):
    def test_it_goes_through_the_reducer_like_native(self):
        compiled = _writer_graph()
        native = compiled.invoke({"log": []})
        out = _run(
            DaprWorkflowGraphRunner(graph=compiled, name="reducer-first"),
            {"log": []},
        )
        self.assertEqual(out.status, "completed", out.error)
        ours = _messages(out.output)
        self.assertEqual([m["content"] for m in ours], ["one", "two", "three"])
        self.assertTrue(all(m.get("id") for m in ours), ours)
        self.assertEqual(len(ours), len(native["messages"]))
        self.assertEqual(out.output["total"], native["total"])

    async def test_updates_match_native_in_channels_values_and_message_fields(self):
        native, ours = await _both(_writer_graph(), "reducer-updates", {"log": []})
        want = [p for m, p in _wire(native) if m == "updates"]
        got = [p for m, p in _wire(ours) if m == "updates"]
        self.assertEqual(len(got), len(want))
        for index, (expected, actual) in enumerate(zip(want, got)):
            with self.subTest(update=index):
                self.assertEqual(actual, expected)


class TestTupleInputMessages(_Golden):
    def _graph(self) -> Any:
        def echo(state: _Writer) -> dict:
            return {"messages": state["messages"]}

        graph = StateGraph(_Writer)
        graph.add_node("echo", echo)
        graph.add_edge(START, "echo")
        graph.add_edge("echo", END)
        return graph.compile()

    def test_a_tuple_is_a_message_with_an_id(self):
        compiled = self._graph()
        native = compiled.invoke({"messages": [("user", "hi")]})
        out = _run(
            DaprWorkflowGraphRunner(graph=compiled, name="tuple-input"),
            {"messages": [("user", "hi")]},
        )
        self.assertEqual(out.status, "completed", out.error)
        ours = _messages(out.output)
        self.assertEqual(len(ours), len(native["messages"]))
        self.assertEqual(ours[0]["content"], "hi")
        self.assertEqual(ours[0]["type"], "human")
        self.assertTrue(ours[0].get("id"))

    async def test_what_a_node_echoes_is_not_streamed_again(self):
        native, ours = await _both(
            self._graph(), "tuple-echo", {"messages": [("user", "hi")]}
        )
        got = [p for m, p in _wire(ours) if m == "updates"]
        want = [p for m, p in _wire(native) if m == "updates"]
        self.assertEqual(got, want)


def _stray_key_graph() -> Any:
    def unk(state: _Log) -> dict:
        return {"log": ["a"], "bogus": 1}

    def only(state: _Log) -> dict:
        return {"bogus": 2}

    def command(state: _Log) -> Command:
        return Command(update={"log": ["c"], "bogus": 3})

    def seen(state: _Log) -> dict:
        # what the node can read: a stray key must not have been stored
        return {"log": sorted(state)}

    graph = StateGraph(_Log)
    for name, fn in (("unk", unk), ("only", only), ("cmd", command), ("seen", seen)):
        graph.add_node(name, fn)  # type: ignore[arg-type]
    graph.add_edge(START, "unk")
    graph.add_edge("unk", "only")
    graph.add_edge("only", "cmd")
    graph.add_edge("cmd", "seen")
    graph.add_edge("seen", END)
    return graph.compile()


class TestKeysThatAreNotChannels(_Golden):
    async def test_updates_match_native(self):
        native, ours = await _both(_stray_key_graph(), "golden-channels", {"log": []})
        want = [p for m, p in _wire(native) if m == "updates"]
        got = [p for m, p in _wire(ours) if m == "updates"]
        self.assertEqual(got, want)

    def test_a_stray_key_is_not_stored_for_later_nodes(self):
        compiled = _stray_key_graph()
        out = _run(
            DaprWorkflowGraphRunner(graph=compiled, name="chan-state"), {"log": []}
        )
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["log"], compiled.invoke({"log": []})["log"])
        self.assertNotIn("bogus", out.channel_state.values)


class _Pydantic(BaseModel):
    x: int = 0
    log: Annotated[list, operator.add] = Field(default_factory=list)
    note: Optional[str] = None


@dataclass
class _Plain:
    x: int = 0
    log: Annotated[list, operator.add] = field(default_factory=list)


def _object_graph(schema: Any, make: Any) -> Any:
    def node(state: Any) -> Any:
        return make()

    graph = StateGraph(schema)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return graph.compile()


class TestStateObjectResults(_Golden):
    async def _check(self, schema: Any, make: Any, name: str, start: Any) -> None:
        compiled = _object_graph(schema, make)
        native, ours = await _both(compiled, name, start)
        self._assert_same_sequence(native, ours)
        updates = [c for m, c in ours if m == "updates"]
        self.assertEqual(list(updates[0]), ["n"])
        self.assertNotIn("__result__", updates[0]["n"])

    async def test_pydantic_instance(self):
        await self._check(
            _Pydantic, lambda: _Pydantic(x=5, log=["a"]), "golden-pyd", {"x": 1}
        )

    async def test_pydantic_fields_left_at_their_default_are_not_written(self):
        # LangGraph writes only what was set or differs from a None default
        await self._check(_Pydantic, lambda: _Pydantic(x=2), "golden-pyd-def", {})

    async def test_dataclass_instance(self):
        await self._check(_Plain, lambda: _Plain(x=5, log=["a"]), "golden-dc", {"x": 1})


class TestTheStateIsWritten(_Golden):
    def test_the_workflow_output_has_the_nodes_fields(self):
        compiled = _object_graph(_Pydantic, lambda: _Pydantic(x=5, log=["a"]))
        runner = DaprWorkflowGraphRunner(graph=compiled, name="golden-pyd-out")
        out = _run(runner, {"x": 1, "log": []})
        self.assertEqual(out.status, "completed", out.error)
        want = compiled.invoke({"x": 1, "log": []})
        self.assertEqual(out.output["x"], want["x"])
        self.assertEqual(out.output["log"], want["log"])


class TestWritesOfAnObject(_Golden):
    def test_a_plain_object_is_still_one_result(self):
        class Thing:
            pass

        thing = Thing()
        self.assertEqual(_result_to_raw_writes(thing), [("__result__", thing)])


def _config_graph(*, graph_config: Any = None, node_metadata: Any = None) -> Any:
    model = _ReactModel()

    def agent(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent, metadata=node_metadata)
    graph.add_node("tools", ToolNode([add]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    compiled = graph.compile()
    return compiled.with_config(graph_config) if graph_config else compiled


def _chunks(items: Any, node: str, kind: type) -> Any:
    return [
        m
        for mode, (m, md) in [i for i in items if i[0] == "messages"]
        if md["langgraph_node"] == node and isinstance(m, kind)
    ]


def _metadata(items: Any, node: str) -> Any:
    return [
        md
        for mode, (m, md) in [i for i in items if i[0] == "messages"]
        if md["langgraph_node"] == node
    ]


class TestGraphConfig(_Golden):
    async def test_metadata_and_tags_of_with_config_reach_every_chunk(self):
        compiled = _config_graph(graph_config={"metadata": {"gm": 1}, "tags": ["gt"]})
        native, ours = await _both(compiled, "meta-graph", ADD_INPUT)
        for node in ("agent", "tools"):
            with self.subTest(node=node):
                self.assertTrue(_metadata(native, node))
                for md_native, md_ours in zip(
                    _metadata(native, node), _metadata(ours, node)
                ):
                    self.assertEqual(md_ours["gm"], 1)
                    self.assertEqual(
                        md_ours.get("tags"), md_native.get("tags"), (md_native, md_ours)
                    )
        self._assert_same_sequence(native, ours)

    async def test_ls_integration_of_the_graph_wins(self):
        compiled = _config_graph(graph_config={"metadata": {"ls_integration": "mine"}})
        native, ours = await _both(compiled, "meta-ls", ADD_INPUT)
        tool = _metadata(ours, "tools")[0]
        self.assertEqual(tool["ls_integration"], "mine")
        self.assertEqual(
            tool["ls_integration"], _metadata(native, "tools")[0]["ls_integration"]
        )

    async def test_the_caller_overrides_the_graph(self):
        # the caller of _both sets foo=bar
        compiled = _config_graph(graph_config={"metadata": {"foo": "graph"}})
        native, ours = await _both(compiled, "meta-caller", ADD_INPUT)
        self.assertEqual(_metadata(native, "tools")[0]["foo"], "bar")
        self.assertEqual(_metadata(ours, "tools")[0]["foo"], "bar")


class TestNodeMetadata(_Golden):
    async def test_add_node_metadata_reaches_that_nodes_chunks_only(self):
        compiled = _config_graph(node_metadata={"custom": "meta", "foo": "node"})
        native, ours = await _both(compiled, "meta-node", ADD_INPUT)
        agent = _metadata(ours, "agent")
        self.assertTrue(agent)
        for md in agent:
            self.assertEqual(md["custom"], "meta")
            self.assertEqual(md["foo"], "node")  # the node's own beats the caller's
        for md in _metadata(ours, "tools"):
            self.assertNotIn("custom", md)
        self._assert_same_sequence(native, ours)

    async def test_the_tool_message_chunk_carries_the_graphs_metadata_too(self):
        compiled = _config_graph(graph_config={"metadata": {"gm": 1}})
        native, ours = await _both(compiled, "meta-toolmsg", ADD_INPUT)
        tool_chunks = [
            md
            for mode, (m, md) in [i for i in ours if i[0] == "messages"]
            if isinstance(m, ToolMessage)
        ]
        self.assertEqual(len(tool_chunks), 1)
        self.assertEqual(tool_chunks[0]["gm"], 1)


def _subgraph_graph() -> Any:
    model = _StreamingModel()

    def inner(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    sub = StateGraph(MessagesState)
    sub.add_node("inner", inner)
    sub.add_edge(START, "inner")
    sub.add_edge("inner", END)

    outer = StateGraph(MessagesState)
    outer.add_node("outer", sub.compile())
    outer.add_edge(START, "outer")
    outer.add_edge("outer", END)
    return outer.compile()


class TestSubgraphNode(_Golden):
    async def test_messages_are_the_outer_nodes_output_not_the_inner_tokens(self):
        native, ours = await _both(_subgraph_graph(), "golden-sub", INPUT)
        messages = [i[1] for i in ours if i[0] == "messages"]
        self.assertEqual([m[1]["langgraph_node"] for m in messages], ["outer"])
        self.assertEqual(messages[0][0].content, "Hello world")
        self.assertNotIn(
            "inner", " ".join(m[1]["langgraph_checkpoint_ns"] for m in messages)
        )
        self._assert_same_sequence(native, ours)

    def test_the_final_state_has_each_message_once_as_native_does(self):
        compiled = _subgraph_graph()
        runner = DaprWorkflowGraphRunner(graph=compiled, name="golden-sub-out")
        out = _run(runner, INPUT)
        self.assertEqual(out.status, "completed", out.error)
        got = [(m["type"], m["content"]) for m in out.output["messages"]]
        want = [(m.type, m.content) for m in compiled.invoke(INPUT)["messages"]]
        self.assertEqual(got, want)


def _map_reduce(*, goto: bool = False) -> Any:
    def work(arg: dict) -> dict:
        return {"out": [arg["i"] * 10]}

    def fan(state: _Out) -> Any:
        sends = [Send("work", {"i": i}) for i in (1, 2, 3)]
        return Command(goto=sends) if goto else {"out": []}

    graph = StateGraph(_Out)
    graph.add_node("fan", fan)
    graph.add_node("work", work)
    graph.add_edge(START, "fan")
    if not goto:
        graph.add_conditional_edges(
            "fan", lambda state: [Send("work", {"i": i}) for i in (1, 2, 3)], ["work"]
        )
    graph.add_edge("work", END)
    return graph.compile()


def _updates(items: List) -> List[str]:
    return sorted(
        json.dumps(p, sort_keys=True, default=str)
        for m, p in _wire(items)
        if m == "updates"
    )


class TestMapReduce(_Golden):
    def test_every_send_runs_the_node_on_its_argument(self):
        for goto in (False, True):
            with self.subTest(goto=goto):
                self.setUp()
                compiled = _map_reduce(goto=goto)
                out = _run(
                    DaprWorkflowGraphRunner(graph=compiled, name=f"send-{goto}"),
                    {"out": []},
                )
                self.assertEqual(out.status, "completed", out.error)
                self.assertEqual(
                    sorted(out.output["out"]),
                    sorted(compiled.invoke({"out": []})["out"]),
                )
                self.assertEqual(sorted(out.output["out"]), [10, 20, 30])

    async def test_one_updates_chunk_per_send_like_native(self):
        for goto in (False, True):
            with self.subTest(goto=goto):
                self.setUp()
                native, ours = await _both(
                    _map_reduce(goto=goto), f"send-stream-{goto}", {"out": []}
                )
                self.assertEqual(_updates(ours), _updates(native))
                works = [p for m, p in ours if m == "updates" and "work" in p]
                self.assertEqual(len(works), 3)

    async def test_the_sends_are_push_tasks_in_the_metadata(self):
        from langchain_core.messages import AIMessageChunk

        from .test_streaming import _StreamingModel

        model = _StreamingModel()

        def work(arg: dict) -> dict:
            model.invoke("hi")
            return {"out": [arg["i"]]}

        graph = StateGraph(_Out)
        graph.add_node("work", work)
        graph.add_conditional_edges(
            START, lambda state: [Send("work", {"i": i}) for i in (1, 2)], ["work"]
        )
        graph.add_edge("work", END)
        native, ours = await _both(graph.compile(), "send-meta", {"out": []})

        def push(items: List) -> List[Any]:
            return sorted(
                (
                    md["langgraph_triggers"],
                    tuple(md["langgraph_path"][:1]),
                    md["langgraph_step"],
                )
                for m, (c, md) in [i for i in items if i[0] == "messages"]
                if isinstance(c, AIMessageChunk)
            )

        self.assertEqual(push(ours), push(native))
        self.assertTrue(all(t == ("__pregel_push",) for t, _, _ in push(ours)))

    def test_a_send_of_a_node_that_a_static_edge_also_reaches_runs_both(self):
        def work(state: Any) -> dict:
            return {"out": ["arg" if "i" in state else "state"]}

        graph = StateGraph(_Out)
        graph.add_node("a", lambda state: {"out": ["a"]})
        graph.add_node("work", work)
        graph.add_edge(START, "a")
        graph.add_edge("a", "work")
        graph.add_conditional_edges("a", lambda s: [Send("work", {"i": 1})], ["work"])
        graph.add_edge("work", END)
        compiled = graph.compile()
        out = _run(
            DaprWorkflowGraphRunner(graph=compiled, name="send-both"), {"out": []}
        )
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(
            sorted(out.output["out"]), sorted(compiled.invoke({"out": []})["out"])
        )


class TestPrebuiltAgentToolCalls(_Golden):
    """``create_agent`` sends each tool call to ``tools`` as a ``Send``."""

    def _agent(self) -> Any:
        from langchain.agents import create_agent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        class Model(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "two-calls"

            def bind_tools(self, tools: Any, **kw: Any) -> "Model":
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kw):
                if any(m.type == "tool" for m in messages):
                    msg = AIMessage(content="done")
                else:
                    msg = AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "add", "args": {"a": 1, "b": 2}, "id": "c1"},
                            {"name": "add", "args": {"a": 3, "b": 4}, "id": "c2"},
                        ],
                    )
                return ChatResult(generations=[ChatGeneration(message=msg)])

        return create_agent(Model(), tools=[add])

    async def test_two_tool_calls_are_two_tools_updates(self):
        try:
            compiled = self._agent()
        except ImportError:
            self.skipTest("langchain.agents is not installed")
        native, ours = await _both(
            compiled,
            "send-create-agent",
            {"messages": [{"role": "user", "content": "go"}]},
        )
        tools = [p for m, p in ours if m == "updates" and "tools" in p]
        want = [p for m, p in native if m == "updates" and "tools" in p]
        self.assertEqual(len(want), 2)
        self.assertEqual(len(tools), 2)


def _send_router_graph(*, pull_too: bool = False) -> Any:
    model = _StreamingModel()

    def work(arg: Any) -> dict:
        get_stream_writer()({"work": arg.get("i") if isinstance(arg, dict) else None})
        return {"out": ["w"]}

    def route(state: _Out) -> str:
        get_stream_writer()("route")
        model.invoke("which way?")
        return END

    def fan(state: _Out) -> Any:
        return Command(
            goto=[Send("work", {"i": i}) for i in (1, 2, 3)]
            + (["work"] if pull_too else [])
        )

    graph = StateGraph(_Out)
    graph.add_node("fan", fan)
    graph.add_node("work", work)
    graph.add_edge(START, "fan")
    graph.add_conditional_edges("work", route, [END])
    return graph.compile()


def _tasks(items: List) -> List[str]:
    """The (path, triggers) of every token chunk, whichever task's it is."""
    return sorted(
        json.dumps([md["langgraph_path"], md["langgraph_triggers"]], default=str)
        for m, (c, md) in [i for i in _wire(items) if i[0] == "messages"]
        for _ in [0]
    )


def _canonical_by_task(items: List) -> List[str]:
    """``_canonical`` with each token's message id named by its task's path.

    Native runs the Send tasks at once, so which task's tokens get the first
    message id depends on the order they happened to run in.
    """
    out = []
    for mode, payload in _wire(items):
        if mode == "messages":
            chunk, md = payload
            chunk = {**chunk, "kwargs": dict(chunk["kwargs"])}
            chunk["kwargs"]["id"] = json.dumps(md["langgraph_path"], default=str)
            payload = [chunk, md]
        out.append(json.dumps((mode, payload), sort_keys=True, default=str))
    return sorted(out)


class TestRouterOfSendTasks(_Golden):
    async def test_each_router_streams_as_its_own_task_without_a_reset(self):
        native, ours = await _both(_send_router_graph(), "send-router", {"out": []})
        self.assertNotIn("node_reset", [m for m, _ in ours])
        self.assertEqual(_canonical_by_task(ours), _canonical_by_task(native))

    async def test_a_pull_task_and_send_tasks_of_one_node(self):
        native, ours = await _both(
            _send_router_graph(pull_too=True), "send-router-pull", {"out": []}
        )
        self.assertNotIn("node_reset", [m for m, _ in ours])
        # which task a message id goes to depends on the order the tasks ran in
        self.assertEqual(_tasks(ours), _tasks(native))
        self.assertEqual(
            sorted(c for m, c in ours if m == "custom" and isinstance(c, str)),
            ["route"] * 4,
        )

    async def test_the_routers_tokens_carry_the_push_path_and_trigger(self):
        _, ours = await _both(_send_router_graph(), "send-router-meta", {"out": []})
        routed = [
            md
            for m, (c, md) in [i for i in ours if i[0] == "messages"]
            if isinstance(c, AIMessageChunk)
        ]
        self.assertEqual(sorted({md["langgraph_path"][1] for md in routed}), [0, 1, 2])
        self.assertTrue(
            all(md["langgraph_triggers"] == ("__pregel_push",) for md in routed)
        )


class _GotoState(TypedDict, total=False):
    count: int
    log: Annotated[list, operator.add]


def _goto_runner(name: str, build: Any) -> DaprWorkflowGraphRunner:
    graph = StateGraph(_GotoState)
    build(graph)
    runner = DaprWorkflowGraphRunner(graph=graph.compile(), name=name)
    runner._started = True
    return runner


def _goto_b(graph: Any) -> None:
    def a(state: _GotoState) -> Command[Literal["b"]]:
        return Command(update={"count": 5, "log": ["a"]}, goto="b")

    def b(state: _GotoState) -> dict:
        return {"count": state["count"] + 1, "log": ["b"]}

    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge(START, "a")
    graph.add_edge("b", END)


class TestGoto(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_the_workflow_follows_goto(self):
        runner = _goto_runner("goto-wf", _goto_b)
        out = _run(runner, {"count": 0, "log": []})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.executed, [["a"], ["b"]])
        self.assertEqual(out.output["count"], 6)

    async def test_the_stream_shows_the_target_node_like_native_astream(self):
        runner = _goto_runner("goto-stream", _goto_b)
        native = [
            c
            async for c in runner.graph.astream(
                {"count": 0, "log": []}, stream_mode="updates"
            )
        ]
        with mock.patch.object(runner, "run_async", _fake_run_async(runner)):
            ours = await _collect(
                runner.astream(
                    {"count": 0, "log": []},
                    workflow_id="wf-goto",
                    stream_mode="updates",
                )
            )
        self.assertEqual([list(c) for c in ours], [list(c) for c in native])
        self.assertEqual(ours, native)

    async def test_goto_a_list_fans_out(self):
        def build(graph: Any) -> None:
            def a(state: _GotoState) -> Command[Literal["b", "c"]]:
                return Command(update={"log": ["a"]}, goto=["b", "c"])

            graph.add_node("a", a)
            graph.add_node("b", lambda state: {"log": ["b"]})
            graph.add_node("c", lambda state: {"log": ["c"]})
            graph.add_edge(START, "a")
            graph.add_edge("b", END)
            graph.add_edge("c", END)

        out = _run(_goto_runner("goto-fan", build), {"log": []})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.executed[0], ["a"])
        self.assertCountEqual(out.executed[1], ["b", "c"])

    async def test_goto_a_node_that_a_static_edge_also_reaches_runs_it_once(self):
        def build(graph: Any) -> None:
            def a(state: _GotoState) -> Command[Literal["b"]]:
                return Command(update={"log": ["a"]}, goto="b")

            graph.add_node("a", a)
            graph.add_node("b", lambda state: {"log": ["b"]})
            graph.add_edge(START, "a")
            graph.add_edge("a", "b")
            graph.add_edge("b", END)

        out = _run(_goto_runner("goto-dup", build), {"log": []})
        self.assertEqual(out.executed, [["a"], ["b"]])

    async def test_goto_end_ends_the_run(self):
        def build(graph: Any) -> None:
            def a(state: _GotoState) -> Command:
                return Command(update={"log": ["a"]}, goto=END)

            graph.add_node("a", a)
            graph.add_edge(START, "a")

        out = _run(_goto_runner("goto-end", build), {"log": []})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.executed, [["a"]])


class TestSendInGoto(_RunnerTestCase):
    def test_a_send_in_command_goto_runs_its_node_on_the_argument(self):
        class State(TypedDict, total=False):
            x: int

        def a(state: State) -> Command[Literal["w"]]:
            return Command(update={"x": 1}, goto=Send("w", {"x": 2}))

        graph = StateGraph(State)
        graph.add_node("a", a)
        graph.add_node("w", lambda state: {"x": state["x"] + 10})
        graph.add_edge(START, "a")
        graph.add_edge("w", END)
        compiled = graph.compile()
        runner = DaprWorkflowGraphRunner(graph=compiled, name="send-goto")
        out = _run(runner, {"x": 0})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.executed, [["a"], ["w#0"]])
        # the node read the Send argument (2), not the channel state (1)
        self.assertEqual(out.output["x"], 12)
        self.assertEqual(out.output["x"], compiled.invoke({"x": 0})["x"])
