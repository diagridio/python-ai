# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A reducer channel holds the types its nodes wrote, not their JSON.

State such as ``Annotated[set, operator.or_]`` or ``Annotated[tuple,
operator.add]`` failed the run (the reducer was handed a string or a list where
it needs a set or a tuple), and a list of messages under ``operator.add`` came
back to the next node as dicts.
"""

import operator
from typing import Any, List

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .test_runner import _RunnerTestCase, _run


class _Sets(TypedDict, total=False):
    tags: Annotated[set, operator.or_]
    frozen: Annotated[frozenset, operator.or_]
    seen: List[str]


class _Tuples(TypedDict, total=False):
    pair: Annotated[tuple, operator.add]
    seen: str


class _History(TypedDict, total=False):
    history: Annotated[List[AnyMessage], operator.add]
    seen: str


def _two_nodes(schema: Any, first: Any, second: Any, reader: Any) -> Any:
    graph = StateGraph(schema)
    graph.add_node("a", first)
    graph.add_node("b", second)
    graph.add_node("read", reader)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", "read")
    graph.add_edge("read", END)
    return graph.compile()


class TestReducerValueTypes(_RunnerTestCase):
    def test_a_set_channel_reduces_with_or(self):
        compiled = _two_nodes(
            _Sets,
            lambda s: {"tags": {"a"}, "frozen": frozenset({1})},
            lambda s: {"tags": {"b"}, "frozen": frozenset({2})},
            lambda s: {"seen": [type(s["tags"]).__name__, type(s["frozen"]).__name__]},
        )
        native = compiled.invoke({})
        out = _run(DaprWorkflowGraphRunner(graph=compiled, name="rv-set"), {})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(sorted(out.output["tags"]), sorted(native["tags"]))
        self.assertEqual(sorted(out.output["frozen"]), [1, 2])
        self.assertEqual(out.output["seen"], ["set", "frozenset"])

    def test_a_tuple_channel_reduces_with_add(self):
        compiled = _two_nodes(
            _Tuples,
            lambda s: {"pair": (1,)},
            lambda s: {"pair": (2,)},
            lambda s: {"seen": type(s["pair"]).__name__},
        )
        native = compiled.invoke({})
        out = _run(DaprWorkflowGraphRunner(graph=compiled, name="rv-tuple"), {})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(list(out.output["pair"]), list(native["pair"]))
        self.assertEqual(out.output["seen"], "tuple")

    def test_messages_in_a_list_channel_stay_messages_for_the_next_node(self):
        compiled = _two_nodes(
            _History,
            lambda s: {"history": [AIMessage("x")]},
            lambda s: {"history": [AIMessage("y")]},
            lambda s: {
                "seen": "|".join(
                    f"{type(m).__name__}:{m.content}" for m in s["history"]
                )
            },
        )
        native = compiled.invoke({"history": []})
        out = _run(
            DaprWorkflowGraphRunner(graph=compiled, name="rv-history"),
            {"history": []},
        )
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["seen"], native["seen"])
        self.assertEqual(out.output["seen"], "AIMessage:x|AIMessage:y")
        self.assertEqual(
            [m["content"] for m in out.output["history"]],
            [m.content for m in native["history"]],
        )


class TestStoredValues(_RunnerTestCase):
    def test_plain_json_is_stored_as_it_is_and_the_rest_round_trips(self):
        import datetime
        import json

        from diagrid.agent.langgraph.values import (
            coerce_to_default,
            decode_value,
            encode_value,
            is_encoded,
        )

        for plain in (None, 1, 2.5, "s", [1, "a", None], {"k": [1, {"j": True}]}):
            self.assertIs(encode_value(plain), plain)
        for rich in (
            {"a"},
            frozenset({1}),
            datetime.date(2020, 1, 1),
            [AIMessage("x")],
        ):
            stored = encode_value(rich)
            self.assertTrue(is_encoded(stored), rich)
            json.dumps(stored)
            self.assertEqual(decode_value(stored), rich)
        # a tuple has no JSON of its own: it comes back as a list, then as the
        # channel's own type
        self.assertEqual(coerce_to_default(decode_value(encode_value((1,))), ()), (1,))
        self.assertEqual(coerce_to_default([1], []), [1])
