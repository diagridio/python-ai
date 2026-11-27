# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A task's Command goto Sends come before the Sends of its conditional edges."""

import itertools
import operator
from typing import Any, List

from langchain_core.messages import AIMessageChunk
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .test_runner import _clear_dapr_registration, _run, _RunnerTestCase
from .golden_harness import _both, _Golden


class _Log(TypedDict, total=False):
    log: Annotated[list, operator.add]


def _native_and_ours(graph: StateGraph, graph_input: dict, name: str):
    _clear_dapr_registration()
    compiled = graph.compile()
    native = compiled.invoke(graph_input)
    out = _run(DaprWorkflowGraphRunner(graph=compiled, name=name), graph_input)
    return native, out


class TestGotoOrder(_Golden):
    def test_goto_sends_come_before_a_branchs_sends(self):
        def fan(state: _Log) -> Command:
            return Command(goto=[Send("w", {"i": "goto"})])

        graph = StateGraph(_Log)
        graph.add_node("fan", fan)
        graph.add_node("w", lambda arg: {"log": [str(arg["i"])]})
        graph.add_edge(START, "fan")
        graph.add_conditional_edges("fan", lambda s: [Send("w", {"i": "cond"})], ["w"])
        graph.add_edge("w", END)
        native, out = _native_and_ours(graph, {"log": ["fan"]}, "numbering-goto")
        self.assertEqual(native["log"], ["fan", "goto", "cond"])
        self.assertEqual(out.output["log"], native["log"])
