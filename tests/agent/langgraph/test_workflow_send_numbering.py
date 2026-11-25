# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``Send`` tasks are numbered over the whole step, in the order LangGraph uses."""

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


class TestSendNumbering(_Golden):
    async def test_two_sending_sources_number_their_tasks_over_the_step(self):
        from .test_streaming import _StreamingModel

        model = _StreamingModel()

        def work(arg: dict) -> dict:
            model.invoke("hi")
            return {"log": [arg["i"]]}

        def sends(prefix: str) -> Any:
            return lambda state: [Send("work", {"i": f"{prefix}{k}"}) for k in (1, 2)]

        graph = StateGraph(_Log)
        graph.add_node("f1", lambda s: {})
        graph.add_node("f2", lambda s: {})
        graph.add_node("work", work)
        graph.add_edge(START, "f1")
        graph.add_edge(START, "f2")
        graph.add_conditional_edges("f1", sends("a"), ["work"])
        graph.add_conditional_edges("f2", sends("b"), ["work"])
        graph.add_edge("work", END)
        native, ours = await _both(graph.compile(), "numbering-paths", {"log": []})

        def paths(items: List) -> List[Any]:
            return sorted(
                {
                    tuple(md["langgraph_path"])
                    for m, (c, md) in [i for i in items if i[0] == "messages"]
                    if isinstance(c, AIMessageChunk)
                }
            )

        self.assertEqual(paths(native), [("__pregel_push", i, False) for i in range(4)])
        self.assertEqual(paths(ours), paths(native))
