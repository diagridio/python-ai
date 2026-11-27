# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The native-vs-Dapr comparison harness the golden streaming tests share.

Each graph runs through LangGraph's own ``astream`` and through the runner's
``astream`` (the real ``run_async``, ``_poll_workflow`` and ``agent_workflow``
with a fake workflow client standing in for Dapr); the tests compare the two.
"""

import json
import operator
import re
import threading
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple
from unittest import mock

from dapr.ext.workflow import WorkflowStatus
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumpd
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .streaming_helpers import _collect, _IdCtx, _StreamingModel
from .test_runner import _RunnerTestCase

# LangGraph names a task by uuid; Dapr by an integer. Only the node matters.
_TASK_NS = re.compile(r"^(\w+):[0-9a-f-]+$")


def _normalize(items: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    """Make two runs comparable: stable message ids, no task uuids."""
    ids: Dict[str, str] = {}

    def fix(value: Any) -> Any:
        if isinstance(value, BaseMessage):
            copy = value.model_copy()
            if copy.id is not None:
                copy.id = ids.setdefault(copy.id, f"msg-{len(ids)}")
            return copy
        if isinstance(value, str):
            return _TASK_NS.sub(r"\1:<task>", value)
        if isinstance(value, dict):
            return {k: fix(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(fix(v) for v in value)
        if isinstance(value, list):
            return [fix(v) for v in value]
        return value

    return [(mode, fix(payload)) for mode, payload in items]


MODES = ["messages", "updates", "custom"]


ADD_INPUT = {"messages": [{"role": "user", "content": "add 2 and 3"}]}


class _FakeWorkflowClient:
    """Runs ``agent_workflow`` on a thread when scheduled; the poll sees its state."""

    def __init__(self, runner: DaprWorkflowGraphRunner) -> None:
        self._runner = runner
        self._done: Dict[str, Any] = {}

    def schedule_new_workflow(self, *, workflow: Any, input: Any, instance_id: str):
        def work() -> None:
            from diagrid.agent.langgraph.models import (
                GraphWorkflowInput,
            )

            wf_input = GraphWorkflowInput.from_dict(json.loads(json.dumps(input)))
            self._done[instance_id] = _run_input(self._runner, wf_input, instance_id)

        threading.Thread(target=work, daemon=True).start()

    def get_workflow_state(self, *, instance_id: str):
        output = self._done.get(instance_id)
        if output is None:
            return SimpleNamespace(
                runtime_status=WorkflowStatus.RUNNING,
                serialized_output=None,
                serialized_custom_status=None,
                failure_details=None,
            )
        return SimpleNamespace(
            runtime_status=WorkflowStatus.COMPLETED,
            serialized_output=json.dumps(output),
            serialized_custom_status=None,
            failure_details=None,
        )


def _run_input(runner: Any, wf_input: Any, instance_id: str) -> Dict[str, Any]:
    from diagrid.agent.langgraph.workflow import agent_workflow

    ctx = _IdCtx(runner, instance_id)
    workflow = agent_workflow(ctx, json.loads(json.dumps(wf_input.to_dict())))
    result = None
    try:
        while True:
            result = workflow.send(result)
    except StopIteration as stop:
        return stop.value  # type: ignore[no-any-return]


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    get_stream_writer()({"adding": [a, b]})
    return a + b


class _ReactModel(BaseChatModel):
    """Asks for ``add(2, 3)`` once, streaming the call, then streams its answer."""

    @property
    def _llm_type(self) -> str:
        return "react"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ReactModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="The sum is 5"))]
        )

    def _stream(self, messages, stop=None, run_manager=None, **kw) -> Iterator:
        if any(isinstance(m, ToolMessage) for m in messages):
            chunks = [AIMessageChunk(content=t) for t in ("The ", "sum ", "is 5")]
        else:
            chunks = [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "add",
                            "args": '{"a": 2, "b": 3}',
                            "id": "c1",
                            "index": 0,
                        }
                    ],
                )
            ]
        for chunk in chunks:
            gen = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content or "", chunk=gen)
            yield gen


async def _both(
    compiled: Any, name: str, graph_input: Any, modes: List[str] = MODES
) -> Tuple[List, List]:
    native = [
        item
        async for item in compiled.astream(
            graph_input,
            {
                "configurable": {"thread_id": "t1"},
                "tags": ["x"],
                "metadata": {"foo": "bar"},
            },
            stream_mode=modes,
        )
    ]
    runner = DaprWorkflowGraphRunner(graph=compiled, name=name)
    runner._started = True
    runner._workflow_client = _FakeWorkflowClient(runner)  # type: ignore[assignment]
    with mock.patch("diagrid.agent.langgraph.workflow.when_all", side_effect=list):
        ours = await _collect(
            runner.astream(
                graph_input,
                thread_id="t1",
                config={"tags": ["x"], "metadata": {"foo": "bar"}},
                workflow_id=f"wf-{name}",
                stream_mode=modes,
                poll_interval=0.01,
            )
        )
    return native, ours


def _wire(items: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    return [(mode, dumpd(payload)) for mode, payload in _normalize(items)]


class _Golden(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    def _assert_same_sequence(self, native: List, ours: List) -> None:
        want, got = _wire(native), _wire(ours)
        self.assertEqual([m for m, _ in got], [m for m, _ in want])
        for index, (expected, actual) in enumerate(zip(want, got)):
            with self.subTest(chunk=index, mode=expected[0]):
                self.assertEqual(actual, expected)

    def _steps(self, items: List) -> List[int]:
        return [i[1][1]["langgraph_step"] for i in items if i[0] == "messages"]


class _Log(TypedDict, total=False):
    log: Annotated[list, operator.add]


class _Out(TypedDict, total=False):
    out: Annotated[list, operator.add]


def _canonical(items: List) -> List[str]:
    """The chunks as sorted JSON: nodes of one step run at once, so order is free."""
    return sorted(json.dumps(i, sort_keys=True, default=str) for i in _wire(items))


class _State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    count: int


def _golden_graph() -> Any:
    model = _StreamingModel()

    def chat(state: _State) -> dict:
        get_stream_writer()({"progress": 1})
        return {"messages": [model.invoke(state["messages"])]}

    def counter(state: _State) -> dict:
        get_stream_writer()("counting")
        return {"count": 1}

    def quiet(state: _State) -> None:
        return None

    graph = StateGraph(_State)
    graph.add_node("chat", chat)
    graph.add_node("counter", counter)
    graph.add_node("quiet", quiet)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", "counter")
    graph.add_edge("counter", "quiet")
    graph.add_edge("quiet", END)
    return graph.compile()


def _chain(length: int) -> Any:
    graph = StateGraph(_Log)
    names = [f"n{i}" for i in range(length)]
    for name in names:
        graph.add_node(name, lambda state, name=name: {"log": [name]})
    graph.add_edge(START, names[0])
    for a, b in zip(names, names[1:]):
        graph.add_edge(a, b)
    graph.add_edge(names[-1], END)
    return graph.compile()


def _runner(compiled: Any, name: str) -> DaprWorkflowGraphRunner:
    runner = DaprWorkflowGraphRunner(graph=compiled, name=name)
    runner._started = True
    runner._workflow_client = _FakeWorkflowClient(runner)  # type: ignore[assignment]
    return runner


class _FakeDaprTest(_Golden):
    def setUp(self):
        super().setUp()
        patch = mock.patch(
            "diagrid.agent.langgraph.workflow.when_all", side_effect=list
        )
        patch.start()
        self.addCleanup(patch.stop)
