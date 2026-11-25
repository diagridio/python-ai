# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Shared helpers for the streaming tests: fake model, fake run, SSE and serve."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple
from unittest import mock

from dapr.ext.workflow import WorkflowStatus
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime

from diagrid.agent.core.workflow import runner as core_runner
from diagrid.agent.core.workflow.stream import InProcessSink, StreamClosed, StreamEvent
from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.models import (
    ChannelState,
    ExecuteNodeInput,
    GraphWorkflowOutput,
)
from diagrid.agent.langgraph.streaming import STREAM_RUN_KEY

from .test_runner import _InlineWorkflowContext, _run, _RunnerTestCase


def parse_sse(text: str) -> List[Tuple[str, Any]]:
    """``(event, data)`` for each frame of a ``text/event-stream`` body."""
    events = []
    for frame in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.split("\n"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def serve_app(runner: DaprWorkflowGraphRunner, **serve_kwargs: Any) -> Any:
    """The FastAPI app ``runner.serve()`` builds, without starting a server."""
    captured: Dict[str, Any] = {}
    with (
        mock.patch("uvicorn.run", lambda app, **kw: captured.update(app=app)),
        mock.patch.object(runner, "_setup_telemetry"),
        mock.patch.object(runner, "_setup_serve_defaults"),
    ):
        runner.serve(**serve_kwargs)
    return captured["app"]


TOKENS = ["Hel", "lo ", "wor", "ld"]


class _StreamingModel(BaseChatModel):
    """Chat model with a real ``_stream``; ``invoke`` streams when a handler asks."""

    @property
    def _llm_type(self) -> str:
        return "fake-streaming"

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kw):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage("".join(TOKENS)))]
        )

    def _stream(
        self, messages: List[BaseMessage], stop=None, run_manager=None, **kw
    ) -> Iterator[ChatGenerationChunk]:
        for tok in TOKENS:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=tok))
            if run_manager:
                run_manager.on_llm_new_token(tok, chunk=chunk)
            yield chunk


class _Ctx:
    def __init__(self, workflow_id: str, task_id: int = 1):
        self.workflow_id = workflow_id
        self.orchestration_id = workflow_id  # what dapr's activity context reads
        self.task_id = task_id


class _IdCtx(_InlineWorkflowContext):
    """Inline context whose activities see a chosen workflow id."""

    def __init__(self, runner, workflow_id: str):
        super().__init__(runner)
        self._workflow_id = workflow_id

    def call_activity(self, activity: Any, *, input: Any, retry_policy: Any = None):
        activity_name = activity if isinstance(activity, str) else activity.__name__
        self.scheduled.append(activity_name)
        fn = (
            self._registry.get_activity(activity)
            if isinstance(activity, str)
            else activity
        )
        out = fn(_Ctx(self._workflow_id), json.loads(json.dumps(input)))
        return json.loads(json.dumps(out))


def _chat_graph(
    *,
    with_runtime: bool = False,
    name: str = "stream",
    writes: bool = True,
    stream_registry: Any = None,
) -> Any:
    model = _StreamingModel()

    def plain(state: MessagesState) -> dict:
        if writes:
            get_stream_writer()({"progress": 1})
        return {"messages": [model.invoke(state["messages"])]}

    def with_rt(state: MessagesState, runtime: Runtime) -> dict:
        if writes:
            get_stream_writer()({"progress": 1})
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", with_rt if with_runtime else plain)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return DaprWorkflowGraphRunner(
        graph=graph.compile(), name=name, stream_registry=stream_registry
    )


def _fake_run_async(runner: DaprWorkflowGraphRunner):
    """Stand-in for ``run_async``: runs the workflow on a worker thread."""

    async def run_async(
        input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
    ):
        yield {"type": "workflow_started", "workflow_id": workflow_id}
        out = await asyncio.to_thread(_run, runner, input, _IdCtx(runner, workflow_id))
        yield {
            "type": "workflow_completed",
            "workflow_id": workflow_id,
            "output": out.output,
        }

    return run_async


INPUT = {"messages": [{"role": "user", "content": "hi"}]}


async def _collect(agen) -> list:
    return [item async for item in agen]


_real_sleep = asyncio.sleep


async def _instant_sleep(delay: float, *args: Any) -> None:
    await _real_sleep(0)


class _Unavailable(ConnectionError):
    """Stands in for the gRPC UNAVAILABLE error of a restarting sidecar."""


def _state(status: WorkflowStatus, output: Any = None) -> Any:
    return SimpleNamespace(
        runtime_status=status,
        serialized_output=json.dumps(output) if output is not None else None,
        serialized_custom_status=None,
        failure_details=None,
    )


def _completed_state() -> Any:
    out = GraphWorkflowOutput(
        output={"x": 1}, channel_state=ChannelState(), steps=1, status="completed"
    )
    return _state(WorkflowStatus.COMPLETED, out.to_dict())


class _PollCase(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    """A runner whose workflow client answers each status poll from a script."""

    def _runner(self, polls: List[Any], name: str):
        runner = _chat_graph(name=name)
        runner._started = True
        client = mock.Mock()
        client.get_workflow_state.side_effect = polls
        runner._workflow_client = client
        for patch in (
            mock.patch.object(core_runner, "POLL_RETRY_BASE_SECONDS", 0),
            # the SSE path polls every 0.5 s whatever the test asks for
            mock.patch.object(core_runner.asyncio, "sleep", _instant_sleep),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        return runner, client


def _payload(
    node: str = "n", step: int = 0, *, streamed: bool = True
) -> Dict[str, Any]:
    """The input of an ``execute_node_activity`` call, for a streamed run or not."""
    return ExecuteNodeInput(
        node_name=node,
        channel_state=ChannelState(),
        config={STREAM_RUN_KEY: True} if streamed else None,
        step=step,
    ).to_dict()


FAILED_ERROR = {
    "message": "activity blew up",
    "error_type": "RuntimeError",
    "stack_trace": "Traceback ...",
}


def _scripted_run_async(*events: Dict[str, Any]):
    """A ``run_async`` that yields ``events`` (``thread_id`` filled in)."""

    async def run_async(
        input, *, thread_id=None, config=None, workflow_id=None, poll_interval=0.5
    ):
        for event in events:
            yield (
                {"thread_id": thread_id, **event}
                if event["type"] == "workflow_started"
                else dict(event)
            )

    return run_async


STARTED = {"type": "workflow_started", "workflow_id": "wf-fixed", "graph_name": "g"}


FAILED = {"type": "workflow_failed", "workflow_id": "wf-fixed", "error": FAILED_ERROR}


def _completed(status: str, error: Any = None) -> Dict[str, Any]:
    return {
        "type": "workflow_completed",
        "workflow_id": "wf-fixed",
        "output": {"x": 1},
        "steps": 2,
        "status": status,
        **({"error": error} if error else {}),
    }


class _ScriptedCase(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    def _runner(self, events: List[Dict[str, Any]], name: str):
        runner = _chat_graph(name=name)
        runner._started = True
        patch = mock.patch.object(runner, "run_async", _scripted_run_async(*events))
        patch.start()
        self.addCleanup(patch.stop)
        return runner

    def _client(self, runner, **serve_kwargs):
        from fastapi.testclient import TestClient

        return TestClient(serve_app(runner, **serve_kwargs))


ENDINGS: Dict[str, Dict[str, Any]] = {
    "status-error": {
        "events": [STARTED, _completed("error", "Node 'x' failed: nope")],
        "message": "Node 'x' failed: nope",
    },
    "max-steps": {
        "events": [STARTED, _completed("max_steps_reached", "Max steps (3) reached")],
        "message": "Max steps (3) reached",
    },
    "workflow-failed": {
        "events": [STARTED, FAILED],
        "message": "activity blew up",
        "error": FAILED_ERROR,
    },
    "workflow-error": {
        "events": [
            STARTED,
            {"type": "workflow_error", "workflow_id": "wf-fixed", "error": "gone"},
        ],
        "message": "gone",
        "error_type": "WorkflowError",
    },
    "terminated": {
        "events": [STARTED, {"type": "workflow_terminated", "workflow_id": "wf-fixed"}],
        "message": "Workflow was terminated",
        "error_type": "WorkflowTerminated",
    },
    "no-final-event": {
        "events": [STARTED],
        "message": "run ended without a result",
        "error_type": "NoResult",
        # the id is the one the run was started with, not the scripted one
        "workflow_id": None,
    },
}


def _event(
    attempt: str, seq: int, mode: str = "messages", node: str = "n", step: int = 0
) -> StreamEvent:
    """One chunk of a node's stream; its data names the attempt and sequence."""
    return StreamEvent("wf", step, node, attempt, seq, (), mode, f"{attempt}-{seq}")


async def _drain(sink: InProcessSink) -> List[Any]:
    """Everything the sink delivers, up to and including its close marker."""
    await asyncio.sleep(0)
    items: List[Any] = []
    while True:
        item = await sink.get()
        items.append(item)
        if isinstance(item, StreamClosed):
            return items
