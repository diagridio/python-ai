"""E2E: thread_id propagation into the LangGraph ``Runtime`` via Dapr Workflow.

Proves end-to-end, against a real Dapr sidecar, that ``execute_node_activity``
injects a populated ``Runtime`` so a node can read
``runtime.execution_info.thread_id`` — using the stock
``DaprWorkflowDeepAgentRunner`` with NO ``PatchedDaprWorkflowDeepAgentRunner``
subclass. On the unpatched code the node would observe ``execution_info is
None`` (native LangGraph leaves it unset without a checkpointer), so this test
fails before the fix and passes after it.

Deterministic, no LLM required (marked ``integration``, not ``ollama``).
"""

import uuid
from typing import TypedDict

import pytest

from tests.e2e.conftest import clear_dapr_registration


@pytest.mark.integration
def test_node_receives_thread_id_via_runtime() -> None:
    """A node executed as a Dapr activity sees the workflow's thread_id."""
    from langgraph.graph import START, END, StateGraph
    from langgraph.runtime import Runtime

    from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner

    clear_dapr_registration()

    class _State(TypedDict, total=False):
        trigger: str
        captured_exec_thread_id: str

    def capture_runtime(state: _State, runtime: Runtime) -> dict:
        # ``runtime`` is injected by RunnableCallable.invoke(); its
        # ``execution_info`` is populated by execute_node_activity (the fix).
        ei = runtime.execution_info
        return {
            "captured_exec_thread_id": ei.thread_id if ei else "NO_EXECUTION_INFO",
        }

    graph = StateGraph(_State)
    graph.add_node("capture", capture_runtime)
    graph.add_edge(START, "capture")
    graph.add_edge("capture", END)
    compiled = graph.compile()

    runner = DaprWorkflowDeepAgentRunner(
        agent=compiled,
        name="e2e-threadid-test",
        max_steps=5,
    )
    thread_id = f"e2e-tid-{uuid.uuid4().hex[:8]}"
    try:
        runner.start()
        result = runner.invoke(
            input={"trigger": "go"},
            thread_id=thread_id,
            timeout=60,
        )

        assert result is not None, "invoke returned None — workflow never completed"
        assert result.get("captured_exec_thread_id") == thread_id, (
            "node did not receive runtime.execution_info.thread_id; got "
            f"{result.get('captured_exec_thread_id')!r}, expected {thread_id!r}"
        )
    finally:
        runner.shutdown()
        clear_dapr_registration()
