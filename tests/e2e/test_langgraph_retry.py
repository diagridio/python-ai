"""E2E test for deterministic LangGraph node retry via Dapr RetryPolicy.

Unlike agent framework retry tests (where tool exceptions are caught by
``execute_tool_activity`` and returned as error messages), LangGraph node
exceptions propagate directly through ``execute_node_activity``. Dapr's
``RetryPolicy(max_number_of_attempts=3)`` retries the activity automatically.

This test is **deterministic** — no LLM needed.
"""

import threading
import uuid
from typing import List, TypedDict

import pytest

from tests.e2e.conftest import clear_dapr_registration

# Module-level counters track node invocations across Dapr activity calls.
# Protected by a lock because Dapr may dispatch activities on a thread pool.
_lock = threading.Lock()
_counts: dict[str, int] = {"node1": 0, "node2": 0, "node3": 0}


def _reset_counts() -> None:
    """Reset all node execution counters."""
    with _lock:
        for key in _counts:
            _counts[key] = 0


# ---------------------------------------------------------------------------
# Graph state and node definitions
# ---------------------------------------------------------------------------


class RetryState(TypedDict):
    messages: List[str]
    step: int


def _node_one(state: RetryState) -> dict:
    """First node — always succeeds."""
    with _lock:
        _counts["node1"] += 1
    return {
        "messages": state["messages"] + ["Node 1 completed"],
        "step": state["step"] + 1,
    }


def _node_two(state: RetryState) -> dict:
    """Second node — fails on first 2 attempts, succeeds on 3rd.

    Dapr's RetryPolicy retries the activity automatically.
    """
    with _lock:
        _counts["node2"] += 1
        current = _counts["node2"]
    if current <= 2:
        raise ConnectionError(f"Simulated failure (attempt {current})")
    return {
        "messages": state["messages"] + ["Node 2 completed after retries"],
        "step": state["step"] + 1,
    }


def _node_three(state: RetryState) -> dict:
    """Third node — always succeeds."""
    with _lock:
        _counts["node3"] += 1
    return {
        "messages": state["messages"] + ["Node 3 completed"],
        "step": state["step"] + 1,
    }


# ---------------------------------------------------------------------------
# Test: Node retry with Dapr RetryPolicy
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_langgraph_node_retry() -> None:
    """Test that Dapr retries a failing node activity automatically.

    Graph: START -> node_one -> node_two -> node_three -> END.
    node_two raises ConnectionError on attempts 1-2, succeeds on attempt 3.
    Dapr's RetryPolicy(max_number_of_attempts=3) handles the retry.

    Assertions:
    - Workflow completes with step == 3
    - node_one executed exactly once (not re-executed after checkpoint)
    - node_two executed exactly 3 times (2 failures + 1 success)
    - node_three executed exactly once (after node_two succeeds)
    """
    from langgraph.graph import END, START, StateGraph

    from diagrid.agent.langgraph import DaprWorkflowGraphRunner

    _reset_counts()

    graph = StateGraph(RetryState)
    graph.add_node("node_one", _node_one)
    graph.add_node("node_two", _node_two)
    graph.add_node("node_three", _node_three)
    graph.add_edge(START, "node_one")
    graph.add_edge("node_one", "node_two")
    graph.add_edge("node_two", "node_three")
    graph.add_edge("node_three", END)
    compiled = graph.compile()

    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        max_steps=50,
        name="e2e-langgraph-retry-test",
    )
    try:
        runner.start()
        result = runner.invoke(
            input={"messages": ["Start"], "step": 0},
            thread_id=f"e2e-retry-{uuid.uuid4().hex[:8]}",
            timeout=60,
        )

        assert result is not None, "invoke returned None"
        assert result["step"] == 3, f"expected step=3, got {result['step']}"

        assert _counts["node1"] == 1, (
            f"node_one should execute once, got {_counts['node1']}"
        )
        assert _counts["node2"] == 3, (
            f"node_two should execute 3 times (2 failures + 1 success), "
            f"got {_counts['node2']}"
        )
        assert _counts["node3"] == 1, (
            f"node_three should execute once, got {_counts['node3']}"
        )
    finally:
        runner.shutdown()
        clear_dapr_registration()
