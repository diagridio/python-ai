"""E2E tests for LangGraph graphs with Dapr Workflow execution."""

import uuid
from typing import Dict, List, Literal, Optional, TypedDict

import pytest

from tests.e2e.conftest import clear_dapr_registration


# ---------------------------------------------------------------------------
# Graph state definitions
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    messages: list[str]
    counter: int


class LlmState(TypedDict):
    messages: list[str]
    counter: int
    llm_response: str


# ---------------------------------------------------------------------------
# Pure graph node functions (no LLM)
# ---------------------------------------------------------------------------


def process_node(state: SimpleState) -> dict:
    """Process the input and add a message."""
    return {
        "messages": state["messages"] + ["processed by node A"],
        "counter": state["counter"] + 1,
    }


def validate_node(state: SimpleState) -> dict:
    """Validate the processed data."""
    return {
        "messages": state["messages"] + ["validated by node B"],
        "counter": state["counter"] + 1,
    }


def finalize_node(state: SimpleState) -> dict:
    """Finalize the output."""
    return {
        "messages": state["messages"] + ["finalized by node C"],
        "counter": state["counter"] + 1,
    }


# ---------------------------------------------------------------------------
# Test A: Pure graph execution (no LLM needed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_langgraph_pure_graph_execution() -> None:
    """Test that Dapr durably executes graph nodes as activities.

    This test runs without an LLM — pure Dapr workflow execution.
    StateGraph with 3 nodes: process -> validate -> finalize.
    """
    from langgraph.graph import END, START, StateGraph

    from diagrid.agent.langgraph import DaprWorkflowGraphRunner

    graph = StateGraph(SimpleState)
    graph.add_node("process", process_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "process")
    graph.add_edge("process", "validate")
    graph.add_edge("validate", "finalize")
    graph.add_edge("finalize", END)
    compiled = graph.compile()

    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        name="e2e-langgraph-pure-test",
    )
    try:
        runner.start()
        result = runner.invoke(
            input={"messages": ["hello"], "counter": 0},
            thread_id=f"e2e-langgraph-pure-{uuid.uuid4().hex[:8]}",
            timeout=60,
        )

        assert result is not None, "invoke returned None"
        assert result["counter"] == 3, f"expected counter=3, got {result['counter']}"
        assert len(result["messages"]) == 4, (
            f"expected 4 messages (1 initial + 3 nodes), got {len(result['messages'])}"
        )
        assert result["messages"][0] == "hello"
        assert "processed by node A" in result["messages"]
        assert "validated by node B" in result["messages"]
        assert "finalized by node C" in result["messages"]
    finally:
        runner.shutdown()
        clear_dapr_registration()


# ---------------------------------------------------------------------------
# Test B: Graph with LLM node (Ollama)
# ---------------------------------------------------------------------------


def _make_llm_node(model: str, endpoint: str):
    """Create an LLM node function that calls Ollama via openai."""
    import openai

    client = openai.OpenAI(base_url=endpoint, api_key="ollama")

    def llm_node(state: LlmState) -> dict:
        """Node that calls the LLM through the OpenAI-compatible API."""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        llm_text = response.choices[0].message.content or "no response"
        return {
            "messages": state["messages"] + [f"llm said: {llm_text}"],
            "counter": state["counter"] + 1,
            "llm_response": llm_text,
        }

    return llm_node


def _process_llm_node(state: LlmState) -> dict:
    """Pure processing node before LLM."""
    return {
        "messages": state["messages"] + ["pre-processed"],
        "counter": state["counter"] + 1,
    }


def _finalize_llm_node(state: LlmState) -> dict:
    """Pure finalization node after LLM."""
    return {
        "messages": state["messages"] + ["finalized"],
        "counter": state["counter"] + 1,
    }


@pytest.mark.ollama
@pytest.mark.integration
def test_langgraph_with_llm_node(
    ollama_model: str,
    ollama_endpoint: str,
) -> None:
    """Test that a graph node calling the LLM works through Dapr.

    Graph: process_node (pure) -> llm_node (calls openai) -> finalize_node (pure).
    """
    from langgraph.graph import END, START, StateGraph

    from diagrid.agent.langgraph import DaprWorkflowGraphRunner

    llm_node = _make_llm_node(ollama_model, ollama_endpoint)

    graph = StateGraph(LlmState)
    graph.add_node("process", _process_llm_node)
    graph.add_node("llm", llm_node)
    graph.add_node("finalize", _finalize_llm_node)
    graph.add_edge(START, "process")
    graph.add_edge("process", "llm")
    graph.add_edge("llm", "finalize")
    graph.add_edge("finalize", END)
    compiled = graph.compile()

    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        name="e2e-langgraph-llm-test",
    )
    try:
        runner.start()
        result = runner.invoke(
            input={
                "messages": ["summarize: hello world"],
                "counter": 0,
                "llm_response": "",
            },
            thread_id=f"e2e-langgraph-llm-{uuid.uuid4().hex[:8]}",
            timeout=120,
        )

        assert result is not None, "invoke returned None"
        assert result["counter"] == 3, f"expected counter=3, got {result['counter']}"
        assert "llm_response" in result, "llm_response key missing from result"
        assert result["llm_response"], "llm_response was empty"
    finally:
        runner.shutdown()
        clear_dapr_registration()


# ---------------------------------------------------------------------------
# Conditional graph state and node functions (from conditional_graph.py)
# ---------------------------------------------------------------------------


class ConditionalState(TypedDict):
    input_value: int
    path_taken: str
    result: str


def _classifier_node(state: ConditionalState) -> dict:
    """Classify the input and determine the path."""
    if state["input_value"] > 100:
        path = "high"
    elif state["input_value"] > 50:
        path = "medium"
    else:
        path = "low"
    return {"path_taken": path}


def _high_value_processor(state: ConditionalState) -> dict:
    """Process high value inputs."""
    return {"result": f"High value processed: {state['input_value'] * 2}"}


def _medium_value_processor(state: ConditionalState) -> dict:
    """Process medium value inputs."""
    return {"result": f"Medium value processed: {state['input_value'] * 1.5}"}


def _low_value_processor(state: ConditionalState) -> dict:
    """Process low value inputs."""
    return {"result": f"Low value processed: {state['input_value']}"}


def _conditional_finalizer(state: ConditionalState) -> dict:
    """Finalize the processing."""
    return {"result": f"Final: {state['result']} (path: {state['path_taken']})"}


def _route_by_value(state: ConditionalState) -> Literal["high", "medium", "low"]:
    """Route to appropriate processor based on input value."""
    if state["input_value"] > 100:
        return "high"
    elif state["input_value"] > 50:
        return "medium"
    else:
        return "low"


# ---------------------------------------------------------------------------
# Test C: Conditional graph routing (from conditional_graph.py)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "input_value,expected_path",
    [(150, "high"), (75, "medium"), (25, "low")],
    ids=["high-path", "medium-path", "low-path"],
)
def test_langgraph_conditional_routing(input_value: int, expected_path: str) -> None:
    """Test Dapr's durable execution of conditional edge routing.

    Graph: START -> classifier -> [high|medium|low] -> finalizer -> END.
    Each parametrized case verifies the correct path is taken.
    """
    from langgraph.graph import END, START, StateGraph

    from diagrid.agent.langgraph import DaprWorkflowGraphRunner

    graph = StateGraph(ConditionalState)
    graph.add_node("classifier", _classifier_node)
    graph.add_node("high", _high_value_processor)
    graph.add_node("medium", _medium_value_processor)
    graph.add_node("low", _low_value_processor)
    graph.add_node("finalizer", _conditional_finalizer)
    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier",
        _route_by_value,
        {"high": "high", "medium": "medium", "low": "low"},
    )
    graph.add_edge("high", "finalizer")
    graph.add_edge("medium", "finalizer")
    graph.add_edge("low", "finalizer")
    graph.add_edge("finalizer", END)
    compiled = graph.compile()

    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        name=f"e2e-langgraph-cond-{expected_path}",
    )
    try:
        runner.start()
        result = runner.invoke(
            input={"input_value": input_value, "path_taken": "", "result": ""},
            thread_id=f"e2e-cond-{expected_path}-{uuid.uuid4().hex[:8]}",
            timeout=60,
        )

        assert result is not None, "invoke returned None"
        assert result["path_taken"] == expected_path, (
            f"expected path '{expected_path}', got '{result['path_taken']}'"
        )
        assert "Final:" in result["result"], (
            f"expected finalizer output, got '{result['result']}'"
        )
        assert expected_path in result["result"], (
            f"expected path name in result, got '{result['result']}'"
        )
    finally:
        runner.shutdown()
        clear_dapr_registration()


# ---------------------------------------------------------------------------
# ReAct agent state and node functions (from react_agent.py)
# ---------------------------------------------------------------------------


def _search_tool(query: str) -> str:
    """Simulate a search tool."""
    mock_results = {
        "weather tokyo": "Tokyo: Sunny, 22C",
        "weather paris": "Paris: Cloudy, 18C",
        "population japan": "Japan population: 125 million",
        "capital france": "Capital of France: Paris",
    }
    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return value
    return f"No results found for: {query}"


def _calculator_tool(expression: str) -> str:
    """Simulate a calculator tool."""
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid expression"
    try:
        result = eval(expression)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


class ReactState(TypedDict):
    question: str
    thoughts: List[str]
    tool_calls: List[Dict]
    tool_results: List[str]
    current_step: int
    final_answer: Optional[str]
    should_continue: bool


def _reasoning_node(state: ReactState) -> dict:
    """Reason about the question and decide next action."""
    question = state["question"].lower()
    step = state["current_step"]
    tool_results = state.get("tool_results", [])

    if step == 0:
        if "weather" in question:
            thought = "User is asking about weather. I should use the search tool."
            city = question.split()[-1] if question.split() else "tokyo"
            tool_call = {"tool": "search", "args": f"weather {city}"}
            return {
                "thoughts": state["thoughts"] + [thought],
                "tool_calls": state["tool_calls"] + [tool_call],
                "current_step": step + 1,
                "should_continue": True,
            }
        elif any(op in question for op in ["+", "-", "*", "/"]):
            thought = "User wants a calculation."
            expr = "".join(c for c in question if c in "0123456789+-*/() ")
            tool_call = {"tool": "calculator", "args": expr.strip()}
            return {
                "thoughts": state["thoughts"] + [thought],
                "tool_calls": state["tool_calls"] + [tool_call],
                "current_step": step + 1,
                "should_continue": True,
            }
        else:
            thought = "I'll search for information about this question."
            tool_call = {"tool": "search", "args": question}
            return {
                "thoughts": state["thoughts"] + [thought],
                "tool_calls": state["tool_calls"] + [tool_call],
                "current_step": step + 1,
                "should_continue": True,
            }
    elif step == 1 and tool_results:
        thought = f"I have the tool results: {tool_results[-1]}. I can now answer."
        return {
            "thoughts": state["thoughts"] + [thought],
            "current_step": step + 1,
            "should_continue": False,
        }
    else:
        return {
            "thoughts": state["thoughts"] + ["Reached max reasoning steps."],
            "should_continue": False,
        }


def _tool_execution_node(state: ReactState) -> dict:
    """Execute the pending tool call."""
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {"tool_results": state.get("tool_results", [])}

    tool_call = tool_calls[-1]
    tool_name = tool_call.get("tool", "")
    tool_args = tool_call.get("args", "")

    if tool_name == "search":
        result = _search_tool(tool_args)
    elif tool_name == "calculator":
        result = _calculator_tool(tool_args)
    else:
        result = f"Unknown tool: {tool_name}"

    return {"tool_results": state.get("tool_results", []) + [result]}


def _answer_node(state: ReactState) -> dict:
    """Generate the final answer."""
    tool_results = state.get("tool_results", [])
    if tool_results:
        answer = f"Based on my research: {tool_results[-1]}"
    else:
        answer = "I couldn't find specific information to answer your question."
    return {"final_answer": answer}


def _should_continue(state: ReactState) -> Literal["tools", "answer"]:
    """Decide whether to continue with tools or generate answer."""
    if state.get("should_continue", True):
        return "tools"
    return "answer"


# ---------------------------------------------------------------------------
# Test D: ReAct agent pattern (from react_agent.py)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_langgraph_react_agent() -> None:
    """Test Dapr's handling of graph cycles (loop-back edges).

    Graph: START -> reasoning -> [tools -> reasoning]* -> answer -> END.
    Exercises conditional loop-back, a different Dapr workflow execution pattern.
    """
    from langgraph.graph import END, START, StateGraph

    from diagrid.agent.langgraph import DaprWorkflowGraphRunner

    graph = StateGraph(ReactState)
    graph.add_node("reasoning", _reasoning_node)
    graph.add_node("tools", _tool_execution_node)
    graph.add_node("answer", _answer_node)
    graph.add_edge(START, "reasoning")
    graph.add_conditional_edges(
        "reasoning",
        _should_continue,
        {"tools": "tools", "answer": "answer"},
    )
    graph.add_edge("tools", "reasoning")
    graph.add_edge("answer", END)
    compiled = graph.compile()

    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        max_steps=20,
        name="e2e-langgraph-react-test",
    )
    try:
        runner.start()
        result = runner.invoke(
            input={
                "question": "What is the weather in Tokyo?",
                "thoughts": [],
                "tool_calls": [],
                "tool_results": [],
                "current_step": 0,
                "final_answer": None,
                "should_continue": True,
            },
            thread_id=f"e2e-react-{uuid.uuid4().hex[:8]}",
            timeout=60,
        )

        assert result is not None, "invoke returned None"
        assert len(result["thoughts"]) >= 2, (
            f"expected >=2 thoughts, got {len(result['thoughts'])}"
        )
        assert len(result["tool_calls"]) >= 1, (
            f"expected >=1 tool call, got {len(result['tool_calls'])}"
        )
        assert result["final_answer"] is not None, "final_answer was None"
        assert "Tokyo" in result["final_answer"], (
            f"expected Tokyo in answer, got '{result['final_answer']}'"
        )
    finally:
        runner.shutdown()
        clear_dapr_registration()
