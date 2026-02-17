"""
ReAct-Style Agent with Dapr Workflow Example

This example demonstrates a ReAct-style agent pattern using LangGraph
with durable execution via Dapr Workflows. The agent can use tools
and iterates until it has enough information to respond.

Prerequisites:
    - Redis running on localhost:6379
    - Install dependencies: pip install diagrid langgraph

Run with:
    cd diagrid/agent/langgraph/examples
    dapr run --app-id langgraph-react --resources-path ./components -- python3 react_agent.py
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from diagrid.agent.langgraph import DaprWorkflowGraphRunner


# Simulated tool implementations
def search_tool(query: str) -> str:
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


def calculator_tool(expression: str) -> str:
    """Simulate a calculator tool."""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid expression"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


# Define the state schema
class AgentState(TypedDict):
    # The user's question
    question: str
    # Accumulated thoughts/reasoning
    thoughts: List[str]
    # Tool calls made
    tool_calls: List[Dict]
    # Tool results
    tool_results: List[str]
    # Current step in reasoning
    current_step: int
    # Final answer (when ready)
    final_answer: Optional[str]
    # Whether we should continue
    should_continue: bool


# Define node functions
def reasoning_node(state: AgentState) -> dict:
    """Reason about the question and decide next action."""
    print(f"  [reasoning] Step {state['current_step']}", flush=True)
    print(f"  [reasoning] Question: {state['question']}", flush=True)

    question = state["question"].lower()
    step = state["current_step"]
    tool_results = state.get("tool_results", [])

    # Simple rule-based reasoning for demo purposes
    if step == 0:
        # First step: analyze the question
        if "weather" in question:
            thought = "User is asking about weather. I should use the search tool."
            tool_call = {"tool": "search", "args": f"weather {question.split()[-1]}"}
            return {
                "thoughts": state["thoughts"] + [thought],
                "tool_calls": state["tool_calls"] + [tool_call],
                "current_step": step + 1,
                "should_continue": True,
            }
        elif "calculate" in question or any(
            op in question for op in ["+", "-", "*", "/"]
        ):
            thought = "User wants a calculation. I should use the calculator."
            # Extract expression (simplified)
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
        # We have tool results, formulate final answer
        thought = f"I have the tool results: {tool_results[-1]}. I can now answer."
        return {
            "thoughts": state["thoughts"] + [thought],
            "current_step": step + 1,
            "should_continue": False,  # Ready to finalize
        }

    else:
        # Safety fallback
        return {
            "thoughts": state["thoughts"] + ["Reached max reasoning steps."],
            "should_continue": False,
        }


def tool_execution_node(state: AgentState) -> dict:
    """Execute the pending tool call."""
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {"tool_results": state.get("tool_results", [])}

    # Execute the most recent tool call
    tool_call = tool_calls[-1]
    tool_name = tool_call.get("tool", "")
    tool_args = tool_call.get("args", "")

    print(f"  [tool_execution] Executing: {tool_name}({tool_args})", flush=True)

    if tool_name == "search":
        result = search_tool(tool_args)
    elif tool_name == "calculator":
        result = calculator_tool(tool_args)
    else:
        result = f"Unknown tool: {tool_name}"

    print(f"  [tool_execution] Result: {result}", flush=True)

    return {
        "tool_results": state.get("tool_results", []) + [result],
    }


def answer_node(state: AgentState) -> dict:
    """Generate the final answer."""
    print("  [answer] Generating final answer", flush=True)

    tool_results = state.get("tool_results", [])

    # Construct answer from gathered information
    if tool_results:
        answer = f"Based on my research: {tool_results[-1]}"
    else:
        answer = "I couldn't find specific information to answer your question."

    print(f"  [answer] Final answer: {answer}", flush=True)

    return {"final_answer": answer}


# Routing function
def should_continue(state: AgentState) -> Literal["tools", "answer"]:
    """Decide whether to continue with tools or generate answer."""
    if state.get("should_continue", True):
        return "tools"
    return "answer"


def build_agent_graph() -> StateGraph:
    """Build the ReAct-style agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tools", tool_execution_node)
    graph.add_node("answer", answer_node)

    # Add edges
    graph.add_edge(START, "reasoning")

    # Conditional edge after reasoning
    graph.add_conditional_edges(
        "reasoning",
        should_continue,
        {
            "tools": "tools",
            "answer": "answer",
        },
    )

    # After tool execution, go back to reasoning
    graph.add_edge("tools", "reasoning")

    # Answer leads to END
    graph.add_edge("answer", END)

    return graph


async def run_agent(runner: DaprWorkflowGraphRunner, question: str, thread_id: str):
    """Run the agent with a question."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"Question: {question}", flush=True)
    print(f"{'=' * 60}", flush=True)

    input_state = {
        "question": question,
        "thoughts": [],
        "tool_calls": [],
        "tool_results": [],
        "current_step": 0,
        "final_answer": None,
        "should_continue": True,
    }

    async for event in runner.run_async(
        input=input_state,
        thread_id=thread_id,
    ):
        event_type = event["type"]

        if event_type == "workflow_started":
            print(f"[Workflow] Started: {event.get('workflow_id')}", flush=True)

        elif event_type == "workflow_status_changed":
            status = event.get("status", "")
            if "RUNNING" in str(status):
                print("[Workflow] Processing...", flush=True)

        elif event_type == "workflow_completed":
            print("\n[Workflow] Completed!", flush=True)
            output = event.get("output", {})

            print("\nThinking process:", flush=True)
            for i, thought in enumerate(output.get("thoughts", []), 1):
                print(f"  {i}. {thought}", flush=True)

            print("\nTool calls made:", flush=True)
            for tc in output.get("tool_calls", []):
                print(f"  - {tc.get('tool')}({tc.get('args')})", flush=True)

            print(f"\nFinal Answer: {output.get('final_answer', 'N/A')}", flush=True)

        elif event_type == "workflow_failed":
            print(f"[Workflow] Failed: {event.get('error')}", flush=True)


async def main():
    print("=" * 60, flush=True)
    print("LangGraph with Dapr Workflow - ReAct Agent Example", flush=True)
    print("=" * 60, flush=True)

    # Build and compile the graph
    graph = build_agent_graph()
    compiled = graph.compile()

    print("\nAgent structure:", flush=True)
    print("  START -> reasoning -> [tools -> reasoning]* -> answer -> END", flush=True)
    print("\nAvailable tools:", flush=True)
    print("  - search: Look up information", flush=True)
    print("  - calculator: Perform calculations", flush=True)
    print("-" * 60, flush=True)

    # Create the Dapr Workflow runner
    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        max_steps=20,  # Allow multiple reasoning cycles
        name="react_agent",
    )

    # Start the workflow runtime
    print("\nStarting Dapr Workflow runtime...", flush=True)
    runner.start()
    print("Runtime started!", flush=True)

    try:
        # Test with different questions
        await run_agent(runner, "What is the weather in Tokyo?", "react-001")
        await run_agent(runner, "Calculate 25 * 4 + 10", "react-002")
        await run_agent(runner, "What is the capital of France?", "react-003")

        print("\n" + "=" * 60, flush=True)
        print("All agent runs complete!", flush=True)
        print("=" * 60, flush=True)

    finally:
        # Shutdown the runtime
        print("\nShutting down workflow runtime...", flush=True)
        runner.shutdown()
        print("Done!", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
