# Diagrid Agent Smolagents Extension

This is the Diagrid Agent extension for Hugging Face `smolagents` agents using Dapr Workflows.

This extension enables durable execution of smolagents `ToolCallingAgent`
instances using Dapr Workflows. Each model call and each tool execution runs
as a separate Dapr Workflow activity, providing:

- **Fault tolerance**: Agents automatically resume from the last successful activity on failure
- **Durability**: Agent state is persisted and can survive process restarts
- **Observability**: Full visibility into agent execution through Dapr's workflow APIs

## Why `ToolCallingAgent` and not `CodeAgent`?

smolagents ships two agent types. `ToolCallingAgent` asks the model for
structured, individually-dispatched JSON tool calls — each one maps cleanly
to a single Dapr activity. `CodeAgent` asks the model to write a Python code
blob per step and executes the whole blob in one shot in a sandboxed
interpreter; any number of tool calls inside that blob are just ordinary
Python function calls with no discrete boundary to checkpoint. Only
`ToolCallingAgent` is supported here.

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

## Installation

```bash
pip install "diagrid[smolagents]"
```

## Quick Start

```python
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from diagrid.agent.smolagents import DaprWorkflowAgentRunner

@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query.
    """
    return f"Results for: {query}"

# Build the agent as usual
agent = ToolCallingAgent(
    model=OpenAIServerModel(model_id="gpt-4o-mini"),
    tools=[search_web],
)

# Create a Dapr workflow runner
runner = DaprWorkflowAgentRunner(
    agent=agent,
    name="my-agent",
)

# Start the workflow runtime
runner.start()

try:
    # Run the agent - each model call and tool execution is now a durable activity
    async for event in runner.run_async(
        task="Hello, please help me with...",
        session_id="my-session",
    ):
        print(event)
finally:
    runner.shutdown()
```

## How It Works

The extension wraps a smolagents `ToolCallingAgent`'s loop in a Dapr Workflow:

1. **Workflow Start**: When you call `run_async()`, a new Dapr Workflow instance is created
2. **Model Activity**: Each model call is executed as a durable activity
3. **Tool Activities**: Each tool execution is a separate durable activity, run in parallel when the model requests multiple tool calls in one turn
4. **Termination**: When the model calls the built-in `final_answer` tool, the workflow returns its argument as the result — matching smolagents' own bare-name-match termination convention
5. **Checkpointing**: After each activity, the workflow state is checkpointed
6. **Recovery**: On failure, the workflow resumes from the last successful activity

Conversation history is kept as plain `system`/`user`/`assistant` text turns
rather than replaying smolagents' own `AgentMemory`/`ActionStep` objects —
smolagents itself flattens tool calls and their results into plain text
turns before sending them to the model, so this keeps durable workflow state
simple without losing fidelity.

## Architecture

```text
DaprWorkflowAgentRunner (orchestrates the agent loop)
+-- Activity: call_model_activity()          # Get next action from the model
+-- Activity: execute_tool_activity()        # First tool call
+-- Activity: execute_tool_activity()        # Second tool call (parallel with the first)
+-- Activity: call_model_activity()          # Model processes tool results
+-- ... continues until the model calls final_answer
```

## Requirements

- Python >= 3.11
- Dapr >= 1.17.3
- `smolagents` >= 1.26.0
- A Dapr state store component configured
