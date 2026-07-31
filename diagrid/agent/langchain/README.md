# Diagrid Agent LangChain Extension

This is the Diagrid Agent extension for LangChain agents using Dapr Workflows.

This extension enables durable execution of LangChain (`langchain_core`) chat
models and tools using Dapr Workflows. Each chat-model call and each tool
execution runs as a separate Dapr Workflow activity, providing:

- **Fault tolerance**: Agents automatically resume from the last successful activity on failure
- **Durability**: Agent state is persisted and can survive process restarts
- **Observability**: Full visibility into agent execution through Dapr's workflow APIs

## Why not `langchain.agents.create_agent`?

LangChain 1.x's own `create_agent` compiles directly to a LangGraph graph —
that path is already covered by this repo's `diagrid.agent.langgraph`
extension. This extension instead wraps `langchain_core` primitives (a
`BaseChatModel` and a list of `BaseTool`) directly and reimplements the
tool-calling loop as a Dapr Workflow, the same way the Strands and Google ADK
extensions reimplement their frameworks' agent loops.

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

## Installation

```bash
pip install "diagrid[langchain]"
```

## Quick Start

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from diagrid.agent.langchain import DaprWorkflowAgentRunner

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Create a Dapr workflow runner around plain langchain_core building blocks
runner = DaprWorkflowAgentRunner(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search_web],
    name="my-agent",
    system_prompt="You are a helpful assistant.",
)

# Start the workflow runtime
runner.start()

try:
    # Run the agent - each chat-model call and tool execution is now a durable activity
    async for event in runner.run_async(
        task="Hello, please help me with...",
        session_id="my-session",
    ):
        print(event)
finally:
    runner.shutdown()
```

## How It Works

The extension wraps a LangChain model + tools agent loop in a Dapr Workflow:

1. **Workflow Start**: When you call `run_async()`, a new Dapr Workflow instance is created
2. **LLM Activity**: Each chat-model call is executed as a durable activity
3. **Tool Activities**: Each tool execution is a separate durable activity, run in parallel when the model requests multiple tool calls in one turn
4. **Checkpointing**: After each activity, the workflow state is checkpointed
5. **Recovery**: On failure, the workflow resumes from the last successful activity

## Architecture

```text
DaprWorkflowAgentRunner (orchestrates the agent loop)
+-- Activity: call_llm_activity()          # Get next action from the chat model
+-- Activity: execute_tool_activity()      # First tool call
+-- Activity: execute_tool_activity()      # Second tool call (parallel with the first)
+-- Activity: call_llm_activity()          # Model processes tool results
+-- ... continues until the model returns a final response with no tool calls
```

## Requirements

- Python >= 3.11
- Dapr >= 1.17.3
- `langchain-core` >= 1.3.2
- A Dapr state store component configured
