# Diagrid

**Durable AI Agents with Diagrid Catalyst**

The `diagrid` package is the primary SDK for building durable, fault-tolerant AI agents using Diagrid Catalyst. It integrates seamlessly with popular agent frameworks, wrapping them in Dapr Workflows to ensure your agents can recover from failures, persist state across restarts, and scale effectively.

## Features

- **Multi-Framework Support:** Native integrations for LangGraph, CrewAI, Google ADK, Strands, and OpenAI Agents.
- **Durability:** Agent state is automatically persisted. If your process crashes, the agent resumes from the last successful step.
- **Fault Tolerance:** Built-in retries and error handling powered by Dapr.
- **Observability:** Deep insights into agent execution, tool calls, and state transitions.

## Installation

Install the base package along with the extension for your chosen framework:

```bash
# For LangGraph
pip install "diagrid[langgraph]"

# For CrewAI
pip install "diagrid[crewai]"

# For Google ADK
pip install "diagrid[adk]"

# For Strands
pip install "diagrid[strands]"

# For OpenAI Agents
pip install "diagrid[openai_agents]"
```

## Prerequisites

- **Python:** 3.11 or higher
- **Dapr:** [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/) installed and initialized (`dapr init`).

## Quick Start

### LangGraph

Wrap your LangGraph `CompiledGraph` with `DaprWorkflowGraphRunner` to make it durable.

```python
from langgraph.graph import StateGraph, START, END
from diagrid.agent.langgraph import DaprWorkflowGraphRunner

# ... Define your graph nodes and state ...
graph = StateGraph(State)
# ... Add nodes and edges ...
compiled_graph = graph.compile()

# Run durably
runner = DaprWorkflowGraphRunner(graph=compiled_graph)
runner.start()

result = runner.invoke(
    input={"messages": ["Hello"]},
    thread_id="thread-1"
)
print(result)

runner.shutdown()
```

### CrewAI

Wrap your CrewAI `Agent` with `DaprWorkflowAgentRunner`.

```python
from crewai import Agent, Task
from diagrid.agent.crewai import DaprWorkflowAgentRunner

agent = Agent(role="Researcher", goal="Find data", ...)
task = Task(description="Research AI", agent=agent)

# Run durably
runner = DaprWorkflowAgentRunner(agent=agent)
runner.start()

# Execute asynchronously
import asyncio
asyncio.run(runner.run_async(task=task, session_id="session-1"))

runner.shutdown()
```

### Google ADK

Use `DaprWorkflowAgentRunner` to execute Google ADK agents as workflows.

```python
from google.adk.agents import LlmAgent
from diagrid.agent.adk import DaprWorkflowAgentRunner

agent = LlmAgent(name="my_agent", model="gemini-2.0-flash", ...)

runner = DaprWorkflowAgentRunner(agent=agent, state_store_name="agent-store")

# Run the agent loop
async for event in runner.run_async(user_message="Hello", session_id="session-1"):
    print(event)
```

### Strands

Use the `DurableAgent` wrapper for Strands.

```python
from strands import Agent
from diagrid.agent.strands import DurableAgent

agent = Agent(model="us.amazon.nova-pro-v1:0", ...)
durable_agent = DurableAgent(agent)

result = durable_agent("What is the weather?")
print(result)
```

## How It Works

This SDK leverages [Dapr Workflows](https://docs.dapr.io/developing-applications/building-blocks/workflow/) to orchestrate agent execution.
1.  **Orchestration:** The agent's control loop is modeled as a workflow.
2.  **Activities:** Each tool execution or LLM call is modeled as a durable activity.
3.  **State Store:** Dapr saves the workflow state to a configured state store (e.g., Redis, CosmosDB) after every step.
