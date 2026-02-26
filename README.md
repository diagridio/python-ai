# Diagrid

**Durable AI Agents with Diagrid Catalyst**

The `diagrid` package is the primary SDK for building durable, fault-tolerant AI agents using [Diagrid Catalyst](https://www.diagrid.io/catalyst). It integrates seamlessly with popular agent frameworks, wrapping them in Dapr Workflows to ensure your agents can recover from failures, persist state across restarts, and scale effectively.

Get started with [Catalyst for free](https://diagrid.ws/get-catalyst).

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

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

## CLI

The `diagrid` package includes the `diagridpy` CLI — a tool for setting up your local development environment and deploying agents to Kubernetes with a single command.

### `diagridpy init`

Bootstraps a complete local development environment in one step:

1. **Authenticates** with Diagrid Catalyst (browser-based device code flow or API key)
2. **Creates a Catalyst project** to manage your agent's AppID and connection details
3. **Clones a quickstart template** for your chosen framework into a new directory
4. **Provisions a local Kubernetes cluster** using [kind](https://kind.sigs.k8s.io/)
5. **Installs the `catalyst-agents` Helm chart** — Dapr, observability stack, Redis, and LLM backend
6. **Creates a Catalyst AppID** with a provisioned API token

```bash
# Initialize with the default framework (dapr-agents)
diagridpy init my-project

# Initialize with a specific framework
diagridpy init my-project --framework langgraph

# Use an API key instead of browser auth
diagridpy init my-project --framework crewai --api-key <YOUR_KEY>
```

Supported frameworks: `dapr-agents`, `langgraph`, `crewai`, `adk`, `strands`, `openai-agents`

### `diagridpy deploy`

Builds your agent image, loads it into the local cluster, and deploys it with the correct Catalyst connection details automatically injected as environment variables.

```bash
# Build and deploy from the current directory (requires a Dockerfile)
diagridpy deploy

# Deploy and immediately trigger the agent with a prompt
diagridpy deploy --trigger "Plan a trip to Paris"

# Override image name, tag, or target project
diagridpy deploy --image my-agent --tag v1 --project my-project
```

Run `diagridpy --help` or `diagridpy <command> --help` to see all available options.

## Quick Start

### LangGraph

Wrap your LangGraph `StateGraph` with `DaprWorkflowGraphRunner` to make it durable.

```python
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from diagrid.agent.langgraph import DaprWorkflowGraphRunner


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}, 72F"


tools = [get_weather]
tools_by_name = {t.name: t for t in tools}
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def call_model(state: MessagesState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def call_tools(state: MessagesState) -> dict:
    last_message = state["messages"][-1]
    results = []
    for tc in last_message.tool_calls:
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
    return {"messages": results}


def should_use_tools(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tools)
graph.add_edge("tools", "agent")

runner = DaprWorkflowGraphRunner(graph=graph.compile())
runner.serve(
    port=int(os.environ.get("APP_PORT", "5001")),
    input_mapper=lambda req: {"messages": [HumanMessage(content=req["task"])]},
)
```

### CrewAI

Wrap your CrewAI `Agent` with `DaprWorkflowAgentRunner`.

```python
import os

from crewai import Agent
from crewai.tools import tool
from diagrid.agent.crewai import DaprWorkflowAgentRunner


@tool("Get weather")
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}, 72F"


agent = Agent(
    role="Assistant",
    goal="Help users",
    backstory="Expert assistant",
    tools=[get_weather],
    llm="openai/gpt-4o-mini",
)
runner = DaprWorkflowAgentRunner(agent=agent)
runner.serve(port=int(os.environ.get("APP_PORT", "5001")))
```

### Google ADK

Use `DaprWorkflowAgentRunner` to execute Google ADK agents as workflows.

```python
import os

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from diagrid.agent.adk import DaprWorkflowAgentRunner


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}, 72F"


agent = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    tools=[FunctionTool(get_weather)],
)
runner = DaprWorkflowAgentRunner(agent=agent)
runner.serve(port=int(os.environ.get("APP_PORT", "5001")))
```

### Strands

Use the `DaprWorkflowAgentRunner` wrapper for Strands.

```python
import os

from strands import Agent, tool
from strands.models.openai import OpenAIModel
from diagrid.agent.strands import DaprWorkflowAgentRunner


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 72F"


agent = Agent(
    model=OpenAIModel(model_id="gpt-4o-mini"),
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)
runner = DaprWorkflowAgentRunner(agent=agent)
runner.serve(port=int(os.environ.get("APP_PORT", "5001")))
```

### OpenAI Agents

Use the `DaprWorkflowAgentRunner` wrapper for OpenAI Agents.

```python
import os

from agents import Agent, function_tool
from diagrid.agent.openai_agents import DaprWorkflowAgentRunner


@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}, 72F"


agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4o-mini",
    tools=[get_weather],
)
runner = DaprWorkflowAgentRunner(agent=agent)
runner.serve(port=int(os.environ.get("APP_PORT", "5001")))
```

## How It Works

This SDK leverages [Dapr Workflows](https://docs.dapr.io/developing-applications/building-blocks/workflow/) to orchestrate agent execution.
1.  **Orchestration:** The agent's control loop is modeled as a workflow.
2.  **Activities:** Each tool execution or LLM call is modeled as a durable activity.
3.  **State Store:** Dapr saves the workflow state to a configured state store (e.g., Redis, CosmosDB) after every step.
