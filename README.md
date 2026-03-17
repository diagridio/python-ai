# Durable Workflows for AI agents

**Make your AI agents resilient to failure and outages**

The `diagrid` package is an extension SDK for the open-source [Dapr](https://github.com/dapr/dapr) project to build durable, fault-tolerant AI agents. It integrates seamlessly with popular agent frameworks, wrapping them in Dapr Workflows to ensure agents can recover from failures, persist state across restarts, and scale effectively.

Get started with [Diagrid Catalyst for free](https://diagrid.ws/get-catalyst).

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

## Features

- **Multi-Framework Support:** Native integrations for LangGraph, CrewAI, Google ADK, Strands, PydanticAI and OpenAI Agents.
- **Durability:** Agent state is automatically persisted in the database of your choice. If your process crashes, the agent resumes from the last successful step.
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

# For Pydantic AI
pip install "diagrid[pydantic_ai]"

# For OpenAI Agents
pip install "diagrid[openai_agents]"

# For LangChain Deep Agents
pip install "diagrid[deepagents]"
```

## Prerequisites

- **Python:** 3.11 or higher

## Getting Started with Diagrid Catalyst

Diagrid Catalyst is a fully managed workflow engine for AI agents, built on the open-source CNCF Dapr Workflow project. It's the easiest way to test the different agentic integrations for free.

See [quickstarts](https://docs.diagrid.io/getting-started/quickstarts/ai-agents/) to get started in less than 5 minutes.

## How It Works

This SDK leverages [Dapr Workflows](https://docs.dapr.io/developing-applications/building-blocks/workflow/) to orchestrate agent execution.
1.  **Orchestration:** The agent's control loop is modeled as a workflow.
2.  **Activities:** Each tool execution or LLM call is modeled as a durable activity.
3.  **State Store:** Dapr saves the workflow state to a configured state store (e.g., Redis, CosmosDB) after every step.

Your code can run anywhere (local machine, Kubernetes, EC2, etc.) while the fully managed workflow engine takes care of the agent's execution state, making it crash-proof and resilient to any outage or failure.
