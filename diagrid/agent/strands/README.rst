Diagrid Agent Strands Extension
================================

This is the Diagrid Agent extension for Strands agents using Dapr Workflows.

This extension enables durable execution of Strands agents using Dapr Workflows.
Each tool execution in an agent runs as a separate Dapr Workflow activity, providing:

- **Fault tolerance**: Agents automatically resume from the last successful activity on failure
- **Durability**: Agent state is persisted and can survive process restarts
- **Observability**: Full visibility into agent execution through Dapr's workflow APIs

Installation
------------

.. code-block:: bash

    pip install diagrid

Prerequisites
-------------

1. Dapr installed and initialized (``dapr init``)
2. A running Dapr sidecar (``dapr run`` or Kubernetes)

Quick Start
-----------

.. code-block:: python

    from strands import Agent, tool
    from diagrid.agent.strands import DurableAgent

    @tool
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(
        model="us.amazon.nova-pro-v1:0",
        tools=[search],
    )

    durable = DurableAgent(agent)
    result = durable("Search for AI papers")

Running with Dapr
-----------------

.. code-block:: bash

    dapr run --app-id strands-agent --dapr-grpc-port 50001 -- python3 your_script.py

How It Works
------------

The extension wraps Strands agent execution in a Dapr Workflow:

1. **DurableAgent**: Each tool call becomes a separate Dapr activity
2. **DaprAgentWorkflow**: Wraps an agent for workflow-based execution
3. **DaprWorkflowToolExecutor**: Custom executor that dispatches tools as activities
4. **DaprStateSessionManager**: Persists agent state to Dapr state stores

Each iteration of the agent loop is checkpointed, so if the process fails,
the workflow resumes from the last successful activity.

Advanced Usage
--------------

Using the decorator pattern:

.. code-block:: python

    from diagrid.agent.strands import dapr_agent_workflow

    @dapr_agent_workflow(workflow_name="my_agent")
    def create_agent() -> Agent:
        return Agent(model="us.amazon.nova-pro-v1:0", tools=[my_tool])

    workflow = create_agent()
    workflow.register(workflow_runtime)

State persistence:

.. code-block:: python

    from diagrid.agent.strands import DaprStateSessionManager

    session_manager = DaprStateSessionManager(
        store_name="agent-workflow",
        session_id="user-123",
    )

    agent = Agent(
        model="us.amazon.nova-pro-v1:0",
        hooks=[session_manager],
    )

License
-------

Apache License 2.0
