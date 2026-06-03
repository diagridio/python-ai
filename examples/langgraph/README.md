# Diagrid Agent LangGraph Examples

## Prerequisites

1. **Dapr** [running locally](https://docs.dapr.io/getting-started/install-dapr-cli/):
   ```bash
   dapr init
   ```

2. **Install dependencies** (from the repo root):
   ```bash
   uv sync --all-packages --extra langgraph
   ```

## Running the Examples

### Simple Graph Example

A basic example demonstrating a linear graph with two nodes:

```bash
cd examples
dapr run --app-id langgraph-simple --resources-path ./components -- python3 simple_graph.py
```

### Conditional Routing Example

Demonstrates conditional edges based on state:

```bash
cd examples
dapr run --app-id langgraph-conditional --resources-path ./components -- python3 conditional_graph.py
```

### ReAct Agent Example

A more complex example showing a ReAct-style agent with tool execution:

```bash
cd examples
dapr run --app-id langgraph-react --resources-path ./components -- python3 react_agent.py
```

### Crash Recovery Test

Demonstrates Dapr Workflow fault tolerance by simulating a process crash mid-execution. The test creates a 3-node graph, crashes during node 2 on the first run, and verifies that Dapr automatically resumes and completes the workflow on the second run.

```bash
# Clean up any previous test state
rm -f /tmp/langgraph_crash_test_state.json

# First run (will crash during node 2):
cd examples
dapr run --app-id langgraph-crash-test --resources-path ./components -- python3 test_crash_recovery.py

# Second run (Dapr auto-resumes and completes):
dapr run --app-id langgraph-crash-test --resources-path ./components -- python3 test_crash_recovery.py
```

On the second run you should see `TEST PASSED: Crash recovery worked!` confirming that the workflow resumed from where it left off.

## What It Does

The examples demonstrate how LangGraph nodes are executed as durable Dapr Workflow activities:

1. Each node function becomes a separate Dapr activity
2. State (channel values) is serialized and passed between activities
3. The workflow orchestrates the Pregel execution loop
4. If the process crashes, execution resumes from the last completed node

## Architecture

```
User Input
    |
    v
DaprWorkflowGraphRunner
    |
    v
START WORKFLOW: langgraph_workflow
    |
    +--> Activity: execute_node (node_a)
    |         |
    |         v
    +--> Activity: execute_node (node_b)
    |         |
    |         v
    +--> ... (continues until END)
    |
    v
Return final state
```

Each activity is checkpointed by Dapr, providing durability guarantees.
