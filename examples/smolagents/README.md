# Diagrid Agent Smolagents Extension Examples

## Prerequisites

1. **Dapr** installed and initialized:
   ```bash
   dapr init
   ```

2. **Redis** running on localhost:6379:
   ```bash
   docker run -d --name redis -p 6379:6379 redis:latest
   ```

3. **Install dependencies** (from the repo root):
   ```bash
   uv sync --all-packages --extra smolagents
   ```

4. **OPENAI_API_KEY** environment variable set:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

## Running the Examples

### Simple Agent

A simple example demonstrating a smolagents `ToolCallingAgent` running as a durable Dapr Workflow:

```bash
cd examples/smolagents
dapr run --app-id smolagents-agent --resources-path ./resources -- python3 simple_agent.py
```

The agent has three tools (`search_web`, `calculate`, `get_weather`) and responds to a weather/calculation query. Each model call and each tool call becomes a separate durable activity.

### Crash Recovery Test

Demonstrates Dapr workflow durability by crashing the process mid-execution and resuming on restart:

```bash
# Clean up any previous test state first:
rm -f /tmp/smolagents_crash_test_state.json

# First run (will crash during tool 2):
dapr run --app-id smolagents-crash-test --resources-path ./resources -- python3 test_crash_recovery.py

# Second run (Dapr auto-resumes and completes):
dapr run --app-id smolagents-crash-test --resources-path ./resources -- python3 test_crash_recovery.py
```

The test creates an agent with 3 sequential tools. On the first run, it crashes during tool 2. On the second run, Dapr automatically resumes the workflow from where it left off and completes all 3 tools.

### Retry Test

Demonstrates Dapr's activity-level retry policy recovering from a transient tool failure without any application-level catch:

```bash
rm -f /tmp/smolagents_retry_test_state.json
dapr run --app-id smolagents-retry-test --resources-path ./resources -- python3 test_retry.py
```

Tool 2 raises a `ConnectionError` on its first 2 invocations and succeeds on the 3rd. The exception propagates out of `execute_tool_activity` uncaught, so Dapr's retry policy (3 attempts, exponential backoff) retries that exact activity automatically — tool 1's already-checkpointed result is never re-executed.

## What It Does

The examples demonstrate how smolagents model calls and tool executions run as durable Dapr Workflow activities:

1. Each tool call becomes a separate Dapr activity
2. The model call itself is also a durable activity
3. Conversation state is kept as plain text turns and passed between activities
4. The workflow orchestrates the agent loop (call model -> execute tools -> repeat) until the model calls the built-in `final_answer` tool
5. If the process crashes, execution resumes from the last completed activity

## Architecture

```
User Task
    |
    v
DaprWorkflowAgentRunner
    |
    v
START WORKFLOW: dapr.smolagents.<name>.workflow
    |
    +--> Activity: call_model_activity (get next action from the model)
    |         |
    |         v
    +--> Activity: execute_tool_activity (tool call 1)
    +--> Activity: execute_tool_activity (tool call 2)
    |         |
    |         v
    +--> Activity: call_model_activity (with tool results)
    |         |
    |         v
    +--> ... (continues until the model calls final_answer)
    |
    v
Yield workflow events to caller
```

Each workflow activity is checkpointed by Dapr, providing durability guarantees.
