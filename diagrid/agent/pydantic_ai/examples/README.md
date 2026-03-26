# Diagrid Pydantic AI Extension Examples

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
   uv sync --all-packages --extra pydantic_ai
   ```

4. **OPENAI_API_KEY** environment variable set:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

## Running the Examples

### Simple Agent

A basic example demonstrating a Pydantic AI agent with tools running as durable Dapr Workflow activities:

```bash
cd examples
dapr run --app-id pydantic-ai-agent --resources-path ./components -- python3 simple_agent.py
```

The agent has three tools (`get_weather`, `search_web`, `get_datetime`) and responds to a weather/news query. Each tool call becomes a separate durable activity.

### Crash Recovery Test

Demonstrates Dapr Workflow fault tolerance by simulating a process crash mid-execution. The test creates an agent with three sequential tools, crashes during tool 2 on the first run, and verifies that Dapr automatically resumes and completes the workflow on the second run.

```bash
# Clean up any previous test state
rm -f /tmp/pydantic_ai_crash_test_state.json

# First run (will crash during tool 2):
cd examples
dapr run --app-id pydantic-ai-crash-test --resources-path ./components -- python3 test_crash_recovery.py

# Second run (Dapr auto-resumes and completes):
dapr run --app-id pydantic-ai-crash-test --resources-path ./components -- python3 test_crash_recovery.py
```

On the second run you should see `TEST PASSED: Crash recovery worked!` confirming the workflow resumed from where it left off.

### Retry Test

Demonstrates Dapr's automatic activity retry policy. Tool 2 raises a `ConnectionError` on the first two attempts and succeeds on the third. Dapr retries the activity automatically without re-executing tool 1.

```bash
rm -f /tmp/pydantic_ai_retry_test_state.json
cd examples
dapr run --app-id pydantic-ai-retry-test --resources-path ./components -- python3 test_retry.py
```

You should see `TEST PASSED: Retry worked!` with tool 1 executed once, tool 2 executed three times, and tool 3 executed once.

## What It Does

The examples demonstrate how Pydantic AI tool executions are run as durable Dapr Workflow activities:

1. Each tool call becomes a separate Dapr activity
2. The LLM call itself is also a durable activity
3. Conversation state (messages) is serialized and passed between activities
4. The workflow orchestrates the agent loop (call LLM → execute tools → repeat)
5. If the process crashes, execution resumes from the last completed activity

## Architecture

```
User Message
    |
    v
DaprWorkflowAgentRunner
    |
    v
START WORKFLOW: pydantic_ai_agent_workflow
    |
    +--> Activity: call_llm_activity (get next action from LLM)
    |         |
    |         v
    +--> Activity: execute_tool_activity (tool call 1)
    +--> Activity: execute_tool_activity (tool call 2)
    |         |
    |         v
    +--> Activity: call_llm_activity (get next action)
    |         |
    |         v
    +--> ... (continues until final response)
    |
    v
Return final_response
```

Each activity is checkpointed by Dapr, providing durability guarantees.
