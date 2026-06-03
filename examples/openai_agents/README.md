# Diagrid OpenAI Agents SDK Extension Examples

## Prerequisites

1. **Dapr** [running locally](https://docs.dapr.io/getting-started/install-dapr-cli/):
   ```bash
   dapr init
   ```

2. **OpenAI API Key** set in environment:
   ```bash
   export OPENAI_API_KEY=your-api-key
   ```

3. **Install dependencies** (from the repo root):
   ```bash
   uv sync --all-packages --extra openai_agents
   ```

## Running the Examples

### Simple Agent Example

A basic example demonstrating an OpenAI Agents SDK agent with tools running as durable Dapr Workflow activities:

```bash
cd examples
dapr run --app-id openai-agents-demo --resources-path ./components -- python3 simple_agent.py
```

The agent has three tools (`get_weather`, `search_web`, `get_datetime`) and responds to a weather/news query. Each tool call becomes a separate durable activity.

### Crash Recovery Test

Demonstrates Dapr Workflow fault tolerance by simulating a process crash mid-execution. The test creates an agent with three sequential tools, crashes the process during tool 2 on the first run, and verifies that Dapr automatically resumes and completes the workflow on the second run.

```bash
# Clean up any previous test state
rm -f /tmp/openai_agents_crash_test_state.json

# First run (will crash during tool 2):
cd examples
dapr run --app-id openai-agents-crash-test --resources-path ./components -- python3 test_crash_recovery.py

# Second run (Dapr auto-resumes and completes):
dapr run --app-id openai-agents-crash-test --resources-path ./components -- python3 test_crash_recovery.py
```

On the second run you should see `TEST PASSED: Crash recovery worked!` confirming that the workflow resumed from where it left off.

## What It Does

The examples demonstrate how OpenAI Agents SDK tool executions are run as durable Dapr Workflow activities:

1. Each tool call becomes a separate Dapr activity
2. The LLM call itself is also a durable activity
3. Conversation state (messages) is serialized and passed between activities
4. The workflow orchestrates the agent loop (call LLM -> execute tools -> repeat)
5. If the process crashes, execution resumes from the last completed activity

## Architecture

```
User Message
    |
    v
DaprWorkflowAgentRunner
    |
    v
START WORKFLOW: openai_agents_workflow
    |
    +--> Activity: call_llm (OpenAI Chat Completions API)
    |         |
    |         v
    +--> Activity: execute_tool (tool call 1)  \
    +--> Activity: execute_tool (tool call 2)   > parallel via when_all
    |         |
    |         v
    +--> Activity: call_llm (get next action)
    |         |
    |         v
    +--> ... (continues until final response)
    |
    v
Return final response
```

Each activity is checkpointed by Dapr, providing durability guarantees.
