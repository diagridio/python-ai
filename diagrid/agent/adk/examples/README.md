# Diagrid ADK Extension Examples

## Prerequisites

1. **Redis** running on localhost:6379:
   ```bash
   docker run -d --name redis -p 6379:6379 redis:latest
   ```

2. **Google API Key** set in environment:
   ```bash
   export GOOGLE_API_KEY=your-api-key
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Examples

Activate your virtual environment before running (so `dapr run` picks up the right Python):

```bash
source .venv/bin/activate
```

### Simple Agent Example

A basic example demonstrating an ADK agent with tools running as durable Dapr Workflow activities:

```bash
cd examples
dapr run --app-id adk-agent --resources-path ./components -- python3 simple_agent.py
```

The agent has three tools (`get_weather`, `get_time`, `calculate`) and responds to a weather/time query about Tokyo. Each tool call becomes a separate durable activity.

### Crash Recovery Test

Demonstrates Dapr Workflow fault tolerance by simulating a process crash mid-execution. The test creates an agent with three sequential tools, crashes the process during tool 2 on the first run, and verifies that Dapr automatically resumes and completes the workflow on the second run.

```bash
# Clean up any previous test state
rm -f /tmp/adk_crash_test_state.json

# First run (will crash during tool 2):
cd examples
dapr run --app-id adk-crash-test --resources-path ./components -- python3 test_crash_recovery.py

# Second run (Dapr auto-resumes and completes):
dapr run --app-id adk-crash-test --resources-path ./components -- python3 test_crash_recovery.py
```

On the second run you should see `TEST PASSED: Crash recovery worked!` confirming that the workflow resumed from where it left off.

## What It Does

The examples demonstrate how ADK tool executions are run as durable Dapr Workflow activities:

1. Each tool call becomes a separate Dapr activity
2. The LLM call itself is also a durable activity
3. Conversation state (messages) is serialized and passed between activities
4. The workflow orchestrates the agent loop (call LLM -> execute tools -> repeat)
5. If the process crashes, execution resumes from the last completed activity

## Architecture

```
User Input
    |
    v
DaprWorkflowAgentRunner
    |
    v
START WORKFLOW: adk_agent_workflow
    |
    +--> Activity: call_llm (get next action from Gemini)
    |         |
    |         v
    +--> Activity: execute_tool (tool call 1)
    +--> Activity: execute_tool (tool call 2)
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
