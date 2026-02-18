# Diagrid Agent Strands Extension Examples

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
   pip install diagrid
   ```

4. **OPENAI_API_KEY** environment variable set:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

## Running the Examples

### Simple Agent

A simple example demonstrating a Strands agent with tools running as a durable Dapr Workflow:

```bash
cd examples
dapr run --app-id strands-agent --resources-path ./components -- python3 simple_agent.py
```

The agent has three tools (`search_web`, `calculate`, `get_weather`) and responds to a weather/calculation query. The entire agent invocation becomes a durable workflow activity.

### Crash Recovery Test

Demonstrates Dapr workflow durability by crashing the process mid-execution and resuming on restart:

```bash
# Clean up any previous test state first:
rm -f /tmp/strands_crash_test_state.json

# First run (will crash during tool 2):
dapr run --app-id strands-crash-test --resources-path ./components -- python3 test_crash_recovery.py

# Second run (Dapr auto-resumes and completes):
dapr run --app-id strands-crash-test --resources-path ./components -- python3 test_crash_recovery.py
```

The test creates an agent with 3 sequential tools. On the first run, it crashes during tool 2. On the second run, Dapr automatically resumes the workflow from where it left off and completes all 3 tools.

## Architecture

```
User Prompt
    |
    v
DaprWorkflowAgentRunner
    |
    v
START WORKFLOW: strands_agent_workflow
    |
    +--> Activity: run_agent (invoke Strands agent)
    |         |
    |         v
    |    Agent calls tools, gets LLM responses
    |         |
    |         v
    +--> Return result
    |
    v
Yield workflow events to caller
```

Each workflow activity is checkpointed by Dapr, providing durability guarantees.
