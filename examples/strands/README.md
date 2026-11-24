# Diagrid Agent Strands Extension Examples

## Prerequisites

Check the Dapr runtime version (1.17 or later) and activate the repo's virtual environment first. See [Shared prerequisites](../README.md#shared-prerequisites).

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
   uv sync --all-packages --extra strands
   ```

4. **OPENAI_API_KEY** environment variable set:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

## Running the Examples

### Simple Agent

A simple example demonstrating a Strands agent with tools running as a durable Dapr Workflow:

```bash
cd examples/strands
dapr run --app-id strands-agent --resources-path ./resources -- python3 simple_agent.py
```

The agent has three tools (`search_web`, `calculate`, `get_weather`) and responds to a weather/calculation query. Each LLM call and each tool call becomes a separate durable workflow activity.

### Crash Recovery Test

Demonstrates Dapr workflow durability by crashing the process mid-execution and resuming on restart:

```bash
# Clean up any previous test state first:
rm -f /tmp/strands_crash_test_state.json

# First run (will crash during tool 2):
dapr run --app-id strands-crash-test --resources-path ./resources -- python3 test_crash_recovery.py

# Second run (Dapr auto-resumes and completes):
dapr run --app-id strands-crash-test --resources-path ./resources -- python3 test_crash_recovery.py
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
START WORKFLOW: dapr.strands.<name>.workflow
    |
    +--> Activity: strands_call_model (get next action from the LLM)
    |         |
    |         v
    +--> Activity: strands_execute_tool (one per requested tool)
    |         |
    |         v
    +--> ... (loops until the model returns a final response)
    |
    v
Yield workflow events to caller
```

Each workflow activity is checkpointed by Dapr, providing durability guarantees.

## Conversation history

`DaprWorkflowAgentRunner` treats `session_id` as one conversation. A second
call with the same `session_id` sees the earlier turns, like repeated calls on
one plain Strands `Agent`. Use a new `session_id` to start fresh.

```python
runner.run_sync(task="My name is Ada.", session_id="chat-1")
runner.run_sync(task="What is my name?", session_id="chat-1")  # sees the first turn
```

- History is stored under `strands-<runner name>-<session_id>-messages`, so two
  runners with different names never share it.
- It is saved after each successful run and trimmed to the agent's
  `SlidingWindowConversationManager.window_size` (default 40).
- For durable memory, add a Dapr state store component named `agent-memory`
  (or pass `state_store=` to the runner). History then survives restarts and is
  shared between replicas:

  ```yaml
  apiVersion: dapr.io/v1alpha1
  kind: Component
  metadata:
    name: agent-memory
  spec:
    type: state.redis
    version: v1
    metadata:
      - name: redisHost
        value: "localhost:6379"
      - name: redisPassword
        value: ""
  ```

- Without a state store, history stays in the runner process and a warning is
  logged: it does not survive a restart and is not shared between replicas.
- The agent's own `messages` and any Strands session manager are not used by
  the runner.
- Callers that reuse one fixed `session_id` for unrelated tasks now carry
  history between those tasks.

`DurableAgent` is a single conversation, like `Agent`: each call starts from
`agent.messages` and writes the result back to it.
