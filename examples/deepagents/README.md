# Diagrid Deep Agents Extension Examples

## Prerequisites

1. **Dapr** [running locally](https://docs.dapr.io/getting-started/install-dapr-cli/):
   ```bash
   dapr init
   ```

2. **Install dependencies** (from the repo root):
   ```bash
   uv sync --all-packages --extra all
   ```

3. **OPENAI_API_KEY** environment variable set:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

## Running the Examples

### Simple Agent

A Deep Agent with tools running as a durable Dapr Workflow:

```bash
cd examples
dapr run --app-id deep-agent --resources-path ./components -- python3 simple_agent.py
```

The agent uses `create_deep_agent()` with `get_weather` and `search_web` tools. Since Deep Agents compile to standard LangGraph graphs, each node in the graph becomes a durable Dapr workflow activity.

### Crash Recovery Test

Demonstrates Dapr workflow durability by crashing the process mid-execution and resuming on restart:

```bash
# Clean up any previous test state first:
rm -f /tmp/deepagents_crash_test_state.json

# First run (will crash during tool 2):
dapr run --app-id deep-agent-crash-test --resources-path ./components -- python3 test_crash_recovery.py

# Second run (Dapr auto-resumes and completes):
dapr run --app-id deep-agent-crash-test --resources-path ./components -- python3 test_crash_recovery.py
```

### Incident Response with Sandbox

A full incident response agent that combines **durable execution**, **sandbox access**, and a **5-step runbook**. The sandbox is pre-seeded with simulated production infrastructure (application logs, database logs, service configs, and mitigation scripts). The agent reads real files and executes real shell commands inside the sandbox as part of the incident response workflow.

The demo runs **twice** to show crash recovery:

```bash
# Clean up any previous state:
rm -f /tmp/incident_response_demo_state.json

# First run — completes steps 1-2, crashes during step 3:
dapr run --app-id incident-agent --resources-path ./components -- python3 demo_incident_response.py

# Second run — Dapr resumes from step 3, completes the remaining steps:
dapr run --app-id incident-agent --resources-path ./components -- python3 demo_incident_response.py
```

**What happens:**

| Run | Steps | Behavior |
|-----|-------|----------|
| 1 | 1. Diagnose (reads sandbox logs) → 2. Page on-call → 3. Crash | Process exits mid-mitigation. Steps 1-2 are checkpointed. |
| 2 | 3. Mitigate (runs sandbox script) → 4. Notify → 5. Postmortem (writes to sandbox) | Dapr resumes at step 3. On-call team is NOT paged again. |

**Sandbox filesystem layout:**

```
/var/log/payments-api/app.log      — application error logs (connection pool exhaustion)
/var/log/postgres/pg-prod-01.log   — database connection logs
/etc/payments-api/config.yaml      — service configuration (pool_size, replicas)
/opt/runbooks/payments-api/restart_pool.sh — mitigation script (executed in step 3)
/opt/templates/postmortem.md       — postmortem template (used in step 5)
/opt/postmortems/INC-2024-0892.md  — generated postmortem (written in step 5)
```

The agent also receives Deep Agents sandbox tools (`ls`, `read_file`, `write_file`, `execute`, `grep`, `glob`) via the `backend` parameter, which it can use to inspect the environment between runbook steps.

### Retry Test

Demonstrates Dapr's automatic activity retry policy. Tool 2 raises a `ConnectionError` on the first two attempts and succeeds on the third:

```bash
rm -f /tmp/deepagents_retry_test_state.json
dapr run --app-id deep-agent-retry-test --resources-path ./components -- python3 test_retry.py
```

## How It Works

Deep Agents (`create_deep_agent()`) compile to standard LangGraph `CompiledStateGraph` objects. The `DaprWorkflowDeepAgentRunner` is a thin wrapper around the existing LangGraph Dapr integration:

```
create_deep_agent()
    |
    v
CompiledStateGraph (standard LangGraph)
    |
    v
DaprWorkflowDeepAgentRunner (extends DaprWorkflowGraphRunner)
    |
    v
Dapr Workflow with durable activities
```

### Architecture

```
User Message
    |
    v
DaprWorkflowDeepAgentRunner
    |
    v
START WORKFLOW: dapr.langgraph.<name>.workflow
    |
    +--> Activity: execute_node (Deep Agent planning node)
    |         |
    |         v
    +--> Activity: execute_node (tool execution node)
    |         |
    |         v
    +--> Activity: evaluate_condition (should continue?)
    |         |
    |         v
    +--> Activity: execute_node (next node...)
    |         |
    |         v
    +--> ... (continues until END)
    |
    v
Return final output
```

Each activity is checkpointed by Dapr, providing durability guarantees. If the process crashes, Dapr resumes from the last completed activity.

### Sub-Agent Workflows (AsyncSubAgent)

Demonstrates durable multi-agent orchestration where each sub-agent runs as its own Dapr workflow, and the supervisor is also durable. Uses the deepagents `AsyncSubAgent` middleware for inter-agent communication over the Agent Protocol (HTTP).

**Architecture:**

```
Supervisor (DaprWorkflowDeepAgentRunner)
  |
  +-- start_async_task("researcher") --> HTTP --> Researcher (DaprWorkflowDeepAgentRunner)
  |     check_async_task(task_id)    <-- HTTP <--              separate Dapr Workflow
  |
  +-- start_async_task("analyst")    --> HTTP --> Analyst (DaprWorkflowDeepAgentRunner)
        check_async_task(task_id)    <-- HTTP <--              separate Dapr Workflow
```

Each agent runs in its own process with its own Dapr sidecar. Start them in three separate terminals:

```bash
# Terminal 1 -- Researcher sub-agent (port 8001)
dapr run --app-id researcher --app-port 8001 --resources-path ./components -- \
    python3 subagent_workflows.py researcher

# Terminal 2 -- Analyst sub-agent (port 8002)
dapr run --app-id analyst --app-port 8002 --resources-path ./components -- \
    python3 subagent_workflows.py analyst

# Terminal 3 -- Supervisor (delegates to sub-agents via HTTP)
dapr run --app-id supervisor --resources-path ./components -- \
    python3 subagent_workflows.py supervisor
```

The supervisor uses `create_deep_agent()` with `AsyncSubAgent` specs. Each sub-agent is wrapped in `DaprWorkflowDeepAgentRunner` and exposes Agent Protocol endpoints via a thin FastAPI adapter. Every LLM call and tool invocation across all three agents is a durable Dapr workflow activity.
