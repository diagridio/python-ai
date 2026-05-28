# Claude Managed Agent SDK Extension Examples

## Prerequisites

1. **Dapr** installed and initialized:
   ```bash
   dapr init
   ```

2. **Redis** running on localhost:6379:
   ```bash
   docker run -d --name redis -p 6379:6379 redis:latest
   ```

3. **Install dependencies**:
   ```bash
   pip install diagrid claude-agent-sdk anthropic
   ```

4. **ANTHROPIC_API_KEY** environment variable set:
   ```bash
   export ANTHROPIC_API_KEY=sk-...
   ```

## Running the Examples

### Simple Agent

A simple example demonstrating a Claude agent with tools running as a durable Dapr Workflow:

```bash
cd examples
dapr run --app-id claude-agents-demo --resources-path ./components -- python3 simple_agent.py
```

The agent has three tools (`get_weather`, `search_web`, `get_datetime`). The
entire agent invocation runs as a single workflow; each LLM turn and each tool
call is its own durable activity.

### Crash Recovery Test

Demonstrates Dapr workflow durability by crashing the process mid-execution
and resuming on restart:

```bash
# Clean up any previous test state first:
rm -f /tmp/claude_agents_crash_test_state.json

# First run (will crash during tool 2):
dapr run --app-id claude-agents-crash-test --resources-path ./components -- python3 test_crash_recovery.py

# Second run (Dapr auto-resumes and completes):
dapr run --app-id claude-agents-crash-test --resources-path ./components -- python3 test_crash_recovery.py
```

The test creates an agent with three sequential tools. On the first run, it
crashes during tool 2. On the second run, Dapr automatically resumes the
workflow from the last completed activity and finishes all three tools.

## Architecture

```
User Prompt
    |
    v
DaprWorkflowAgentRunner
    |
    v
START WORKFLOW: dapr.claudeagents.<name>.workflow
    |
    +--> Activity: claude_call_llm  (turn 1, Anthropic messages.create)
    |         |
    |         v
    |    Assistant returns text + tool_use blocks
    |
    +--> Activity: claude_execute_tool (one per tool_use, fanned out in parallel)
    |
    +--> Activity: claude_call_llm  (turn 2, with tool results)
    |         |
    |         v
    |    Assistant returns final text
    |
    v
Yield workflow_completed event to caller
```

Each LLM call and each tool call is a Dapr workflow activity, checkpointed
independently — a crash mid-tool resumes from the last completed activity.
