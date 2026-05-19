# Diagrid HolmesGPT Integration

Durable execution of [HolmesGPT](https://github.com/HolmesGPT/holmesgpt) investigations on Dapr Workflows. Each LLM iteration and each tool call becomes a Dapr Workflow activity, providing:

- **Fault tolerance** — investigations resume from the last completed activity after sidecar or process restarts.
- **Durable approvals** — `requires_approval` tool pauses become `wait_for_external_event`, surviving days if needed.
- **Polling SSE bridge** — event tape stored in a Dapr state store; clients reconnect with `Last-Event-Id` and resume.
- **No upstream changes** — HolmesGPT is consumed as a library; the integration is non-invasive.

## Prerequisites

```bash
pip install diagrid[holmesgpt]                # pulls holmesgpt + agent-core
brew install dapr/tap/dapr-cli && dapr init   # one-time
```

> **Note**: this extra is intentionally **not** part of `diagrid[all]`. HolmesGPT
> ships strict pins on fastapi, uvicorn, cachetools, mcp, httpx, and other
> transitive deps that conflict with the looser constraints used by the other
> agent integrations. Install it in a dedicated environment.

`dapr init` is enough — the integration writes its event tape to the **same** Dapr state store that the workflow runtime already uses for actor state (the default `statestore` component created by `dapr init`). No second component is required.

The store is whatever Dapr is configured for — Redis is the default, but any Dapr-supported actor state store works (PostgreSQL, MongoDB, MySQL, SQLite, Cosmos DB, etcd, …). The integration only uses `save_state` / `get_bulk_state` and per-key `ttlInSeconds` metadata, all of which are part of Dapr's portable state API.

## Quick start

```python
from diagrid.agent.holmesgpt import DaprWorkflowHolmesRunner

runner = DaprWorkflowHolmesRunner(
    name="sre-agent",
    config_path="~/.holmes/config.yaml",  # optional
    model="anthropic/claude-sonnet-4-5-20250929",  # optional
    max_steps=10,
)
runner.start()

# Same call shape as `holmes ask "<question>"`, but every LLM iteration
# and tool call runs as a durable Dapr workflow activity. The HolmesGPT
# system prompt, toolset instructions, skills, and global instructions
# are all rendered exactly the way the `holmes` CLI / HTTP server render
# them — `ask()` delegates to `build_chat_messages`.
result = runner.ask("Why is checkout-api crashlooping?")
print(result["final"]["content"])

runner.shutdown()
```

For lower-level control (pre-rendered messages, no system prompt injection), use `runner.invoke(messages=...)` instead — same semantics, just skips HolmesGPT's prompt builder.

Run the example with a sidecar:

```bash
dapr run --app-id holmes-cli -- python diagrid/agent/holmesgpt/examples/basic.py
```

## HTTP server with SSE

```python
from diagrid.agent.holmesgpt import DaprWorkflowHolmesRunner

DaprWorkflowHolmesRunner(name="sre-agent").serve(port=5001)
```

Endpoints:

| Method | Path                                         | Purpose                                  |
|--------|----------------------------------------------|------------------------------------------|
| POST   | `/investigations`                            | Schedule a new investigation             |
| GET    | `/investigations/{id}`                       | Workflow status                          |
| GET    | `/investigations/{id}/stream`                | SSE event tape (supports `Last-Event-Id`)|
| POST   | `/investigations/{id}/approve`               | Approve / reject a paused tool           |
| POST   | `/investigations/{id}/frontend_result`       | Submit a frontend-executed tool result   |

Event types streamed:

- `iteration_started`, `iteration_completed`
- `start_tool_calling`, `tool_calling_result`
- `approval_required`, `ai_answer_end`
- Workflow-level: `workflow_started`, `workflow_completed`, `workflow_failed`, `workflow_terminated`

## Architecture

```
┌────────────────────────┐
│ Your service           │
│ ┌────────────────────┐ │       ┌─────────────────────┐
│ │ FastAPI            │ │       │ Dapr Workflow       │
│ │  /investigations   │─│──────▶│  investigation_     │
│ │  /…/stream  ←──────│─│──┐    │  workflow           │
│ │  /…/approve  ──────│─│──┼──▶ │   ├─ call_llm_act.  │
│ └────────────────────┘ │  │    │   └─ invoke_tool_a. │
│                        │  │    └──────┬──────────────┘
│ Dapr state store ◀─────│──┘           │ writes
│  "holmes-events"  ─────│──────────────┘ events
└────────────────────────┘
```

- **Workflow** allocates monotonic event seqs and orchestrates the loop.
- **Activities** call HolmesGPT primitives (`LLM.completion`, `Tool.invoke`) and write events to the per-instance tape.
- **SSE handler** polls the tape (default 300 ms) and forwards events to the browser.

## Configuration

| Environment variable      | Default          | Purpose                                                     |
|---------------------------|------------------|-------------------------------------------------------------|
| `HOLMES_EVENTS_STORE`     | `statestore`     | Dapr state store component name (defaults to the actor state store) |
| `HOLMES_EVENTS_PREFIX`    | `holmes.stream`  | State key prefix for tape entries                           |
| `HOLMES_EVENTS_TTL`       | `86400`          | Event TTL in seconds (0 disables)                           |

All three are also overridable via `DaprWorkflowHolmesRunner(events_store_name=…, events_key_prefix=…, events_ttl_seconds=…)`. The default collocates the tape on the same component Dapr uses for workflow durability, so both share a single failure domain — if the workflow can persist, the tape can persist. Point them at separate components only if you have a reason to (e.g. isolate UI traffic on a different backend).

## Approval flow

When a HolmesGPT tool returns `APPROVAL_REQUIRED` or `FRONTEND_PAUSE`:

1. The workflow emits an `approval_required` event onto the tape.
2. The workflow blocks on `wait_for_external_event(f"resume:{tool_call_id}")`.
3. The client POSTs to `/investigations/{id}/approve` (or `/frontend_result`).
4. The runner calls `raise_workflow_event` to deliver the decision.
5. The workflow re-invokes the tool with `user_approved=True` and continues.

This is durable: a process restart while the workflow is paused does not lose the pause state — the next `raise_workflow_event` call resumes it.

## Notes & limitations

- The agent loop is mirrored from HolmesGPT (`call_stream` in `holmes/core/tool_calling_llm.py`). Pin a HolmesGPT version and re-validate when bumping.
- Token-level streaming inside an LLM iteration is not preserved — only iteration-level events. This matches HolmesGPT's existing UX granularity.
- HolmesGPT compaction events are currently elided. Add a `record_event` call in `call_llm_activity` if you need them surfaced.
