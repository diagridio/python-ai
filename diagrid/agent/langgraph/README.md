# Diagrid Agent LangGraph

Durable execution of LangGraph graphs using Dapr Workflows.

This package enables durable execution of LangGraph graphs using Dapr Workflows.
Each node execution in a graph runs as a separate Dapr Workflow activity, providing:

- **Fault tolerance**: Graphs automatically resume from the last successful node on failure
- **Durability**: Graph state is persisted and can survive process restarts
- **Observability**: Full visibility into graph execution through Dapr's workflow APIs

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

## Installation

```bash
pip install "diagrid[langgraph]"
```

## Quick Start

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from diagrid.agent.langgraph import DaprWorkflowGraphRunner

# Define your state
class State(TypedDict):
    messages: list[str]
    count: int

# Define node functions
def process_node(state: State) -> dict:
    return {
        "messages": state["messages"] + ["processed"],
        "count": state["count"] + 1
    }

def validate_node(state: State) -> dict:
    return {"messages": state["messages"] + ["validated"]}

# Build the graph
graph = StateGraph(State)
graph.add_node("process", process_node)
graph.add_node("validate", validate_node)
graph.add_edge(START, "process")
graph.add_edge("process", "validate")
graph.add_edge("validate", END)
compiled = graph.compile()

# Run with Dapr Workflows
runner = DaprWorkflowGraphRunner(graph=compiled, name="my-graph")
runner.start()

try:
    result = runner.invoke(
        input={"messages": ["start"], "count": 0},
        thread_id="my-thread-123",
    )
    print(result)
    # Output: {"messages": ["start", "processed", "validated"], "count": 1}
finally:
    runner.shutdown()
```

## Async Usage

```python
import asyncio

async def main():
    runner = DaprWorkflowGraphRunner(graph=compiled, name="my-graph")
    runner.start()

    try:
        async for event in runner.run_async(
            input={"messages": ["start"], "count": 0},
            thread_id="my-thread-123",
        ):
            print(f"Event: {event['type']}")
            if event["type"] == "workflow_completed":
                print(f"Output: {event['output']}")
    finally:
        runner.shutdown()

asyncio.run(main())
```

## Streaming

```python
from diagrid.agent.langgraph import (
    DaprWorkflowGraphRunner,
    GraphRunError,  # raised when a streamed run fails
    StreamControl,  # what `on_control` receives
    StreamSinkRegistry,  # the type of the `stream_registry` argument
)
```

`astream` yields what `CompiledStateGraph.astream` yields, while each node runs
as a Dapr activity: LLM tokens (`messages`), node updates (`updates`) and
whatever a node writes with `get_stream_writer()` (`custom`).

```python
async for mode, chunk in runner.astream(
    {"messages": [{"role": "user", "content": "hi"}]},
    stream_mode=["messages", "updates", "custom"],
):
    if mode == "messages":
        message_chunk, metadata = chunk
        print(message_chunk.content, end="")
```

A single mode yields bare chunks, as native `astream` does, and a list of modes
yields `(mode, chunk)`. Events about the stream itself are `StreamControl`
objects (`kind`, `data`):

| `kind` | `data` | Meaning |
|---|---|---|
| `workflow_started` | `{"workflow_id", "thread_id"}` | Always the first event: the ids to use with `terminate_workflow` or `get_workflow_status`. Delivered only to `on_control`, never yielded. |
| `node_reset` | `{"node", "step"}` | A node was retried; drop its partial output. `step` is the `langgraph_step` of that node's `messages` metadata (numbered from 1). |
| `stream_unavailable` | `{"workflow_id", "reason", "missing_steps", "missing_nodes"}` | The run finished but a node's chunks never arrived (see the limit below). `missing_nodes` is a list of `{"step", "node"}`, one per node of a step that streamed nothing, even when other nodes of that step did. Steps are numbered from 1, like `langgraph_step`. |
| `stream_dropped` | `{"dropped": n}` | The consumer fell behind and `n` token or custom chunks were dropped. `updates` are never dropped. |

Pass `on_control=callback` to `astream` to receive them. A list of modes also
yields them as `(kind, data)` tuples; a single mode never does, so
`async for message, metadata in runner.astream(..., stream_mode="messages")`
never meets one.

Dapr retries a failed node activity and the chunks the failed attempt already
sent cannot be taken back (native `astream` has no retries). A single-mode
consumer that passes no `on_control` therefore sees the failed attempt's partial
text followed by the retry's; a warning is logged for each such `node_reset`.
To discard the partial output, pass `on_control` or use a list of modes and drop
what the node sent before its `node_reset`.

A run that fails (a node or condition error, max steps, a terminated workflow,
no result) raises `GraphRunError`; its `.final` is the full terminal event.

To stop reading early, close the stream. A bare `break` does not close an async
generator that is still referenced, so use `aclosing`:

```python
from contextlib import aclosing

async with aclosing(runner.astream(graph_input, stream_mode="messages")) as stream:
    async for message, metadata in stream:
        if done(message):
            break
```

Closing the stream does not cancel the Dapr workflow, which runs to completion.

### Over HTTP

`serve()` answers `POST /agent/run` with `"stream": true` as server-sent events.
`stream_mode` (a mode or a list; default all three) picks the chunks; an
unsupported one is refused with a `422` before the stream starts. The first
event is `workflow_started`. An idle stream sends a `: keepalive` comment every
15 seconds so a proxy does not cut it, and the response sets `Cache-Control:
no-cache` and `X-Accel-Buffering: no` so an nginx or ingress in front does not
buffer the tokens. The
messages and updates are serialized with `langchain_core.load.dumpd`, so
`langchain_core.load.load` restores them. The stream ends with
`workflow_completed` or `workflow_failed`; both carry `workflow_id` and
`thread_id`, a failure carries the error, and `dropped` is set when chunks were
dropped. The run's result is also published to `publish_topic`; it is exactly what
a non-streaming call returns and publishes for the same run, so a run that ends
with `status: "error"` is published as that call reports it (`workflow_completed`
with the error string) even though the stream's last event is `workflow_failed`.

A failed status poll (the sidecar restarting, a dropped call) is retried for
about a minute and does not fail the run. If the stream still cannot follow it,
it ends with `workflow_failed` whose `error.error_type` is `PollError` and
`workflow_still_running` is `true`: the workflow may yet complete, nothing is
published for it, and `GET /agent/run/{workflow_id}` tells how it ended.

`stream` and `stream_mode` are reserved keys on a streaming request: they are
removed before the graph (or your `input_mapper`) sees it. On a plain request
they are left alone.

### Limits

- **Single replica.** Chunks travel from the activity to the HTTP connection
  through an in-process queue. If an activity runs in another process (several
  replicas of the app, or after a restart) its chunks are not delivered: the
  activity logs a warning once per run and the stream sends `stream_unavailable`
  before it ends. The run and its result are not affected. Delivery across
  replicas is not implemented yet.
- **A disconnect does not cancel the run.** When the client goes away the stream
  is closed, but the Dapr workflow keeps running to completion. Terminate it with
  `terminate_workflow` if you need it stopped. The first event of every stream is
  `workflow_started` with the `workflow_id` and `thread_id`, so a client that
  loses its connection can still `GET /agent/run/{workflow_id}` or ask for the
  run to be terminated. With `astream`, `on_control` receives the same event.
  When `publish_topic` is set, the server keeps following the run after the
  client left and publishes its result when it ends, as it does for a client
  that stays.

- **`Send` payloads.** A node run through `Send("node", arg)` (or `goto=Send(...)`)
  reads the channel state, not `arg`; this is logged once per edge. A node that
  reads the argument gets different data than under LangGraph.

## Architecture

```text
User's LangGraph
      |
      v
DaprWorkflowGraphRunner
      |
      v
START WORKFLOW: dapr.langgraph.<name>.workflow
      |
      +--> Activity: execute_node_activity (node_a)
      |         |
      |         v
      +--> Activity: execute_node_activity (node_b)
      |         |
      |         v
      +--> ... (continues until END)
      |
      v
Return final state
```

Each node execution is a durable Dapr Workflow activity with:
- Automatic retries on failure
- State persistence between activities
- Full observability through Dapr APIs

## Requirements

- Python >= 3.11
- Dapr >= 1.17.3
- LangGraph >= 1.1.10
- Running Dapr sidecar

## Links

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Dapr](https://dapr.io/)
