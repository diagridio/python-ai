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

## Conversations

Pass the same `thread_id` to each run of a conversation, and send only the new
message:

```python
runner.invoke({"messages": [{"role": "user", "content": "hi"}]}, thread_id="chat-1")
result = runner.invoke(
    {"messages": [{"role": "user", "content": "and again"}]}, thread_id="chat-1"
)
# result["messages"] holds both turns, as with a LangGraph checkpointer
```

That is all. `thread_id` may also come as `config["configurable"]["thread_id"]`
or, over HTTP, as `thread_id` in the request body. Each run starts from the
state the thread's last run ended with and applies its input to it through the
channels' reducers, as LangGraph does with a checkpointer. The state is kept in
the app's Dapr state store (`agent-memory`, which a Catalyst app has), keyed by
thread. To keep it elsewhere, compile the graph with a LangGraph checkpointer
(`graph.compile(checkpointer=MemorySaver())`) or pass `checkpointer=` to the
runner (a `DaprMemoryCheckpointer` on another store, or a LangGraph saver with
the synchronous `get_tuple` and `put`). With none of these, `thread_id` only
names the run and each run starts from its input alone.

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
from langgraph.types import Command  # resumes a run paused by interrupt()
```

`astream` yields what `CompiledStateGraph.astream` yields, while each node runs
as a Dapr activity. The stream modes are:

| Mode | Chunk |
|---|---|
| `messages` | LLM tokens and the messages a node returns, as `(message_chunk, metadata)` |
| `updates` | `{node: update}`, one per node, with the values as the node returned them |
| `custom` | whatever a node writes with `get_stream_writer()` |
| `values` | the graph's output state after each step that wrote an output channel (and after the input, when it wrote one) |
| `tasks` | a task's start (`id`, `name`, `input`, `triggers`) when its node starts, and its result (`result`, `error`, `interrupts`) after its `updates` chunk |
| `debug` | the `tasks` events wrapped as `{step, timestamp, type, payload}` |

`checkpoints` is refused (`ValueError` from `astream`, a `422` over HTTP): the
runner keeps no per-step LangGraph checkpoints, because the Dapr workflow
history is the run's durable record, so there are no checkpoint ids a client
could pass back (a thread keeps only the state its last run ended with, see
[Conversations](#conversations)).
For the same reason `debug` holds only the task events, which is what native
`debug` holds for a graph compiled without a checkpointer. A client that wants
the full state reads it from `values`, from the run's final event, or from `GET
/agent/run/{workflow_id}`.

The `values` chunk of step s is sent by the first activity of step s+1 to
start, which is handed that state, so it comes one scheduling hop later than in
LangGraph but in the same place in the stream. The state after the last step is
sent before the final event; a run that failed in a node or a router sends no
value for that step, and a run cut by its step limit sends it, as LangGraph
does. A task that fails sends no result: Dapr retries it and `node_reset`
covers its partial output.

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
| `workflow_started` | `{"workflow_id", "thread_id"}` | The first event, sent once the workflow has been scheduled, so `GET /agent/run/{workflow_id}` finds the run as soon as the event arrives (a run that could not be scheduled ends with `workflow_failed` and has none): the ids to use with `terminate_workflow` or `get_workflow_status`. Delivered only to `on_control`, never yielded. |
| `node_reset` | `{"node", "step"}`, plus `"path"` for a `Send` task | A node was retried; drop its partial output. `node` and `step` are the `langgraph_node` and `langgraph_step` of that node's `messages` metadata (`step` is numbered from 1). A task started by a `Send` also carries `path`, the `langgraph_path` of its chunks (`["__pregel_push", index, false]`), which tells apart the tasks of one node in a step. A retry of a conditional edge's function also carries `branch`, the edge's function name. The function runs inside its node's activity, before the node's `updates` chunk, as in LangGraph; if it raises there, it runs again in an activity of its own, after that chunk, and only what the function sent is stale, not the node's own output. |
| `stream_unavailable` | `{"workflow_id", "reason", "missing_steps", "missing_nodes"}` | The run finished but a node's chunks never arrived (see the limit below). `missing_nodes` is a list of `{"step", "node"}`, one per node of a step that streamed nothing, even when other nodes of that step did. Steps are numbered from 1, like `langgraph_step`. A `Send` task is named `node#k` there (`k` is its position among the step's `Send` tasks). A conditional edge's function that ran in an activity of its own (a START edge's, or one that raised inside its node's activity and ran again) is listed as `node:branch:<condition>` (`node#k:branch:<condition>` after a `Send` task, `__start__:branch:<condition>` at step 0 for a START edge), where `<condition>` is the name the runner registered the edge under. |
| `stream_purged` | `{"workflow_id", "step"}` | A reconnect or a late reader found that part of the run's event log outlived its TTL (state-store transport only), from `step` on: those chunks cannot be replayed. Sent once; the chunks still in the log, the live ones and the final event follow. |
| `thread_not_saved` | `{"workflow_id", "thread_id", "error"}` | The run ended, but its thread's state could not be saved (after retries), so the next run on the thread starts from the state before this one. The run's output stands: the final event follows, with `thread_saved: false` and `thread_error`, which a plain `POST /agent/run` and `GET /agent/run/{workflow_id}` return too. |
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
no result) raises `GraphRunError`; its `.final` is the full terminal event. A run
that hits its step limit raises an error that is also LangGraph's
`GraphRecursionError`, so `except GraphRecursionError` works as it does with
native LangGraph. `invoke` raises the same errors for such a run, instead of
returning its partial output.

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

`serve()` answers `POST /agent/run` with `"stream": true` (the JSON boolean;
any other value is a plain request and the key reaches the graph) as
server-sent events.
`stream_mode` (a mode or a list; default `messages`, `updates` and `custom`)
picks the chunks; an unsupported one, `checkpoints` included, is refused with a
`422` before the stream starts. The first
event is `workflow_started`. An idle stream sends a `: keepalive` comment every
15 seconds so a proxy does not cut it, and the response sets `Cache-Control:
no-cache` and `X-Accel-Buffering: no` so an nginx or ingress in front does not
buffer the tokens. The
chunks are serialized with `langchain_core.load.dumpd`, so
`langchain_core.load.load` restores the messages in them. The stream ends with
`workflow_completed` or `workflow_failed` (or `workflow_interrupted`, see
[Interrupts](#interrupts)); both carry `workflow_id` and
`thread_id`, a failure carries the error, and `dropped` is set when chunks were
dropped. The run's result is also published to `publish_topic`; it is exactly what
a non-streaming call returns and publishes for the same run, so a run that ends
with `status: "error"` is published as that call reports it (`workflow_completed`
with the error string) even though the stream's last event is `workflow_failed`.

A failed status poll that looks transient (a gRPC or connection error: the
sidecar restarting, a dropped call) is retried for about a minute and does not
fail the run. If the stream still cannot follow it, it ends with
`workflow_failed` whose `error.error_type` is `PollError` and
`workflow_still_running` is `true`: the workflow may yet complete, nothing is
published for it, and `GET /agent/run/{workflow_id}` tells how it ended. A
failure of the stream itself after `workflow_started` is reported the same way
with `error_type` `StreamError`.

A chunk that cannot be encoded (a `custom` value whose `to_dict` raises, say) is
sent as `{"unserializable": "<its repr>"}`, counted in the final event's
`dropped`, and does not end the stream.

`stream` and `stream_mode` are reserved keys on a streaming request: they are
removed before the graph (or your `input_mapper`) sees it. On a plain request
they are left alone.

`GET /agent/run/{workflow_id}` reports a run by the id in `workflow_started`:
`status` is the workflow's runtime status, and once the run has completed the
body also has the `output` and `steps` a plain `POST /agent/run` returns for
it, `run_status` (`completed`, `error` or `max_steps_reached`) and `error`
when there is one. A workflow that failed carries its `error`; a run paused by
`interrupt()` has `interrupted: true` and its `interrupts` (see
[Interrupts](#interrupts)). An unknown id is a `404`.

Every event carries an SSE `id`. With the state-store transport (see
[Streaming across replicas](#streaming-across-replicas)) the id is a cursor
into the run's event log that any replica can resume from: `v1.start` on
`workflow_started`, `v1.end` on the final event, and in between the position
after that chunk (a control event repeats the id of the chunk before it). With
the in-process transport the events are numbered from 1 on each connection.

#### Reconnecting

A client that lost its stream (a network drop, a proxy timeout, the serving
process restarting) reconnects with `GET /agent/run/{workflow_id}/stream`,
sending the last id it received as the `Last-Event-ID` header (a browser
`EventSource` does this by itself) or, when it cannot set headers, as the
`last_event_id` query parameter. Any replica answers it. The response is the
same SSE stream from that point on: `workflow_started`, then the chunks the
client has not seen, then the live ones, then the final event with the run's
result. Nothing is repeated and nothing is skipped. Without an id, or with one
that does not parse, the whole run is replayed; with `v1.end` only the final
event is sent. A run that already ended is replayed the same way, for as long
as its log is kept (an hour by default). `stream_mode` (repeat the parameter,
or separate modes with commas) picks the chunks as on `POST`; an unsupported
one is a `422`, and an unknown run a `404`. If a node was retried after the
client saw part of its output, the stream sends `node_reset` for it before the
new attempt's chunks. If part of the log has expired, the stream sends
`stream_purged` once and goes on with what is still there and what comes
live (a run resumed after waiting on `interrupt()` for longer than the batch
TTL streams everything after the resume), then the final event.

On the in-process transport nothing can be replayed: the endpoint answers with
`workflow_started`, then `stream_unavailable` with reason
`transport_not_replayable` (reconnecting needs a state store), then waits for
the run and sends the final event with its result.

In Python, `runner.astream_resume(workflow_id, last_event_id=None,
stream_mode=...)` does the same and yields what `astream` yields.

#### Streaming across replicas

How chunks get from a node's activity to the stream is the runner's stream
transport (`DaprWorkflowGraphRunner(stream_transport=...)`). You normally do
not pick one:

- **`StateLogTransport` (the default whenever the app has a state store).**
  Every activity writes its chunks to an event log in a Dapr state store, in
  small batches (every 100 ms, every 64 chunks, and when the node ends), and
  the replica that serves a stream reads them back. So a stream carries every
  node's chunks whichever replica ran it, a client can reconnect to any
  replica, and a slow client lags instead of losing chunks. The runner uses
  the app's `agent-memory` state store if it has one, else the state store the
  workflows themselves use; every workflow app has that one, so on Catalyst
  there is nothing to add: no component, no variable, no argument. Keys look
  like `lgstream:<workflow id>:<step>:<task>:...` with `:` between path-escaped
  parts, batches expire after an hour and their index after a day. Chunks
  arrive about 150 to 250 ms later than in process, in small bursts. A failed
  write is retried with the next batch, and a node's last write (its `updates`
  chunk), which no batch follows, is retried up to four times with backoff; a
  write never fails a node. If a node's
  chunks still never reach the log, the stream ends with `stream_unavailable`
  whose `reason` is `write_failed`. A store that cannot be read ends the
  stream with `workflow_failed` (`error_type` `StreamReadError`), never with an
  empty stream. `StateLogStreamConfig` sets the store, key prefix, TTLs, batch
  sizes and read interval:

  ```python
  from diagrid.agent.core.workflow.stream_log import (
      StateLogStreamConfig,
      StateLogTransport,
  )

  runner = DaprWorkflowGraphRunner(
      graph=compiled,
      name="my-graph",
      stream_transport=StateLogTransport(
          StateLogStreamConfig(store_name="statestore", event_ttl_s=600)
      ),
  )
  ```

- **`InProcessTransport` (no state store, or a `stream_registry` given).**
  Chunks go straight from the activity to the stream through a queue in the
  process. Only the nodes that run in the process serving the stream are
  streamed (a single replica), nothing can be replayed, and a stream cannot be
  resumed after a drop or a restart.

#### Behind a proxy

Anything between the client and the app (nginx, an ingress, a load balancer)
must pass the response through as it is written, not buffer it. The response
already sends `X-Accel-Buffering: no` and `Cache-Control: no-cache`, which
nginx and the nginx ingress honour; for any other proxy, turn response
buffering off for `POST /agent/run`, and keep its idle timeout above 15
seconds so the keepalive comments hold the connection open. A client should
expect events to arrive one by one as the nodes produce them, comment lines
(`: keepalive`) to ignore while a node is quiet, and the stream to end right
after `workflow_completed` or `workflow_failed`. If tokens arrive all at once
at the end of the run, something in front of the app is buffering.

### Interrupts

A node can pause the run with LangGraph's `interrupt(value)`, as in a graph
with a checkpointer: the workflow is the checkpoint, so no checkpointer is
needed for this.

- The stream sends LangGraph's `updates` chunk `{"__interrupt__": [...]}` (each
  interrupt's `value` and `id`; `Interrupt` objects in `astream`) and ends with
  the event `workflow_interrupted` (`workflow_id`, `thread_id`, `interrupts`).
  `invoke` returns `{"__interrupt__": [Interrupt(...)]}`.
- The workflow waits durably for the answer: in its history, not in process
  memory, so the app may restart or scale meanwhile. The other nodes of the
  step keep their results. `GET /agent/run/{workflow_id}` shows
  `interrupted: true` and the `interrupts` while it waits.
- Resume it with `POST /agent/run/{workflow_id}/resume` and the body
  `{"resume": <value>}` (add `"stream": true`, and `stream_mode` if you like,
  for SSE; without it the response is the run's result, as a plain `POST`
  returns it). In Python: `runner.astream(Command(resume=value),
  workflow_id=...)`. The interrupted node runs again in the same run, on the
  same thread, and `interrupt()` returns the value; a dict keyed by interrupt
  ids answers several interrupts one by one. The stream of a resume carries
  what happens after it (starting with `node_reset` for the node that runs
  again) and ends with the final event, or with `workflow_interrupted` when a
  node asks again. A body without `resume` is a `422`, an unknown run a `404`,
  and a run that is not waiting a `409`.

### Limits

- **Single replica without a state store.** On the in-process transport chunks
  travel from the activity to the HTTP connection through a queue in the
  process. If an activity runs in another process (several replicas of the
  app, or after a restart) its chunks are not delivered: the activity logs a
  warning once per run and the stream sends `stream_unavailable` before it
  ends. The run and its result are not affected. Give the app a state store to
  stream across replicas and to reconnect (see
  [Streaming across replicas](#streaming-across-replicas)).
- **A replay shows a node's last attempt.** When a node was retried, a client
  that was connected saw the first attempt's chunks and then `node_reset`; a
  replay from the start (or a reconnect from before the retry) has the attempt
  that finished only.
- **Node code must not count attempts in process memory.** A retried node may
  run in another process, so a counter kept in a module or a closure does not
  see the earlier attempt. Keep such state in the graph's state or a store.
- **A disconnect does not cancel the run.** The stream closes but the Dapr
  workflow runs to completion; `terminate_workflow` stops it, and the ids in
  `workflow_started` let a client that lost its connection reconnect to its
  stream, `GET /agent/run/{workflow_id}` or terminate the run. When `publish_topic` is set,
  the server keeps following the run after the client left and publishes its
  result when it ends.

- **`Send` tasks.** Every `Send("node", arg)` (from a conditional edge or
  `goto=Send(...)`) runs the node once, as its own activity, with `arg` as its
  input, and streams with the `__pregel_push` trigger and path as LangGraph does
  (`create_agent`'s tool calls and map-reduce graphs). In `stream_unavailable`'s
  `missing_nodes` such a task is named `node#k`, `k` being its position among
  the step's `Send` tasks. The
  `arg` travels through the workflow as the checkpoint serializer's bytes, so it
  must be something that serializer handles.
- **State values keep their types between nodes.** A value of a state channel
  that is not plain JSON (a message, a `datetime`, a `set`, a pydantic object) is
  stored with the checkpoint serializer, so the next node and router read what
  was written, as in LangGraph. It must be something that serializer handles.
  A run's output (the `invoke` result and the final event) is JSON: such a value
  appears there as its dump (a `set` as a sorted list, a `datetime` as a string).
- **The `config` of `astream`, `invoke` and `run_async` travels as JSON.**
  `tags`, `metadata`, `configurable` and `recursion_limit` work; `recursion_limit`
  bounds the run's steps as LangGraph counts them: that many node steps run and
  then the run fails with `GraphRecursionError` (N sequential nodes need a limit
  of N + 1 to complete). The runner's `max_steps` applies when it is absent. A
  `recursion_limit` and `configurable` bound with `compiled.with_config(...)`
  apply unless the call sets its own; a `max_steps` you pass to the runner
  still caps a limit that is only bound to the graph (`create_agent` binds
  9999), so it stays a guard against a runaway loop. `callbacks` and any value that is not JSON
  raise `ValueError` before a run starts.

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
