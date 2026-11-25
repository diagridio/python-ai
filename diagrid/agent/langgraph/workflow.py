# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr Workflow definitions for durable LangGraph execution."""

import asyncio
import base64
import contextlib
import copy
import dataclasses
import functools
import logging
import threading
import uuid
from datetime import timedelta
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from dapr.ext.workflow import (
    DaprWorkflowContext,
    WorkflowActivityContext,
    RetryPolicy,
    when_all,
)

from langchain_core.messages import (
    AnyMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.runnables.config import run_in_executor, set_config_context
from langgraph.graph.message import add_messages
from pydantic import BaseModel, TypeAdapter, ValidationError
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite, Send

try:
    from langgraph.runtime import ExecutionInfo
except ImportError:  # pragma: no cover - older langgraph without ExecutionInfo
    ExecutionInfo = None  # type: ignore[assignment,misc]

try:
    from langgraph._internal._constants import CONFIG_KEY_RUNTIME, CONF
except ImportError:  # pragma: no cover - constant moved/renamed across versions
    CONFIG_KEY_RUNTIME = "__pregel_runtime"
    CONF = "configurable"

from diagrid.agent.core.workflow.stream import (
    StreamSinkRegistry,
    default_registry,
    get_stream_sink,
)

from .streaming import STREAM_RUN_KEY, NodeStreamer, StaticConfig
from .models import (
    ChannelState,
    ExecuteNodeInput,
    ExecuteNodeOutput,
    EvaluateConditionInput,
    EvaluateConditionOutput,
    GraphConfig,
    GraphWorkflowInput,
    GraphWorkflowOutput,
    NodeWrite,
)

import inspect

logger = logging.getLogger(__name__)


# Global registries - populated by the runner
_node_registry: Dict[str, Callable] = {}
_condition_registry: Dict[str, Callable] = {}
_channel_reducers: Dict[str, Callable] = {}
# The value a reducer channel holds before its first write (``[]`` for a list).
_channel_defaults: Dict[str, Any] = {}
_NO_DEFAULT: Any = object()
# The graph's state channels: a node's other keys are not writes, as in LangGraph.
# ``None`` until a runner registers them (then nothing is dropped).
_state_channels: Optional[frozenset] = None
# What a node and a conditional edge are handed: the channels of its input schema
# and the function that coerces the state dict to a schema class (pydantic or
# dataclass state). Conditions are keyed by their name.
_node_inputs: Dict[str, "_InputShape"] = {}
_condition_inputs: Dict[str, "_InputShape"] = {}
_serializer: Optional[Any] = None
_default_graph_config: Optional["GraphConfig"] = None
_default_input_mapper: Optional[Callable] = None
_default_max_steps: int = 100
# Every trigger channel of each node, as LangGraph lists them in a task's metadata.
_node_triggers: Dict[str, Tuple[str, ...]] = {}
# The metadata and tags the graph and each node give their tasks (see StaticConfig).
_node_static: Dict[str, StaticConfig] = {}

# The key a write that replaces a reducer channel's value is stored under.
OVERWRITE = "__overwrite__"

# Special node names
START = "__start__"
END = "__end__"

# Each node runs as an activity named after the node, so the workflow UI shows
# the graph's own step names. Instances started before this change scheduled
# every node as ``execute_node_activity``; the patch keeps them replaying.
NODE_ACTIVITY_NAMES_PATCH = "langgraph-node-activity-names"
GENERIC_NODE_ACTIVITY = "execute_node_activity"
CONDITION_ACTIVITY = "evaluate_condition_activity"
# What LangGraph names the trigger (and path) of a task started by a ``Send``.
PUSH_TRIGGER = "__pregel_push"


def send_key(node: str, slot: int) -> str:
    """The name a ``Send`` task's chunks and ``executed`` entry go by.

    Several ``Send`` tasks may run the same node in one step; each needs a name
    of its own so their attempts and ``updates`` are not taken for one node's.
    """
    return f"{node}#{slot}"


_GENERIC_ACTIVITY_NAMES = frozenset({GENERIC_NODE_ACTIVITY, CONDITION_ACTIVITY})


def node_activity_name(node_name: str) -> str:
    """Return the activity name a node is scheduled under.

    A node named like one of the generic activities keeps running through
    ``execute_node_activity`` rather than shadowing it.
    """
    if node_name in _GENERIC_ACTIVITY_NAMES:
        return GENERIC_NODE_ACTIVITY
    return node_name


def make_node_activity(
    node_name: str,
    stream_registry: Optional[StreamSinkRegistry] = None,
) -> Callable[[WorkflowActivityContext, Dict[str, Any]], Dict[str, Any]]:
    """Build the activity registered for one node.

    Dapr registers each function object once, so every node gets its own
    wrapper around ``execute_node_activity``. ``stream_registry`` is where the
    wrapper looks for the sink of a streamed run (default: the module's).
    """

    def node_activity(
        ctx: WorkflowActivityContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return execute_node_activity(ctx, input_data, stream_registry=stream_registry)

    return node_activity


def register_node(name: str, func: Callable) -> None:
    """Register a node function for use by the execute_node activity."""
    _node_registry[name] = func


def register_node_triggers(name: str, triggers: List[str]) -> None:
    """Record the channels that trigger a node (for ``langgraph_triggers``)."""
    _node_triggers[name] = tuple(sorted(triggers))


def register_node_static_config(name: str, static: StaticConfig) -> None:
    """Record the graph's and the node's own metadata and tags for ``name``."""
    _node_static[name] = static


def get_node_static_config(name: str) -> Optional[StaticConfig]:
    """What :func:`register_node_static_config` recorded for ``name``."""
    return _node_static.get(name)


def get_node_triggers(name: str) -> Optional[Tuple[str, ...]]:
    """The trigger channels of a node, sorted as LangGraph reports them."""
    return _node_triggers.get(name)


def get_registered_node(name: str) -> Optional[Callable]:
    """Get a registered node function by name."""
    return _node_registry.get(name)


def register_condition(name: str, func: Callable) -> None:
    """Register a condition function for conditional edges."""
    _condition_registry[name] = func


def get_registered_condition(name: str) -> Optional[Callable]:
    """Get a registered condition function by name."""
    return _condition_registry.get(name)


def register_channel_reducer(
    channel: str, reducer: Callable, default: Any = _NO_DEFAULT
) -> None:
    """Register a reducer function for a channel.

    ``default`` is what the channel holds before its first write; the first
    write is reduced into it, as LangGraph does. Without one it is stored as is.
    """
    _channel_reducers[channel] = reducer
    if default is _NO_DEFAULT:
        _channel_defaults.pop(channel, None)
    else:
        _channel_defaults[channel] = default


@dataclass(frozen=True)
class _InputShape:
    """The channels a node reads and the mapper that builds the object it takes."""

    channels: Optional[Tuple[str, ...]] = None
    mapper: Optional[Callable[[Dict[str, Any]], Any]] = None

    def apply(self, state: Dict[str, Any]) -> Any:
        """``state`` cut to the channels and coerced to the schema class."""
        channels = self.channels
        if channels is not None:
            state = {k: v for k, v in state.items() if k in channels}
        return self.mapper(state) if self.mapper else state


def _input_shape(channels: Any, mapper: Optional[Callable]) -> _InputShape:
    """The shape for an input schema's ``channels`` (a single root channel is not cut)."""
    names = [channels] if isinstance(channels, str) else list(channels or ())
    keep = tuple(names) if names and "__root__" not in names else None
    return _InputShape(channels=keep, mapper=mapper)


def register_node_input(name: str, channels: Any, mapper: Optional[Callable]) -> None:
    """Record what node ``name`` is handed as its state."""
    _node_inputs[name] = _input_shape(channels, mapper)


def register_condition_input(
    name: str, channels: Any, mapper: Optional[Callable]
) -> None:
    """Record what the conditional edge ``name`` is handed as its state."""
    _condition_inputs[name] = _input_shape(channels, mapper)


def register_state_channels(channels: Iterable[str]) -> None:
    """Record the graph's state channels, the only keys a node can write."""
    global _state_channels
    _state_channels = frozenset(channels)


def _keep_channel(channel: str) -> bool:
    """Whether a node's key is a state channel (everything is, before any is set)."""
    names = _state_channels
    return names is None or "__root__" in names or channel in names


def get_channel_reducer(channel: str) -> Optional[Callable]:
    """Get the reducer for a channel."""
    return _channel_reducers.get(channel)


def set_serializer(serializer: Any) -> None:
    """Set the serializer for channel values."""
    global _serializer
    _serializer = serializer


def get_serializer() -> Optional[Any]:
    """Get the serializer."""
    return _serializer


def set_default_graph_config(
    config: "GraphConfig",
    *,
    input_mapper: Optional[Callable] = None,
    max_steps: int = 100,
) -> None:
    """Store the default graph config for orchestrator-initiated calls."""
    global _default_graph_config, _default_input_mapper, _default_max_steps
    _default_graph_config = config
    _default_input_mapper = input_mapper
    _default_max_steps = max_steps


def _close_langsmith_parent_trace_async(config: Optional[Dict[str, Any]]) -> None:
    """Close the parent LangSmith trace in a background thread (non-blocking)."""
    if not config:
        return
    run_id = config.get("langsmith_run_id")
    if not run_id:
        return

    import threading

    def _do_close() -> None:
        try:
            import os

            if os.environ.get("LANGSMITH_TRACING", "").lower() not in ("true", "1"):
                return
            from langsmith import Client
            from datetime import datetime, timezone

            client = Client()
            client.update_run(run_id, end_time=datetime.now(timezone.utc))
            client.flush()
        except Exception as e:
            logger.debug(f"Failed to close parent LangSmith trace: {e}")

    threading.Thread(target=_do_close, daemon=True).start()


def _langsmith_parent(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the parent "LangGraph" trace's dotted order when tracing is on.

    Activities run on Dapr's thread pool, so this also clears any run tree a
    previous activity left in the thread's context.
    """
    import os

    dotted_order = (config or {}).get("langsmith_dotted_order")
    if not dotted_order or os.environ.get("LANGSMITH_TRACING", "").lower() not in (
        "true",
        "1",
    ):
        return None

    # The parent run-tree ContextVar is a private symbol that langsmith
    # has renamed across versions (_PARENT_RUN_TREE -> _PARENT_RUN_TREE_REF
    # in 0.7.31); look it up dynamically and degrade gracefully if it moves.
    try:
        from langsmith import run_helpers as _ls_run_helpers

        _parent_run_tree = getattr(
            _ls_run_helpers, "_PARENT_RUN_TREE_REF", None
        ) or getattr(_ls_run_helpers, "_PARENT_RUN_TREE", None)
        if _parent_run_tree is not None:
            _parent_run_tree.set(None)
    except (ImportError, AttributeError):
        pass
    return str(dotted_order)


def clear_registries() -> None:
    """Clear all registries."""
    _node_registry.clear()
    _condition_registry.clear()
    _channel_reducers.clear()
    _channel_defaults.clear()
    global _state_channels
    _state_channels = None
    _node_triggers.clear()
    _node_static.clear()
    _node_inputs.clear()
    _condition_inputs.clear()
    global _serializer, _default_graph_config, _default_input_mapper, _default_max_steps
    _serializer = None
    _default_graph_config = None
    _default_input_mapper = None
    _default_max_steps = 100


def _workflow_input(
    ctx: DaprWorkflowContext, input_data: Dict[str, Any]
) -> GraphWorkflowInput:
    """The workflow's input: a full ``GraphWorkflowInput`` or a ``{"task": ...}``."""
    if "graph_config" in input_data:
        # Normal internal call with full GraphWorkflowInput
        return GraphWorkflowInput.from_dict(input_data)

    # Orchestrator call with simple {"task": "..."} input
    if _default_graph_config is None:
        raise ValueError(
            "Received simple task input but no default graph config is set. "
            "Ensure the runner has been started before the workflow is invoked."
        )
    # Build graph input from the task using the stored mapper
    if _default_input_mapper:
        graph_input = _default_input_mapper(input_data)
    elif "task" in input_data:
        graph_input = {"messages": [{"role": "user", "content": input_data["task"]}]}
    else:
        graph_input = input_data

    channel_state = ChannelState(
        values=_serialize_input(
            graph_input, lambda c, i: _replay_safe_id(ctx.instance_id, i, c)
        ),
        versions={k: 1 for k in graph_input.keys()},
        updated_channels=list(graph_input.keys()),
    )
    return GraphWorkflowInput(
        graph_config=_default_graph_config,
        channel_state=channel_state,
        step=0,
        max_steps=_default_max_steps,
        thread_id=str(ctx.instance_id),
    )


@dataclass
class _Run:
    """What every step of one ``agent_workflow`` execution shares."""

    ctx: DaprWorkflowContext
    graph_config: GraphConfig
    channel_state: ChannelState
    config: Optional[Dict[str, Any]]
    thread_id: Optional[str]
    retry_policy: RetryPolicy
    node_activities_by_name: bool
    #: The nodes that ran in each step, reported on the output so a streaming
    #: consumer can tell which node's chunks never arrived.
    executed: List[List[str]] = field(default_factory=list)

    def finish(
        self, step: int, status: str, error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Close the run's trace and build the workflow output."""
        if not self.ctx.is_replaying:
            _close_langsmith_parent_trace_async(self.config)
        return GraphWorkflowOutput(
            output=_extract_output(
                self.channel_state, self.graph_config.output_channels
            ),
            channel_state=self.channel_state,
            steps=step,
            executed=self.executed,
            status=status,
            **({"error": error} if error else {}),
        ).to_dict()


def agent_workflow(
    ctx: DaprWorkflowContext, input_data: Dict[str, Any]
) -> Generator[Any, Any, Dict[str, Any]]:
    """Dapr Workflow that orchestrates LangGraph execution.

    This workflow implements the Pregel/BSP execution model:
    1. Determine which nodes should run based on updated channels
    2. Execute all triggered nodes in parallel (as activities)
    3. Apply writes to channels
    4. Repeat until no more nodes are triggered or END is reached

    Args:
        ctx: The Dapr workflow context
        input_data: Dictionary containing GraphWorkflowInput data

    Returns:
        GraphWorkflowOutput as a dictionary
    """
    workflow_input = _workflow_input(ctx, input_data)

    # Ask before the first yield: a replay of an older instance must see the
    # patch as absent, and the answer is fixed for the rest of the execution.
    node_activities_by_name = ctx.is_patched(NODE_ACTIVITY_NAMES_PATCH)

    run = _Run(
        ctx=ctx,
        graph_config=workflow_input.graph_config,
        channel_state=workflow_input.channel_state,
        config=workflow_input.config,
        thread_id=workflow_input.thread_id,
        retry_policy=RetryPolicy(
            max_number_of_attempts=3,
            first_retry_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            max_retry_interval=timedelta(seconds=30),
        ),
        node_activities_by_name=node_activities_by_name,
    )
    max_steps = workflow_input.max_steps
    step = 0

    pending_nodes, pending_sends, error = yield from _start_nodes(run)
    if error:
        return run.finish(step, "error", f"Condition evaluation failed: {error}")
    # Joins: how many of each waiting edge's sources have run since it last fired.
    joined: List[Set[str]] = [set() for _ in run.graph_config.waiting_edges]

    # Main execution loop
    while True:
        # Log workflow step for debugging
        if not ctx.is_replaying:
            print(
                f"  [WORKFLOW] Step {step}, pending_nodes={pending_nodes}", flush=True
            )

        # END ends the run only when nothing else is pending: another branch of
        # the step may still have nodes to run, as in LangGraph.
        nodes_to_execute = [n for n in pending_nodes if n != END]
        if not nodes_to_execute and not pending_sends:
            return run.finish(step, "completed")

        # The bound is checked after the end of the run, so a graph that needs
        # exactly ``max_steps`` steps completes instead of being cut at the end.
        if step >= max_steps:
            break

        run.executed.append(
            [
                *nodes_to_execute,
                *[send_key(t["node"], slot) for slot, t in enumerate(pending_sends)],
            ]
        )

        node_outputs = yield from _run_nodes(run, nodes_to_execute, pending_sends, step)
        failed = next((o for o in node_outputs if o.error), None)
        if failed:
            return run.finish(
                step, "error", f"Node '{failed.node_name}' failed: {failed.error}"
            )

        # Apply writes to channels
        try:
            updated_channels = _apply_outputs(run.channel_state, node_outputs)
        except ReducerError as e:
            return run.finish(step, "error", str(e))
        run.channel_state.updated_channels = updated_channels

        ran = list(
            dict.fromkeys([*nodes_to_execute, *[t["node"] for t in pending_sends]])
        )
        pending_nodes, pending_sends, error = yield from _next_nodes(
            run, ran, node_outputs, joined, step
        )
        if error:
            return run.finish(step, "error", f"Condition evaluation failed: {error}")
        step += 1

    # Max steps reached
    return run.finish(step, "max_steps_reached", f"Max steps ({max_steps}) reached")


def _apply_outputs(
    channel_state: ChannelState, node_outputs: List[ExecuteNodeOutput]
) -> List[str]:
    """Apply the writes of a step's nodes; the channels they updated, in order."""
    updated_channels: List[str] = []
    for output in node_outputs:
        for write in output.writes:
            _apply_write(channel_state, write.channel, write.value)
            if write.channel not in updated_channels:
                updated_channels.append(write.channel)
    return updated_channels


_StartOrNext = Tuple[List[str], List[Dict[str, Any]], Optional[str]]


def _start_nodes(run: _Run) -> Generator[Any, Any, _StartOrNext]:
    """The nodes (and ``Send`` tasks) the START edges lead to, and a condition error.

    A conditional START edge is evaluated on the input, as LangGraph does.
    """
    graph_config = run.graph_config
    if graph_config.entry_points is None:  # a config from before entry_points
        return [graph_config.entry_point], [], None
    pending_nodes = list(graph_config.entry_points)
    pending_sends: List[Dict[str, Any]] = []
    for edge in graph_config.edges:
        if edge.source != START or not edge.condition:
            continue
        cond_output = yield from _evaluate_edge(
            run.ctx,
            edge,
            START,
            run.channel_state,
            run.config,
            run.thread_id,
            run.retry_policy,
            -1,  # before the first node, so the first node is step 1
        )
        if cond_output.error:
            return pending_nodes, pending_sends, cond_output.error
        for n in cond_output.next_nodes:
            if n not in pending_nodes:
                pending_nodes.append(n)
        pending_sends.extend(_indexed(cond_output.sends))
    return pending_nodes, pending_sends, None


def _indexed(sends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``sends`` numbered by their position in the list one source returned."""
    return [{**send, "index": i} for i, send in enumerate(sends)]


def _run_nodes(
    run: _Run,
    nodes_to_execute: List[str],
    sends: List[Dict[str, Any]],
    step: int,
) -> Generator[Any, Any, List[ExecuteNodeOutput]]:
    """Run the nodes and ``Send`` tasks of one step in parallel, each as an activity."""
    node_tasks = []
    invocations: List[Tuple[str, Optional[Dict[str, Any]]]] = [
        *[(name, None) for name in nodes_to_execute],
        *[
            (t["node"], {"arg": t["arg"], "index": t["index"], "slot": slot})
            for slot, t in enumerate(sends)
        ],
    ]
    for node_name, send in invocations:
        node_input = ExecuteNodeInput(
            node_name=node_name,
            channel_state=run.channel_state,
            config=run.config,
            thread_id=run.thread_id,
            step=step,
            send=send,
        )
        activity: Any = execute_node_activity
        if run.node_activities_by_name:
            activity = node_activity_name(node_name)
        node_tasks.append(
            run.ctx.call_activity(
                activity,
                input=node_input.to_dict(),
                retry_policy=run.retry_policy,
            )
        )

    # Wait for all node executions to complete
    task_results = yield when_all(node_tasks)
    return [ExecuteNodeOutput.from_dict(result) for result in task_results]


def _next_nodes(
    run: _Run,
    nodes_to_execute: List[str],
    node_outputs: List[ExecuteNodeOutput],
    joined: List[Set[str]],
    step: int,
) -> Generator[Any, Any, _StartOrNext]:
    """The nodes and ``Send`` tasks the next step runs, and a condition error.

    They follow from the edges of the nodes that ran, the joins those nodes
    complete, and the ``Command(goto=...)`` they returned.
    """
    next_nodes: List[str] = []
    next_sends: List[Dict[str, Any]] = []
    for node_name in nodes_to_execute:
        for edge in run.graph_config.edges:
            if edge.source != node_name:
                continue
            if not edge.condition:
                # Direct edge
                if edge.target not in next_nodes:
                    next_nodes.append(edge.target)
                continue
            # Evaluate conditional edge (as activity)
            cond_output = yield from _evaluate_edge(
                run.ctx,
                edge,
                node_name,
                run.channel_state,
                run.config,
                run.thread_id,
                run.retry_policy,
                step,
            )
            if cond_output.error:
                return next_nodes, next_sends, cond_output.error
            for n in cond_output.next_nodes:
                if n not in next_nodes:
                    next_nodes.append(n)
            next_sends.extend(_indexed(cond_output.sends))

    # A join fires, once, in the step after the last of its sources ran.
    for seen, join in zip(joined, run.graph_config.waiting_edges):
        seen.update(n for n in nodes_to_execute if n in join.sources)
        if seen.issuperset(join.sources):
            seen.clear()
            if join.target not in next_nodes:
                next_nodes.append(join.target)

    # Nodes a Command(goto=...) asked for run next as well, as in LangGraph.
    for output in node_outputs:
        for target in output.goto:
            if target not in next_nodes:
                next_nodes.append(target)
        next_sends.extend(_indexed(output.sends))
    return next_nodes, next_sends, None


def _evaluate_edge(
    ctx: DaprWorkflowContext,
    edge: Any,
    source_node: str,
    channel_state: ChannelState,
    config: Optional[Dict[str, Any]],
    thread_id: Optional[str],
    retry_policy: RetryPolicy,
    step: int,
) -> Generator[Any, Any, EvaluateConditionOutput]:
    """Evaluate a conditional edge as an activity; ``yield from`` it."""
    cond_input = EvaluateConditionInput(
        source_node=source_node,
        condition_name=edge.condition,
        channel_state=channel_state,
        config=config,
        thread_id=thread_id,
        path_map=edge.path_map,
        step=step,
    )
    # By name, so the runner's own activity (the one that knows its stream
    # registry) runs, not the module-level function.
    cond_result = yield ctx.call_activity(
        CONDITION_ACTIVITY,
        input=cond_input.to_dict(),
        retry_policy=retry_policy,
    )
    return EvaluateConditionOutput.from_dict(cond_result)


def _build_node_runtime(
    thread_id: Optional[str],
    ctx: WorkflowActivityContext,
    node_name: str,
    stream_writer: Optional[Callable[[Any], None]] = None,
) -> Runtime:
    """Build a LangGraph ``Runtime`` carrying execution info for the node.

    Sandbox / state backends read ``runtime.execution_info.thread_id`` to
    resolve which workflow owns a given tool call. Constructing a bare
    ``Runtime()`` drops that linkage, so we populate ``execution_info``
    explicitly. Falls back to a bare ``Runtime()`` when ``ExecutionInfo`` is
    unavailable or its constructor signature differs across langgraph versions
    (a broad ``except`` keeps a version bump from hard-failing the node).

    Args:
        thread_id: The execution thread identifier to expose to consumers.
        ctx: The workflow activity context (provides workflow/task ids).
        node_name: The node being executed (used as the checkpoint namespace).
        stream_writer: Receives ``get_stream_writer()`` output; omitted when
            the run is not streaming (the library no-op stays in place).

    Returns:
        A ``Runtime`` with ``execution_info`` populated, or a bare ``Runtime``.
    """
    extra: Dict[str, Any] = {"stream_writer": stream_writer} if stream_writer else {}
    if ExecutionInfo is None:
        return Runtime(**extra)
    try:
        execution_info = ExecutionInfo(
            checkpoint_id=str(getattr(ctx, "workflow_id", "") or ""),
            checkpoint_ns=node_name,
            task_id=str(getattr(ctx, "task_id", "") or ""),
            thread_id=thread_id,
        )
        return Runtime(execution_info=execution_info, **extra)
    except Exception:  # pragma: no cover - signature drift / partial ctx
        return Runtime(**extra)


def _is_async_only(runnable: Any) -> bool:
    """Whether ``runnable`` wraps only an ``async def`` (no sync ``func``).

    ``RunnableCallable.invoke()`` raises for these, so they must be run through
    ``ainvoke()`` instead.
    """
    return (
        getattr(runnable, "func", None) is None
        and getattr(runnable, "afunc", None) is not None
    )


def _has_async_entry(runnable: Any) -> bool:
    """Whether ``runnable`` has a real async entry point, which a node should use.

    ``ToolNode`` has a sync ``func`` and an ``afunc``; its sync one cannot run
    an ``async def`` tool (MCP adapter tools are such), its async one runs both,
    as native ``astream`` does. A plain sync node also gets an ``afunc``, but
    only a ``run_in_executor`` wrapper around its function: that is not one.
    """
    afunc = getattr(runnable, "afunc", None)
    if afunc is None:
        return False
    return not (isinstance(afunc, functools.partial) and afunc.func is run_in_executor)


@dataclass
class _NodeCall:
    """Everything one execution of a node needs, gathered by the activity."""

    ctx: WorkflowActivityContext
    node_input: ExecuteNodeInput
    node_func: Callable
    state: Any
    config: Dict[str, Any]
    streamer: Optional[NodeStreamer]
    #: Side-channel writes queued via CONFIG_KEY_SEND while the node runs.
    pending_sends: list

    @property
    def node_name(self) -> str:
        return self.node_input.node_name

    @property
    def stream_writer(self) -> Optional[Callable[[Any], None]]:
        return self.streamer.writer if self.streamer else None


def _make_activity_read(
    channel_state: ChannelState, pending_sends: list
) -> Callable[..., Any]:
    """Stand-in for CONFIG_KEY_READ inside a Dapr activity.

    In normal LangGraph execution the Pregel engine injects a read
    function bound to live BaseChannel objects.  Here the activity
    only has a serialized ``channel_state``, so we read from that.

    It is a ``functools.partial`` whose second argument is the channel mapping,
    as Pregel's is: the prebuilt tool node reads ``read.args[1]`` for the
    channel names when ``create_agent`` sends it a bare list of tool calls.
    """
    return functools.partial(
        _read_activity_channels, pending_sends, channel_state.values
    )


def _read_activity_channels(
    pending_sends: list,
    channels: Dict[str, Any],
    select: Any,
    fresh: bool = False,
) -> Any:
    if fresh:
        # Apply any pending sends to a snapshot before reading.
        merged = dict(channels)
        for ch, val in pending_sends:
            reducer = get_channel_reducer(ch)
            if reducer and ch in merged:
                try:
                    merged[ch] = reducer(merged[ch], val)
                except Exception:
                    merged[ch] = val
            else:
                merged[ch] = val
        source = merged
    else:
        source = channels

    if isinstance(select, str):
        return source.get(select)
    return {k: source.get(k) for k in select if k in source}


class _BackgroundLoop:
    """One event loop per process, in a daemon thread, that async nodes run on.

    An async client a node keeps (a chat model's HTTP connection pool) is bound
    to the loop it was first used on. A loop made and closed for each activity
    would break that client for the next activity, so every coroutine runs on
    this one, which is never closed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _running_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                threading.Thread(
                    target=loop.run_forever, name="langgraph-node-loop", daemon=True
                ).start()
                self._loop = loop
            return self._loop

    def run(self, coro: Any) -> Any:
        """Run ``coro`` to completion on the loop and return its result.

        ``run_coroutine_threadsafe`` starts the task in a copy of the calling
        thread's context, so the ambient config ``set_config_context`` set for
        the node is what the coroutine sees.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._running_loop())
        return future.result()


_background_loop = _BackgroundLoop()


def _await_if_coroutine(r: Any) -> Any:
    """Run ``r`` to completion on the process's node loop if it is a coroutine."""
    if asyncio.iscoroutine(r):
        return _background_loop.run(r)
    return r


def _runnable_config(call: _NodeCall) -> Dict[str, Any]:
    """The config a Runnable node is invoked with.

    Copies before mutating so we never write back into the shared config dict
    handed to this activity (immutability).
    """
    _config = dict(call.config) if call.config else {}
    _config[CONF] = dict(_config.get(CONF) or {})

    thread_id = (
        call.node_input.thread_id
        or _config.get("thread_id")
        or getattr(call.ctx, "workflow_id", None)
    )
    # Expose thread_id in the conventional LangGraph location too,
    # so middleware reading config["configurable"]["thread_id"] works.
    _config[CONF].setdefault("thread_id", thread_id)

    # Ensure the config has a Runtime object for deep agent middleware nodes
    # that require it (runtime injection via RunnableCallable). It also lets
    # sandbox/state backends resolve which workflow owns a tool call via
    # runtime.execution_info.thread_id.
    if CONFIG_KEY_RUNTIME not in _config[CONF]:
        _config[CONF][CONFIG_KEY_RUNTIME] = _build_node_runtime(
            thread_id, call.ctx, call.node_name, call.stream_writer
        )
    if call.streamer:
        _config = call.streamer.patch_config(_config)

    # Inject CONFIG_KEY_READ / CONFIG_KEY_SEND so that deepagents
    # StateBackend (sandbox tools) can read/write the "files"
    # channel when running inside a Dapr workflow activity.
    try:
        from langgraph._internal._constants import (
            CONFIG_KEY_READ,
            CONFIG_KEY_SEND,
        )

        _config[CONF][CONFIG_KEY_READ] = _make_activity_read(
            call.node_input.channel_state, call.pending_sends
        )
        _config[CONF][CONFIG_KEY_SEND] = call.pending_sends.extend
    except ImportError:
        pass
    return _config


def _invoke_runnable(call: _NodeCall, config: Dict[str, Any]) -> Any:
    """Call a Runnable node; one with an async entry returns a coroutine to await."""
    if _has_async_entry(call.node_func):
        return call.node_func.ainvoke(call.state, config=config)  # type: ignore[attr-defined]
    return call.node_func.invoke(call.state, config=config)  # type: ignore[attr-defined]


def _call_plain(call: _NodeCall) -> Any:
    """Call a plain function node (sync or async) and return its result."""
    sig = inspect.signature(call.node_func)
    params = list(sig.parameters.keys())
    if len(params) >= 2 and "config" in params:
        out = call.node_func(call.state, config=call.config)
    else:
        out = call.node_func(call.state)
    return _await_if_coroutine(out)


def _plain_node_config(call: _NodeCall) -> Dict[str, Any]:
    """Ambient config for a plain function node.

    Plain callables get no config injected, so what a node reaches through the
    ambient config (``get_stream_writer()``, the stream handler and the task
    metadata of a streamed run) is supplied here. Without a streamer the
    runtime carries LangGraph's no-op writer, as ``invoke`` does natively.
    """
    thread_id = call.node_input.thread_id or getattr(call.ctx, "workflow_id", None)
    runtime = _build_node_runtime(
        thread_id, call.ctx, call.node_name, call.stream_writer
    )
    configurable = {**(call.config.get(CONF) or {}), CONFIG_KEY_RUNTIME: runtime}
    config: Dict[str, Any] = {CONF: configurable}
    return call.streamer.patch_config(config) if call.streamer else config


def _run_node(call: _NodeCall) -> Any:
    """Run the node and return its raw result."""
    node_func = call.node_func
    if hasattr(node_func, "invoke") and callable(getattr(node_func, "invoke")):
        # A LangChain Runnable (e.g. RunnableCallable): call .invoke() so that
        # argument injection (runtime, config, store) works.
        node_config = _runnable_config(call)

        def run() -> Any:
            return _await_if_coroutine(_invoke_runnable(call, node_config))

        ambient = node_config
    else:

        def run() -> Any:
            return _call_plain(call)

        ambient = _plain_node_config(call)

    # RunnableCallable.invoke only sets the ambient config when tracing;
    # get_stream_writer() and the stream handler need it, and a node that calls
    # get_stream_writer() must find a (no-op) writer when the run is not streamed.
    with set_config_context(ambient) as ctx_:  # type: ignore[arg-type]
        return ctx_.run(run)


def _warn_stream_sink_missing(
    registry: StreamSinkRegistry, workflow_id: str, node_name: str
) -> None:
    """Say, once per run, that a streamed run's chunks are going nowhere."""
    if not registry.first_report(workflow_id):
        return
    if registry.was_closed(workflow_id):
        # This process had the stream and it ended (the client left, or the
        # run's stream was closed): not a second replica, nothing to act on.
        logger.debug(
            "Run %s: its stream was closed in this process (its consumer left "
            "or the stream ended); chunks of node %r are not delivered",
            workflow_id,
            node_name,
        )
        return
    logger.warning(
        "Run %s is streamed but this process has no stream sink for it, so the "
        "chunks of node %r (and of any node that runs here) are not delivered. "
        "The activity is running in another process (several replicas) or the "
        "run outlived the process that streams it. Cross-replica streaming is "
        "not supported yet; the run itself is unaffected.",
        workflow_id,
        node_name,
    )


def _make_streamer(
    ctx: WorkflowActivityContext,
    node_input: ExecuteNodeInput,
    config: Dict[str, Any],
    registry: Optional[StreamSinkRegistry],
) -> Optional[NodeStreamer]:
    """The streamer for this attempt of the node, or ``None`` when not streaming.

    Chunks are streamed only when a consumer registered a sink for this run;
    otherwise nothing about the node's execution changes.
    """
    send = node_input.send
    return _streamer_for(
        ctx,
        node_input.node_name,
        node_input.step or 0,
        node_input.thread_id,
        config,
        registry,
        # a Send is a task of its own, keyed apart from the node's other runs
        sink_node=None
        if send is None
        else send_key(node_input.node_name, send["slot"]),
        send_index=None if send is None else send["index"],
    )


def _streamer_for(
    ctx: WorkflowActivityContext,
    node_name: str,
    step: int,
    thread_id: Optional[str],
    config: Dict[str, Any],
    registry: Optional[StreamSinkRegistry],
    sink_node: Optional[str] = None,
    send_index: Optional[int] = None,
) -> Optional[NodeStreamer]:
    """A streamer that emits as ``node_name`` at ``step``, or ``None``.

    ``send_index`` is set for a node that runs for a ``Send``: LangGraph tags
    such a task with the ``__pregel_push`` trigger and path.
    """
    sink = get_stream_sink(getattr(ctx, "workflow_id", None), registry=registry)
    if sink is None:
        if config.get(STREAM_RUN_KEY):
            _warn_stream_sink_missing(
                registry or default_registry, str(ctx.workflow_id), node_name
            )
        return None
    return NodeStreamer(
        sink,
        workflow_id=str(ctx.workflow_id),
        step=step,
        node=node_name,
        task_id=getattr(ctx, "task_id", ""),
        thread_id=thread_id or config.get("thread_id"),
        triggers=_task_triggers(node_name, send_index),
        static=get_node_static_config(node_name),
        sink_node=sink_node,
        path=None if send_index is None else (PUSH_TRIGGER, send_index, False),
    )


def _task_triggers(
    node_name: str, send_index: Optional[int]
) -> Optional[Tuple[str, ...]]:
    """The ``langgraph_triggers`` of a task: what LangGraph lists for it.

    A ``Send`` task is triggered by the push channel, and a conditional START
    edge's function (which is no node, so has no registered triggers) by START.
    """
    if send_index is not None:
        return (PUSH_TRIGGER,)
    if node_name == START:
        return (START,)
    return get_node_triggers(node_name)


def _run_with_stream_hooks(call: _NodeCall, parent: Optional[str]) -> Any:
    """Run the node, telling its streamer when it starts, finishes or fails.

    ``parent`` is the LangSmith parent trace to nest the node's run under.
    """
    streamer = call.streamer
    if streamer:
        streamer.node_started(call.state, call.config)
    try:
        if parent:
            import langsmith as ls

            with ls.trace(
                name=call.node_name,
                run_type="chain",
                parent=parent,
                inputs={"step": call.node_name},
            ):
                result = _run_node(call)
        else:
            result = _run_node(call)
    except BaseException as exc:
        if streamer:
            streamer.node_failed(exc)
        raise
    if streamer:
        streamer.node_finished(result)
    return result


def _node_output(call: _NodeCall, result: Any) -> Dict[str, Any]:
    """The activity's output for a node that returned ``result``.

    Converts the result to writes, appends any side-channel writes queued via
    CONFIG_KEY_SEND (e.g. StateBackend writing to the "files" channel from
    sandbox tools), and streams the node's update.
    """
    raw_writes = [*_result_to_raw_writes(result), *call.pending_sends]
    if call.streamer:
        call.streamer.updates(raw_writes)
    goto, sends = _result_to_gotos(result)
    return ExecuteNodeOutput(
        node_name=call.node_name,
        writes=_serialize_writes(raw_writes),
        goto=goto,
        sends=sends,
    ).to_dict()


def execute_node_activity(
    ctx: WorkflowActivityContext,
    input_data: Dict[str, Any],
    *,
    stream_registry: Optional[StreamSinkRegistry] = None,
) -> Dict[str, Any]:
    """Activity that executes a single LangGraph node.

    This activity:
    1. Gets the node function from the registry
    2. Reconstructs the state from channel values
    3. Executes the node function
    4. Returns the writes produced

    Args:
        ctx: The workflow activity context
        input_data: Dictionary containing ExecuteNodeInput data
        stream_registry: Where to look up the sink of a streamed run
            (default: the module's registry).

    Returns:
        ExecuteNodeOutput as a dictionary
    """
    node_input = ExecuteNodeInput.from_dict(input_data)
    node_name = node_input.node_name

    # Log activity execution for debugging
    print(f"  [ACTIVITY] Executing node '{node_name}' as Dapr activity", flush=True)

    # Get the node function from registry
    node_func = get_registered_node(node_name)
    if node_func is None:
        return ExecuteNodeOutput(
            node_name=node_name,
            error=f"Node '{node_name}' not found in registry",
        ).to_dict()

    # Extract tracing config
    config = node_input.config or {}

    # Collect side-channel writes from CONFIG_KEY_SEND (used by StateBackend
    # to write to the "files" channel from sandbox tools).
    call = _NodeCall(
        ctx=ctx,
        node_input=node_input,
        node_func=node_func,
        state=_node_state(node_input),
        config=config,
        streamer=_make_streamer(ctx, node_input, config, stream_registry),
        pending_sends=[],
    )

    # Run under the LangSmith child trace so LLM calls nest under the parent
    # "LangGraph" trace created by runner.invoke().
    result = _run_with_stream_hooks(call, _langsmith_parent(config))

    return _node_output(call, result)


def _node_state(node_input: ExecuteNodeInput) -> Any:
    """What the node takes as input: the channel state, or a ``Send``'s argument."""
    if node_input.send is None:
        state = _reconstruct_state(node_input.channel_state)
        shape = _node_inputs.get(node_input.node_name)
        return shape.apply(state) if shape else state
    return _decode_arg(node_input.send["arg"])


def make_condition_activity(
    stream_registry: Optional[StreamSinkRegistry] = None,
) -> Callable[[WorkflowActivityContext, Dict[str, Any]], Dict[str, Any]]:
    """Build the conditional-edge activity of a runner (see ``make_node_activity``)."""

    def evaluate_condition(
        ctx: WorkflowActivityContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return evaluate_condition_activity(
            ctx, input_data, stream_registry=stream_registry
        )

    return evaluate_condition


def _call_condition(
    ctx: WorkflowActivityContext,
    cond_input: EvaluateConditionInput,
    cond_func: Callable,
    state: Any,
    registry: Optional[StreamSinkRegistry],
) -> Any:
    """Call a conditional edge's function and return what it routed to.

    In a streamed run the function's LLM tokens and ``get_stream_writer()``
    events are sent as its source node's (that is what LangGraph does), keyed
    apart from the node's own chunks.
    """
    config = dict(cond_input.config or {})
    config[CONF] = dict(config.get(CONF) or {})
    thread_id = (
        cond_input.thread_id
        or config.get("thread_id")
        or getattr(ctx, "workflow_id", None)
    )
    config[CONF].setdefault("thread_id", thread_id)
    streamer = None
    if cond_input.step is not None:
        streamer = _streamer_for(
            ctx,
            cond_input.source_node,
            cond_input.step,
            cond_input.thread_id,
            config,
            registry,
            # one key per edge: a node's several conditional edges are several
            # tasks, and a key they share would read as one retrying
            sink_node=f"{cond_input.source_node}:branch:{cond_input.condition_name}",
        )
    if CONFIG_KEY_RUNTIME not in config[CONF]:
        config[CONF][CONFIG_KEY_RUNTIME] = _build_node_runtime(
            thread_id,
            ctx,
            cond_input.condition_name,
            streamer.writer if streamer else None,
        )
    if streamer:
        config = streamer.patch_config(config)

    # If the condition is a LangChain Runnable (e.g. RunnableCallable from Deep
    # Agents branches), call .invoke() so runtime injection works.
    is_runnable = hasattr(cond_func, "invoke") and callable(
        getattr(cond_func, "invoke")
    )

    def run() -> Any:
        if not is_runnable:
            return _await_if_coroutine(cond_func(state))
        if _is_async_only(cond_func):
            # A coroutine; run to completion
            return _await_if_coroutine(cond_func.ainvoke(state, config=config))  # type: ignore[attr-defined]
        return _await_if_coroutine(cond_func.invoke(state, config=config))  # type: ignore[attr-defined]

    # Runnable.invoke only sets the ambient config when tracing, and a plain
    # function never does: get_stream_writer() and the stream handler need it.
    with set_config_context(config) as ctx_:  # type: ignore[arg-type]
        return ctx_.run(run)


def evaluate_condition_activity(
    ctx: WorkflowActivityContext,
    input_data: Dict[str, Any],
    *,
    stream_registry: Optional[StreamSinkRegistry] = None,
) -> Dict[str, Any]:
    """Activity that evaluates a conditional edge.

    Args:
        ctx: The workflow activity context
        input_data: Dictionary containing EvaluateConditionInput data
        stream_registry: The runner's registry, where the sink of a streamed
            run is (default: the module's).

    Returns:
        EvaluateConditionOutput as a dictionary
    """
    cond_input = EvaluateConditionInput.from_dict(input_data)

    # Get the condition function from registry
    cond_func = get_registered_condition(cond_input.condition_name)
    if cond_func is None:
        return EvaluateConditionOutput(
            error=f"Condition '{cond_input.condition_name}' not found in registry",
        ).to_dict()

    # Reconstruct state from channel values
    state: Any = _reconstruct_state(cond_input.channel_state)
    shape = _condition_inputs.get(cond_input.condition_name)
    if shape:
        state = shape.apply(state)

    # Evaluate under the parent "LangGraph" trace. A Runnable condition (every
    # add_conditional_edges path is one) traces itself, so without a parent
    # each evaluation lands in LangSmith as a separate top-level trace.
    parent = _langsmith_parent(cond_input.config)
    if parent:
        import langsmith as ls

        trace_ctx: Any = ls.tracing_context(parent=parent)
    else:
        trace_ctx = contextlib.nullcontext()

    with trace_ctx:
        result = _call_condition(ctx, cond_input, cond_func, state, stream_registry)

    # Result should be a node name, list of node names, or Send objects.
    # LangGraph Send objects have a .node attribute with the target node name.
    # Any other value goes through the edge's path_map when it has one, as in
    # LangGraph: a router may return labels ("continue") rather than node names.
    path_map = cond_input.path_map
    items = list(result) if isinstance(result, (list, tuple)) else [result]

    next_nodes = []
    sends = []
    for item in items:
        if isinstance(item, Send):
            sends.append(_send_task(item))
        elif path_map is None:
            next_nodes.append(item if isinstance(item, str) else str(item))
        elif str(item) in path_map:
            next_nodes.append(path_map[str(item)])
        else:
            return EvaluateConditionOutput(
                error=(
                    f"Condition '{cond_input.condition_name}' returned {item!r}, "
                    f"which is not in its path_map {sorted(path_map)}"
                ),
            ).to_dict()

    return EvaluateConditionOutput(next_nodes=next_nodes, sends=sends).to_dict()


def _get_triggered_nodes(
    graph_config: GraphConfig,
    channel_state: ChannelState,
    pending_nodes: List[str],
    step: int,
) -> List[str]:
    """Determine which nodes should execute in this step.

    Args:
        graph_config: The graph configuration
        channel_state: Current channel state
        pending_nodes: Nodes explicitly pending execution
        step: Current step number

    Returns:
        List of node names to execute
    """
    # On first step, start with entry point
    if step == 0 and not pending_nodes:
        return [graph_config.entry_point]

    # Return pending nodes if any
    if pending_nodes:
        return pending_nodes

    return []


def _replay_safe_id(instance_id: Any, index: int, channel: str = "messages") -> str:
    """The id of input message ``index`` of ``channel``: same on every replay."""
    key = (
        f"{instance_id}:{index}"
        if channel == "messages"
        else (f"{instance_id}:{channel}:{index}")
    )
    return str(uuid.uuid5(uuid.NAMESPACE_OID, key))


def _message_id(channel: str, index: int) -> str:
    """A fresh id for the message at ``index`` of the input (outside a workflow)."""
    return str(uuid.uuid4())


def _serialize_input(
    input_data: Dict[str, Any],
    message_id: Callable[[str, int], str] = _message_id,
) -> Dict[str, Any]:
    """Serialize input values for channel storage.

    A message written to an ``add_messages`` channel (a message, a string, a
    pair, a dict, or a list of them) is stored as a message dict with an id, as
    the reducer gives it when LangGraph writes the input. A node that returns
    the messages it was handed (a subgraph does) is then understood to return
    the same messages, not new ones: the stream does not send them again and the
    state does not hold them twice.

    ``message_id`` makes the id of the message at an index of a channel.
    Workflow code is replayed from its first line for every decision, so it
    passes an id that is the same on every replay; random ids are for callers
    outside a workflow.
    """
    return {
        channel: (
            _serialize_messages(value, functools.partial(message_id, channel))
            if channel == "messages" or _is_messages_channel(channel)
            else _serialize_value(value)
        )
        for channel, value in input_data.items()
    }


def _extract_output(
    channel_state: ChannelState, output_channels: List[str]
) -> Dict[str, Any]:
    """Extract output values from the specified channels.

    Args:
        channel_state: Current channel state
        output_channels: Channels to extract

    Returns:
        Dictionary of channel name to value
    """
    values = {**channel_state.values, **_unwritten_defaults(channel_state.values)}
    return {
        channel: values[channel] for channel in output_channels if channel in values
    }


class ReducerError(Exception):
    """A channel's reducer raised while a write was applied; the run fails."""


def _apply_write(channel_state: ChannelState, channel: str, value: Any) -> None:
    """Apply a write to a channel, using reducer if available.

    Args:
        channel_state: Channel state to update
        channel: Channel name
        value: Value to write
    """
    reducer = get_channel_reducer(channel)
    if reducer and isinstance(value, dict) and set(value) == {OVERWRITE}:
        # Overwrite replaces the channel's value without calling the reducer.
        channel_state.values[channel] = value[OVERWRITE]
        channel_state.versions[channel] = channel_state.versions.get(channel, 0) + 1
        return
    if reducer and channel not in channel_state.values:
        # The first write is reduced into the channel's default, so add_messages
        # gives its messages ids and objects, as it does natively.
        if channel in _channel_defaults:
            channel_state.values[channel] = copy.deepcopy(_channel_defaults[channel])
    if reducer and channel in channel_state.values:
        # Apply reducer to combine values
        current = channel_state.values[channel]
        if _is_messages_channel(channel):
            # add_messages turns dicts into messages with convert_to_messages,
            # which files a dump's metadata under additional_kwargs; hand it
            # the messages themselves.
            current, value = _as_messages(current), _as_messages(value)
        try:
            channel_state.values[channel] = reducer(current, value)
            # Re-serialize the reducer output so channel state stays
            # JSON-compatible (reducers like add_messages may return
            # BaseMessage objects from dict inputs).
            reduced = channel_state.values[channel]
            if isinstance(reduced, list):
                channel_state.values[channel] = [
                    _serialize_value(item) for item in reduced
                ]
        except Exception as e:
            # LangGraph fails the run on a reducer that raises; overwriting the
            # channel would hide the error and differ between a run and a replay.
            raise ReducerError(f"Reducer failed for channel '{channel}': {e}") from e
    else:
        # No reducer or first value, just set
        channel_state.values[channel] = value

    # Increment version
    channel_state.versions[channel] = channel_state.versions.get(channel, 0) + 1


def _unwritten_defaults(values: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of the default of every reducer channel that ``values`` lacks."""
    return {
        channel: copy.deepcopy(default)
        for channel, default in _channel_defaults.items()
        if channel not in values
    }


def _reconstruct_state(channel_state: ChannelState) -> Dict[str, Any]:
    """Reconstruct the state dictionary from channel values.

    Args:
        channel_state: Channel state

    Returns:
        State dictionary
    """
    serializer = get_serializer()
    state = {}

    for channel, value in channel_state.values.items():
        # Try to deserialize if needed
        if serializer and isinstance(value, (str, bytes)):
            try:
                if isinstance(value, str) and value.startswith(('{"', "[")):
                    import json

                    state[channel] = json.loads(value)
                else:
                    state[channel] = value
            except Exception:
                state[channel] = value
        else:
            state[channel] = value

    # A reducer channel nobody has written yet holds its default, as it does in
    # LangGraph: a node reads ``state["log"]`` as ``[]``, not as a missing key.
    state.update(_unwritten_defaults(channel_state.values))

    # Reconstruct LangChain message objects from serialized dicts.
    # Nodes (e.g. Deep Agent middleware) expect BaseMessage instances, not
    # plain dicts, so every add_messages channel (whatever it is named) is
    # converted back. Items may be {"type": "human", ...} or
    # {"role": "user", ...} dicts.
    for channel, value in state.items():
        if channel == "messages" or _is_messages_channel(channel):
            state[channel] = _as_messages(value)

    return state


_MESSAGE_ADAPTER: TypeAdapter = TypeAdapter(AnyMessage)


def _as_message(item: Any) -> Any:
    """A stored message dict as the message it was dumped from.

    A dump (``type`` and the message's own fields) is validated as the message
    class, which keeps ``response_metadata``, ``usage_metadata`` and the like.
    ``convert_to_messages`` would take those fields for ``additional_kwargs``.
    Anything else (a ``role`` dict, a ``RemoveMessage``) goes through it.
    """
    if isinstance(item, dict) and "type" in item and "role" not in item:
        try:
            return _MESSAGE_ADAPTER.validate_python(item)
        except ValidationError:
            pass
    return convert_to_messages([item])[0]


def _as_messages(value: Any) -> Any:
    """``value``'s message dicts as message objects (left as is if it is not that)."""
    if isinstance(value, list) and value and isinstance(value[0], dict):
        try:
            return [_as_message(item) for item in value]
        except Exception:
            return value
    return value


def _result_to_raw_writes(result: Any) -> List[Tuple[str, Any]]:
    """Convert a node result to ``(channel, value)`` pairs, values untouched.

    This is what LangGraph itself records as the node's channel writes (and
    shows in ``stream_mode="updates"``); :func:`_result_to_writes` serializes
    the same pairs for the workflow state.

    Args:
        result: The result from a node function

    Returns:
        ``(channel, value)`` pairs in write order
    """
    if result is None:
        return []

    # Handle LangGraph Command objects (used by Deep Agents and newer LangGraph)
    # Single Command
    if isinstance(result, Command):
        return _command_to_raw_writes(result)

    # List of Commands
    if isinstance(result, list) and result and isinstance(result[0], Command):
        writes: List[Tuple[str, Any]] = []
        for cmd in result:
            writes.extend(_command_to_raw_writes(cmd))
        return writes

    if isinstance(result, dict):
        return [(k, v) for k, v in result.items() if _keep_channel(k)]

    # An instance of a state schema (a Pydantic model or a dataclass): LangGraph
    # writes its fields.
    fields = _state_object_writes(result)
    if fields is not None:
        return fields

    # Non-dict results - write to a default channel
    return [("__result__", result)]


def _state_object_writes(result: Any) -> Optional[List[Tuple[str, Any]]]:
    """``(field, value)`` pairs of a Pydantic model or dataclass instance.

    The rule LangGraph applies to a node that returns an instance of its state
    schema (``get_update_as_tuples``): every field is a write, except that for a
    Pydantic model a field whose value and default are both ``None`` is left out
    unless the node set it. ``None`` for anything else.
    """
    if isinstance(result, BaseModel):
        model = type(result)
        keep = result.model_fields_set
        return [
            (name, value)
            for name, info in model.model_fields.items()
            for value in [getattr(result, name, None)]
            if value is not None or info.default is not None or name in keep
        ]
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return [(f.name, getattr(result, f.name)) for f in dataclasses.fields(result)]
    return None


def _encode_arg(arg: Any) -> Dict[str, Any]:
    """A ``Send`` argument as JSON that gives the same object back.

    The node takes the argument as its input, as LangGraph hands it over, so
    messages inside it must stay messages: the checkpoint serializer keeps them.
    """
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        kind, data = JsonPlusSerializer().dumps_typed(arg)
        return {"typed": [kind, base64.b64encode(data).decode("ascii")]}
    except ImportError:  # pragma: no cover - the extra is always installed
        return {"json": _serialize_value(arg)}


def _decode_arg(encoded: Dict[str, Any]) -> Any:
    if "typed" in encoded:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        kind, data = encoded["typed"]
        return JsonPlusSerializer().loads_typed((kind, base64.b64decode(data)))
    return encoded.get("json")


def _send_task(send: Any) -> Dict[str, Any]:
    """A ``Send`` as JSON: the node to run and the argument it takes as input."""
    return {"node": send.node, "arg": _encode_arg(send.arg)}


def _result_to_gotos(result: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
    """The nodes and ``Send`` tasks a ``Command(goto=...)`` result asks to run next.

    ``END`` is skipped (it ends that branch, which running nothing does). Every
    ``Send`` is a task of its own, with its argument as the node's input.
    """
    if isinstance(result, Command):
        commands: List[Any] = [result]
    elif isinstance(result, list) and result and isinstance(result[0], Command):
        commands = result
    else:
        return [], []
    targets: List[str] = []
    sends: List[Dict[str, Any]] = []
    for cmd in commands:
        goto = getattr(cmd, "goto", None) or ()
        for item in [goto] if isinstance(goto, (str, Send)) else goto:
            if isinstance(item, Send):
                sends.append(_send_task(item))
                continue
            name = str(item)
            if name != END and name not in targets:
                targets.append(name)
    return targets, sends


def _command_to_raw_writes(cmd: Any) -> List[Tuple[str, Any]]:
    """Extract ``(channel, value)`` pairs from the ``update`` of a Command.

    This helper only inspects ``cmd.update``; ``goto`` is read by
    :func:`_result_to_gotos`.

    Args:
        cmd: A LangGraph Command instance whose ``update`` field encodes channel writes.

    Returns:
        ``(channel, value)`` pairs derived from ``cmd.update``
    """
    update = getattr(cmd, "update", None)
    if update is None:
        return []

    if isinstance(update, dict):
        return [(k, v) for k, v in update.items() if _keep_channel(k)]
    if isinstance(update, (list, tuple)):
        # update can be a list of (channel, value) tuples
        return [
            (item[0], item[1])
            for item in update
            if isinstance(item, tuple) and len(item) == 2 and _keep_channel(item[0])
        ]
    return []


def _is_messages_channel(channel: str) -> bool:
    """Whether ``channel`` is reduced by ``add_messages`` (whatever it is named)."""
    return get_channel_reducer(channel) is add_messages


def _new_message_id(index: int) -> str:
    """A fresh id for a message a node wrote (the activity's result is recorded)."""
    return str(uuid.uuid4())


def _serialize_messages(
    value: Any, message_id: Callable[[int], str] = _new_message_id
) -> List[Any]:
    """What ``add_messages`` takes as one write, as JSON message dicts with ids.

    Mirrors the reducer: a single message is a list of one, and each item (a
    string, a ``(role, content)`` pair, a dict or a message) becomes a message.
    A message without an id gets ``message_id(index)`` here, because
    ``add_messages`` would otherwise make a random one inside the orchestrator,
    different on every replay.
    """
    items = value if isinstance(value, list) else [value]
    messages = [message_chunk_to_message(m) for m in convert_to_messages(items)]
    return [
        _serialize_value(
            m if m.id is not None else m.model_copy(update={"id": message_id(i)})
        )
        for i, m in enumerate(messages)
    ]


def _overwrite_of(value: Any) -> Tuple[bool, Any]:
    """Whether ``value`` is an ``Overwrite`` (in any form LangGraph takes), and its value."""
    if isinstance(value, Overwrite):
        return True, value.value
    if isinstance(value, dict):
        if len(value) == 1 and OVERWRITE in value:
            return True, value[OVERWRITE]
        if value.get("type") == OVERWRITE and "value" in value:
            return True, value["value"]
    return False, None


def _serialize_channel_value(
    channel: str,
    value: Any,
    message_id: Callable[[int], str] = _new_message_id,
) -> Any:
    """``value`` as the JSON a write to (or an input of) ``channel`` is stored as.

    An ``Overwrite`` is kept as ``{"__overwrite__": value}`` so that applying the
    write can bypass the reducer (``_apply_write``).
    """
    is_overwrite, inner = _overwrite_of(value)
    if is_overwrite:
        return {OVERWRITE: _serialize_channel_value(channel, inner, message_id)}
    if _is_messages_channel(channel):
        return _serialize_messages(value, message_id)
    return _serialize_value(value)


def _serialize_writes(raw_writes: List[Tuple[str, Any]]) -> List[NodeWrite]:
    return [
        NodeWrite(channel=channel, value=_serialize_channel_value(channel, value))
        for channel, value in raw_writes
    ]


def _result_to_writes(result: Any) -> List[NodeWrite]:
    """Convert a node result to a list of serialized writes.

    Args:
        result: The result from a node function

    Returns:
        List of NodeWrite objects
    """
    return _serialize_writes(_result_to_raw_writes(result))


def _serialize_value(value: Any) -> Any:
    """Serialize a value for storage.

    Args:
        value: Value to serialize

    Returns:
        Serialized value (JSON-compatible)
    """
    # Handle common types
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]

    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}

    # Try common serialization methods
    if hasattr(value, "model_dump"):
        # Pydantic v2
        return value.model_dump()
    if hasattr(value, "dict"):
        # Pydantic v1
        return value.dict()
    if hasattr(value, "to_dict"):
        return value.to_dict()

    # Handle LangChain messages
    try:
        from langchain_core.messages import BaseMessage

        if isinstance(value, BaseMessage):
            return {
                "type": value.type,
                "content": value.content,
                "additional_kwargs": value.additional_kwargs,
                "id": value.id,
            }
    except ImportError:
        pass

    # Fall back to string representation
    return str(value)
