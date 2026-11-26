# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr Workflow definitions for durable LangGraph execution."""

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
    Union,
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
from langchain_core.runnables.config import set_config_context
from langgraph.graph.message import add_messages
from pydantic import BaseModel, TypeAdapter, ValidationError
from langgraph.runtime import Runtime
from langgraph.errors import GraphInterrupt, ParentCommand
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
    StreamSink,
    StreamSinkRegistry,
    StreamTransport,
)

from .activity_stream import (
    activity_writer,
    publish_run_status,
    transport_of,
)
from .interrupts import (
    ROUTER_INTERRUPT_ERROR,
    DroppedInterrupt,
    TaskResumes,
    interrupt_config,
    interrupt_payloads,
    nested_interrupts,
    router_interrupt_conf,
    settle_interrupts,
    without_resume_writes,
)
from .node_routes import route_in_node
from .runnable_call import (  # noqa: F401 (re-exported)
    _BackgroundLoop,
    _await_if_coroutine,
    _background_loop,
    _has_async_entry,
    _invoke_runnable,
)
from .stream_attempts import record_attempts, with_attempt
from .stream_values import state_values
from .streaming import STREAM_RUN_KEY, NodeStreamer, StaticConfig
from .task_identity import TaskIdentity
from .thread_state import THREAD_STATE_PATCH, load_thread, save_thread
from .values import (
    coerce_to_default,
    decode_value,
    encode_value,
    is_encoded,
    is_plain_json,
)
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
# The graph's state channels: a node's other keys are dropped, not written.
# ``None`` until a runner registers them (then nothing is dropped).
_state_channels: Optional[frozenset] = None
# What a node and a conditional edge are handed: the channels of its input schema
# and the function that coerces the state dict to a schema class (pydantic or
# dataclass state). Conditions are keyed by their name.
_node_inputs: Dict[str, "_InputShape"] = {}
_condition_inputs: Dict[str, "_InputShape"] = {}
# The managed values the graph's state asks for (name -> ``REMAINING_STEPS`` or
# ``IS_LAST_STEP``); they are given to nodes and routers, never stored.
_managed_channels: Dict[str, str] = {}
_serializer: Optional[Any] = None
_default_graph_config: Optional["GraphConfig"] = None
_default_input_mapper: Optional[Callable] = None
_default_max_steps: int = 100
_default_strict_limit: bool = False
# The config a run started without one gets: the graph's own (``with_config``).
_default_run_config: Optional[Dict[str, Any]] = None
# Every trigger channel of each node, for ``langgraph_triggers`` in a task's metadata.
_node_triggers: Dict[str, Tuple[str, ...]] = {}
# The metadata and tags the graph and each node give their tasks (see StaticConfig).
_node_static: Dict[str, StaticConfig] = {}

# The managed values of LangGraph's own that are supplied (``langgraph.managed``).
REMAINING_STEPS = "remaining_steps"
IS_LAST_STEP = "is_last_step"

# The key a write that replaces a reducer channel's value is stored under.
OVERWRITE = "__overwrite__"

# The one channel of a graph whose whole state is a single value
# (``StateGraph(Annotated[list, operator.add])``).
ROOT = "__root__"

# Special node names
START = "__start__"
END = "__end__"

# Each node runs as an activity named after the node, so the workflow UI shows
# the graph's own step names. A replay must schedule what the history holds, so
# an instance that scheduled every node as ``execute_node_activity`` keeps doing
# so: the patch is what tells the two apart.
NODE_ACTIVITY_NAMES_PATCH = "langgraph-node-activity-names"
# A step's nodes run sorted by name (LangGraph's task order) and END does not
# end the run while other branches of the step are pending. An instance that
# scheduled its nodes in the order the edges were found, and ended at END, must
# replay that way: the patch is what tells the two apart.
STEP_ORDER_PATCH = "langgraph-step-order"
# A node's conditional edges run inside the node's activity, as LangGraph runs a
# router inside its node's task. An instance that scheduled a condition activity
# after each node must replay that way: the patch is what tells the two apart.
ROUTE_IN_NODE_PATCH = "langgraph-route-in-node"
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
    stream: Union[StreamTransport, StreamSinkRegistry, None] = None,
) -> Callable[[WorkflowActivityContext, Dict[str, Any]], Dict[str, Any]]:
    """Build the activity registered for one node.

    Dapr registers each function object once, so every node gets its own
    wrapper around ``execute_node_activity``. ``stream`` is the runner's
    stream transport (a registry stands for the in-process one over it;
    default: the module's registry).
    """
    transport = transport_of(stream)

    def node_activity(
        ctx: WorkflowActivityContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return execute_node_activity(ctx, input_data, stream_transport=transport)

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
    write is reduced into it. Without one it is stored as is.
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
    #: The state is one channel (``__root__``) and the node takes its value.
    root: bool = False

    def apply(self, state: Dict[str, Any]) -> Any:
        """``state`` cut to the channels and coerced to the schema class."""
        if self.root:
            return state.get(ROOT)
        channels = self.channels
        if channels is not None:
            state = {k: v for k, v in state.items() if k in channels}
        return self.mapper(state) if self.mapper else state


def _input_shape(channels: Any, mapper: Optional[Callable]) -> _InputShape:
    """The shape for an input schema's ``channels`` (a single root channel is not cut)."""
    names = [channels] if isinstance(channels, str) else list(channels or ())
    keep = tuple(names) if names and ROOT not in names else None
    return _InputShape(channels=keep, mapper=mapper, root=ROOT in names)


def register_node_input(name: str, channels: Any, mapper: Optional[Callable]) -> None:
    """Record what node ``name`` is handed as its state."""
    _node_inputs[name] = _input_shape(channels, mapper)


def register_condition_input(
    name: str, channels: Any, mapper: Optional[Callable]
) -> None:
    """Record what the conditional edge ``name`` is handed as its state."""
    _condition_inputs[name] = _input_shape(channels, mapper)


def register_managed_channel(name: str, kind: str) -> None:
    """Record that state key ``name`` is a managed value of ``kind``.

    ``kind`` is ``REMAINING_STEPS`` or ``IS_LAST_STEP``.
    """
    _managed_channels[name] = kind


def _with_managed(
    state: Dict[str, Any], recursion_limit: Optional[int], step: Optional[int]
) -> Dict[str, Any]:
    """``state`` with the managed values of a task at workflow step ``step``.

    LangGraph counts ``RemainingSteps`` down from the limit: ``limit - 1 - step``
    for a task whose step is numbered from 0, and ``IsLastStep`` is true when one
    remains. Nothing is added when the workflow did not say what the limit is.
    """
    if not _managed_channels or recursion_limit is None or step is None:
        return state
    remaining = recursion_limit - 1 - step
    values = {REMAINING_STEPS: remaining, IS_LAST_STEP: remaining == 1}
    return {**state, **{n: values[k] for n, k in _managed_channels.items()}}


def register_state_channels(channels: Iterable[str]) -> None:
    """Record the graph's state channels, the only keys a node can write."""
    global _state_channels
    _state_channels = frozenset(channels)


def _keep_channel(channel: str) -> bool:
    """Whether a node's key is a state channel (everything is, before any is set)."""
    names = _state_channels
    return names is None or ROOT in names or channel in names


def _is_root_state() -> bool:
    """Whether the graph's whole state is the one ``__root__`` channel."""
    return _state_channels is not None and ROOT in _state_channels


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
    strict_limit: bool = False,
    run_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Store the default graph config for orchestrator-initiated calls.

    ``run_config`` is the LangGraph config such a run starts with: what a runner
    entry point makes of the config bound to the graph (``with_config``).
    """
    global _default_graph_config, _default_input_mapper, _default_max_steps
    global _default_run_config, _default_strict_limit
    _default_graph_config = config
    _default_input_mapper = input_mapper
    _default_max_steps = max_steps
    _default_strict_limit = strict_limit
    _default_run_config = copy.deepcopy(run_config) if run_config else None


def _close_langsmith_parent_trace_async(config: Optional[Dict[str, Any]]) -> None:
    """Close the parent LangSmith trace in a background thread (non-blocking)."""
    if not config:
        return
    run_id = config.get("langsmith_run_id")
    if not run_id:
        return

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
    _managed_channels.clear()
    _node_inputs.clear()
    _condition_inputs.clear()
    global _serializer, _default_graph_config, _default_input_mapper, _default_max_steps
    global _default_run_config, _default_strict_limit
    _default_run_config = None
    _default_strict_limit = False
    _serializer = None
    _default_graph_config = None
    _default_input_mapper = None
    _default_max_steps = 100


def _workflow_input(
    ctx: DaprWorkflowContext, input_data: Dict[str, Any]
) -> GraphWorkflowInput:
    """The workflow's input: a full ``GraphWorkflowInput`` or a ``{"task": ...}``."""
    if "graph_config" in input_data:
        return GraphWorkflowInput.from_dict(input_data)

    # An orchestrator starts the workflow with just {"task": "..."}.
    if _default_graph_config is None:
        raise ValueError(
            "Received simple task input but no default graph config is set. "
            "Ensure the runner has been started before the workflow is invoked."
        )
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
        strict_limit=_default_strict_limit,
        config=_orchestrated_config(str(ctx.instance_id)),
        thread_id=str(ctx.instance_id),
    )


def _orchestrated_config(thread_id: str) -> Optional[Dict[str, Any]]:
    """The config of a run an orchestrator started: the graph's own, if it has one."""
    if _default_run_config is None:
        return None
    return {**copy.deepcopy(_default_run_config), "thread_id": thread_id}


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
    #: Whether a step's nodes run sorted by name (see :data:`STEP_ORDER_PATCH`).
    step_order: bool
    #: The run's bound as a LangGraph ``recursion_limit`` (what ``RemainingSteps``
    #: counts against): the runner's own ``max_steps`` lets a run of exactly that
    #: many steps complete, which is a limit one above it.
    recursion_limit: Optional[int] = None
    #: Whether a node's activity evaluates its conditional edges
    #: (see :data:`ROUTE_IN_NODE_PATCH`).
    route_in_node: bool = False
    #: Whether the run starts from its thread's state and saves its own as the
    #: thread's (see :data:`THREAD_STATE_PATCH`).
    thread_state: bool = False
    #: The nodes that ran in each step, reported on the output so a streaming
    #: consumer can tell which node's chunks never arrived.
    executed: List[List[str]] = field(default_factory=list)
    #: Deferred nodes that were triggered and wait for nothing else to be pending.
    held: List[str] = field(default_factory=list)
    #: The conditional edges evaluated in their own activity, ``[step, sink
    #: key]``, so a streaming consumer can tell which router's chunks never
    #: arrived.
    branches: List[List[Any]] = field(default_factory=list)
    #: The interrupt the run waits on (``{"seq", "step", "interrupts"}``).
    interrupt: Optional[Dict[str, Any]] = None
    #: How many times the run was interrupted so far.
    interrupt_seq: int = 0
    #: The resume the run took last, ``{"seq", "id"}``.
    resumed: Optional[Dict[str, Any]] = None
    #: The attempt of each unit's output the run used (see ``stream_attempts``).
    attempts: Dict[str, str] = field(default_factory=dict)

    @property
    def output_channels(self) -> Optional[List[str]]:
        """What an activity of a streamed run sends ``values`` chunks of."""
        if not (self.config or {}).get(STREAM_RUN_KEY):
            return None
        return list(self.graph_config.output_channels)

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
            branches=self.branches or None,
            attempts=self.attempts or None,
            status=status,
            error=error or None,
        ).to_dict()


#: The nodes, the ``Send`` tasks and the error a search for the next step ends with.
_StartOrNext = Tuple[List[str], List[Dict[str, Any]], Optional[str]]


def _new_run(ctx: DaprWorkflowContext, workflow_input: GraphWorkflowInput) -> _Run:
    """The shared state of one ``agent_workflow`` execution."""
    # Ask before the first yield: a replay of an older instance must see the
    # patch as absent, and the answer is fixed for the rest of the execution.
    node_activities_by_name = ctx.is_patched(NODE_ACTIVITY_NAMES_PATCH)
    step_order = ctx.is_patched(STEP_ORDER_PATCH)
    route_in_node = ctx.is_patched(ROUTE_IN_NODE_PATCH)
    thread_state = workflow_input.thread_state and ctx.is_patched(THREAD_STATE_PATCH)
    return _Run(
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
        step_order=step_order,
        recursion_limit=(
            workflow_input.max_steps
            if workflow_input.strict_limit
            else workflow_input.max_steps + 1
        ),
        route_in_node=route_in_node,
        thread_state=thread_state,
    )


def agent_workflow(
    ctx: DaprWorkflowContext, input_data: Dict[str, Any]
) -> Generator[Any, Any, Dict[str, Any]]:
    """Dapr Workflow that runs a LangGraph graph step by step (Pregel's BSP model).

    Each step runs the triggered nodes in parallel as activities, applies their
    writes to the channels, and finds the next step's nodes from the edges. The
    run ends when nothing is triggered, END is reached or a step bound is hit;
    the result is a ``GraphWorkflowOutput`` as a dictionary.

    A run on a thread with a checkpointer starts from the thread's last state
    with the input applied to it, and keeps the state it ends with as the
    thread's: both through activities (see ``thread_state``).
    """
    workflow_input = _workflow_input(ctx, input_data)
    run = _new_run(ctx, workflow_input)
    thread_id = str(run.thread_id or ctx.instance_id)
    if run.thread_state:
        run.channel_state = yield from load_thread(
            ctx, thread_id, run.channel_state, run.retry_policy
        )
    output = yield from _run_graph(run, workflow_input)
    if run.thread_state:
        output = yield from save_thread(ctx, thread_id, output, run.retry_policy)
    return output


def _run_graph(
    run: _Run, workflow_input: GraphWorkflowInput
) -> Generator[Any, Any, Dict[str, Any]]:
    """Run the graph's steps until it ends; the workflow output as a dict."""
    max_steps = workflow_input.max_steps
    step = 0

    pending_nodes, pending_sends, error = yield from _start_nodes(run)
    if error:
        return run.finish(step, "error", f"Condition evaluation failed: {error}")
    pending_nodes = _hold_deferred(run, pending_nodes, pending_sends)
    # Joins: how many of each waiting edge's sources have run since it last fired.
    joined: List[Set[str]] = [set() for _ in run.graph_config.waiting_edges]

    while True:
        _log_step(run, step, pending_nodes)
        nodes_to_execute, run_ends = _step_nodes(
            pending_nodes, pending_sends, run.step_order
        )
        # A recursion_limit is checked before the end of the run: a run that
        # has taken that many steps fails, whatever follows.
        if workflow_input.strict_limit and step >= max_steps:
            break
        if run_ends:
            return run.finish(step, "completed")

        # The runner's own bound is checked after the end of the run, so a graph
        # that needs exactly ``max_steps`` steps completes instead of being cut
        # at the end.
        if step >= max_steps:
            break

        pending_nodes, pending_sends, error = yield from _run_step(
            run, nodes_to_execute, pending_sends, joined, step
        )
        if error:
            return run.finish(step, "error", error)
        pending_nodes = _hold_deferred(run, pending_nodes, pending_sends)
        step += 1

    return run.finish(step, "max_steps_reached", f"Max steps ({max_steps}) reached")


def _log_step(run: _Run, step: int, pending_nodes: List[str]) -> None:
    """Print the step about to run, once: not again on a replay of it."""
    if not run.ctx.is_replaying:
        print(f"  [WORKFLOW] Step {step}, pending_nodes={pending_nodes}", flush=True)


def _run_step(
    run: _Run,
    nodes: List[str],
    sends: List[Dict[str, Any]],
    joined: List[Set[str]],
    step: int,
) -> Generator[Any, Any, _StartOrNext]:
    """Run a step's nodes and ``Send`` tasks, apply their writes, find the next step.

    Returns the next step's nodes and ``Send`` tasks, and the message the run
    ends with if a node failed, a reducer raised or a condition failed.
    """
    run.executed.append(
        [*nodes, *[send_key(t["node"], slot) for slot, t in enumerate(sends)]]
    )
    publish_run_status(run)
    outputs = yield from _run_nodes(run, nodes, sends, step)
    outputs = yield from settle_interrupts(
        run,
        outputs,
        step,
        lambda resumes: _rerun_nodes(run, nodes, sends, step, resumes),
        publish_run_status,
    )
    record_attempts(run.attempts, step, run.executed[-1], outputs)
    failed = next((o for o in outputs if o.error), None)
    if failed:
        return [], [], f"Node '{failed.node_name}' failed: {failed.error}"

    tasks = [
        _Task(
            node=output.node_name,
            output=output,
            slot=i - len(nodes) if i >= len(nodes) else None,
        )
        for i, output in enumerate(outputs)
    ]
    # A conditional edge reads the state with only its own task's writes
    # applied, so a step that runs several tasks keeps the state they start
    # from.
    before = _state_before_step(run, tasks)
    try:
        run.channel_state.updated_channels = _apply_outputs(run.channel_state, outputs)
    except ReducerError as e:
        return [], [], str(e)
    next_nodes, next_sends, error = yield from _next_nodes(
        run, tasks, before, joined, step
    )
    return (
        next_nodes,
        next_sends,
        f"Condition evaluation failed: {error}" if error else None,
    )


def _step_nodes(
    pending_nodes: List[str], pending_sends: List[Dict[str, Any]], step_order: bool
) -> Tuple[List[str], bool]:
    """The nodes to run in a step, and whether the run ends instead.

    END ends the run only when nothing else is pending: another branch of the
    step may still have nodes to run. The nodes run sorted by
    name (LangGraph's task order), so the order their writes apply in does not
    depend on the order the edges were found in.

    An instance started before :data:`STEP_ORDER_PATCH` (``step_order`` false)
    ends at END and runs the nodes in the order they were found: its history
    holds activities scheduled that way, and a replay must schedule the same.
    """
    if step_order:
        nodes = sorted(n for n in pending_nodes if n != END)
        return nodes, not nodes and not pending_sends
    nodes = [n for n in pending_nodes if n != END]
    return nodes, (END in pending_nodes or not nodes) and not pending_sends


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


@dataclass
class _Frontier:
    """The nodes and ``Send`` tasks found to run next, in the order found."""

    nodes: List[str] = field(default_factory=list)
    sends: List[Dict[str, Any]] = field(default_factory=list)

    def add_nodes(self, names: Iterable[str]) -> None:
        for name in names:
            if name not in self.nodes:
                self.nodes.append(name)

    def result(self, error: Optional[str] = None) -> _StartOrNext:
        return self.nodes, self.sends, error


def _start_nodes(run: _Run) -> Generator[Any, Any, _StartOrNext]:
    """The nodes (and ``Send`` tasks) the START edges lead to, and a condition error.

    A conditional START edge is evaluated on the input.
    """
    graph_config = run.graph_config
    if graph_config.entry_points is None:  # an input persisted without entry_points
        return [graph_config.entry_point], [], None
    found = _Frontier(nodes=list(graph_config.entry_points))
    for edge in graph_config.edges:
        if edge.source != START or not edge.condition:
            continue
        # step -1: before the first node, so the first node is step 1
        error = yield from _follow_edge(
            run, found, edge, START, run.channel_state, -1, None
        )
        if error:
            return found.result(error)
    return found.result()


def _hold_deferred(
    run: _Run, pending_nodes: List[str], pending_sends: List[Dict[str, Any]]
) -> List[str]:
    """The nodes to run next, with deferred ones held back until nothing else is.

    A node added with ``defer=True`` runs once, when no other node or ``Send``
    task is pending (the end of the run, unless it starts more), however many
    of its sources ran before.
    """
    deferred = set(run.graph_config.deferred)
    if not deferred:
        return pending_nodes
    ready = []
    for node in pending_nodes:
        if node not in deferred:
            ready.append(node)
        elif node not in run.held:
            run.held.append(node)
    if not any(n != END for n in ready) and not pending_sends and run.held:
        ready, run.held = [*ready, *run.held], []
    return ready


@dataclass
class _Task:
    """One node (or ``Send``) execution of a step, with what it returned."""

    node: str
    output: ExecuteNodeOutput
    #: Position among the step's ``Send`` tasks; ``None`` for a node's own run.
    slot: Optional[int] = None


def _state_before_step(run: _Run, tasks: List["_Task"]) -> Optional[ChannelState]:
    """A copy of the channel state to give each task's routers, if they need one.

    With one task, the state after its writes is what its routers read, so
    nothing is kept. With several, a task's routers must not see its siblings'
    writes, so the state before the step's writes is kept. A task whose
    activity evaluated its routers needs nothing.
    """
    if len(tasks) < 2:
        return None
    has_router = any(
        e.condition and e.source == t.node
        for t in tasks
        if not t.output.routes_done
        for e in run.graph_config.edges
    )
    return copy.deepcopy(run.channel_state) if has_router else None


def _state_for_task(
    run: _Run, before: Optional[ChannelState], task: _Task
) -> ChannelState:
    """The state a task's conditional edges read: the step's start plus its writes."""
    if before is None:
        return run.channel_state
    state = copy.deepcopy(before)
    for write in task.output.writes:
        _apply_write(state, write.channel, write.value)
    return state


def _invocations(
    nodes: List[str], sends: List[Dict[str, Any]]
) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """A step's tasks in order: its nodes, then its ``Send`` tasks."""
    return [
        *[(name, None) for name in nodes],
        *[(t["node"], {"arg": t["arg"], "slot": slot}) for slot, t in enumerate(sends)],
    ]


def _call_node(
    run: _Run,
    node_name: str,
    send: Optional[Dict[str, Any]],
    step: int,
    resume: Optional[List[Any]] = None,
) -> Any:
    """Schedule one task of ``step`` as its activity."""
    node_input = ExecuteNodeInput(
        node_name=node_name,
        channel_state=run.channel_state,
        config=run.config,
        thread_id=run.thread_id,
        step=step,
        send=send,
        recursion_limit=run.recursion_limit,
        routes=_node_routes(run, node_name) if run.route_in_node else None,
        output_channels=run.output_channels,
        resume=resume,
    )
    activity: Any = execute_node_activity
    if run.node_activities_by_name:
        activity = node_activity_name(node_name)
    return run.ctx.call_activity(
        activity, input=node_input.to_dict(), retry_policy=run.retry_policy
    )


def _run_nodes(
    run: _Run,
    nodes_to_execute: List[str],
    sends: List[Dict[str, Any]],
    step: int,
) -> Generator[Any, Any, List[ExecuteNodeOutput]]:
    """Run the nodes and ``Send`` tasks of one step in parallel, each as an activity."""
    node_tasks = [
        _call_node(run, name, send, step)
        for name, send in _invocations(nodes_to_execute, sends)
    ]
    task_results = yield when_all(node_tasks)
    return [ExecuteNodeOutput.from_dict(result) for result in task_results]


def _rerun_nodes(
    run: _Run,
    nodes: List[str],
    sends: List[Dict[str, Any]],
    step: int,
    resumes: Dict[int, List[Any]],
) -> Generator[Any, Any, Dict[int, ExecuteNodeOutput]]:
    """Run the step's tasks at ``resumes``' positions again, with their resume values."""
    invocations = _invocations(nodes, sends)
    positions = sorted(resumes)
    node_tasks = [
        _call_node(run, *invocations[i], step, resume=resumes[i]) for i in positions
    ]
    task_results = yield when_all(node_tasks)
    return {
        i: ExecuteNodeOutput.from_dict(result)
        for i, result in zip(positions, task_results)
    }


def _node_routes(run: _Run, node_name: str) -> Optional[List[Dict[str, Any]]]:
    """The conditional edges of ``node_name``, for its activity to evaluate."""
    routes = [
        {"condition": edge.condition, "path_map": edge.path_map}
        for edge in run.graph_config.edges
        if edge.source == node_name and edge.condition
    ]
    return routes or None


def _next_nodes(
    run: _Run,
    tasks: List[_Task],
    before: Optional[ChannelState],
    joined: List[Set[str]],
    step: int,
) -> Generator[Any, Any, _StartOrNext]:
    """The nodes and ``Send`` tasks the next step runs, and a condition error.

    They follow from the edges of the tasks that ran, the joins those tasks
    complete, and the ``Command(goto=...)`` they returned. A conditional edge is
    evaluated once per task, on the state with only that task's writes (see
    :func:`_state_for_task`). Tasks are taken in the order LangGraph applies
    them, and each one's ``goto`` before its conditional edges, so the ``Send``
    tasks come out numbered as LangGraph numbers them. A task whose activity
    evaluated its conditional edges reports where they lead.
    """
    found = _Frontier()
    for task in tasks:
        found.add_nodes(task.output.goto)
        found.sends.extend(task.output.sends)
        for edge in run.graph_config.edges:
            if edge.source != task.node:
                continue
            if not edge.condition:
                found.add_nodes([edge.target])
                continue
            if task.output.routes_done:
                continue
            try:
                state = _state_for_task(run, before, task)
            except ReducerError as e:
                return found.result(str(e))
            error = yield from _follow_edge(
                run, found, edge, task.node, state, step, task.slot
            )
            if error:
                return found.result(error)
        if task.output.routes_done:
            if task.output.route_error:
                return found.result(task.output.route_error)
            found.add_nodes(task.output.next_nodes)
            found.sends.extend(task.output.route_sends)

    # A join fires, once, in the step after the last of its sources ran.
    ran = {task.node for task in tasks}
    for seen, join in zip(joined, run.graph_config.waiting_edges):
        seen.update(n for n in ran if n in join.sources)
        if seen.issuperset(join.sources):
            seen.clear()
            found.add_nodes([join.target])
    return found.result()


def _follow_edge(
    run: _Run,
    found: _Frontier,
    edge: Any,
    source_node: str,
    channel_state: ChannelState,
    step: int,
    send_slot: Optional[int],
) -> Generator[Any, Any, Optional[str]]:
    """Evaluate a conditional edge as an activity and add where it leads to ``found``.

    Returns the condition's error, if it failed. ``send_slot`` is the position
    of the source task among the step's ``Send`` tasks, when it is one.
    """
    cond_input = EvaluateConditionInput(
        source_node=source_node,
        condition_name=edge.condition,
        channel_state=channel_state,
        config=run.config,
        thread_id=run.thread_id,
        path_map=edge.path_map,
        step=step,
        send_slot=send_slot,
        recursion_limit=run.recursion_limit,
        # a START edge reads the input: the state of the run's first values chunk
        output_channels=run.output_channels if source_node == START else None,
    )
    run.branches.append([step, _branch_key(cond_input)])
    publish_run_status(run)
    # By name, so the runner's own activity (the one that knows its stream
    # registry) runs, not the module-level function.
    cond_result = yield run.ctx.call_activity(
        CONDITION_ACTIVITY,
        input=cond_input.to_dict(),
        retry_policy=run.retry_policy,
    )
    output = EvaluateConditionOutput.from_dict(cond_result)
    record_attempts(run.attempts, step, [_branch_key(cond_input)], [output])
    if output.error:
        return output.error
    found.add_nodes(output.next_nodes)
    found.sends.extend(output.sends)
    return None


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
    #: Where the node's routers write their chunks, in a streamed run.
    sink: Optional[StreamSink] = None
    #: The task's resume values, shared by the node, its subgraphs and routers.
    resumes: TaskResumes = field(default_factory=TaskResumes)

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
        _read_activity_channels,
        pending_sends,
        {name: decode_value(v) for name, v in channel_state.values.items()},
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


def _is_runnable(obj: Any) -> bool:
    """Whether ``obj`` is a LangChain Runnable, whose ``invoke`` injects config."""
    return hasattr(obj, "invoke") and callable(getattr(obj, "invoke"))


def _node_config(
    base: Optional[Dict[str, Any]],
    *,
    thread_id: Optional[str],
    ctx: WorkflowActivityContext,
    name: str,
    streamer: Optional[NodeStreamer],
    identity: Optional[TaskIdentity] = None,
) -> Dict[str, Any]:
    """The config a node or conditional edge runs under, built from ``base``.

    ``base`` is copied, never written back into, as it is the config handed to
    the activity. The result carries the thread id where middleware reads it
    (``configurable["thread_id"]``), a ``Runtime`` (deep agent middleware
    requires one, and sandbox/state backends resolve the owning workflow from
    ``runtime.execution_info.thread_id``) and, in a streamed run, the stream
    handler. ``identity`` sets the task's ``langgraph_*`` metadata, streamed
    run or not, as the Pregel loop sets it on every task.
    """
    config = dict(base) if base else {}
    config[CONF] = dict(config.get(CONF) or {})
    resolved = thread_id or config.get("thread_id") or getattr(ctx, "workflow_id", None)
    config[CONF].setdefault("thread_id", resolved)
    if identity is not None:
        static = get_node_static_config(identity.node) or StaticConfig()
        config["metadata"] = identity.config_metadata(config, static, resolved)
    if CONFIG_KEY_RUNTIME not in config[CONF]:
        config[CONF][CONFIG_KEY_RUNTIME] = _build_node_runtime(
            resolved, ctx, name, streamer.writer if streamer else None
        )
    return streamer.patch_config(config) if streamer else config


def _task_identity(
    ctx: WorkflowActivityContext, node_name: str, step: int, send_index: Optional[int]
) -> TaskIdentity:
    """The task a node (or one of its routers) runs as, as Pregel names it."""
    return TaskIdentity(
        step=step,
        node=node_name,
        task_id=getattr(ctx, "task_id", ""),
        triggers=_task_triggers(node_name, send_index),
        path=None if send_index is None else (PUSH_TRIGGER, send_index, False),
    )


def _call_identity(call: _NodeCall) -> TaskIdentity:
    """The task this node execution runs as."""
    send = call.node_input.send
    return _task_identity(
        call.ctx,
        call.node_name,
        call.node_input.step or 0,
        None if send is None else send["slot"],
    )


def _runnable_config(call: _NodeCall) -> Dict[str, Any]:
    """The config a Runnable node is invoked with."""
    config = _node_config(
        call.config,
        thread_id=call.node_input.thread_id,
        ctx=call.ctx,
        name=call.node_name,
        streamer=call.streamer,
        identity=_call_identity(call),
    )
    # Inject CONFIG_KEY_READ / CONFIG_KEY_SEND so that deepagents
    # StateBackend (sandbox tools) can read/write the "files"
    # channel when running inside a Dapr workflow activity.
    try:
        from langgraph._internal._constants import (
            CONFIG_KEY_READ,
            CONFIG_KEY_SEND,
        )

        config[CONF][CONFIG_KEY_READ] = _make_activity_read(
            call.node_input.channel_state, call.pending_sends
        )
        config[CONF][CONFIG_KEY_SEND] = call.pending_sends.extend
    except ImportError:
        pass
    config[CONF].update(_interrupt_conf(call))
    return config


def _interrupt_conf(call: _NodeCall) -> Dict[str, Any]:
    """What ``interrupt()`` reads from the config: the task's scratchpad and name."""
    node_input = call.node_input
    send = node_input.send
    workflow_id = getattr(call.ctx, "workflow_id", "")
    return interrupt_config(
        namespace=(
            f"{call.node_name}:{workflow_id}:{node_input.step or 0}:"
            f"{'' if send is None else send['slot']}"
        ),
        step=node_input.step or 0,
        stop=node_input.recursion_limit,
        resume=node_input.resume,
        send=call.pending_sends.extend,
        resumes=call.resumes,
    )


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
    runtime carries LangGraph's no-op writer.
    """
    config = _node_config(
        {CONF: call.config.get(CONF)},
        thread_id=call.node_input.thread_id,
        ctx=call.ctx,
        name=call.node_name,
        streamer=call.streamer,
        identity=_call_identity(call),
    )
    config[CONF].update(_interrupt_conf(call))
    return config


def _run_node(call: _NodeCall) -> Any:
    """Run the node and return its raw result."""
    node_func = call.node_func
    if _is_runnable(node_func):
        # A LangChain Runnable (e.g. RunnableCallable): call .invoke() so that
        # argument injection (runtime, config, store) works.
        node_config = _runnable_config(call)

        def run() -> Any:
            return _await_if_coroutine(
                _invoke_runnable(node_func, call.state, node_config)
            )

        ambient = node_config
    else:

        def run() -> Any:
            return _call_plain(call)

        ambient = _plain_node_config(call)

    # RunnableCallable.invoke only sets the ambient config when tracing;
    # get_stream_writer() and the stream handler need it, and a node that calls
    # get_stream_writer() must find a (no-op) writer when the run is not streamed.
    with set_config_context(ambient) as ctx_:  # type: ignore[arg-type]
        try:
            result = ctx_.run(run)
        except ParentCommand as exc:
            return _parent_command_result(exc, call.node_name)
    return nested_interrupts(result, call.resumes, call.node_name)


def _parent_command_result(exc: ParentCommand, node_name: str) -> Command:
    """The node's result when a subgraph it runs hands over to its parent graph.

    A node inside a subgraph returns ``Command(graph=Command.PARENT)`` to send an
    update and a ``goto`` to the graph the subgraph is a node of (the swarm and
    supervisor handoff). The subgraph runs here, inside its node's activity, and
    LangGraph ends its run with the command addressed to the graph that called it
    (``""``) or to the node by name; the command is then what the node returned.
    Anything else is addressed above this graph, which has nowhere to send it, so
    the error stands, as it does in LangGraph.
    """
    cmd = exc.args[0] if exc.args else None
    if isinstance(cmd, Command) and cmd.graph in ("", node_name):
        return dataclasses.replace(cmd, graph=None)
    raise exc


def _make_streamer(
    ctx: WorkflowActivityContext,
    node_input: ExecuteNodeInput,
    config: Dict[str, Any],
    sink: Optional[StreamSink],
) -> Optional[NodeStreamer]:
    """The streamer for this attempt of the node, or ``None`` when not streaming.

    Chunks are streamed only when the transport gave this activity a writer;
    otherwise nothing about the node's execution changes.
    """
    send = node_input.send
    return _streamer_for(
        ctx,
        node_input.node_name,
        node_input.step or 0,
        node_input.thread_id,
        config,
        sink,
        # a Send is a task of its own, keyed apart from the node's other runs
        sink_node=None
        if send is None
        else send_key(node_input.node_name, send["slot"]),
        send_index=None if send is None else send["slot"],
    )


def _streamer_for(
    ctx: WorkflowActivityContext,
    node_name: str,
    step: int,
    thread_id: Optional[str],
    config: Dict[str, Any],
    sink: Optional[StreamSink],
    sink_node: Optional[str] = None,
    send_index: Optional[int] = None,
    branch: Optional[str] = None,
) -> Optional[NodeStreamer]:
    """A streamer that emits as ``node_name`` at ``step`` into ``sink``, or ``None``.

    ``send_index`` is set for a node that runs for a ``Send``: LangGraph tags
    such a task with the ``__pregel_push`` trigger and path.
    """
    if sink is None:
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
        branch=branch,
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
        streamer.task_started(call.state, call.config)
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
    sandbox tools), evaluates the node's conditional edges when the workflow
    asked for that, and then streams the node's update, as LangGraph does.
    """
    raw_writes = [
        *_result_to_raw_writes(result),
        *without_resume_writes(call.pending_sends),
    ]
    writes = _serialize_writes(raw_writes)
    routed: Dict[str, Any] = {}
    if call.node_input.routes:
        routed = route_in_node(
            call.ctx, call.node_input, writes, call.sink, _interrupt_conf(call)
        ).output_fields()
    if call.streamer:
        call.streamer.updates(raw_writes)
        call.streamer.task_finished(raw_writes)
    goto, sends = _result_to_gotos(result)
    return ExecuteNodeOutput(
        node_name=call.node_name,
        writes=writes,
        goto=goto,
        sends=sends,
        **routed,
    ).to_dict()


def execute_node_activity(
    ctx: WorkflowActivityContext,
    input_data: Dict[str, Any],
    *,
    stream_registry: Optional[StreamSinkRegistry] = None,
    stream_transport: Optional[StreamTransport] = None,
) -> Dict[str, Any]:
    """Run one node, or one ``Send`` task, and return the writes it produced.

    The node runs under a config that carries a ``Runtime`` and, when someone
    streams the run, the handler that sends its chunks to the writer
    ``stream_transport`` gives this execution (``stream_registry`` stands for
    the in-process transport over it; default the module's registry). An
    exception propagates, so Dapr retries the activity.
    """
    node_input = ExecuteNodeInput.from_dict(input_data)
    node_name = node_input.node_name

    print(f"  [ACTIVITY] Executing node '{node_name}' as Dapr activity", flush=True)

    node_func = get_registered_node(node_name)
    if node_func is None:
        return ExecuteNodeOutput(
            node_name=node_name,
            error=f"Node '{node_name}' not found in registry",
        ).to_dict()

    config = node_input.config or {}
    send = node_input.send
    writer = activity_writer(
        ctx,
        transport_of(stream_transport or stream_registry),
        node_input.step or 0,
        node_name if send is None else send_key(node_name, send["slot"]),
        config,
    )
    done = False
    try:
        call = _NodeCall(
            ctx=ctx,
            node_input=node_input,
            node_func=node_func,
            state=_node_state(node_input),
            config=config,
            streamer=_make_streamer(ctx, node_input, config, writer),
            pending_sends=[],
            sink=writer,
            resumes=TaskResumes(node_input.resume),
        )
        _send_values(
            ctx,
            node_input.channel_state,
            node_input.output_channels,
            node_input.step or 0,
            writer,
        )
        # Under the LangSmith parent trace the runner opened, so LLM calls nest in it.
        try:
            result = _run_with_stream_hooks(call, _langsmith_parent(config))
            output = _node_output(call, result)
        except GraphInterrupt as paused:
            # Not a failure: the task waits for a resume and runs again then.
            interrupts = interrupt_payloads(paused)
            if call.streamer:
                call.streamer.interrupted(interrupts)
            return with_attempt(
                ExecuteNodeOutput(node_name=node_name, interrupts=interrupts).to_dict(),
                writer,
            )
        except DroppedInterrupt as e:  # retrying would drop it again
            return ExecuteNodeOutput(node_name=node_name, error=str(e)).to_dict()
        done = True
        return with_attempt(output, writer)
    finally:
        if writer is not None:
            # synchronous: the unit's log is complete before the activity returns
            writer.finish(done=done)


def _send_values(
    ctx: WorkflowActivityContext,
    channel_state: ChannelState,
    output_channels: Optional[List[str]],
    step: int,
    sink: Optional[StreamSink],
) -> None:
    """Offer the state this activity was handed as the ``values`` chunk of ``step``.

    It is the state after the previous step (the input for step 0), which
    every activity of ``step`` is handed: the sink keeps the first offer, which
    comes before any of the step's own chunks.
    """
    if not output_channels or sink is None:
        return
    workflow_id = getattr(ctx, "workflow_id", None)
    event = state_values(str(workflow_id), step, channel_state, output_channels)
    if event is not None:
        sink.put(event)


def _node_state(node_input: ExecuteNodeInput) -> Any:
    """What the node takes as input: the channel state, or a ``Send``'s argument."""
    if node_input.send is None:
        state = _with_managed(
            _reconstruct_state(node_input.channel_state),
            node_input.recursion_limit,
            node_input.step,
        )
        shape = _node_inputs.get(node_input.node_name)
        return shape.apply(state) if shape else state
    return _decode_arg(node_input.send["arg"])


def make_condition_activity(
    stream: Union[StreamTransport, StreamSinkRegistry, None] = None,
) -> Callable[[WorkflowActivityContext, Dict[str, Any]], Dict[str, Any]]:
    """Build the conditional-edge activity of a runner (see ``make_node_activity``)."""
    transport = transport_of(stream)

    def evaluate_condition(
        ctx: WorkflowActivityContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return evaluate_condition_activity(ctx, input_data, stream_transport=transport)

    return evaluate_condition


def _branch_key(cond_input: EvaluateConditionInput) -> str:
    """The sink key of a conditional edge's chunks, apart from its task's own."""
    task = (
        cond_input.source_node
        if cond_input.send_slot is None
        else send_key(cond_input.source_node, cond_input.send_slot)
    )
    return f"{task}:branch:{cond_input.condition_name}"


def _call_condition(
    ctx: WorkflowActivityContext,
    cond_input: EvaluateConditionInput,
    cond_func: Callable,
    state: Any,
    sink: Optional[StreamSink],
    interrupt_conf: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call a conditional edge's function and return what it routed to.

    In a streamed run the function's LLM tokens and ``get_stream_writer()``
    events are sent as its source node's, keyed
    apart from the node's own chunks.
    """
    config = cond_input.config or {}
    streamer = None
    if cond_input.step is not None:
        streamer = _streamer_for(
            ctx,
            cond_input.source_node,
            cond_input.step,
            cond_input.thread_id,
            config,
            sink,
            # one key per edge and task: a node's several conditional edges, and
            # its several Send tasks, are several tasks, and a key they share
            # would read as one retrying
            sink_node=_branch_key(cond_input),
            send_index=cond_input.send_slot,
            branch=cond_input.condition_name,
        )
    config = _node_config(
        config,
        thread_id=cond_input.thread_id,
        ctx=ctx,
        name=cond_input.condition_name,
        streamer=streamer,
        identity=_task_identity(
            ctx, cond_input.source_node, cond_input.step or 0, cond_input.send_slot
        ),
    )
    config[CONF].update(interrupt_conf or router_interrupt_conf(cond_input))

    def run() -> Any:
        # A LangChain Runnable (e.g. RunnableCallable from Deep Agents
        # branches) goes through invoke() so runtime injection works.
        if not _is_runnable(cond_func):
            return _await_if_coroutine(cond_func(state))
        return _await_if_coroutine(_invoke_runnable(cond_func, state, config))

    # Runnable.invoke only sets the ambient config when tracing, and a plain
    # function never does: get_stream_writer() and the stream handler need it.
    with set_config_context(config) as ctx_:  # type: ignore[arg-type]
        result = ctx_.run(run)
    if streamer:
        streamer.branch_finished()
    return result


def evaluate_condition_activity(
    ctx: WorkflowActivityContext,
    input_data: Dict[str, Any],
    *,
    stream_registry: Optional[StreamSinkRegistry] = None,
    stream_transport: Optional[StreamTransport] = None,
) -> Dict[str, Any]:
    """Evaluate a conditional edge; the output names where it routes, or the error.

    A streamed run's chunks go to the writer ``stream_transport`` gives this
    execution (see :func:`execute_node_activity`).
    """
    cond_input = EvaluateConditionInput.from_dict(input_data)
    writer = activity_writer(
        ctx,
        transport_of(stream_transport or stream_registry),
        cond_input.step if cond_input.step is not None else 0,
        _branch_key(cond_input),
        cond_input.config or {},
    )
    done = False
    try:
        _send_values(
            ctx,
            cond_input.channel_state,
            cond_input.output_channels,
            0,
            writer,
        )
        try:
            output = evaluate_condition(ctx, cond_input, writer).to_dict()
        except GraphInterrupt:
            output = EvaluateConditionOutput(error=ROUTER_INTERRUPT_ERROR).to_dict()
        done = True
        return with_attempt(output, writer)
    finally:
        if writer is not None:
            writer.finish(done=done)


def evaluate_condition(
    ctx: WorkflowActivityContext,
    cond_input: EvaluateConditionInput,
    sink: Optional[StreamSink],
    interrupt_conf: Optional[Dict[str, Any]] = None,
) -> EvaluateConditionOutput:
    """Call a conditional edge's function and say where it routes, or the error.

    What the condition activity does, and what a node activity does for each
    of its node's conditional edges. An exception of the function propagates.
    """
    cond_func = get_registered_condition(cond_input.condition_name)
    if cond_func is None:
        return EvaluateConditionOutput(
            error=f"Condition '{cond_input.condition_name}' not found in registry",
        )

    state: Any = _with_managed(
        _reconstruct_state(cond_input.channel_state),
        cond_input.recursion_limit,
        cond_input.step,
    )
    shape = _condition_inputs.get(cond_input.condition_name)
    if shape:
        state = shape.apply(state)

    # Under the parent "LangGraph" trace. A Runnable condition (every
    # add_conditional_edges path is one) traces itself, so without a parent
    # each evaluation lands in LangSmith as a separate top-level trace.
    with _under_parent_trace(cond_input.config):
        result = _call_condition(
            ctx, cond_input, cond_func, state, sink, interrupt_conf
        )
    return _route(cond_input, result)


def _under_parent_trace(config: Optional[Dict[str, Any]]) -> Any:
    """A context that nests LangSmith runs under the run's parent trace, if it has one."""
    parent = _langsmith_parent(config)
    if not parent:
        return contextlib.nullcontext()
    import langsmith as ls

    return ls.tracing_context(parent=parent)


def _route(cond_input: EvaluateConditionInput, result: Any) -> EvaluateConditionOutput:
    """Where a conditional edge's return value sends the run.

    ``result`` is a node name, a list of them, or ``Send`` objects. Any other
    value goes through the edge's ``path_map`` when it has one: a router may
    return labels (``"continue"``) rather than node names.
    """
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
            )
    return EvaluateConditionOutput(next_nodes=next_nodes, sends=sends)


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
            if _is_messages_channel(channel)
            else _store_value(value)
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
    values = {
        **{name: _output_value(v) for name, v in channel_state.values.items()},
        **{
            name: _output_json(default)
            for name, default in _unwritten_defaults(channel_state.values).items()
        },
    }
    return {
        channel: values[channel] for channel in output_channels if channel in values
    }


def _output_value(value: Any) -> Any:
    """A stored value as the JSON a run's output carries."""
    return _output_json(decode_value(value)) if is_encoded(value) else value


def _output_json(value: Any) -> Any:
    """``value`` as JSON for a run's output: a set becomes a (sorted) list.

    The output is JSON, so it cannot hold the set or the object a node wrote;
    a nested message or model becomes its dump (see :func:`_serialize_value`).
    """
    if isinstance(value, (set, frozenset)):
        items = [_output_json(v) for v in value]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(value, (list, tuple)):
        return [_output_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _output_json(v) for k, v in value.items()}
    return _serialize_value(value)


class ReducerError(Exception):
    """A channel's reducer raised while a write was applied; the run fails."""


def _apply_write(channel_state: ChannelState, channel: str, value: Any) -> None:
    """Apply a write to a channel, through its reducer if it has one."""
    reducer = get_channel_reducer(channel)
    if reducer and isinstance(value, dict) and set(value) == {OVERWRITE}:
        # Overwrite replaces the channel's value without calling the reducer.
        channel_state.values[channel] = value[OVERWRITE]
        channel_state.versions[channel] = channel_state.versions.get(channel, 0) + 1
        return
    if reducer and channel not in channel_state.values:
        # The first write is reduced into the channel's default, so add_messages
        # gives its messages ids and objects.
        if channel in _channel_defaults:
            channel_state.values[channel] = copy.deepcopy(_channel_defaults[channel])
    if reducer and channel in channel_state.values:
        default = _channel_defaults.get(channel)
        current = coerce_to_default(
            decode_value(channel_state.values[channel]), default
        )
        value = coerce_to_default(decode_value(value), default)
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
            if _is_messages_channel(channel) and isinstance(reduced, list):
                channel_state.values[channel] = [
                    _serialize_value(item) for item in reduced
                ]
            else:
                channel_state.values[channel] = _store_value(reduced)
        except Exception as e:
            # LangGraph fails the run on a reducer that raises; overwriting the
            # channel would hide the error and differ between a run and a replay.
            raise ReducerError(f"Reducer failed for channel '{channel}': {e}") from e
    else:
        channel_state.values[channel] = value

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
        if is_encoded(value):
            state[channel] = coerce_to_default(
                decode_value(value), _channel_defaults.get(channel)
            )
            continue
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

    # A reducer channel nobody has written yet holds its default: a node reads ``state["log"]`` as ``[]``, not as a missing key.
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
    Anything else (a ``role`` dict, a ``RemoveMessage``) goes through it. A
    ``ChatMessage`` dump has both a ``type`` (``"chat"``) and a ``role`` of its
    own, so a ``role`` next to ``type="chat"`` does not make it the shorthand.
    """
    if (
        isinstance(item, dict)
        and "type" in item
        and (item["type"] == "chat" or "role" not in item)
    ):
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

    if isinstance(result, Command):
        return _command_to_raw_writes(result)

    # A root state has no keys: whatever else the node returns is the channel's
    # write, a dict included.
    if _is_root_state() and not (
        isinstance(result, list) and result and isinstance(result[0], Command)
    ):
        return [(ROOT, result)]

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

    if _is_root_state() and not isinstance(update, dict):
        if not (
            isinstance(update, (list, tuple))
            and update
            and all(isinstance(i, tuple) and len(i) == 2 for i in update)
        ):
            return [(ROOT, update)]
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
    return _store_value(value)


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


def _store_value(value: Any) -> Any:
    """``value`` as it is kept in the channel state: plain JSON, or typed (see values.py).

    What the serializer cannot write falls back to :func:`_serialize_value`.
    """
    stored = encode_value(value)
    if is_encoded(stored) or is_plain_json(stored):
        return stored
    return _serialize_value(value)


def _serialize_value(value: Any) -> Any:
    """``value`` as JSON-compatible data for storage."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]

    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}

    if hasattr(value, "model_dump"):
        # Pydantic v2
        return value.model_dump()
    if hasattr(value, "dict"):
        # Pydantic v1
        return value.dict()
    if hasattr(value, "to_dict"):
        return value.to_dict()

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

    return str(value)
