# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runner for executing LangGraph graphs as Dapr Workflows."""

import asyncio
import json
import logging
import time
import uuid
from contextlib import aclosing
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from dapr.ext.workflow import WorkflowStatus
from langchain_core.runnables.config import DEFAULT_RECURSION_LIMIT
from langgraph.types import Command

from diagrid.agent.core.types.type import SupportedFrameworks
from diagrid.agent.core.workflow import BaseWorkflowRunner
from diagrid.agent.core.workflow.runner import fold_run_event
from diagrid.agent.core.workflow.stream import (
    InProcessTransport,
    StreamSinkRegistry,
    StreamTransport,
)
from .models import (
    ChannelState,
    EdgeConfig,
    GraphConfig,
    GraphWorkflowInput,
    GraphWorkflowOutput,
    NodeConfig,
    WaitingEdgeConfig,
)
from .streaming import (
    STREAM_RUN_KEY,
    StaticConfig,
    FAILED_STATUSES,
    StreamControl,
    normalize_final,
    normalize_stream_mode,
    run_error,
)
from .stream_events import (
    FINAL_EVENTS,
    stream_failure_event,
    route_astream_item,
    started_event,
    without_executed,
)
from .langsmith_trace import ParentTrace, start_parent_trace
from .run_status import completed_result, interrupted_result
from .activity_stream import parse_status
from .interrupts import INTERRUPT_KEY, as_interrupts, interrupted_event
from .reconnect import add_reconnect_route, resume_events
from .resume_run import add_resume_route, resume_run_events
from .stream_drain import SseEncoder, drain, parse_output
from .stream_source import (
    custom_status,
    open_source,
    resolve_transport,
)
from .thread_state import (
    LOAD_THREAD_ACTIVITY,
    SAVE_THREAD_ACTIVITY,
    make_load_activity,
    make_save_activity,
    resolve_thread_store,
)
from .limits import StepBound, step_bound
from .workflow import (
    IS_LAST_STEP,
    REMAINING_STEPS,
    agent_workflow,
    make_condition_activity,
    GENERIC_NODE_ACTIVITY,
    make_node_activity,
    node_activity_name,
    register_node,
    register_node_static_config,
    register_condition_input,
    register_node_input,
    register_node_triggers,
    register_condition,
    register_channel_reducer,
    _NO_DEFAULT,
    register_managed_channel,
    register_state_channels,
    set_default_graph_config,
    set_serializer,
    clear_registries,
    START,
    END,
    ROOT,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

_TRANSPORT_KEYS = ("stream", "stream_mode")

#: The runner's step bound when it is given none (see ``max_steps``).
DEFAULT_MAX_STEPS = 100


# What a reducer channel holds before its first write; LangGraph's own marker
# when the channel type has no empty value.
try:
    from langgraph.channels.binop import MISSING as _LANGGRAPH_MISSING
except ImportError:  # pragma: no cover - a LangGraph without the marker
    _LANGGRAPH_MISSING = _NO_DEFAULT


def _managed_kinds(managed: Dict[str, Any]) -> Dict[str, str]:
    """The state keys that are LangGraph's ``RemainingSteps`` or ``IsLastStep``.

    Any other managed value (one a library defines) is not supplied.
    """
    try:
        from langgraph.managed.is_last_step import (
            IsLastStepManager,
            RemainingStepsManager,
        )
    except ImportError:  # pragma: no cover - a LangGraph without them
        return {}
    kinds: Dict[str, str] = {}
    for name, spec in managed.items():
        cls = getattr(spec, "cls", spec)
        if isinstance(cls, type) and issubclass(cls, RemainingStepsManager):
            kinds[name] = REMAINING_STEPS
        elif isinstance(cls, type) and issubclass(cls, IsLastStepManager):
            kinds[name] = IS_LAST_STEP
    return kinds


def _check_run_config(config: Optional[Dict[str, Any]]) -> None:
    """Refuse a LangGraph config the workflow cannot carry, before a run starts.

    The config travels to the activities as JSON, so ``callbacks`` (handler
    objects) and any other value that is not JSON cannot be passed; an
    invalid ``recursion_limit`` is refused too. Raises ``ValueError``.
    """
    if not config:
        return
    if config.get("callbacks"):
        raise ValueError(
            "config['callbacks'] is not supported: the config travels to the "
            "workflow as JSON. Attach handlers to the models inside the nodes."
        )
    limit = config.get("recursion_limit")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError(
            f"config['recursion_limit'] must be a positive integer, got {limit!r}"
        )
    for key, value in config.items():
        items = (
            value.items()
            if key == "configurable" and isinstance(value, dict)
            else [(key, value)]
        )
        for name, item in items:
            try:
                json.dumps(item)
            except (TypeError, ValueError) as e:
                where = (
                    f"config['configurable'][{name!r}]"
                    if key == "configurable"
                    else f"config[{name!r}]"
                )
                raise ValueError(
                    f"{where} is not JSON serializable, and the config travels to "
                    f"the workflow as JSON: {e}"
                ) from e


def _merge_graph_config(
    graph_config: Optional[Dict[str, Any]], config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """``config`` over the config bound to the graph (``compiled.with_config(...)``).

    LangGraph merges the two for every run: the bound ``configurable`` is the
    base of the caller's, and the bound ``recursion_limit`` applies unless the
    caller sets one (LangGraph takes the default, 25, for no limit). The bound
    ``metadata`` and ``tags`` reach the tasks through ``StaticConfig``, and bound
    ``callbacks`` (handler objects) cannot travel to the workflow as JSON.
    Returns a new dict.
    """
    bound = graph_config or {}
    merged = dict(config or {})
    configurable = {
        **(bound.get("configurable") or {}),
        **(merged.get("configurable") or {}),
    }
    if configurable:
        merged["configurable"] = configurable
    limit = bound.get("recursion_limit")
    if limit is not None and merged.get("recursion_limit") in (
        None,
        DEFAULT_RECURSION_LIMIT,
    ):
        merged["recursion_limit"] = limit
    return merged


def _run_identity(
    config: Optional[Dict[str, Any]],
    merged: Dict[str, Any],
    thread_id: Optional[str],
    workflow_id: Optional[str],
) -> Tuple[str, str, Dict[str, Any]]:
    """``(thread_id, workflow_id, config)`` for a run, whichever way it is started.

    The thread id is the one given, else ``configurable["thread_id"]`` of
    ``merged`` (the caller's config over the graph's own, already checked), else
    new. The config handed to the run is the caller's ``config``, which the run
    merges with the graph's own again, carrying that one id so the id the run
    reports is the one its nodes see.
    """
    thread = str(
        thread_id or (merged.get("configurable") or {}).get("thread_id") or uuid.uuid4()
    )
    run_config = {
        **(config or {}),
        "configurable": {
            **((config or {}).get("configurable") or {}),
            "thread_id": thread,
        },
    }
    return thread, workflow_id or f"graph-{thread}-{uuid.uuid4().hex[:8]}", run_config


async def _wait_scheduled(scheduled: asyncio.Event, pump: "asyncio.Task[None]") -> bool:
    """Wait until the workflow is scheduled or the pump ends; whether it was."""
    waiter = asyncio.ensure_future(scheduled.wait())
    try:
        await asyncio.wait({waiter, pump}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
    return scheduled.is_set()


@dataclass(frozen=True)
class _RunSetup:
    """What a run needs once its identity, config and input are settled."""

    workflow_id: str
    thread_id: str
    #: The config the workflow runs with, ``langsmith_*`` keys included.
    config: Dict[str, Any]
    #: The LangSmith parent trace to close if the run is abandoned.
    trace: Optional[ParentTrace]
    #: The workflow's input, as scheduled.
    workflow_input: Dict[str, Any]


class DaprWorkflowGraphRunner(BaseWorkflowRunner):
    """Runner that executes LangGraph graphs as Dapr Workflows.

    This runner wraps a compiled LangGraph and executes it using Dapr Workflows,
    making each node execution a durable activity. This provides:

    - Fault tolerance: Graphs automatically resume from the last successful node
    - Durability: Graph state persists and survives process restarts
    - Observability: Full visibility into execution through Dapr's workflow APIs

    Example:
        ```python
        from langgraph.graph import StateGraph, START, END
        from diagrid.agent.langgraph import DaprWorkflowGraphRunner

        # Define your graph
        class State(TypedDict):
            messages: list[str]

        def node_a(state: State) -> dict:
            return {"messages": state["messages"] + ["from A"]}

        graph = StateGraph(State)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        compiled = graph.compile()

        # Create runner and start
        runner = DaprWorkflowGraphRunner(graph=compiled, name="my-graph")
        runner.start()

        # Run the graph
        async for event in runner.run_async(
            input={"messages": ["hello"]},
            thread_id="thread-123",
        ):
            print(event)

        runner.shutdown()
        ```

    Attributes:
        graph: The compiled LangGraph
        workflow_runtime: The Dapr WorkflowRuntime instance
        workflow_client: The Dapr WorkflowClient for managing workflows
    """

    #: Framework identity used for the agent registry and the workflow name
    #: (``dapr.<framework>.<name>.workflow``). Subclasses that wrap a different
    #: framework on top of the LangGraph runtime (e.g. Deep Agents) override this.
    _REGISTRY_FRAMEWORK: SupportedFrameworks = SupportedFrameworks.LANGGRAPH

    def __init__(
        self,
        graph: "CompiledStateGraph",
        *,
        name: str,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_steps: Optional[int] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        instructions: Optional[str] = None,
        registry_config: Optional[Any] = None,
        stream_registry: Optional[StreamSinkRegistry] = None,
        checkpointer: Optional[Any] = None,
        stream_transport: Optional[StreamTransport] = None,
    ):
        """Initialize the runner.

        Args:
            graph: The compiled LangGraph to execute
            name: Required name for the graph
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_steps: Maximum number of steps before stopping (default: 100).
                A ``recursion_limit`` bound to the graph (``with_config``) takes
                over when this is not given; when it is, it caps that limit.
            role: Optional role description for registry (e.g. "Schedule Planner")
            goal: Optional goal description for registry
            instructions: Optional system prompt for registry, also used as the
                goal when ``goal`` is not given. Without it, the registry looks
                for a module-level ``SystemMessage`` in node globals.
            registry_config: Optional registry configuration for metadata extraction
            stream_registry: Where streamed runs register their sinks when
                they stream in process. Giving one picks the in-process
                transport. Defaults to a new registry owned by this runner.
            checkpointer: Where a thread's state is kept between runs, so a run
                with the same ``thread_id`` continues the conversation: a
                ``DaprMemoryCheckpointer`` or a LangGraph checkpoint saver with
                the synchronous API. Defaults to the checkpointer the graph was
                compiled with, else the app's Dapr state store
                (``agent-memory``) when it has one; with none, ``thread_id`` is
                only a label.
            stream_transport: How a streamed run's chunks get from its
                activities to its stream. By default the state-store event
                log (``StateLogTransport``) when the app has a state store,
                which any replica reads and a client can reconnect to, and
                the in-process transport otherwise.
        """
        self._graph = graph
        self._max_steps = DEFAULT_MAX_STEPS if max_steps is None else max_steps
        self._max_steps_given = max_steps is not None
        self._stream_registry = (
            stream_registry if stream_registry is not None else StreamSinkRegistry()
        )
        self._given_transport = stream_transport
        self._given_registry = stream_registry
        # Pumps of streams whose consumer left while a result is still to be
        # published; held here so they are not collected before they finish.
        self._detached_pumps: Set["asyncio.Task[None]"] = set()

        super().__init__(
            name,
            framework=self._REGISTRY_FRAMEWORK,
            host=host,
            port=port,
            max_iterations=self._max_steps,
        )

        if name:
            self._graph._diagrid_name = name  # type: ignore[attr-defined]
        if role:
            self._graph._diagrid_role = role  # type: ignore[attr-defined]
        if goal:
            self._graph._diagrid_goal = goal  # type: ignore[attr-defined]
        if instructions:
            self._graph._diagrid_instructions = instructions  # type: ignore[attr-defined]
        # Surface the runner's step budget so the registry reflects the actual
        # max_iterations instead of a hardcoded default (read by DeepAgentsMapper).
        self._graph._diagrid_max_steps = self._max_steps  # type: ignore[attr-defined]

        self._thread_store = resolve_thread_store(
            checkpointer, self._graph, self._state_store
        )
        self._stream_transport = resolve_transport(
            self._given_transport,
            self._given_registry,
            getattr(self._state_store, "store_name", None) or self._workflow_store_name,
            self._stream_registry,
        )
        if isinstance(self._stream_transport, InProcessTransport):
            self._stream_registry = self._stream_transport.registry

        self._register_agent_metadata(
            agent=self._graph,
            framework=self._REGISTRY_FRAMEWORK,
            registry=registry_config,
            name=self._name,
        )

        self._register_workflow_components()

        # Register graph components (nodes, channels, serializer) first.
        # This calls clear_registries() internally.
        self._register_graph_components()
        # Then extract graph config, which also registers conditions.
        # Must run AFTER _register_graph_components so conditions are not wiped.
        self._graph_config = self._extract_graph_config()
        self._register_node_activities()

    def _register_workflow_components(self) -> None:
        """Register workflow and activities on the workflow runtime."""
        self._workflow_runtime.register_workflow(
            agent_workflow, name=self.workflow_name
        )
        self._workflow_runtime.register_activity(
            make_node_activity(GENERIC_NODE_ACTIVITY, self._stream_transport),
            name=GENERIC_NODE_ACTIVITY,
        )
        self._workflow_runtime.register_activity(
            make_condition_activity(self._stream_transport),
            name="evaluate_condition_activity",
        )
        # Registered even without a store, so an instance that scheduled them
        # still replays on a runner that has none.
        self._workflow_runtime.register_activity(
            make_load_activity(self._thread_store), name=LOAD_THREAD_ACTIVITY
        )
        self._workflow_runtime.register_activity(
            make_save_activity(self._thread_store), name=SAVE_THREAD_ACTIVITY
        )

    def _register_node_activities(self) -> None:
        """Register one activity per graph node, named after the node."""
        for node in self._graph_config.nodes:
            name = node_activity_name(node.name)
            if name == GENERIC_NODE_ACTIVITY:
                continue
            self._workflow_runtime.register_activity(
                make_node_activity(node.name, self._stream_transport), name=name
            )

    def _extract_graph_config(self) -> GraphConfig:
        """Extract configuration from the compiled graph."""
        nodes = []
        edges = []
        finish_points = []

        graph_nodes = getattr(self._graph, "nodes", {})
        for node_name, node_spec in graph_nodes.items():
            if node_name in (START, END, "__start__", "__end__"):
                continue

            triggers = []
            channels_read = []
            channels_write: List[str] = []

            if hasattr(node_spec, "triggers"):
                triggers = list(node_spec.triggers) if node_spec.triggers else []
            if hasattr(node_spec, "channels"):
                channels_read = (
                    [node_spec.channels]
                    if isinstance(node_spec.channels, str)
                    else list(node_spec.channels or [])
                )

            register_node_triggers(node_name, triggers)
            register_node_input(
                node_name, node_spec.channels, getattr(node_spec, "mapper", None)
            )
            register_node_static_config(node_name, self._static_config(node_spec))
            nodes.append(
                NodeConfig(
                    name=node_name,
                    triggers=triggers,
                    channels_read=channels_read,
                    channels_write=channels_write,
                )
            )

        builder = getattr(self._graph, "builder", None)

        if builder:
            graph_edges: Any = getattr(builder, "edges", set())
            for source, target in graph_edges:
                source_name = source if source != "__start__" else START
                target_name = target if target != "__end__" else END

                if target_name == END:
                    if source_name not in finish_points:
                        finish_points.append(source_name)

                edges.append(
                    EdgeConfig(
                        source=source_name,
                        target=target_name,
                    )
                )

            branches = getattr(builder, "branches", {})
            for source, branch_dict in branches.items():
                for branch_name, branch_spec in branch_dict.items():
                    path_func = getattr(branch_spec, "path", None)

                    if path_func:
                        condition_name = f"{source}_{branch_name}_condition"
                        register_condition(condition_name, path_func)
                        self._register_condition_input(
                            condition_name, source, branch_spec
                        )

                        # ``ends`` is the edge's path_map ({return value: node}),
                        # either given to add_conditional_edges or inferred from
                        # the condition's Literal return type. Keys are stringified
                        # because the map crosses JSON to reach the activity.
                        ends = getattr(branch_spec, "ends", None)
                        edges.append(
                            EdgeConfig(
                                source=source if source != "__start__" else START,
                                target="",  # Determined at runtime
                                condition=condition_name,
                                path_map=(
                                    {str(k): v for k, v in ends.items()}
                                    if ends
                                    else None
                                ),
                            )
                        )

        entry_points = sorted(
            {e.target for e in edges if e.source == START and not e.condition}
        )
        has_start_branch = any(e.source == START and e.condition for e in edges)
        if not entry_points and not has_start_branch and nodes:
            entry_points = [nodes[0].name]
        entry_point = (
            entry_points[0] if entry_points else (nodes[0].name if nodes else "")
        )

        waiting_edges = [
            WaitingEdgeConfig(sources=sorted(sources), target=target)
            for sources, target in sorted(
                getattr(builder, "waiting_edges", None) or (),
                key=lambda w: (sorted(w[0]), w[1]),
            )
        ]

        input_channels = []
        output_channels = []

        if hasattr(self._graph, "input_channels"):
            ic = self._graph.input_channels
            input_channels = [ic] if isinstance(ic, str) else list(ic or [])
        if hasattr(self._graph, "output_channels"):
            oc = self._graph.output_channels
            output_channels = [oc] if isinstance(oc, str) else list(oc or [])

        if not input_channels:
            input_channels = list(getattr(self._graph, "channels", {}).keys())
        if not output_channels:
            output_channels = list(getattr(self._graph, "channels", {}).keys())

        return GraphConfig(
            name=self._name,
            nodes=nodes,
            edges=edges,
            entry_point=entry_point,
            finish_points=finish_points,
            input_channels=input_channels,
            output_channels=output_channels,
            entry_points=entry_points,
            waiting_edges=waiting_edges,
            deferred=sorted(
                name
                for name, spec in (getattr(builder, "nodes", None) or {}).items()
                if getattr(spec, "defer", False)
            ),
        )

    def _run_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """The config a run starts with: the caller's over the graph's own.

        Checked for what cannot travel to the workflow before anything starts.
        """
        merged = _merge_graph_config(getattr(self._graph, "config", None), config)
        _check_run_config(merged)
        return merged

    def _step_bound(self, caller_config: Optional[Dict[str, Any]]) -> StepBound:
        """The step limit of a run started with ``caller_config``."""
        return step_bound(
            caller_config,
            getattr(self._graph, "config", None),
            self._max_steps,
            max_steps_given=self._max_steps_given,
        )

    def _prepare_run(
        self,
        input: Any,
        thread_id: Optional[str],
        config: Optional[Dict[str, Any]],
        workflow_id: Optional[str],
    ) -> _RunSetup:
        """Everything ``invoke`` and ``run_async`` settle before scheduling the run."""
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        thread_id, workflow_id, caller_config = _run_identity(
            config, self._run_config(config), thread_id, workflow_id
        )
        bound = self._step_bound(config)
        run_config = self._run_config(caller_config)
        run_config["thread_id"] = thread_id

        # The parent trace is passed to the activities via config. The caller's
        # finally closes it if the run is abandoned; on normal completion
        # workflow.py closes it too (a benign double close).
        trace = start_parent_trace(input, thread_id)
        if trace:
            run_config.update(trace.config)

        channels = self._channel_input(input)
        workflow_input = GraphWorkflowInput(
            graph_config=self._graph_config,
            channel_state=ChannelState(
                values=self._serialize_input(channels),
                versions={k: 1 for k in channels.keys()},
                updated_channels=list(channels.keys()),
            ),
            step=0,
            max_steps=bound.max_steps,
            strict_limit=bound.strict,
            pending_nodes=[],
            config=run_config,
            thread_id=thread_id,
            thread_state=self._thread_store is not None,
        ).to_dict()
        json.dumps(workflow_input)
        return _RunSetup(workflow_id, thread_id, run_config, trace, workflow_input)

    def _register_condition_input(
        self, condition_name: str, source: str, branch_spec: Any
    ) -> None:
        """Record the state a conditional edge's function is handed.

        LangGraph reads it through the edge's input schema (taken from the
        annotation of the function's parameter), else its source node's, else the
        graph's state schema: the channels of that schema, coerced to the schema
        class for pydantic and dataclass state. A function with no input schema
        of its own is also given the keys its source node wrote, so the state is
        not cut to the channels then (unless it is coerced to a class).
        """
        builder = self._graph.builder
        node = builder.nodes.get(source)
        schema = branch_spec.input_schema or (
            node.input_schema if node is not None else builder.state_schema
        )
        channels = builder.schemas.get(schema)
        if channels is None:
            return
        mapper = getattr(self._graph, "schema_to_mapper", {}).get(schema)
        cut = branch_spec.input_schema is not None or mapper is not None
        register_condition_input(
            condition_name, list(channels) if cut else None, mapper
        )

    def _static_config(self, node_spec: Any) -> StaticConfig:
        """The metadata and tags the compiled graph and this node give its tasks."""
        graph_config = getattr(self._graph, "config", None) or {}
        return StaticConfig(
            graph_metadata=dict(graph_config.get("metadata") or {}),
            graph_tags=tuple(graph_config.get("tags") or ()),
            node_metadata=dict(getattr(node_spec, "metadata", None) or {}),
            node_tags=tuple(getattr(node_spec, "tags", None) or ()),
        )

    def _register_graph_components(self) -> None:
        """Register graph nodes and other components in the global registries."""
        clear_registries()

        graph_nodes = getattr(self._graph, "nodes", {})
        for node_name, node_spec in graph_nodes.items():
            if node_name in (START, END, "__start__", "__end__"):
                continue

            # Register the Runnable wrapper (RunnableCallable, ToolNode, ...),
            # not its raw ``func``: execute_node_activity calls ``.invoke()`` on
            # it, which injects ``config``/``runtime``/``store`` into nodes that
            # declare them. ``async def`` nodes have no ``func`` at all.
            node_func = None

            if hasattr(node_spec, "bound"):
                node_func = node_spec.bound
            elif hasattr(node_spec, "runnable"):
                node_func = node_spec.runnable
            elif callable(node_spec):
                node_func = node_spec

            if node_func is not None:
                register_node(node_name, node_func)
                logger.info(f"Registered node: {node_name}")
            else:
                logger.warning(f"Could not extract callable for node: {node_name}")

        channels = getattr(self._graph, "channels", {})
        for channel_name, channel in channels.items():
            # BinaryOperatorAggregate uses "operator" not "reducer"
            reducer = getattr(channel, "reducer", None) or getattr(
                channel, "operator", None
            )
            if reducer:
                default = getattr(channel, "value", _NO_DEFAULT)
                if default is _LANGGRAPH_MISSING:
                    default = _NO_DEFAULT
                register_channel_reducer(channel_name, reducer, default)
                logger.info(f"Registered reducer for channel: {channel_name}")

        # A node's keys that are not state channels are dropped
        # (CompiledStateGraph's ``_get_updates``).
        state_channels = getattr(
            getattr(self._graph, "builder", None), "channels", None
        )
        if isinstance(state_channels, dict):
            managed = getattr(self._graph.builder, "managed", None) or {}
            register_state_channels([*state_channels, *managed])
            for name, kind in _managed_kinds(managed).items():
                register_managed_channel(name, kind)

        try:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

            set_serializer(JsonPlusSerializer())
        except ImportError:
            logger.warning(
                "JsonPlusSerializer not available, using basic serialization"
            )

    # ------------------------------------------------------------------
    # LangGraph-specific shutdown (no chat client)
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shutdown the workflow runtime."""
        if not self._started:
            return

        self._workflow_runtime.shutdown()
        self._started = False
        logger.info("Dapr Workflow runtime stopped")

    # ------------------------------------------------------------------
    # LangGraph-specific execution methods
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Any,
        *,
        thread_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
        timeout: Optional[float] = None,
    ) -> Any:
        """Run the graph synchronously and return the final output.

        Args:
            input: Input values for the graph channels
            thread_id: Thread identifier for the execution
            config: Optional LangGraph config dict. It travels to the workflow
                as JSON: ``tags``, ``metadata``, ``configurable`` and
                ``recursion_limit`` work (the limit bounds the run's steps as
                LangGraph counts them; ``recursion_limit`` and ``configurable``
                bound to the graph with ``with_config`` apply unless the caller
                sets its own). ``callbacks`` and any value that is not JSON
                raise ``ValueError`` before a run starts.
            workflow_id: Optional workflow instance ID (generated if not provided)
            poll_interval: How often to poll for workflow status (seconds)
            timeout: Optional timeout in seconds

        Returns:
            Final output values from the graph

        Raises:
            GraphRecursionError: The run hit its step limit (the error is also a
                :class:`GraphRunError`).
            GraphRunError: The run failed (a node or a condition raised).
            RuntimeError: The workflow failed or was terminated.
        """
        run = self._prepare_run(input, thread_id, config, workflow_id)
        workflow_id, thread_id, trace = run.workflow_id, run.thread_id, run.trace

        logger.info(f"Starting workflow: {workflow_id}")
        assert self._workflow_client is not None
        self._workflow_client.schedule_new_workflow(
            workflow=agent_workflow,
            input=run.workflow_input,
            instance_id=workflow_id,
        )

        start_time = time.time()
        try:
            while True:
                time.sleep(poll_interval)

                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(
                        f"Workflow {workflow_id} timed out after {timeout}s"
                    )

                state = self._workflow_client.get_workflow_state(
                    instance_id=workflow_id
                )

                if state is None:
                    raise RuntimeError(f"Workflow {workflow_id} state not found")

                paused = interrupted_event(
                    parse_status(state.serialized_custom_status), workflow_id
                )
                if state.runtime_status == WorkflowStatus.RUNNING and paused:
                    # As LangGraph's invoke: the interrupts, under __interrupt__.
                    return {INTERRUPT_KEY: list(as_interrupts(paused["interrupts"]))}

                if state.runtime_status == WorkflowStatus.COMPLETED:
                    # Workflow completed normally; workflow.py already closes the trace.
                    trace = None
                    if state.serialized_output:
                        output_dict = (
                            json.loads(state.serialized_output)
                            if isinstance(state.serialized_output, str)
                            else state.serialized_output
                        )
                        output = GraphWorkflowOutput.from_dict(output_dict)
                        if output.status in FAILED_STATUSES:
                            # Not a result: a run that hit its step limit or
                            # failed in a node raises.
                            raise run_error(
                                normalize_final(
                                    {
                                        "type": "workflow_completed",
                                        "status": output.status,
                                        "error": output.error,
                                        "output": output.output,
                                    },
                                    workflow_id=workflow_id,
                                    thread_id=thread_id,
                                )
                            )
                        if self._is_root_state() and ROOT in output.output:
                            return output.output[ROOT]
                        return output.output
                    return {}

                elif state.runtime_status == WorkflowStatus.FAILED:
                    error_msg = "Workflow failed"
                    if state.failure_details:
                        error_msg = getattr(
                            state.failure_details, "message", str(state.failure_details)
                        )
                    raise RuntimeError(error_msg)

                elif state.runtime_status == WorkflowStatus.TERMINATED:
                    raise RuntimeError(f"Workflow {workflow_id} was terminated")
        finally:
            if trace:
                trace.end()

    async def run_async(
        self,
        input: Any,
        *,
        thread_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Run the graph asynchronously with event streaming.

        Args:
            input: Input values for the graph channels
            thread_id: Thread identifier for the execution
            config: Optional LangGraph config dict, with the rules of
                :meth:`invoke`: JSON only, ``callbacks`` and values that are not
                JSON raise ``ValueError``, and ``recursion_limit`` bounds the steps
            workflow_id: Optional workflow instance ID (generated if not provided)
            poll_interval: How often to poll for workflow status (seconds)

        Yields:
            Event dictionaries with workflow progress updates
        """
        run = self._prepare_run(input, thread_id, config, workflow_id)
        workflow_id, thread_id, trace = run.workflow_id, run.thread_id, run.trace

        logger.info(f"Starting workflow: {workflow_id}")
        assert self._workflow_client is not None
        # Off the loop: this is a blocking gRPC call (see _poll_workflow).
        await asyncio.to_thread(
            self._workflow_client.schedule_new_workflow,
            workflow=agent_workflow,
            input=run.workflow_input,
            instance_id=workflow_id,
        )

        try:
            yield {
                "type": "workflow_started",
                "workflow_id": workflow_id,
                "thread_id": thread_id,
                "graph_name": self._name,
            }

            # Which nodes ran in each step is for the stream (stream_unavailable),
            # not for a caller of run_async: only a streamed run asks for it.
            async for event in self._follow_run(
                workflow_id,
                thread_id,
                poll_interval=poll_interval,
                streamed=bool(run.config.get(STREAM_RUN_KEY)),
            ):
                yield event

        finally:
            if trace:
                trace.end()

    async def _follow_run(
        self,
        workflow_id: str,
        thread_id: str,
        *,
        poll_interval: float,
        streamed: bool,
        after_seq: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Poll a scheduled run; its status changes, then its terminal event.

        A run paused by ``interrupt()`` (on an interrupt newer than
        ``after_seq``) ends the poll with ``workflow_interrupted``.
        """
        async for event in self._poll_workflow(
            workflow_id,
            thread_id,
            poll_interval=poll_interval,
            parse_output=lambda wf_id, out: parse_output(wf_id, out, streamed),
            # A poll that fails is reported as such (PollError) while the
            # run may still finish: ride out a restarting sidecar first.
            retry_failed_polls=True,
            running_event=lambda state: interrupted_event(
                parse_status(state.serialized_custom_status), workflow_id, after_seq
            ),
        ):
            yield event

    async def astream_resume(
        self,
        workflow_id: str,
        last_event_id: Optional[str] = None,
        *,
        stream_mode: Any = "updates",
        poll_interval: float = 0.5,
        on_control: Optional[Callable[[StreamControl], None]] = None,
    ) -> AsyncIterator[Any]:
        """Stream a run that was already started, from ``last_event_id`` on.

        ``last_event_id`` is the SSE id of the last event the client got (a
        cursor of the state-store transport); ``None`` replays from the start,
        and the end id gives the final event only. Chunks are yielded as
        :meth:`astream` yields them, from any replica, for a run still going
        or one that ended (within the log's TTL). On the in-process transport
        nothing is replayed: ``stream_unavailable`` with reason
        ``transport_not_replayable``, then the run's end. Raises
        ``LookupError`` for an unknown run and :class:`GraphRunError` for a
        run that failed.
        """
        modes, is_list = normalize_stream_mode(stream_mode)
        async with aclosing(
            resume_events(
                self,
                workflow_id,
                last_event_id,
                poll_interval=poll_interval,
                keepalive=False,
            )
        ) as events:
            async for item in events:
                outs, error = route_astream_item(item, modes, is_list, on_control)
                for out in outs:
                    yield out
                if error is not None:
                    raise error

    async def astream(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        stream_mode: Any = "updates",
        subgraphs: bool = False,
        poll_interval: float = 0.5,
        on_control: Optional[Callable[[StreamControl], None]] = None,
    ) -> AsyncIterator[Any]:
        """Stream a run like ``CompiledStateGraph.astream``.

        ``messages``, ``updates``, ``custom``, ``values``, ``tasks`` and
        ``debug`` are supported (``checkpoints`` is refused: there are no
        LangGraph checkpoints); one mode yields bare chunks, a list of modes
        yields ``(mode, chunk)``. The chunks have
        the shape native ``astream`` yields: ``messages`` are
        ``(message_chunk, metadata)``, ``updates`` are ``{node: update}`` with
        the values as the node returned them (``None`` for a node that wrote
        nothing), and a parallel step yields one ``updates`` chunk per node.
        ``config`` follows the rules of :meth:`invoke`, and the run's thread id
        is ``thread_id``, else ``config["configurable"]["thread_id"]``, else
        new: the same id is in ``workflow_started``, the ``messages`` metadata
        and the run's config.

        Events about the stream itself (retries, missing chunks, a consumer that
        fell behind) are :class:`StreamControl` objects, given to ``on_control``
        and, in list mode only, yielded as ``(kind, data)``. Dapr retries a
        failed node activity and its partial chunks cannot be taken back: a
        single-mode consumer without ``on_control`` logs a warning for each
        ``node_reset`` and sees the partial text. A run that fails raises
        :class:`GraphRunError`.

        Close the stream to stop early (``async with aclosing(...)``): a bare
        ``break`` leaves it open and its ``workflow_id`` reserved. Closing does
        not cancel the Dapr workflow, which runs to completion.
        """
        if subgraphs:
            raise NotImplementedError("subgraphs=True is not supported yet")
        modes, is_list = normalize_stream_mode(stream_mode)
        if isinstance(input, Command):
            if input.resume is None or not workflow_id:
                raise ValueError(
                    "astream(Command(resume=...)) resumes a run paused by "
                    "interrupt(): pass the run's workflow_id"
                )
            events: Any = resume_run_events(
                self,
                workflow_id,
                input.resume,
                poll_interval=poll_interval,
                keepalive=False,
            )
        else:
            events = self._stream_events(
                input,
                config=config,
                thread_id=thread_id,
                workflow_id=workflow_id,
                poll_interval=poll_interval,
            )
        # aclosing: if the caller stops early, close the inner stream now (it
        # unregisters the sink) instead of whenever the garbage collector runs.
        async with aclosing(events) as stream:
            async for item in stream:
                outs, error = route_astream_item(item, modes, is_list, on_control)
                for out in outs:
                    yield out
                if error is not None:
                    raise error

    async def _stream_events(
        self,
        input: Any,
        *,
        config: Optional[Dict[str, Any]],
        thread_id: Optional[str],
        workflow_id: Optional[str],
        poll_interval: float,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
        keepalive: bool = False,
    ) -> AsyncGenerator[Any, None]:
        """Yield ``StreamEvent`` chunks, then the terminal event dict.

        The terminal event is always the last item: ``workflow_completed`` or
        ``workflow_failed`` (see :func:`normalize_final`), carrying the run's
        ``workflow_id`` and ``thread_id``.

        ``on_result`` is called once, when the run ends, with the result a plain
        ``POST /agent/run`` would return for it (not the normalized terminal
        event), so a streamed run publishes exactly what a plain one does.

        The first item is a ``workflow_started`` event (``workflow_id`` and
        ``thread_id``). With ``keepalive`` set, :data:`KEEPALIVE` is yielded
        whenever nothing arrived for ``SSE_KEEPALIVE_SECONDS``.
        """
        thread_id, workflow_id, run_config = _run_identity(
            config, self._run_config(config), thread_id, workflow_id
        )
        run_config = {**run_config, STREAM_RUN_KEY: True}
        seen_nodes: Set[Tuple[int, str]] = set()
        source = open_source(
            self._stream_transport,
            workflow_id,
            status=lambda: custom_status(self._workflow_client, workflow_id),
            seen=seen_nodes,
        )
        task: Optional["asyncio.Task[None]"] = None
        scheduled = asyncio.Event()
        try:
            task = asyncio.create_task(
                self._pump_run(
                    source.on_final,
                    input,
                    thread_id,
                    workflow_id,
                    run_config,
                    poll_interval,
                    on_result,
                    scheduled,
                )
            )
            # Only once the workflow exists: a client (or another replica) that
            # looks the run up on this event must find it. Chunks an activity
            # writes meanwhile wait in the sink, so they still come after it.
            if await _wait_scheduled(scheduled, task):
                yield started_event(workflow_id, thread_id)
            async for item in drain(
                source,
                workflow_id=workflow_id,
                seen=seen_nodes,
                keepalive=keepalive,
                output_channels=self._graph_config.output_channels,
            ):
                yield item
        finally:
            # Also reached when the consumer goes away (client disconnect,
            # abandoned generator). That stops this stream only: the Dapr
            # workflow itself keeps running to completion.
            await source.close()
            await self._release_pump(task, detach=on_result is not None)

    async def _pump_run(
        self,
        on_final: Callable[[Dict[str, Any]], None],
        input: Any,
        thread_id: str,
        workflow_id: str,
        run_config: Dict[str, Any],
        poll_interval: float,
        on_result: Optional[Callable[[Dict[str, Any]], None]],
        scheduled: Optional[asyncio.Event] = None,
    ) -> None:
        """Follow the run, publish its result, and hand ``on_final`` the final.

        ``scheduled`` is set once ``run_async`` reports the workflow started,
        which it does after ``schedule_new_workflow`` returned.
        """
        final: Dict[str, Any] = {}
        result: Dict[str, Any] = {}
        try:
            async for event in self.run_async(
                input,
                thread_id=thread_id,
                config=run_config,
                workflow_id=workflow_id,
                poll_interval=poll_interval,
            ):
                if event["type"] == "workflow_started" and scheduled is not None:
                    scheduled.set()
                fold_run_event(result, without_executed(event))
                if event["type"] in FINAL_EVENTS:
                    final = event
            if on_result is not None:
                await asyncio.to_thread(on_result, result)
        except Exception as e:
            logger.exception("Streamed run %s failed", workflow_id)
            final = stream_failure_event(
                e, "PollError", bool(result.get("instance_id"))
            )
        on_final(normalize_final(final, workflow_id=workflow_id, thread_id=thread_id))

    async def _release_pump(
        self, task: Optional["asyncio.Task[None]"], *, detach: bool
    ) -> None:
        """Let a pump whose consumer left finish (``detach``) or stop it."""
        if task is None or task.done():
            return
        if detach:
            # Someone is waiting for the result on a topic: let the pump finish
            # the run and hand it over, with no consumer.
            self._detached_pumps.add(task)
            task.add_done_callback(self._detached_pumps.discard)
        else:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _status_result(self, state: Any) -> Dict[str, Any]:
        """A completed run's result, or a failed workflow's error."""
        return (
            completed_result(state)
            or interrupted_result(state)
            or super()._status_result(state)
        )

    def _is_root_state(self) -> bool:
        """Whether the graph's whole state is one ``__root__`` channel."""
        channels = getattr(getattr(self._graph, "builder", None), "channels", None)
        return isinstance(channels, dict) and list(channels) == [ROOT]

    def _channel_input(self, input: Any) -> Dict[str, Any]:
        """``input`` as values of channels.

        A graph whose state is one value takes that value as its input, so it is
        the value of the ``__root__`` channel.
        """
        if self._is_root_state() and not (
            isinstance(input, dict) and set(input) == {ROOT}
        ):
            return {ROOT: input}
        return input  # type: ignore[no-any-return]

    def _serialize_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize input values for storage."""
        from .workflow import _serialize_input

        return _serialize_input(input)

    # ------------------------------------------------------------------
    # Serve overrides
    # ------------------------------------------------------------------

    def serve(
        self,
        *,
        port: int = 5001,
        host: str = "0.0.0.0",
        input_mapper: Optional["Callable[[dict], Dict[str, Any]]"] = None,
        pubsub_name: Optional[str] = None,
        subscribe_topic: Optional[str] = None,
        publish_topic: Optional[str] = None,
    ) -> None:
        """Start an HTTP server exposing ``/agent/run``.

        ``POST /agent/run`` with ``"stream": true`` answers as server-sent events
        (``workflow_started``, the chunks ``stream_mode`` picks, then
        ``workflow_completed`` or ``workflow_failed``); the streaming README
        has the details. With a state store the stream carries every node
        whichever replica ran it, and ``GET /agent/run/{workflow_id}/stream``
        reconnects to it (``Last-Event-ID``); without one, streaming works on a
        single replica only. ``POST /agent/run/{workflow_id}/resume`` resumes a
        run paused by ``interrupt()``. A client that disconnects ends the
        stream, not the workflow, and with ``publish_topic`` set the result is
        still published when the run ends.

        ``input_mapper`` maps the request dict to the graph's input.
        """
        self._input_mapper = input_mapper
        super().serve(
            port=port,
            host=host,
            pubsub_name=pubsub_name,
            subscribe_topic=subscribe_topic,
            publish_topic=publish_topic,
        )

    def _setup_telemetry(self) -> None:
        from diagrid.agent.core.telemetry import setup_telemetry, instrument_grpc

        setup_telemetry(self.__class__.__name__, config=self._observability_config)
        instrument_grpc(config=self._observability_config)

    def _setup_serve_defaults(self) -> None:
        input_mapper = getattr(self, "_input_mapper", None)
        # A run an orchestrator starts has no caller config: it gets the graph's
        # own, as every entry point of this runner gives it.
        run_config = self._run_config(None)
        bound = self._step_bound(None)
        set_default_graph_config(
            self._graph_config,
            input_mapper=input_mapper,
            max_steps=bound.max_steps,
            strict_limit=bound.strict,
            run_config=run_config,
        )

    async def _serve_run(
        self,
        request: dict,
        session_id: str,  # type: ignore[type-arg]
    ) -> AsyncIterator[dict[str, Any]]:
        thread_id = request.get("thread_id") or request.get("session_id") or session_id

        graph_input = self._request_to_input(request)

        async for event in self.run_async(input=graph_input, thread_id=thread_id):
            yield event

    _supports_serve_stream = True

    _DEFAULT_STREAM_MODES = ["messages", "updates", "custom"]

    def _validate_stream_request(self, request: dict) -> None:  # type: ignore[type-arg]
        """Reject an unsupported ``stream_mode`` before the response starts."""
        normalize_stream_mode(request.get("stream_mode") or self._DEFAULT_STREAM_MODES)

    async def _serve_stream(  # type: ignore[override]
        self,
        request: dict,  # type: ignore[type-arg]
        session_id: str,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Run the graph and yield its stream as ``(event_name, data)`` pairs.

        ``stream_mode`` in the request (a mode or list of modes) selects what
        is sent; the default is all of ``messages``, ``updates`` and ``custom``.
        Ends with ``workflow_completed`` or ``workflow_failed``, whose data is
        the run's full terminal event. The base runner frames the pairs.
        """
        thread_id = request.get("thread_id") or request.get("session_id") or session_id
        encoder: Optional[SseEncoder] = None
        try:
            modes, _ = normalize_stream_mode(
                request.get("stream_mode") or self._DEFAULT_STREAM_MODES
            )
            encoder = SseEncoder(modes, self._stream_transport.replayable)
            graph_input = self._request_to_input(request, streaming=True)
            async with aclosing(
                self._stream_events(
                    graph_input,
                    config=None,
                    thread_id=thread_id,
                    workflow_id=None,
                    poll_interval=0.5,
                    on_result=on_result,
                    keepalive=True,
                )
            ) as events:
                async for item in events:
                    out = encoder.encode(item)
                    if out is not None:
                        yield out
        except Exception as e:
            logger.exception("Streaming run failed")
            workflow_id = encoder.workflow_id if encoder else None
            failed = stream_failure_event(e, "StreamError", bool(workflow_id))
            yield (
                "workflow_failed",
                normalize_final(failed, workflow_id=workflow_id, thread_id=thread_id),
            )

    def _add_routes(self, app: Any) -> None:
        """Reconnect to a run's stream; resume a run paused by ``interrupt()``."""
        add_reconnect_route(app, self)
        add_resume_route(app, self)

    def _request_to_input(
        self,
        request: dict,  # type: ignore[type-arg]
        *,
        streaming: bool = False,
    ) -> Dict[str, Any]:
        """Map a ``POST /agent/run`` body to the graph's input.

        ``stream`` and ``stream_mode`` are transport keys of a streaming
        request and are removed from it (before an ``input_mapper`` sees it, or
        not) so they never reach the graph. On a plain request they are left
        alone: a graph may have a state key of that name.
        """
        if streaming:
            request = {k: v for k, v in request.items() if k not in _TRANSPORT_KEYS}
        input_mapper = getattr(self, "_input_mapper", None)
        if input_mapper:
            return input_mapper(request)  # type: ignore[no-any-return]
        if "messages" not in request:
            task = request.get("task")
            if task:
                return {"messages": [{"role": "user", "content": task}]}
        return request

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stream_transport(self) -> StreamTransport:
        """How this runner's streamed runs get their chunks (see ``__init__``)."""
        return self._stream_transport

    @property
    def stream_registry(self) -> StreamSinkRegistry:
        """Where this runner's streamed runs register their sinks."""
        return self._stream_registry

    @property
    def graph(self) -> "CompiledStateGraph":
        """The LangGraph being executed."""
        return self._graph

    @property
    def graph_config(self) -> GraphConfig:
        """The extracted graph configuration."""
        return self._graph_config
