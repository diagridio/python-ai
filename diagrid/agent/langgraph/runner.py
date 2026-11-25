# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runner for executing LangGraph graphs as Dapr Workflows."""

import asyncio
import json
import logging
import time
import uuid
from contextlib import aclosing
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

from diagrid.agent.core.types.type import SupportedFrameworks
from diagrid.agent.core.workflow import BaseWorkflowRunner
from diagrid.agent.core.workflow.runner import fold_run_event
from diagrid.agent.core.workflow.stream import (
    CONTROL_MODES,
    KEEPALIVE_EVENT,
    WORKFLOW_STARTED,
    InProcessSink,
    StreamClosed,
    StreamEvent,
    StreamSinkRegistry,
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
    EXECUTED_KEY,
    STREAM_RUN_KEY,
    UPDATES,
    StaticConfig,
    StreamControl,
    normalize_final,
    normalize_stream_mode,
)
from .stream_events import (
    FINAL_EVENTS,
    KEEPALIVE,
    closing_events,
    count_unencodable,
    encode_chunk,
    failure_event,
    next_item,
    route_astream_item,
    started_event,
    without_executed,
)
from .workflow import (
    agent_workflow,
    make_condition_activity,
    GENERIC_NODE_ACTIVITY,
    make_node_activity,
    node_activity_name,
    register_node,
    register_node_static_config,
    register_node_triggers,
    register_condition,
    register_channel_reducer,
    register_state_channels,
    set_default_graph_config,
    set_serializer,
    clear_registries,
    START,
    END,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

_TRANSPORT_KEYS = ("stream", "stream_mode")


# What a reducer channel holds before its first write; LangGraph's own marker
# when the channel type has no empty value.
_NO_DEFAULT: Any = object()
try:
    from langgraph.channels.binop import MISSING as _LANGGRAPH_MISSING
except ImportError:  # pragma: no cover - a LangGraph without the marker
    _LANGGRAPH_MISSING = _NO_DEFAULT


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


def _max_steps_for(config: Optional[Dict[str, Any]], default: int) -> int:
    """The workflow's step bound: ``config['recursion_limit']`` as LangGraph counts it.

    LangGraph allows ``recursion_limit - 1`` node steps (its input is the
    first), so a graph of N sequential nodes needs a limit of N + 1.
    """
    limit = (config or {}).get("recursion_limit")
    return limit - 1 if isinstance(limit, int) else default


def _run_identity(
    config: Optional[Dict[str, Any]],
    thread_id: Optional[str],
    workflow_id: Optional[str],
) -> Tuple[str, str, Dict[str, Any]]:
    """``(thread_id, workflow_id, config)`` for a streamed run.

    The thread id is the one given, else ``config["configurable"]["thread_id"]``,
    else new. The config handed to the run carries that one id, so the id the
    run reports is the one its nodes see.
    """
    _check_run_config(config)
    configurable = (config or {}).get("configurable") or {}
    thread = str(thread_id or configurable.get("thread_id") or uuid.uuid4())
    run_config = {
        **(config or {}),
        "configurable": {**configurable, "thread_id": thread},
        STREAM_RUN_KEY: True,
    }
    return thread, workflow_id or f"graph-{thread}-{uuid.uuid4().hex[:8]}", run_config


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
        max_steps: int = 100,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        instructions: Optional[str] = None,
        registry_config: Optional[Any] = None,
        stream_registry: Optional[StreamSinkRegistry] = None,
    ):
        """Initialize the runner.

        Args:
            graph: The compiled LangGraph to execute
            name: Required name for the graph
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_steps: Maximum number of steps before stopping (default: 100)
            role: Optional role description for registry (e.g. "Schedule Planner")
            goal: Optional goal description for registry
            instructions: Optional system prompt for registry, also used as the
                goal when ``goal`` is not given. Without it, the registry looks
                for a module-level ``SystemMessage`` in node globals.
            registry_config: Optional registry configuration for metadata extraction
            stream_registry: Where streamed runs register their sinks. The
                activities this runner registers write into it. Defaults to a
                new registry owned by this runner.
        """
        self._graph = graph
        self._max_steps = max_steps
        self._stream_registry = (
            stream_registry if stream_registry is not None else StreamSinkRegistry()
        )
        # Pumps of streams whose consumer left while a result is still to be
        # published; held here so they are not collected before they finish.
        self._detached_pumps: Set["asyncio.Task[None]"] = set()

        super().__init__(
            name,
            framework=self._REGISTRY_FRAMEWORK,
            host=host,
            port=port,
            max_iterations=max_steps,
        )

        # Attach metadata hints for the registry mapper
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
        self._graph._diagrid_max_steps = max_steps  # type: ignore[attr-defined]

        # Register metadata
        self._register_agent_metadata(
            agent=self._graph,
            framework=self._REGISTRY_FRAMEWORK,
            registry=registry_config,
            name=self._name,
        )

        # Register workflow and activities
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
            make_node_activity(GENERIC_NODE_ACTIVITY, self._stream_registry),
            name=GENERIC_NODE_ACTIVITY,
        )
        self._workflow_runtime.register_activity(
            make_condition_activity(self._stream_registry),
            name="evaluate_condition_activity",
        )

    def _register_node_activities(self) -> None:
        """Register one activity per graph node, named after the node."""
        for node in self._graph_config.nodes:
            name = node_activity_name(node.name)
            if name == GENERIC_NODE_ACTIVITY:
                continue
            self._workflow_runtime.register_activity(
                make_node_activity(node.name, self._stream_registry), name=name
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
                if default is _NO_DEFAULT or default is _LANGGRAPH_MISSING:
                    register_channel_reducer(channel_name, reducer)
                else:
                    register_channel_reducer(channel_name, reducer, default)
                logger.info(f"Registered reducer for channel: {channel_name}")

        # A node's keys that are not state channels are dropped, as LangGraph
        # does (CompiledStateGraph's ``_get_updates``).
        state_channels = getattr(
            getattr(self._graph, "builder", None), "channels", None
        )
        if isinstance(state_channels, dict):
            managed = getattr(self._graph.builder, "managed", None) or {}
            register_state_channels([*state_channels, *managed])

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
        input: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run the graph synchronously and return the final output.

        Args:
            input: Input values for the graph channels
            thread_id: Thread identifier for the execution
            config: Optional LangGraph config dict
            workflow_id: Optional workflow instance ID (generated if not provided)
            poll_interval: How often to poll for workflow status (seconds)
            timeout: Optional timeout in seconds

        Returns:
            Final output values from the graph
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        thread_id = thread_id or str(uuid.uuid4())
        workflow_id = workflow_id or f"graph-{thread_id}-{uuid.uuid4().hex[:8]}"

        _check_run_config(config)
        config = config or {}
        config["thread_id"] = thread_id

        # Create LangSmith parent trace and pass dotted_order to activities via config.
        # The finally on the polling loop below closes the trace on FAILED/TERMINATED.
        # On normal completion, _close_langsmith_parent_trace_async in workflow.py also
        # closes it (benign double-close since update_run is idempotent).
        _ls_run_id = None
        _ls_client = None
        try:
            import os as _os

            if _os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1"):
                from langsmith import Client as _LsClient
                from datetime import datetime, timezone

                _ls_client = _LsClient()
                _ls_run_id = str(uuid.uuid4())
                _now = datetime.now(timezone.utc)
                _ts = _now.strftime("%Y%m%dT%H%M%S") + f"{_now.microsecond:06d}Z"
                _dotted_order = f"{_ts}{_ls_run_id}"
                _ls_client.create_run(
                    id=_ls_run_id,
                    name="LangGraph",
                    run_type="chain",
                    inputs=input,
                    dotted_order=_dotted_order,
                    trace_id=_ls_run_id,
                    extra={"metadata": {"thread_id": thread_id}},
                )
                config["langsmith_dotted_order"] = _dotted_order
                config["langsmith_run_id"] = _ls_run_id
        except Exception as e:
            logger.debug(f"LangSmith parent trace creation failed: {e}")
            _ls_run_id = None
            _ls_client = None

        channel_state = ChannelState(
            values=self._serialize_input(input),
            versions={k: 1 for k in input.keys()},
            updated_channels=list(input.keys()),
        )

        workflow_input = GraphWorkflowInput(
            graph_config=self._graph_config,
            channel_state=channel_state,
            step=0,
            max_steps=_max_steps_for(config, self._max_steps),
            pending_nodes=[],
            config=config,
            thread_id=thread_id,
        )

        workflow_input_dict = workflow_input.to_dict()
        json.dumps(workflow_input_dict)

        logger.info(f"Starting workflow: {workflow_id}")
        self._workflow_client.schedule_new_workflow(
            workflow=agent_workflow,
            input=workflow_input_dict,
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

                if state.runtime_status == WorkflowStatus.COMPLETED:
                    # Workflow completed normally; workflow.py already closes the trace.
                    _ls_run_id = None
                    if state.serialized_output:
                        output_dict = (
                            json.loads(state.serialized_output)
                            if isinstance(state.serialized_output, str)
                            else state.serialized_output
                        )
                        output = GraphWorkflowOutput.from_dict(output_dict)
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
            if _ls_run_id and _ls_client:
                try:
                    from datetime import datetime, timezone

                    _ls_client.update_run(
                        _ls_run_id, end_time=datetime.now(timezone.utc)
                    )
                    _ls_client.flush()
                except Exception as e:
                    logger.debug(f"LangSmith parent trace close failed: {e}")

    async def run_async(
        self,
        input: Dict[str, Any],
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
            config: Optional LangGraph config dict
            workflow_id: Optional workflow instance ID (generated if not provided)
            poll_interval: How often to poll for workflow status (seconds)

        Yields:
            Event dictionaries with workflow progress updates
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        thread_id = thread_id or str(uuid.uuid4())
        workflow_id = workflow_id or f"graph-{thread_id}-{uuid.uuid4().hex[:8]}"

        _check_run_config(config)
        config = config or {}
        config["thread_id"] = thread_id

        # Create LangSmith parent trace and pass dotted_order to activities via config.
        # The trace is closed by _close_langsmith_parent_trace_async in workflow.py on
        # normal completion. The finally below closes it if the caller abandons the generator.
        _ls_run_id = None
        _ls_client = None
        try:
            import os as _os

            if _os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1"):
                from langsmith import Client as _LsClient
                from datetime import datetime, timezone

                _ls_client = _LsClient()
                _ls_run_id = str(uuid.uuid4())
                _now = datetime.now(timezone.utc)
                _ts = _now.strftime("%Y%m%dT%H%M%S") + f"{_now.microsecond:06d}Z"
                _dotted_order = f"{_ts}{_ls_run_id}"
                _ls_client.create_run(
                    id=_ls_run_id,
                    name="LangGraph",
                    run_type="chain",
                    inputs=input,
                    dotted_order=_dotted_order,
                    trace_id=_ls_run_id,
                    extra={"metadata": {"thread_id": thread_id}},
                )
                config["langsmith_dotted_order"] = _dotted_order
                config["langsmith_run_id"] = _ls_run_id
        except Exception as e:
            logger.debug(f"LangSmith parent trace creation failed: {e}")
            _ls_run_id = None
            _ls_client = None

        channel_state = ChannelState(
            values=self._serialize_input(input),
            versions={k: 1 for k in input.keys()},
            updated_channels=list(input.keys()),
        )

        workflow_input = GraphWorkflowInput(
            graph_config=self._graph_config,
            channel_state=channel_state,
            step=0,
            max_steps=_max_steps_for(config, self._max_steps),
            pending_nodes=[],
            config=config,
            thread_id=thread_id,
        )

        workflow_input_dict = workflow_input.to_dict()
        json.dumps(workflow_input_dict)

        logger.info(f"Starting workflow: {workflow_id}")
        # Off the loop: this is a blocking gRPC call (see _poll_workflow).
        await asyncio.to_thread(
            self._workflow_client.schedule_new_workflow,
            workflow=agent_workflow,
            input=workflow_input_dict,
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
            streamed = bool(config.get(STREAM_RUN_KEY))

            def _parse_output(wf_id: str, output_dict: dict) -> dict:  # type: ignore[type-arg]
                output = GraphWorkflowOutput.from_dict(output_dict)
                with_executed = streamed and bool(output.executed)
                return {
                    "type": "workflow_completed",
                    "workflow_id": wf_id,
                    "output": output.output,
                    "steps": output.steps,
                    "status": output.status,
                    **({"error": output.error} if output.error else {}),
                    **({EXECUTED_KEY: output.executed} if with_executed else {}),
                }

            async for event in self._poll_workflow(
                workflow_id,
                thread_id,
                poll_interval=poll_interval,
                parse_output=_parse_output,
                # A poll that fails is reported as such (PollError) while the
                # run may still finish: ride out a restarting sidecar first.
                retry_failed_polls=True,
            ):
                yield event

        finally:
            if _ls_run_id and _ls_client:
                try:
                    from datetime import datetime, timezone

                    _ls_client.update_run(
                        _ls_run_id, end_time=datetime.now(timezone.utc)
                    )
                    _ls_client.flush()
                except Exception as e:
                    logger.debug(f"LangSmith parent trace close failed: {e}")

    async def astream(
        self,
        input: Dict[str, Any],
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

        ``config`` is positional or keyword, as in LangGraph. The run's thread
        id is ``thread_id`` if given, else ``config["configurable"]["thread_id"]``,
        else a new one; the same id is in the ``workflow_started`` control, the
        ``messages`` metadata and the run's config.

        Supports ``messages``, ``updates`` and ``custom``. A single mode
        yields bare chunks, as native ``astream`` does, and a list of modes
        yields ``(mode, chunk)`` tuples.

        Events about the stream itself are :class:`StreamControl` objects:
        ``node_reset`` (a node was retried; discard its partial output),
        ``stream_unavailable`` (the run finished but chunks for some of its
        nodes never arrived: streaming works only when the node activities run
        in this process, i.e. a single replica) and ``stream_dropped`` (the
        consumer fell behind; ``{"dropped": n}``). Pass ``on_control`` to
        receive them. A list of modes also yields them as ``(kind, data)``
        tuples; a single mode never does, so a consumer written against native
        ``astream`` (``async for message, metadata in ...``) cannot trip over
        one. ``on_control`` also gets a ``workflow_started`` control first, with
        the ``workflow_id`` and ``thread_id`` to use with ``terminate_workflow``
        or ``get_workflow_status``; it is never yielded. A run that fails raises
        :class:`GraphRunError`.

        Dapr retries a failed node activity, and the chunks the failed attempt
        already sent cannot be taken back. Native ``astream`` has no retries,
        so a single-mode consumer that passes no ``on_control`` would see the
        partial text of the failed attempt followed by the retry's. It is not
        left unaware: a warning is logged for each such ``node_reset``. To
        discard the partial output, pass ``on_control`` or use a list of modes
        and drop what the node sent before its ``node_reset``.

        To stop early, close the stream: ``await stream.aclose()``, or iterate
        inside ``async with contextlib.aclosing(runner.astream(...)) as stream``.
        A bare ``break`` does not close an async generator that is still
        referenced, so the run's stream stays open (and its ``workflow_id``
        reserved) until it is closed or garbage collected. Closing does not
        cancel the Dapr workflow, which runs to completion.

        ``config`` is a LangGraph config that travels to the workflow as JSON:
        ``tags``, ``metadata``, ``configurable`` and ``recursion_limit`` work
        (the limit bounds the run's steps as LangGraph counts them; the
        runner's ``max_steps`` applies when it is absent). ``callbacks`` and any
        value that is not JSON raise ``ValueError`` before a run starts.

        Chunks have the shape ``CompiledStateGraph.astream`` yields:
        ``messages`` are ``(message_chunk, metadata)`` and ``updates`` are
        ``{node: update}`` with the values exactly as the node returned them
        (``None`` for a node that wrote nothing); like LangGraph, a parallel
        step yields one ``updates`` chunk per node. Over HTTP
        (``"stream": true``) the same chunks are serialized with
        ``langchain_core.load.dumpd``.
        """
        if subgraphs:
            raise NotImplementedError("subgraphs=True is not supported yet")
        modes, is_list = normalize_stream_mode(stream_mode)
        # aclosing: if the caller stops early, close the inner stream now (it
        # unregisters the sink) instead of whenever the garbage collector runs.
        async with aclosing(
            self._stream_events(
                input,
                config=config,
                thread_id=thread_id,
                workflow_id=workflow_id,
                poll_interval=poll_interval,
            )
        ) as events:
            async for item in events:
                outs, error = route_astream_item(item, modes, is_list, on_control)
                for out in outs:
                    yield out
                if error is not None:
                    raise error

    async def _stream_events(
        self,
        input: Dict[str, Any],
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
            config, thread_id, workflow_id
        )
        sink = self._stream_registry.open(workflow_id)
        task: Optional["asyncio.Task[None]"] = None
        seen_nodes: Set[Tuple[int, str]] = set()
        try:
            # Before the pump schedules the workflow: an activity can write the
            # moment it is scheduled, and the event must come before its chunks.
            sink.put(started_event({}, workflow_id, thread_id))
            task = asyncio.create_task(
                self._pump_run(
                    sink,
                    input,
                    thread_id,
                    workflow_id,
                    run_config,
                    poll_interval,
                    on_result,
                )
            )
            while True:
                item = await next_item(sink, keepalive)
                if isinstance(item, StreamClosed):
                    # Unregister before handing over the result: chunks a late
                    # activity attempt writes from here on are dropped.
                    self._stream_registry.close(workflow_id, sink)
                    for event in closing_events(item, seen_nodes, workflow_id):
                        yield event
                    return
                if isinstance(item, StreamEvent) and item.mode == UPDATES:
                    seen_nodes.add((item.step, item.node))
                yield item
        finally:
            # Also reached when the consumer goes away (client disconnect,
            # abandoned generator). That stops this stream only: the Dapr
            # workflow itself keeps running to completion.
            self._stream_registry.close(workflow_id, sink)
            await self._release_pump(task, detach=on_result is not None)

    async def _pump_run(
        self,
        sink: InProcessSink,
        input: Dict[str, Any],
        thread_id: str,
        workflow_id: str,
        run_config: Dict[str, Any],
        poll_interval: float,
        on_result: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        """Follow the run, publish its result, and close ``sink`` with the final."""
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
                fold_run_event(result, without_executed(event))
                if event["type"] in FINAL_EVENTS:
                    final = event
            if on_result is not None:
                await asyncio.to_thread(on_result, result)
        except Exception as e:
            logger.exception("Streamed run %s failed", workflow_id)
            # If the workflow was started this is the stream failing (its
            # status polls gave out), not the workflow: it may still complete.
            # Say so, and publish nothing: a failure published for a run that
            # then succeeds is a lie.
            started = bool(result.get("instance_id"))
            final = failure_event(e, "PollError" if started else None)
            if started:
                final["workflow_still_running"] = True
        sink.close(normalize_final(final, workflow_id=workflow_id, thread_id=thread_id))

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
        """Start an HTTP server exposing /agent/run endpoints.

        ``POST /agent/run`` with ``"stream": true`` answers as server-sent
        events: ``workflow_started`` first, then the ``messages``, ``updates``
        and ``custom`` chunks picked by ``stream_mode``, then
        ``workflow_completed`` or ``workflow_failed``. The keys ``stream`` and
        ``stream_mode`` are reserved on such a request. Streaming works only on
        a single replica: chunks reach the connection through an in-process
        queue, so a node activity that runs in another process streams nothing
        and the stream ends with ``stream_unavailable``. A client that
        disconnects ends the stream, not the workflow; with ``publish_topic``
        set its result is still published when the run ends.

        Args:
            port: Port to listen on (default: 5001)
            host: Host to bind to (default: 0.0.0.0)
            input_mapper: Optional function to map request dict to graph input.
            pubsub_name: Dapr pub/sub component name.
            subscribe_topic: Topic to subscribe to for incoming tasks.
            publish_topic: Topic to publish results to.
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
        set_default_graph_config(
            self._graph_config,
            input_mapper=input_mapper,
            max_steps=self._max_steps,
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
        workflow_id: Optional[str] = None
        unencodable = 0
        try:
            modes, _ = normalize_stream_mode(
                request.get("stream_mode") or self._DEFAULT_STREAM_MODES
            )
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
                    if item is KEEPALIVE:
                        yield KEEPALIVE_EVENT, None
                    elif isinstance(item, StreamEvent):
                        if item.mode not in CONTROL_MODES and item.mode not in modes:
                            continue
                        if item.mode == WORKFLOW_STARTED:
                            workflow_id = item.data["workflow_id"]
                        wire, ok = encode_chunk(item)
                        unencodable += 0 if ok else 1
                        yield item.mode, wire
                    else:
                        yield item["type"], count_unencodable(item, unencodable)
        except Exception as e:
            logger.exception("Streaming run failed")
            # Once the workflow has started this is the stream failing, not the
            # run: it goes on, and may well succeed (see _pump_run).
            failed = failure_event(e, "StreamError" if workflow_id else None)
            if workflow_id:
                failed["workflow_still_running"] = True
            yield (
                "workflow_failed",
                normalize_final(failed, workflow_id=workflow_id, thread_id=thread_id),
            )

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
