# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runner for executing LangGraph graphs as Dapr Workflows."""

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, TYPE_CHECKING

from dapr.ext.workflow import WorkflowStatus

from diagrid.agent.core.workflow import BaseWorkflowRunner
from .models import (
    ChannelState,
    EdgeConfig,
    GraphConfig,
    GraphWorkflowInput,
    GraphWorkflowOutput,
    NodeConfig,
)
from .workflow import (
    agent_workflow,
    execute_node_activity,
    evaluate_condition_activity,
    register_node,
    register_condition,
    register_channel_reducer,
    set_default_graph_config,
    set_serializer,
    clear_registries,
    START,
    END,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


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
        runner = DaprWorkflowGraphRunner(graph=compiled)
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
        registry_config: Optional[Any] = None,
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
            registry_config: Optional registry configuration for metadata extraction
        """
        self._graph = graph
        self._max_steps = max_steps

        super().__init__(
            name,
            framework="langgraph",
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

        # Register metadata
        self._register_agent_metadata(
            agent=self._graph, framework="langgraph", registry=registry_config
        )

        # Register workflow and activities
        self._register_workflow_components()

        # Extract and register graph components
        self._graph_config = self._extract_graph_config()
        self._register_graph_components()

    def _register_workflow_components(self) -> None:
        """Register workflow and activities on the workflow runtime."""
        self._workflow_runtime.register_workflow(
            agent_workflow, name=self.workflow_name
        )
        self._workflow_runtime.register_activity(
            execute_node_activity, name="execute_node_activity"
        )
        self._workflow_runtime.register_activity(
            evaluate_condition_activity, name="evaluate_condition_activity"
        )

    def _extract_graph_config(self) -> GraphConfig:
        """Extract configuration from the compiled graph."""
        nodes = []
        edges = []
        entry_point = None
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

                if source_name == START:
                    entry_point = target_name
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

                        edges.append(
                            EdgeConfig(
                                source=source if source != "__start__" else START,
                                target="",  # Determined at runtime
                                condition=condition_name,
                            )
                        )

        if not entry_point and nodes:
            entry_point = nodes[0].name

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
            entry_point=entry_point or (nodes[0].name if nodes else ""),
            finish_points=finish_points,
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def _register_graph_components(self) -> None:
        """Register graph nodes and other components in the global registries."""
        clear_registries()

        graph_nodes = getattr(self._graph, "nodes", {})
        for node_name, node_spec in graph_nodes.items():
            if node_name in (START, END, "__start__", "__end__"):
                continue

            node_func = None

            if hasattr(node_spec, "bound"):
                bound = node_spec.bound
                if hasattr(bound, "func"):
                    node_func = bound.func
                elif callable(bound):
                    node_func = bound
            elif hasattr(node_spec, "runnable"):
                runnable = node_spec.runnable
                if hasattr(runnable, "func"):
                    node_func = runnable.func
                elif callable(runnable):
                    node_func = runnable
            elif callable(node_spec):
                node_func = node_spec

            if node_func:
                register_node(node_name, node_func)
                logger.info(f"Registered node: {node_name}")
            else:
                logger.warning(f"Could not extract callable for node: {node_name}")

        channels = getattr(self._graph, "channels", {})
        for channel_name, channel in channels.items():
            if hasattr(channel, "reducer") and channel.reducer:
                register_channel_reducer(channel_name, channel.reducer)
                logger.info(f"Registered reducer for channel: {channel_name}")

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

        config = config or {}
        config["thread_id"] = thread_id

        # Create LangSmith parent trace and pass dotted_order to activities via config.
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
            _ls_client = None
            _ls_run_id = None

        channel_state = ChannelState(
            values=self._serialize_input(input),
            versions={k: 1 for k in input.keys()},
            updated_channels=list(input.keys()),
        )

        workflow_input = GraphWorkflowInput(
            graph_config=self._graph_config,
            channel_state=channel_state,
            step=0,
            max_steps=self._max_steps,
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

        config = config or {}
        config["thread_id"] = thread_id

        # Create LangSmith parent trace and pass dotted_order to activities via config.
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

        channel_state = ChannelState(
            values=self._serialize_input(input),
            versions={k: 1 for k in input.keys()},
            updated_channels=list(input.keys()),
        )

        workflow_input = GraphWorkflowInput(
            graph_config=self._graph_config,
            channel_state=channel_state,
            step=0,
            max_steps=self._max_steps,
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

        yield {
            "type": "workflow_started",
            "workflow_id": workflow_id,
            "thread_id": thread_id,
            "graph_name": self._name,
        }

        def _parse_output(wf_id: str, output_dict: dict) -> dict:  # type: ignore[type-arg]
            output = GraphWorkflowOutput.from_dict(output_dict)
            return {
                "type": "workflow_completed",
                "workflow_id": wf_id,
                "output": output.output,
                "steps": output.steps,
                "status": output.status,
            }

        async for event in self._poll_workflow(
            workflow_id,
            thread_id,
            poll_interval=poll_interval,
            parse_output=_parse_output,
        ):
            yield event

    def _serialize_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize input values for storage."""
        from .workflow import _serialize_value

        return {k: _serialize_value(v) for k, v in input.items()}

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

        setup_telemetry(self.__class__.__name__)
        instrument_grpc()

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

        input_mapper = getattr(self, "_input_mapper", None)
        if input_mapper:
            graph_input = input_mapper(request)
        else:
            graph_input = request
            if "messages" not in request:
                task = request.get("task")
                if task:
                    graph_input = {"messages": [{"role": "user", "content": task}]}

        async for event in self.run_async(input=graph_input, thread_id=thread_id):
            yield event

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> "CompiledStateGraph":
        """The LangGraph being executed."""
        return self._graph

    @property
    def graph_config(self) -> GraphConfig:
        """The extracted graph configuration."""
        return self._graph_config
