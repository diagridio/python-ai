# Copyright 2025 Diagrid Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runner for executing LangGraph graphs as Dapr Workflows."""

import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from dapr.ext.workflow import WorkflowRuntime, DaprWorkflowClient, WorkflowStatus

from .models import (
    ChannelState,
    EdgeConfig,
    GraphConfig,
    GraphWorkflowInput,
    GraphWorkflowOutput,
    NodeConfig,
)
from .workflow import (
    langgraph_workflow,
    execute_node_activity,
    evaluate_condition_activity,
    register_node,
    register_condition,
    register_channel_reducer,
    set_serializer,
    clear_registries,
    START,
    END,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class DaprWorkflowGraphRunner:
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
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_steps: int = 100,
        name: Optional[str] = None,
    ):
        """Initialize the runner.

        Args:
            graph: The compiled LangGraph to execute
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_steps: Maximum number of steps before stopping (default: 100)
            name: Optional name for the graph (inferred if not provided)
        """
        self._graph = graph
        self._max_steps = max_steps
        self._host = host
        self._port = port
        self._name = name or self._infer_graph_name()

        # Create workflow runtime
        self._workflow_runtime = WorkflowRuntime(host=host, port=port)

        # Register workflow and activities
        self._workflow_runtime.register_workflow(
            langgraph_workflow, name="langgraph_workflow"
        )
        self._workflow_runtime.register_activity(
            execute_node_activity, name="execute_node_activity"
        )
        self._workflow_runtime.register_activity(
            evaluate_condition_activity, name="evaluate_condition_activity"
        )

        # Extract and register graph components
        self._graph_config = self._extract_graph_config()
        self._register_graph_components()

        # Create workflow client (for starting/managing workflows)
        self._workflow_client: Optional[DaprWorkflowClient] = None
        self._started = False

    def _infer_graph_name(self) -> str:
        """Infer a name for the graph."""
        # Try to get name from graph or builder
        if hasattr(self._graph, "name") and self._graph.name:
            return self._graph.name
        if hasattr(self._graph, "builder") and hasattr(self._graph.builder, "name"):
            return self._graph.builder.name or "langgraph"
        return "langgraph"

    def _extract_graph_config(self) -> GraphConfig:
        """Extract configuration from the compiled graph."""
        nodes = []
        edges = []
        entry_point = None
        finish_points = []

        # Get nodes from the compiled graph
        graph_nodes = getattr(self._graph, "nodes", {})
        for node_name, node_spec in graph_nodes.items():
            if node_name in (START, END, "__start__", "__end__"):
                continue

            # Extract node triggers and channels
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

        # Get edges
        # Try to access edges from builder or graph
        builder = getattr(self._graph, "builder", None)

        if builder:
            # Get regular edges
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

            # Get conditional edges (branches)
            branches = getattr(builder, "branches", {})
            for source, branch_dict in branches.items():
                for branch_name, branch_spec in branch_dict.items():
                    # Branch spec has the path function and path_map
                    path_func = getattr(branch_spec, "path", None)

                    if path_func:
                        # Register the condition function
                        condition_name = f"{source}_{branch_name}_condition"
                        register_condition(condition_name, path_func)

                        # Create edge for conditional routing
                        edges.append(
                            EdgeConfig(
                                source=source if source != "__start__" else START,
                                target="",  # Determined at runtime
                                condition=condition_name,
                            )
                        )

        # If no entry point found, use first node
        if not entry_point and nodes:
            entry_point = nodes[0].name

        # Get input/output channels
        input_channels = []
        output_channels = []

        if hasattr(self._graph, "input_channels"):
            ic = self._graph.input_channels
            input_channels = [ic] if isinstance(ic, str) else list(ic or [])
        if hasattr(self._graph, "output_channels"):
            oc = self._graph.output_channels
            output_channels = [oc] if isinstance(oc, str) else list(oc or [])

        # Default to state channels if not specified
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

        # Register nodes
        graph_nodes = getattr(self._graph, "nodes", {})
        for node_name, node_spec in graph_nodes.items():
            if node_name in (START, END, "__start__", "__end__"):
                continue

            # Get the actual callable from the node spec
            node_func = None

            if hasattr(node_spec, "bound"):
                # PregelNode has a 'bound' Runnable
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

        # Register channel reducers
        channels = getattr(self._graph, "channels", {})
        for channel_name, channel in channels.items():
            # Check if channel has a reducer
            if hasattr(channel, "reducer") and channel.reducer:
                register_channel_reducer(channel_name, channel.reducer)
                logger.info(f"Registered reducer for channel: {channel_name}")

        # Set up serializer if available
        try:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

            set_serializer(JsonPlusSerializer())
        except ImportError:
            logger.warning(
                "JsonPlusSerializer not available, using basic serialization"
            )

    def start(self) -> None:
        """Start the workflow runtime.

        This must be called before running any workflows. It starts listening
        for workflow work items in the background.
        """
        if self._started:
            return

        self._workflow_runtime.start()
        self._workflow_client = DaprWorkflowClient(host=self._host, port=self._port)
        self._started = True
        logger.info("Dapr Workflow runtime started")

    def shutdown(self) -> None:
        """Shutdown the workflow runtime.

        Call this when you're done running workflows to clean up resources.
        """
        if not self._started:
            return

        self._workflow_runtime.shutdown()
        self._started = False
        logger.info("Dapr Workflow runtime stopped")

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

        Raises:
            RuntimeError: If the runner hasn't been started
            TimeoutError: If timeout is exceeded
            Exception: If the workflow fails
        """
        import time

        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        # Generate IDs
        thread_id = thread_id or str(uuid.uuid4())
        workflow_id = workflow_id or f"graph-{thread_id}-{uuid.uuid4().hex[:8]}"

        # Create initial channel state from input
        channel_state = ChannelState(
            values=self._serialize_input(input),
            versions={k: 1 for k in input.keys()},
            updated_channels=list(input.keys()),
        )

        # Create workflow input
        workflow_input = GraphWorkflowInput(
            graph_config=self._graph_config,
            channel_state=channel_state,
            step=0,
            max_steps=self._max_steps,
            pending_nodes=[],
            config=config,
            thread_id=thread_id,
        )

        # Verify JSON serializable
        workflow_input_dict = workflow_input.to_dict()
        json.dumps(workflow_input_dict)

        # Start the workflow
        logger.info(f"Starting workflow: {workflow_id}")
        self._workflow_client.schedule_new_workflow(
            workflow=langgraph_workflow,
            input=workflow_input_dict,
            instance_id=workflow_id,
        )

        # Poll for completion
        start_time = time.time()
        while True:
            time.sleep(poll_interval)

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Workflow {workflow_id} timed out after {timeout}s")

            state = self._workflow_client.get_workflow_state(instance_id=workflow_id)

            if state is None:
                raise RuntimeError(f"Workflow {workflow_id} state not found")

            if state.runtime_status == WorkflowStatus.COMPLETED:
                # Parse output
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

        Raises:
            RuntimeError: If the runner hasn't been started
        """
        import asyncio

        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        # Generate IDs
        thread_id = thread_id or str(uuid.uuid4())
        workflow_id = workflow_id or f"graph-{thread_id}-{uuid.uuid4().hex[:8]}"

        # Create initial channel state from input
        channel_state = ChannelState(
            values=self._serialize_input(input),
            versions={k: 1 for k in input.keys()},
            updated_channels=list(input.keys()),
        )

        # Create workflow input
        workflow_input = GraphWorkflowInput(
            graph_config=self._graph_config,
            channel_state=channel_state,
            step=0,
            max_steps=self._max_steps,
            pending_nodes=[],
            config=config,
            thread_id=thread_id,
        )

        # Verify JSON serializable
        workflow_input_dict = workflow_input.to_dict()
        json.dumps(workflow_input_dict)

        # Start the workflow
        logger.info(f"Starting workflow: {workflow_id}")
        self._workflow_client.schedule_new_workflow(
            workflow=langgraph_workflow,
            input=workflow_input_dict,
            instance_id=workflow_id,
        )

        # Yield start event
        yield {
            "type": "workflow_started",
            "workflow_id": workflow_id,
            "thread_id": thread_id,
            "graph_name": self._name,
        }

        # Poll for completion
        previous_status = None

        while True:
            await asyncio.sleep(poll_interval)

            state = self._workflow_client.get_workflow_state(instance_id=workflow_id)

            if state is None:
                yield {
                    "type": "workflow_error",
                    "workflow_id": workflow_id,
                    "error": "Workflow state not found",
                }
                break

            # Yield status change events
            if state.runtime_status != previous_status:
                yield {
                    "type": "workflow_status_changed",
                    "workflow_id": workflow_id,
                    "status": str(state.runtime_status),
                    "custom_status": state.serialized_custom_status,
                }
                previous_status = state.runtime_status

            # Check for completion
            if state.runtime_status == WorkflowStatus.COMPLETED:
                output_data = state.serialized_output
                if output_data:
                    try:
                        output_dict = (
                            json.loads(output_data)
                            if isinstance(output_data, str)
                            else output_data
                        )
                        output = GraphWorkflowOutput.from_dict(output_dict)

                        yield {
                            "type": "workflow_completed",
                            "workflow_id": workflow_id,
                            "output": output.output,
                            "steps": output.steps,
                            "status": output.status,
                        }
                    except Exception as e:
                        yield {
                            "type": "workflow_completed",
                            "workflow_id": workflow_id,
                            "raw_output": output_data,
                            "parse_error": str(e),
                        }
                else:
                    yield {
                        "type": "workflow_completed",
                        "workflow_id": workflow_id,
                    }
                break

            elif state.runtime_status == WorkflowStatus.FAILED:
                error_info = None
                if state.failure_details:
                    fd = state.failure_details
                    error_info = {
                        "message": getattr(fd, "message", str(fd)),
                        "error_type": getattr(fd, "error_type", None),
                        "stack_trace": getattr(fd, "stack_trace", None),
                    }
                yield {
                    "type": "workflow_failed",
                    "workflow_id": workflow_id,
                    "error": error_info,
                }
                break

            elif state.runtime_status == WorkflowStatus.TERMINATED:
                yield {
                    "type": "workflow_terminated",
                    "workflow_id": workflow_id,
                }
                break

    def _serialize_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize input values for storage.

        Args:
            input: Input dictionary

        Returns:
            Serialized input
        """
        from .workflow import _serialize_value

        return {k: _serialize_value(v) for k, v in input.items()}

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow.

        Args:
            workflow_id: The workflow instance ID

        Returns:
            Dictionary with workflow status or None if not found
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        state = self._workflow_client.get_workflow_state(instance_id=workflow_id)
        if state is None:
            return None

        return {
            "workflow_id": workflow_id,
            "status": str(state.runtime_status),
            "custom_status": state.serialized_custom_status,
            "created_at": str(state.created_at) if state.created_at else None,
            "last_updated_at": str(state.last_updated_at)
            if state.last_updated_at
            else None,
        }

    def terminate_workflow(self, workflow_id: str) -> None:
        """Terminate a running workflow.

        Args:
            workflow_id: The workflow instance ID
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        self._workflow_client.terminate_workflow(instance_id=workflow_id)
        logger.info(f"Terminated workflow: {workflow_id}")

    def purge_workflow(self, workflow_id: str) -> None:
        """Purge a completed or terminated workflow.

        This removes all workflow state from the state store.

        Args:
            workflow_id: The workflow instance ID
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        self._workflow_client.purge_workflow(instance_id=workflow_id)
        logger.info(f"Purged workflow: {workflow_id}")

    @property
    def graph(self) -> "CompiledStateGraph":
        """The LangGraph being executed."""
        return self._graph

    @property
    def graph_config(self) -> GraphConfig:
        """The extracted graph configuration."""
        return self._graph_config

    @property
    def is_running(self) -> bool:
        """Whether the workflow runtime is running."""
        return self._started
