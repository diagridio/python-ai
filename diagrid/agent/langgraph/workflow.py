# -*- coding: utf-8 -*-

"""
Copyright 2025 Diagrid Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Dapr Workflow definitions for durable LangGraph execution."""

import logging
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from dapr.ext.workflow import (
    DaprWorkflowContext,
    WorkflowActivityContext,
    RetryPolicy,
    when_all,
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

logger = logging.getLogger(__name__)


# Global registries - populated by the runner
_node_registry: Dict[str, Callable] = {}
_condition_registry: Dict[str, Callable] = {}
_channel_reducers: Dict[str, Callable] = {}
_serializer: Optional[Any] = None

# Special node names
START = "__start__"
END = "__end__"


def register_node(name: str, func: Callable) -> None:
    """Register a node function for use by the execute_node activity."""
    _node_registry[name] = func


def get_registered_node(name: str) -> Optional[Callable]:
    """Get a registered node function by name."""
    return _node_registry.get(name)


def register_condition(name: str, func: Callable) -> None:
    """Register a condition function for conditional edges."""
    _condition_registry[name] = func


def get_registered_condition(name: str) -> Optional[Callable]:
    """Get a registered condition function by name."""
    return _condition_registry.get(name)


def register_channel_reducer(channel: str, reducer: Callable) -> None:
    """Register a reducer function for a channel."""
    _channel_reducers[channel] = reducer


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


def clear_registries() -> None:
    """Clear all registries."""
    _node_registry.clear()
    _condition_registry.clear()
    _channel_reducers.clear()
    global _serializer
    _serializer = None


def langgraph_workflow(ctx: DaprWorkflowContext, input_data: Dict[str, Any]) -> Dict[str, Any]:
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
    workflow_input = GraphWorkflowInput.from_dict(input_data)
    graph_config = workflow_input.graph_config
    channel_state = workflow_input.channel_state
    max_steps = workflow_input.max_steps
    config = workflow_input.config

    # Retry policy for activities
    retry_policy = RetryPolicy(
        max_number_of_attempts=3,
        first_retry_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        max_retry_interval=timedelta(seconds=30),
    )

    # Start with entry point
    pending_nodes = [graph_config.entry_point]
    step = 0

    # Main execution loop
    while step < max_steps:
        # Log workflow step for debugging
        if not ctx.is_replaying:
            print(f"  [WORKFLOW] Step {step}, pending_nodes={pending_nodes}", flush=True)

        # Check if END is reached
        if END in pending_nodes or not pending_nodes:
            return GraphWorkflowOutput(
                output=_extract_output(channel_state, graph_config.output_channels),
                channel_state=channel_state,
                steps=step,
                status="completed",
            ).to_dict()

        # Filter out END from nodes to execute
        nodes_to_execute = [n for n in pending_nodes if n != END]
        if not nodes_to_execute:
            return GraphWorkflowOutput(
                output=_extract_output(channel_state, graph_config.output_channels),
                channel_state=channel_state,
                steps=step,
                status="completed",
            ).to_dict()

        # Execute all triggered nodes in parallel (each as a separate activity)
        node_tasks = []
        for node_name in nodes_to_execute:
            node_input = ExecuteNodeInput(
                node_name=node_name,
                channel_state=channel_state,
                config=config,
            )
            task = ctx.call_activity(
                execute_node_activity,
                input=node_input.to_dict(),
                retry_policy=retry_policy,
            )
            node_tasks.append(task)

        # Wait for all node executions to complete
        task_results = yield when_all(node_tasks)

        # Collect outputs and check for errors
        node_outputs: List[ExecuteNodeOutput] = []
        for result_data in task_results:
            output = ExecuteNodeOutput.from_dict(result_data)
            if output.error:
                return GraphWorkflowOutput(
                    output=_extract_output(channel_state, graph_config.output_channels),
                    channel_state=channel_state,
                    steps=step,
                    status="error",
                    error=f"Node '{output.node_name}' failed: {output.error}",
                ).to_dict()
            node_outputs.append(output)

        # Apply writes to channels
        updated_channels = []
        for output in node_outputs:
            for write in output.writes:
                _apply_write(channel_state, write.channel, write.value)
                if write.channel not in updated_channels:
                    updated_channels.append(write.channel)

        channel_state.updated_channels = updated_channels

        # Determine next nodes based on edges from executed nodes
        next_nodes = []
        for node_name in nodes_to_execute:
            for edge in graph_config.edges:
                if edge.source == node_name:
                    if edge.condition:
                        # Evaluate conditional edge (as activity)
                        cond_input = EvaluateConditionInput(
                            source_node=node_name,
                            condition_name=edge.condition,
                            channel_state=channel_state,
                        )
                        cond_result = yield ctx.call_activity(
                            evaluate_condition_activity,
                            input=cond_input.to_dict(),
                            retry_policy=retry_policy,
                        )
                        cond_output = EvaluateConditionOutput.from_dict(cond_result)
                        if cond_output.error:
                            return GraphWorkflowOutput(
                                output=_extract_output(channel_state, graph_config.output_channels),
                                channel_state=channel_state,
                                steps=step,
                                status="error",
                                error=f"Condition evaluation failed: {cond_output.error}",
                            ).to_dict()
                        for n in cond_output.next_nodes:
                            if n not in next_nodes:
                                next_nodes.append(n)
                    else:
                        # Direct edge
                        if edge.target not in next_nodes:
                            next_nodes.append(edge.target)

        # Move to next step
        pending_nodes = next_nodes
        step += 1

    # Max steps reached
    return GraphWorkflowOutput(
        output=_extract_output(channel_state, graph_config.output_channels),
        channel_state=channel_state,
        steps=step,
        status="max_steps_reached",
        error=f"Max steps ({max_steps}) reached",
    ).to_dict()


def execute_node_activity(
    ctx: WorkflowActivityContext, input_data: Dict[str, Any]
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

    try:
        # Reconstruct state from channel values
        state = _reconstruct_state(node_input.channel_state)

        # Execute the node function
        # Node functions can have different signatures:
        # - (state) -> updates
        # - (state, config) -> updates
        import inspect
        sig = inspect.signature(node_func)
        params = list(sig.parameters.keys())

        if len(params) >= 2 and 'config' in params:
            result = node_func(state, config=node_input.config or {})
        else:
            result = node_func(state)

        # Handle async functions
        import asyncio
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(result)
            finally:
                loop.close()

        # Convert result to writes
        writes = _result_to_writes(result)

        return ExecuteNodeOutput(
            node_name=node_name,
            writes=writes,
        ).to_dict()

    except Exception as e:
        logger.error(f"Error executing node '{node_name}': {e}")
        import traceback
        traceback.print_exc()
        return ExecuteNodeOutput(
            node_name=node_name,
            error=str(e),
        ).to_dict()


def evaluate_condition_activity(
    ctx: WorkflowActivityContext, input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Activity that evaluates a conditional edge.

    Args:
        ctx: The workflow activity context
        input_data: Dictionary containing EvaluateConditionInput data

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

    try:
        # Reconstruct state from channel values
        state = _reconstruct_state(cond_input.channel_state)

        # Evaluate the condition
        result = cond_func(state)

        # Handle async functions
        import asyncio
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(result)
            finally:
                loop.close()

        # Result should be a node name or list of node names
        if isinstance(result, str):
            next_nodes = [result]
        elif isinstance(result, (list, tuple)):
            next_nodes = list(result)
        else:
            next_nodes = [str(result)]

        return EvaluateConditionOutput(
            next_nodes=next_nodes,
        ).to_dict()

    except Exception as e:
        logger.error(f"Error evaluating condition '{cond_input.condition_name}': {e}")
        import traceback
        traceback.print_exc()
        return EvaluateConditionOutput(
            error=str(e),
        ).to_dict()


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


def _extract_output(channel_state: ChannelState, output_channels: List[str]) -> Dict[str, Any]:
    """Extract output values from the specified channels.

    Args:
        channel_state: Current channel state
        output_channels: Channels to extract

    Returns:
        Dictionary of channel name to value
    """
    output = {}
    for channel in output_channels:
        if channel in channel_state.values:
            output[channel] = channel_state.values[channel]
    return output


def _apply_write(channel_state: ChannelState, channel: str, value: Any) -> None:
    """Apply a write to a channel, using reducer if available.

    Args:
        channel_state: Channel state to update
        channel: Channel name
        value: Value to write
    """
    reducer = get_channel_reducer(channel)
    if reducer and channel in channel_state.values:
        # Apply reducer to combine values
        try:
            channel_state.values[channel] = reducer(
                channel_state.values[channel], value
            )
        except Exception:
            # If reducer fails, just overwrite
            channel_state.values[channel] = value
    else:
        # No reducer or first value, just set
        channel_state.values[channel] = value

    # Increment version
    channel_state.versions[channel] = channel_state.versions.get(channel, 0) + 1


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
                if isinstance(value, str) and value.startswith(('{"', '[')):
                    import json
                    state[channel] = json.loads(value)
                else:
                    state[channel] = value
            except Exception:
                state[channel] = value
        else:
            state[channel] = value

    return state


def _result_to_writes(result: Any) -> List[NodeWrite]:
    """Convert a node result to a list of writes.

    Args:
        result: The result from a node function

    Returns:
        List of NodeWrite objects
    """
    writes = []

    if result is None:
        return writes

    if isinstance(result, dict):
        for key, value in result.items():
            # Serialize value if needed
            serialized_value = _serialize_value(value)
            writes.append(NodeWrite(channel=key, value=serialized_value))
    else:
        # Non-dict results - write to a default channel
        serialized_value = _serialize_value(result)
        writes.append(NodeWrite(channel="__result__", value=serialized_value))

    return writes


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
    if hasattr(value, 'model_dump'):
        # Pydantic v2
        return value.model_dump()
    if hasattr(value, 'dict'):
        # Pydantic v1
        return value.dict()
    if hasattr(value, 'to_dict'):
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
