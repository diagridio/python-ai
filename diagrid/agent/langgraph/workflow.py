# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr Workflow definitions for durable LangGraph execution."""

import logging
from datetime import timedelta
from typing import Any, Callable, Dict, Generator, List, Optional

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
_default_graph_config: Optional["GraphConfig"] = None
_default_input_mapper: Optional[Callable] = None
_default_max_steps: int = 100

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


def clear_registries() -> None:
    """Clear all registries."""
    _node_registry.clear()
    _condition_registry.clear()
    _channel_reducers.clear()
    global _serializer, _default_graph_config, _default_input_mapper, _default_max_steps
    _serializer = None
    _default_graph_config = None
    _default_input_mapper = None
    _default_max_steps = 100


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
    if "graph_config" in input_data:
        # Normal internal call with full GraphWorkflowInput
        workflow_input = GraphWorkflowInput.from_dict(input_data)
    else:
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
            graph_input = {
                "messages": [{"role": "user", "content": input_data["task"]}]
            }
        else:
            graph_input = input_data

        channel_state = ChannelState(
            values=_serialize_input(graph_input),
            versions={k: 1 for k in graph_input.keys()},
            updated_channels=list(graph_input.keys()),
        )
        workflow_input = GraphWorkflowInput(
            graph_config=_default_graph_config,
            channel_state=channel_state,
            step=0,
            max_steps=_default_max_steps,
        )

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
            print(
                f"  [WORKFLOW] Step {step}, pending_nodes={pending_nodes}", flush=True
            )

        # Check if END is reached
        if END in pending_nodes or not pending_nodes:
            if not ctx.is_replaying:
                _close_langsmith_parent_trace_async(config)
            return GraphWorkflowOutput(
                output=_extract_output(channel_state, graph_config.output_channels),
                channel_state=channel_state,
                steps=step,
                status="completed",
            ).to_dict()

        # Filter out END from nodes to execute
        nodes_to_execute = [n for n in pending_nodes if n != END]
        if not nodes_to_execute:
            if not ctx.is_replaying:
                _close_langsmith_parent_trace_async(config)
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
                if not ctx.is_replaying:
                    _close_langsmith_parent_trace_async(config)
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
                            if not ctx.is_replaying:
                                _close_langsmith_parent_trace_async(config)
                            return GraphWorkflowOutput(
                                output=_extract_output(
                                    channel_state, graph_config.output_channels
                                ),
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
    if not ctx.is_replaying:
        _close_langsmith_parent_trace_async(config)
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

    # Reconstruct state from channel values
    state = _reconstruct_state(node_input.channel_state)

    # Extract tracing config
    config = node_input.config or {}

    def _run_node() -> Any:
        # If the registered object is a LangChain Runnable (e.g. RunnableCallable),
        # call .invoke() so that argument injection (runtime, config, store) works.
        if hasattr(node_func, "invoke") and callable(getattr(node_func, "invoke")):
            # Ensure the config has a Runtime object for deep agent middleware
            # nodes that require it (runtime injection via RunnableCallable).
            _config = dict(config) if config else {}
            if "configurable" not in _config:
                _config["configurable"] = {}
            if "__pregel_runtime" not in _config["configurable"]:
                try:
                    from langgraph.runtime import Runtime

                    _config["configurable"]["__pregel_runtime"] = Runtime()
                except ImportError:
                    pass
            r = node_func.invoke(state, config=_config)
        else:
            import inspect

            sig = inspect.signature(node_func)
            params = list(sig.parameters.keys())

            if len(params) >= 2 and "config" in params:
                r = node_func(state, config=config)
            else:
                r = node_func(state)

        import asyncio

        if asyncio.iscoroutine(r):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                r = loop.run_until_complete(r)
            finally:
                loop.close()
        return r

    # Set up LangSmith child trace so LLM calls nest under the parent
    # "LangGraph" trace created by runner.invoke().
    import os

    dotted_order = config.get("langsmith_dotted_order")
    _use_tracing = dotted_order and os.environ.get("LANGSMITH_TRACING", "").lower() in (
        "true",
        "1",
    )

    if _use_tracing:
        import langsmith as ls

        # Clear any stale context from Dapr's thread pool.
        # _PARENT_RUN_TREE is a private symbol; degrade gracefully if it moves.
        try:
            from langsmith.run_helpers import _PARENT_RUN_TREE

            _PARENT_RUN_TREE.set(None)
        except (ImportError, AttributeError):
            pass

        with ls.trace(
            name=node_name,
            run_type="chain",
            parent=dotted_order,
            inputs={"step": node_name},
        ):
            result = _run_node()
    else:
        result = _run_node()

    # Convert result to writes
    writes = _result_to_writes(result)

    return ExecuteNodeOutput(
        node_name=node_name,
        writes=writes,
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

    # Reconstruct state from channel values
    state = _reconstruct_state(cond_input.channel_state)

    # Evaluate the condition.
    # If the condition is a LangChain Runnable (e.g. RunnableCallable from
    # Deep Agents branches), call .invoke() instead of direct call.
    if hasattr(cond_func, "invoke") and callable(getattr(cond_func, "invoke")):
        result = cond_func.invoke(state)
    else:
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

    # Result should be a node name, list of node names, or Send objects.
    # LangGraph Send objects have a .node attribute with the target node name.
    def _extract_node_name(item: Any) -> str:
        if isinstance(item, str):
            return item
        if hasattr(item, "node"):
            return item.node
        return str(item)

    if isinstance(result, str):
        next_nodes = [result]
    elif isinstance(result, (list, tuple)):
        next_nodes = [_extract_node_name(r) for r in result]
    elif hasattr(result, "node"):
        # Single Send object
        next_nodes = [result.node]
    else:
        next_nodes = [str(result)]

    return EvaluateConditionOutput(
        next_nodes=next_nodes,
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


def _serialize_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize input values for channel storage."""
    return {k: _serialize_value(v) for k, v in input_data.items()}


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
            # Re-serialize the reducer output so channel state stays
            # JSON-compatible (reducers like add_messages may return
            # BaseMessage objects from dict inputs).
            reduced = channel_state.values[channel]
            if isinstance(reduced, list):
                channel_state.values[channel] = [
                    _serialize_value(item) for item in reduced
                ]
        except Exception as e:
            # If reducer fails, just overwrite
            logger.warning(f"Reducer failed for channel '{channel}': {e}")
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
                if isinstance(value, str) and value.startswith(('{"', "[")):
                    import json

                    state[channel] = json.loads(value)
                else:
                    state[channel] = value
            except Exception:
                state[channel] = value
        else:
            state[channel] = value

    # Reconstruct LangChain message objects from serialized dicts.
    # Nodes (e.g. Deep Agent middleware) expect BaseMessage instances, not
    # plain dicts, so we convert the "messages" channel back.
    # Convert when items are dicts (either {"type": "human", ...} or
    # {"role": "user", ...} format).
    if "messages" in state and isinstance(state["messages"], list):
        msgs = state["messages"]
        if msgs and isinstance(msgs[0], dict):
            try:
                from langchain_core.messages import convert_to_messages

                state["messages"] = convert_to_messages(msgs)
            except (ImportError, Exception):
                pass

    return state


def _result_to_writes(result: Any) -> List[NodeWrite]:
    """Convert a node result to a list of writes.

    Args:
        result: The result from a node function

    Returns:
        List of NodeWrite objects
    """
    writes: List[NodeWrite] = []

    if result is None:
        return writes

    # Handle LangGraph Command objects (used by Deep Agents and newer LangGraph)
    try:
        from langgraph.types import Command

        # Single Command
        if isinstance(result, Command):
            return _command_to_writes(result)

        # List of Commands
        if isinstance(result, list) and result and isinstance(result[0], Command):
            for cmd in result:
                writes.extend(_command_to_writes(cmd))
            return writes
    except ImportError:
        pass

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


def _command_to_writes(cmd: Any) -> List[NodeWrite]:
    """Extract channel writes from the ``update`` field of a LangGraph Command.

    This helper only inspects ``cmd.update`` to generate ``NodeWrite`` entries.
    Any ``goto`` / control-flow information on the Command is ignored by this
    workflow runner.

    Args:
        cmd: A LangGraph Command instance whose ``update`` field encodes channel writes.

    Returns:
        List of NodeWrite objects derived from ``cmd.update``
    """
    writes: List[NodeWrite] = []
    update = getattr(cmd, "update", None)
    if update is None:
        return writes

    if isinstance(update, dict):
        for key, value in update.items():
            serialized_value = _serialize_value(value)
            writes.append(NodeWrite(channel=key, value=serialized_value))
    elif isinstance(update, (list, tuple)):
        # update can be a list of (channel, value) tuples
        for item in update:
            if isinstance(item, tuple) and len(item) == 2:
                key, value = item
                serialized_value = _serialize_value(value)
                writes.append(NodeWrite(channel=key, value=serialized_value))

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
