# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Data models for Dapr Workflow LangGraph integration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class NodeConfig:
    """Configuration for a single LangGraph node.

    Attributes:
        name: The node name
        triggers: Channel names that trigger this node
        channels_read: Channels this node reads from
        channels_write: Channels this node writes to
    """

    name: str
    triggers: List[str] = field(default_factory=list)
    channels_read: List[str] = field(default_factory=list)
    channels_write: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "triggers": self.triggers,
            "channels_read": self.channels_read,
            "channels_write": self.channels_write,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeConfig":
        return cls(
            name=data["name"],
            triggers=data.get("triggers", []),
            channels_read=data.get("channels_read", []),
            channels_write=data.get("channels_write", []),
        )


@dataclass
class EdgeConfig:
    """Configuration for a graph edge.

    Attributes:
        source: Source node name
        target: Target node name (or list for conditional)
        condition: Optional condition function name for conditional edges
        path_map: Optional mapping from the condition's return values to node
            names (the ``path_map`` given to ``add_conditional_edges``)
    """

    source: str
    target: str
    condition: Optional[str] = None
    path_map: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "condition": self.condition,
            "path_map": self.path_map,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeConfig":
        return cls(
            source=data["source"],
            target=data["target"],
            condition=data.get("condition"),
            path_map=data.get("path_map"),
        )


@dataclass
class WaitingEdgeConfig:
    """A join: ``add_edge([a, b], target)`` runs ``target`` once every source ran.

    Attributes:
        sources: The nodes that must all have run
        target: The node that runs after the last of them
    """

    sources: List[str]
    target: str

    def to_dict(self) -> Dict[str, Any]:
        return {"sources": list(self.sources), "target": self.target}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WaitingEdgeConfig":
        return cls(sources=list(data["sources"]), target=data["target"])


@dataclass
class GraphConfig:
    """Configuration for the entire LangGraph.

    Attributes:
        name: Graph name
        nodes: List of node configurations
        edges: List of edge configurations
        entry_point: The entry node name (the first of ``entry_points``)
        finish_points: Nodes that lead to END
        input_channels: Channels that accept input
        output_channels: Channels that produce output
        entry_points: Every node a START edge leads to, which all run in the
            first step. Conditional START edges are in ``edges`` and are
            evaluated before it. ``None`` for a config written before this
            field existed: then ``entry_point`` alone starts the run.
        waiting_edges: Joins; a node waits until all their sources have run.
    """

    name: str
    nodes: List[NodeConfig]
    edges: List[EdgeConfig]
    entry_point: str
    finish_points: List[str] = field(default_factory=list)
    input_channels: List[str] = field(default_factory=list)
    output_channels: List[str] = field(default_factory=list)
    entry_points: Optional[List[str]] = None
    waiting_edges: List[WaitingEdgeConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "entry_point": self.entry_point,
            "finish_points": self.finish_points,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "entry_points": self.entry_points,
            "waiting_edges": [w.to_dict() for w in self.waiting_edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphConfig":
        return cls(
            name=data["name"],
            nodes=[NodeConfig.from_dict(n) for n in data["nodes"]],
            edges=[EdgeConfig.from_dict(e) for e in data["edges"]],
            entry_point=data["entry_point"],
            finish_points=data.get("finish_points", []),
            input_channels=data.get("input_channels", []),
            output_channels=data.get("output_channels", []),
            entry_points=data.get("entry_points"),
            waiting_edges=[
                WaitingEdgeConfig.from_dict(w) for w in data.get("waiting_edges") or []
            ],
        )


@dataclass
class ChannelState:
    """Serialized state of all channels.

    Attributes:
        values: Mapping of channel name to serialized value
        versions: Mapping of channel name to version number
        updated_channels: Set of channels updated in current step
    """

    values: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, int] = field(default_factory=dict)
    updated_channels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "versions": self.versions,
            "updated_channels": self.updated_channels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelState":
        return cls(
            values=data.get("values", {}),
            versions=data.get("versions", {}),
            updated_channels=data.get("updated_channels", []),
        )


@dataclass
class NodeWrite:
    """A write operation from a node.

    Attributes:
        channel: The channel to write to
        value: The value to write
    """

    channel: str
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeWrite":
        return cls(
            channel=data["channel"],
            value=data["value"],
        )


@dataclass
class ExecuteNodeInput:
    """Input for the execute_node activity.

    Attributes:
        node_name: Name of the node to execute
        channel_state: Current channel state
        config: Optional LangGraph config dict
        thread_id: Thread identifier for the execution. Propagated into the
            LangGraph ``Runtime`` so sandbox/state backends can resolve which
            workflow owns a given tool call.
    """

    node_name: str
    channel_state: ChannelState
    config: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None
    step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "node_name": self.node_name,
            "channel_state": self.channel_state.to_dict(),
            "config": self.config,
            "thread_id": self.thread_id,
        }
        if self.step is not None:
            data["step"] = self.step
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecuteNodeInput":
        return cls(
            node_name=data["node_name"],
            channel_state=ChannelState.from_dict(data["channel_state"]),
            config=data.get("config"),
            thread_id=data.get("thread_id"),
            step=data.get("step"),
        )


@dataclass
class ExecuteNodeOutput:
    """Output from the execute_node activity.

    Attributes:
        node_name: Name of the executed node
        writes: List of writes produced by the node
        error: Optional error message
        goto: Nodes the node asked to run next through ``Command(goto=...)``,
            in addition to the nodes its edges lead to
    """

    node_name: str
    writes: List[NodeWrite] = field(default_factory=list)
    error: Optional[str] = None
    goto: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "node_name": self.node_name,
            "writes": [w.to_dict() for w in self.writes],
            "error": self.error,
        }
        if self.goto:
            data["goto"] = list(self.goto)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecuteNodeOutput":
        return cls(
            node_name=data["node_name"],
            writes=[NodeWrite.from_dict(w) for w in data.get("writes", [])],
            error=data.get("error"),
            goto=list(data.get("goto") or []),
        )


@dataclass
class EvaluateConditionInput:
    """Input for evaluating a conditional edge.

    Attributes:
        source_node: The source node name
        condition_name: Name of the condition function
        channel_state: Current channel state
        config: Optional LangGraph config dict (used to inject a ``Runtime``
            when the condition is a ``RunnableCallable``)
        thread_id: Thread identifier for the execution
        path_map: Optional mapping from the condition's return values to node
            names (see ``EdgeConfig.path_map``)
    """

    source_node: str
    condition_name: str
    channel_state: ChannelState
    config: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None
    path_map: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node": self.source_node,
            "condition_name": self.condition_name,
            "channel_state": self.channel_state.to_dict(),
            "config": self.config,
            "thread_id": self.thread_id,
            "path_map": self.path_map,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluateConditionInput":
        return cls(
            source_node=data["source_node"],
            condition_name=data["condition_name"],
            channel_state=ChannelState.from_dict(data["channel_state"]),
            config=data.get("config"),
            thread_id=data.get("thread_id"),
            path_map=data.get("path_map"),
        )


@dataclass
class EvaluateConditionOutput:
    """Output from evaluating a conditional edge.

    Attributes:
        next_nodes: List of next node names to execute
        error: Optional error message
    """

    next_nodes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_nodes": self.next_nodes,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluateConditionOutput":
        return cls(
            next_nodes=data.get("next_nodes", []),
            error=data.get("error"),
        )


@dataclass
class GraphWorkflowInput:
    """Input for the main graph workflow.

    Attributes:
        graph_config: Configuration of the graph
        channel_state: Initial/current channel state
        step: Current step number
        max_steps: Maximum steps before stopping
        pending_nodes: Nodes pending execution
        config: Optional LangGraph config dict
        thread_id: Thread identifier for the execution
    """

    graph_config: GraphConfig
    channel_state: ChannelState
    step: int = 0
    max_steps: int = 100
    pending_nodes: List[str] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_config": self.graph_config.to_dict(),
            "channel_state": self.channel_state.to_dict(),
            "step": self.step,
            "max_steps": self.max_steps,
            "pending_nodes": self.pending_nodes,
            "config": self.config,
            "thread_id": self.thread_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphWorkflowInput":
        return cls(
            graph_config=GraphConfig.from_dict(data["graph_config"]),
            channel_state=ChannelState.from_dict(data["channel_state"]),
            step=data.get("step", 0),
            max_steps=data.get("max_steps", 100),
            pending_nodes=data.get("pending_nodes", []),
            config=data.get("config"),
            thread_id=data.get("thread_id"),
        )


@dataclass
class GraphWorkflowOutput:
    """Output from the graph workflow.

    Attributes:
        output: Final output values from output channels
        channel_state: Final channel state
        steps: Total number of steps executed
        status: Workflow status
        error: Optional error message
        executed: The nodes that ran in each step, in order. Lets a streaming
            consumer tell which node's chunks never arrived. ``None`` for an
            output written before the field existed.
    """

    output: Dict[str, Any] = field(default_factory=dict)
    channel_state: Optional[ChannelState] = None
    steps: int = 0
    status: str = "completed"
    error: Optional[str] = None
    executed: Optional[List[List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "output": self.output,
            "channel_state": self.channel_state.to_dict()
            if self.channel_state
            else None,
            "steps": self.steps,
            "status": self.status,
            "error": self.error,
        }
        if self.executed is not None:
            data["executed"] = self.executed
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphWorkflowOutput":
        channel_state = None
        if data.get("channel_state"):
            channel_state = ChannelState.from_dict(data["channel_state"])
        return cls(
            output=data.get("output", {}),
            channel_state=channel_state,
            steps=data.get("steps", 0),
            status=data.get("status", "completed"),
            error=data.get("error"),
            executed=data.get("executed"),
        )
