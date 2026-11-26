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
        deferred: Nodes added with ``defer=True``: one that is triggered waits
            until no other node or ``Send`` task is pending, then runs.
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
    deferred: List[str] = field(default_factory=list)

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
            "deferred": self.deferred,
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
            deferred=list(data.get("deferred") or []),
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
        send: Set when the node runs for a ``Send``: ``{"arg": ..., "slot": k}``,
            the argument the node takes as its input and the ``Send``'s
            position among all of the step's ``Send`` tasks (LangGraph numbers
            its push tasks over the whole step, and that number is in each
            task's ``langgraph_path``). ``None`` for a node that runs on the
            channel state.
        recursion_limit: The run's step limit as a LangGraph ``recursion_limit``
            (what ``RemainingSteps`` and ``IsLastStep`` count against), or
            ``None`` for a workflow that did not send it.
        routes: The node's conditional edges, each ``{"condition", "path_map"}``,
            for the activity to evaluate once the node returns, as LangGraph
            runs a router inside its node's task. ``None`` when the workflow
            evaluates them itself (an instance from before routing moved here).
        output_channels: The graph's output channels, set only in a streamed
            run: the activity sends the state it is handed (the state after the
            previous step) as a ``values`` chunk when the step wrote one of them.
        resume: The values this task's ``interrupt()`` calls return, in order,
            when it runs again after the run was resumed; ``None`` otherwise.
    """

    node_name: str
    channel_state: ChannelState
    config: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None
    step: Optional[int] = None
    send: Optional[Dict[str, Any]] = None
    recursion_limit: Optional[int] = None
    routes: Optional[List[Dict[str, Any]]] = None
    output_channels: Optional[List[str]] = None
    resume: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "node_name": self.node_name,
            "channel_state": self.channel_state.to_dict(),
            "config": self.config,
            "thread_id": self.thread_id,
        }
        if self.step is not None:
            data["step"] = self.step
        if self.send is not None:
            data["send"] = self.send
        if self.recursion_limit is not None:
            data["recursion_limit"] = self.recursion_limit
        if self.routes is not None:
            data["routes"] = self.routes
        if self.output_channels is not None:
            data["output_channels"] = self.output_channels
        if self.resume is not None:
            data["resume"] = self.resume
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecuteNodeInput":
        return cls(
            node_name=data["node_name"],
            channel_state=ChannelState.from_dict(data["channel_state"]),
            config=data.get("config"),
            thread_id=data.get("thread_id"),
            step=data.get("step"),
            send=data.get("send"),
            recursion_limit=data.get("recursion_limit"),
            routes=data.get("routes"),
            output_channels=data.get("output_channels"),
            resume=data.get("resume"),
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
        sends: The ``Send`` objects of that ``goto``, each ``{"node", "arg"}``:
            every one runs its node once, on its argument
        routes_done: Whether the activity evaluated the node's conditional
            edges (``ExecuteNodeInput.routes``). ``False`` when it was not asked
            to or a router raised: the workflow then evaluates them itself.
        next_nodes: Where those conditional edges lead, when they were evaluated
        route_sends: The ``Send`` tasks those conditional edges returned
        route_error: Why routing failed (a ``path_map`` miss, a reducer that
            raised on this task's writes): the run fails with it
    """

    node_name: str
    writes: List[NodeWrite] = field(default_factory=list)
    error: Optional[str] = None
    goto: List[str] = field(default_factory=list)
    sends: List[Dict[str, Any]] = field(default_factory=list)
    routes_done: bool = False
    next_nodes: List[str] = field(default_factory=list)
    route_sends: List[Dict[str, Any]] = field(default_factory=list)
    route_error: Optional[str] = None
    #: The task called ``interrupt()``: ``[{"id", "value"}]``; it wrote nothing
    #: and runs again when the run is resumed.
    interrupts: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "node_name": self.node_name,
            "writes": [w.to_dict() for w in self.writes],
            "error": self.error,
        }
        if self.goto:
            data["goto"] = list(self.goto)
        if self.sends:
            data["sends"] = list(self.sends)
        if self.routes_done:
            data["routes_done"] = True
            data["next_nodes"] = list(self.next_nodes)
            if self.route_sends:
                data["route_sends"] = list(self.route_sends)
            if self.route_error:
                data["route_error"] = self.route_error
        if self.interrupts:
            data["interrupts"] = list(self.interrupts)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecuteNodeOutput":
        return cls(
            node_name=data["node_name"],
            writes=[NodeWrite.from_dict(w) for w in data.get("writes", [])],
            error=data.get("error"),
            goto=list(data.get("goto") or []),
            sends=list(data.get("sends") or []),
            routes_done=bool(data.get("routes_done")),
            next_nodes=list(data.get("next_nodes") or []),
            route_sends=list(data.get("route_sends") or []),
            route_error=data.get("route_error"),
            interrupts=data.get("interrupts"),
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
        step: The workflow step of the source node (``-1`` for a START edge),
            so a streamed run can give the condition's chunks that node's
            step. ``None`` (an older workflow): the condition's chunks are not
            streamed.
        send_slot: Set when the source task ran for a ``Send``: its position
            among the step's ``Send`` tasks. The condition then streams as that
            task, apart from the node's other tasks of the step.
        recursion_limit: The run's step limit as a LangGraph ``recursion_limit``
            (see ``ExecuteNodeInput``).
        output_channels: Set only for a START edge of a streamed run: the
            activity sends the input as the first ``values`` chunk (see
            ``ExecuteNodeInput``).
    """

    source_node: str
    condition_name: str
    channel_state: ChannelState
    config: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None
    path_map: Optional[Dict[str, str]] = None
    step: Optional[int] = None
    send_slot: Optional[int] = None
    recursion_limit: Optional[int] = None
    output_channels: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "source_node": self.source_node,
            "condition_name": self.condition_name,
            "channel_state": self.channel_state.to_dict(),
            "config": self.config,
            "thread_id": self.thread_id,
            "path_map": self.path_map,
        }
        if self.step is not None:
            data["step"] = self.step
        if self.send_slot is not None:
            data["send_slot"] = self.send_slot
        if self.recursion_limit is not None:
            data["recursion_limit"] = self.recursion_limit
        if self.output_channels is not None:
            data["output_channels"] = self.output_channels
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluateConditionInput":
        return cls(
            source_node=data["source_node"],
            condition_name=data["condition_name"],
            channel_state=ChannelState.from_dict(data["channel_state"]),
            config=data.get("config"),
            thread_id=data.get("thread_id"),
            path_map=data.get("path_map"),
            step=data.get("step"),
            send_slot=data.get("send_slot"),
            recursion_limit=data.get("recursion_limit"),
            output_channels=data.get("output_channels"),
        )


@dataclass
class EvaluateConditionOutput:
    """Output from evaluating a conditional edge.

    Attributes:
        next_nodes: List of next node names to execute
        error: Optional error message
        sends: The ``Send`` objects the condition returned, each ``{"node",
            "arg"}``: every one runs its node once, on its argument
    """

    next_nodes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    sends: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "next_nodes": self.next_nodes,
            "error": self.error,
        }
        if self.sends:
            data["sends"] = list(self.sends)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluateConditionOutput":
        return cls(
            next_nodes=data.get("next_nodes", []),
            error=data.get("error"),
            sends=list(data.get("sends") or []),
        )


@dataclass
class GraphWorkflowInput:
    """Input for the main graph workflow.

    Attributes:
        graph_config: Configuration of the graph
        channel_state: Initial/current channel state
        step: Current step number
        max_steps: Maximum steps before stopping
        strict_limit: ``max_steps`` is a LangGraph ``recursion_limit``: a run
            that has taken that many steps fails, even if nothing is pending
            after them. Without it, a run that needs exactly ``max_steps``
            steps completes.
        pending_nodes: Nodes pending execution
        config: Optional LangGraph config dict
        thread_id: Thread identifier for the execution
        thread_state: The run starts from its thread's last state and keeps
            the state it ends with as the thread's (the runner has a
            checkpointer).
    """

    graph_config: GraphConfig
    channel_state: ChannelState
    step: int = 0
    max_steps: int = 100
    strict_limit: bool = False
    pending_nodes: List[str] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None
    thread_state: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "graph_config": self.graph_config.to_dict(),
            "channel_state": self.channel_state.to_dict(),
            "step": self.step,
            "max_steps": self.max_steps,
            "pending_nodes": self.pending_nodes,
            "config": self.config,
            "thread_id": self.thread_id,
        }
        if self.strict_limit:
            data["strict_limit"] = True
        if self.thread_state:
            data["thread_state"] = True
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphWorkflowInput":
        return cls(
            graph_config=GraphConfig.from_dict(data["graph_config"]),
            channel_state=ChannelState.from_dict(data["channel_state"]),
            step=data.get("step", 0),
            max_steps=data.get("max_steps", 100),
            strict_limit=bool(data.get("strict_limit", False)),
            pending_nodes=data.get("pending_nodes", []),
            config=data.get("config"),
            thread_id=data.get("thread_id"),
            thread_state=bool(data.get("thread_state", False)),
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
        branches: The conditional edges evaluated in their own activity, as
            ``[step, sink key]`` (step ``-1`` for a START edge), for the same
            purpose. ``None`` when there were none.
        thread_saved: ``False`` when the run's thread state could not be
            saved (its output stands); ``None`` otherwise.
        thread_error: Why the thread state was not saved.
    """

    output: Dict[str, Any] = field(default_factory=dict)
    channel_state: Optional[ChannelState] = None
    steps: int = 0
    status: str = "completed"
    error: Optional[str] = None
    executed: Optional[List[List[str]]] = None
    branches: Optional[List[List[Any]]] = None
    thread_saved: Optional[bool] = None
    thread_error: Optional[str] = None

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
        if self.branches:
            data["branches"] = self.branches
        if self.thread_saved is not None:
            data["thread_saved"] = self.thread_saved
            data["thread_error"] = self.thread_error
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
            branches=data.get("branches"),
            thread_saved=data.get("thread_saved"),
            thread_error=data.get("thread_error"),
        )
