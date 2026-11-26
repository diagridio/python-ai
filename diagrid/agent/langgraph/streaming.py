# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""LangGraph stream-mode support for node activities.

Reuses what ``CompiledStateGraph.astream`` uses natively: LangGraph's
``StreamMessagesHandler`` (``messages``), ``Runtime.stream_writer``
(``custom``) and the ``langgraph_*`` metadata keys the Pregel loop sets on
every task. Chunks go to a :class:`StreamSink` wrapped in a ``StreamEvent``.
"""

import functools
import itertools
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from langchain_core.load import dumpd
from langchain_core.load.serializable import Serializable
from langgraph.errors import GraphRecursionError

from diagrid.agent.core.workflow.stream import (
    STREAM_UNAVAILABLE,
    UPDATES,
    VALUES,
    StreamEvent,
    StreamSink,
    display_step,
)

from .task_events import (
    DEBUG,
    TASKS,
    debug_event,
    task_id,
    task_metadata,
    task_result,
    task_start,
)

logger = logging.getLogger(__name__)

MESSAGES = "messages"
CUSTOM = "custom"
SUPPORTED_MODES = (MESSAGES, UPDATES, CUSTOM, VALUES, TASKS, DEBUG)
# Modes LangGraph has and the runner refuses, with the reason it gives.
REFUSED_MODES = {
    "checkpoints": (
        "stream_mode 'checkpoints' is not supported: the Dapr workflow runner "
        "keeps no LangGraph checkpoints (the workflow history is the run's "
        "durable record), so there are no checkpoint ids to stream or pass back. "
        "Read the full state from the run's final event or from "
        "GET /agent/run/{workflow_id}."
    ),
}

# Configurable keys LangChain never copies into metadata (they are secrets).
_UNTRACED_CONFIGURABLE = frozenset(("api_key",))

# Key of the run_async terminal event that lists the nodes each step ran. It
# serves the stream (see stream_unavailable_event) and is not part of a result.
EXECUTED_KEY = "executed"
# Key of the run_async terminal event of a streamed run that holds the state it
# ended with, for the stream's last ``values`` chunk; also not part of a result.
FINAL_STATE_KEY = "channel_state"

# Set in the workflow config of a run someone is streaming, so an activity that
# finds no sink to write to knows that is a problem and not just a plain run.
STREAM_RUN_KEY = "diagrid_streamed"


@dataclass(frozen=True)
class StreamControl:
    """An event about the stream itself rather than a chunk of the run.

    ``kind`` is ``workflow_started`` (``{"workflow_id", "thread_id"}``, always
    first), ``node_reset``, ``stream_unavailable`` or ``stream_dropped``;
    ``data`` is its payload. ``astream`` hands these to its ``on_control``
    callback, and in list mode also yields them as ``(kind, data)``, except
    ``workflow_started``, which is only ever given to ``on_control`` and never
    yielded.
    """

    kind: str
    data: Dict[str, Any]


@functools.lru_cache(maxsize=None)
def _own_node_handler_class() -> Any:
    """``StreamMessagesHandler`` limited to the node the activity runs.

    Native LangGraph (``subgraphs=False``) tells a nested graph's runs by their
    checkpoint namespace having more than one segment. Here a node that is a
    compiled subgraph runs as a plain call inside the activity, so the nested
    graph's namespace has a single segment and the stock handler would stream
    its inner nodes' tokens under their own names. Everything whose
    ``langgraph_checkpoint_ns`` is not this node's belongs to a nested graph and
    is ignored; the node's own output message is still sent when it finishes,
    as native does.
    """
    from langgraph.pregel._messages import StreamMessagesHandler

    class OwnNodeMessagesHandler(StreamMessagesHandler):  # type: ignore[misc]
        def __init__(self, *args: Any, own_checkpoint_ns: str, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._own_checkpoint_ns = own_checkpoint_ns

        def _nested(self, metadata: Optional[Dict[str, Any]]) -> bool:
            if not metadata:
                return False
            ns = metadata.get("langgraph_checkpoint_ns")
            return ns is not None and ns != self._own_checkpoint_ns

        def on_chat_model_start(self, *args: Any, **kwargs: Any) -> Any:
            if self._nested(kwargs.get("metadata")):
                return None
            return super().on_chat_model_start(*args, **kwargs)

        def on_chain_start(self, *args: Any, **kwargs: Any) -> Any:
            if self._nested(kwargs.get("metadata")):
                return None
            return super().on_chain_start(*args, **kwargs)

    return OwnNodeMessagesHandler


@dataclass(frozen=True)
class StaticConfig:
    """The metadata and tags the graph itself gives a node's tasks.

    LangGraph merges the compiled graph's own config (``with_config``) under
    the caller's, and the node's ``add_node(metadata=...)`` over both. They
    are fixed by the graph, so a runner registers them once per node.
    """

    graph_metadata: Dict[str, Any] = field(default_factory=dict)
    graph_tags: Tuple[str, ...] = ()
    node_metadata: Dict[str, Any] = field(default_factory=dict)
    node_tags: Tuple[str, ...] = ()

    def metadata(self, caller: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """graph, then the caller's, then the node's own: later wins."""
        return {**self.graph_metadata, **(caller or {}), **self.node_metadata}

    def tags(self, caller: Optional[Sequence[str]]) -> List[str]:
        """The graph's tags, the caller's, the node's, without repeats."""
        merged = [*self.graph_tags, *(caller or ()), *self.node_tags]
        return list(dict.fromkeys(merged))


class NodeStreamer:
    """Streams one activity attempt of one node into a sink."""

    # Logged once per process, not per node of every run.
    _handler_failure_logged = False
    _failure_lock = threading.Lock()

    def __init__(
        self,
        sink: StreamSink,
        *,
        workflow_id: str,
        step: int,
        node: str,
        task_id: Any,
        thread_id: Optional[str] = None,
        triggers: Optional[Sequence[str]] = None,
        static: Optional[StaticConfig] = None,
        sink_node: Optional[str] = None,
        path: Optional[Sequence[Any]] = None,
        branch: Optional[str] = None,
    ) -> None:
        self._path = tuple(path) if path else None
        # The conditional edge whose function this streamer serves, if any.
        self._branch = branch
        # ``sink_node`` is the name chunks are keyed under in the sink when it
        # is not the node's own: a conditional edge's function streams as its
        # source node but must not share that node's attempts and ``updates``.
        self._sink_node = sink_node or node
        self._static = static or StaticConfig()
        self._sink = sink
        self._thread_id = thread_id
        self._workflow_id = workflow_id
        self._step = step
        self._node = node
        self._task_id = task_id
        self._triggers = tuple(triggers) if triggers else None
        self._attempt_id = uuid.uuid4().hex
        self._seq = itertools.count()
        # A node may emit from several threads (parallel tool calls); seq is
        # taken and the chunk handed over under one lock so they stay in order.
        self._emit_lock = threading.Lock()
        self._handler: Any = None
        self._handler_unusable = False
        self._run_id = uuid.uuid4()

    @property
    def metadata(self) -> Dict[str, Any]:
        """The ``langgraph_*`` keys the Pregel loop sets on a task's config."""
        return {
            # Pregel numbers the first node after START as step 1.
            "langgraph_step": display_step(self._step),
            "langgraph_node": self._node,
            # Every trigger channel of the node, sorted, as Pregel lists them.
            "langgraph_triggers": self._triggers or (f"branch:to:{self._node}",),
            "langgraph_path": self._path or ("__pregel_pull", self._node),
            "langgraph_checkpoint_ns": f"{self._node}:{self._task_id}",
            "checkpoint_ns": f"{self._node}:{self._task_id}",
        }

    def _emit(self, ns: Tuple[str, ...], mode: str, data: Any) -> None:
        with self._emit_lock:
            self._sink.put(
                StreamEvent(
                    workflow_id=self._workflow_id,
                    step=self._step,
                    node=self._sink_node,
                    attempt_id=self._attempt_id,
                    seq=next(self._seq),
                    ns=tuple(ns),
                    mode=mode,
                    data=data,
                    display_node=self._node,
                    path=self._path,
                    branch=self._branch,
                )
            )

    def _on_chunk(self, chunk: Tuple[Any, ...]) -> None:
        """Receive ``(ns, mode, payload)`` from ``StreamMessagesHandler``."""
        ns, mode, payload = chunk
        message, metadata = payload
        self._emit(ns, mode, (message, metadata))

    def _messages_handler(self) -> Any:
        """LangGraph's ``StreamMessagesHandler``, or ``None`` if it is unusable.

        It is a private class of langgraph, and the dependency is not pinned
        below the release that may move or change it. When it cannot be built,
        the node still runs and its ``updates`` and ``custom`` chunks still
        stream; only ``messages`` is lost, and that is logged once.
        """
        if self._handler is None and not self._handler_unusable:
            try:
                self._handler = _own_node_handler_class()(
                    self._on_chunk,
                    False,
                    own_checkpoint_ns=self.metadata["langgraph_checkpoint_ns"],
                )
            except (ImportError, TypeError, AttributeError) as e:
                self._handler_unusable = True
                NodeStreamer._log_handler_failure(e)
        return self._handler

    @staticmethod
    def _log_handler_failure(error: BaseException) -> None:
        with NodeStreamer._failure_lock:
            if NodeStreamer._handler_failure_logged:
                return
            NodeStreamer._handler_failure_logged = True
        logger.error(
            "Cannot stream 'messages': langgraph's StreamMessagesHandler is not "
            "usable (%s: %s). Nodes run as usual and 'updates' and 'custom' "
            "still stream, but no LLM tokens are sent. This is expected after a "
            "langgraph release that moved or changed that private class.",
            type(error).__name__,
            error,
        )

    def callbacks(self) -> List[Any]:
        handler = self._messages_handler()
        return [] if handler is None else [handler]

    # The activity calls the node's bound runnable directly, which never
    # starts the chain run Pregel's handler keys node-output messages on
    # (StateGraph nodes are built with trace=False). Drive those two
    # callbacks by hand so ToolNode / hand-built messages stream too.
    def node_started(
        self, node_input: Any, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Open the chain run node-output messages are keyed on.

        Native LangGraph gives that run the task's config metadata (the
        caller's metadata, ``thread_id`` and the other configurable values,
        ``ls_integration``, the ``langgraph_*`` keys), and every message a node
        returns (a ``ToolNode``'s ``ToolMessage``) is sent with it. ``config``
        is the caller's run config.
        """
        handler = self._messages_handler()
        if handler is None:
            return
        config = config or {}
        metadata = {
            **self._configurable_metadata(config),
            "ls_integration": "langgraph",
            **self._static.metadata(config.get("metadata")),
            **{k: v for k, v in self.metadata.items() if k != "checkpoint_ns"},
        }
        handler.on_chain_start(
            {},
            node_input,
            run_id=self._run_id,
            name=self._node,
            metadata=metadata,
            tags=self._static.tags(config.get("tags")),
        )

    def node_finished(self, output: Any) -> None:
        handler = self._messages_handler()
        if handler is not None:
            handler.on_chain_end(output, run_id=self._run_id)

    def node_failed(self, error: BaseException) -> None:
        """Close the chain run :meth:`node_started` opened.

        Without it the message handler keeps that run's state for good.
        """
        handler = self._messages_handler()
        if handler is not None:
            handler.on_chain_error(error, run_id=self._run_id)

    def writer(self, chunk: Any) -> None:
        """``Runtime.stream_writer``: what ``get_stream_writer()`` returns."""
        self._emit((), CUSTOM, chunk)

    @property
    def _event_task_id(self) -> str:
        """The id of this task in its ``tasks`` events."""
        return task_id(self._workflow_id, self._step, self._sink_node)

    def _emit_task(self, payload: Dict[str, Any]) -> None:
        """A ``tasks`` event, and the same event as ``debug`` sends it."""
        self._emit((), TASKS, payload)
        self._emit((), DEBUG, debug_event(display_step(self._step), payload))

    def task_started(
        self, node_input: Any, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send the task's start event, as LangGraph does when a step starts it."""
        config = config or {}
        metadata = task_metadata(
            self._static.metadata(config.get("metadata")),
            self._static.tags(config.get("tags")),
        )
        triggers = self.metadata["langgraph_triggers"]
        self._emit_task(
            task_start(self._event_task_id, self._node, node_input, triggers, metadata)
        )

    def task_finished(self, writes: List[Tuple[str, Any]]) -> None:
        """Send the task's result event: its writes, after its ``updates`` chunk."""
        self._emit_task(task_result(self._event_task_id, self._node, writes))

    def updates(self, writes: List[Tuple[str, Any]]) -> None:
        """Emit the node's ``(channel, value)`` writes as ``{node: update}``.

        The shape is what ``CompiledStateGraph.astream(stream_mode="updates")``
        yields for one node, with the values exactly as the node returned
        them: ``None`` when it wrote nothing, one dict of channel to value, or,
        when a channel is written more than once (a list of Commands, repeated
        ``files`` sends), a list of single-channel dicts so no write is lost.
        """
        update: Any
        channels = [ch for ch, _ in writes]
        if not writes:
            update = None
        elif channels == ["__root__"]:
            # a graph whose whole state is one value: LangGraph reports the value
            update = writes[0][1]
        elif len(set(channels)) == len(channels):
            update = dict(writes)
        else:
            update = [{ch: value} for ch, value in writes]
        self._emit((), UPDATES, {self._node: update})

    def patch_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Copy of ``config`` with the handler and task metadata attached."""
        patched = dict(config)
        existing = patched.get("callbacks")
        if existing is None:
            callbacks: Any = self.callbacks()
        elif isinstance(existing, list):
            callbacks = [*existing, *self.callbacks()]
        else:  # a CallbackManager
            callbacks = existing.copy()
            for handler in self.callbacks():
                callbacks.add_handler(handler, inherit=True)
        patched["callbacks"] = callbacks
        patched["metadata"] = {
            **self._configurable_metadata(patched),
            **self._static.metadata(patched.get("metadata")),
            **self.metadata,
        }
        tags = self._static.tags(patched.get("tags"))
        if tags or "tags" in patched:
            patched["tags"] = tags
        return patched

    def _configurable_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """What native LangGraph copies from ``configurable`` into metadata.

        ``thread_id`` and the other primitive, non-private configurable values,
        so a client keying on ``metadata["thread_id"]`` sees what it would
        from ``CompiledStateGraph.astream``.
        """
        derived = {
            key: value
            for key, value in (config.get("configurable") or {}).items()
            if not key.startswith("__")
            and isinstance(value, (str, int, float, bool))
            and key not in _UNTRACED_CONFIGURABLE
        }
        if self._thread_id:
            derived.setdefault("thread_id", self._thread_id)
        return derived


def normalize_stream_mode(stream_mode: Any) -> Tuple[List[str], bool]:
    """Return ``(modes, is_list)``; reject modes the runner cannot provide."""
    is_list = isinstance(stream_mode, (list, tuple))
    modes = list(stream_mode) if is_list else [stream_mode]
    refused = [
        REFUSED_MODES[m] for m in modes if isinstance(m, str) and m in REFUSED_MODES
    ]
    if refused:
        raise ValueError(refused[0])
    unsupported = [m for m in modes if m not in SUPPORTED_MODES]
    if unsupported:
        raise ValueError(
            f"Unsupported stream_mode {unsupported!r}; "
            f"supported: {list(SUPPORTED_MODES)}"
        )
    return modes, is_list


def _plain_data(value: Any) -> Any:
    """``value`` with everything but LangChain objects reduced to plain data.

    ``dumpd`` replaces what it cannot serialize (a pydantic model, a datetime,
    a UUID) with a ``not_implemented`` stub that holds only a repr. Leaves go
    through the same fallback the run's final output uses
    (``workflow._serialize_value``) so a client sees the same data in a chunk
    and in the final event. ``Serializable`` objects (messages, documents) stay
    as they are for ``dumpd`` to turn into their ``load()``-able form.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Serializable):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain_data(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain_data(v) for k, v in value.items()}
    from diagrid.agent.langgraph.workflow import _serialize_value

    reduced = _serialize_value(value)
    # A model dumps to a dict that can still hold a datetime or a UUID.
    return str(value) if reduced is value else _plain_data(reduced)


def to_wire(mode: str, data: Any) -> Any:
    """JSON-ready form of a chunk for a transport.

    Every chunk of a stream mode can carry LangChain objects (messages in a
    ``values`` state, a task's input) next to arbitrary values a node chose to
    return or write. ``dumpd`` turns
    the objects into the form ``langchain_core.load.load`` restores, so a client
    sees the same objects an in-process ``astream`` consumer does; everything
    else becomes plain data first (see :func:`_plain_data`). Control events go
    as they are.
    """
    if mode in SUPPORTED_MODES:
        return dumpd(_plain_data(data))
    return data


# A finished workflow whose output says it did not get to the end of the graph.
FAILED_STATUSES = ("error", "max_steps_reached")
NO_RESULT_MESSAGE = "run ended without a result"


def failure_message(final: Dict[str, Any]) -> str:
    """The human-readable reason a ``workflow_failed`` event carries."""
    error = final.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "Workflow failed")
    return str(error) if error else "Workflow failed"


class GraphRunError(RuntimeError):
    """A streamed run ended in failure; ``final`` is its terminal event."""

    def __init__(self, final: Dict[str, Any]) -> None:
        super().__init__(failure_message(final))
        self.final = final


class GraphRecursionRunError(GraphRunError, GraphRecursionError):
    """A run cut by its step limit: a :class:`GraphRunError` that is also LangGraph's.

    ``except GraphRecursionError`` (the usual guard around an agent loop) catches
    it, as it catches the error native LangGraph raises.
    """


def run_error(final: Dict[str, Any]) -> GraphRunError:
    """The exception for a ``workflow_failed`` terminal event."""
    if final.get("status") == "max_steps_reached":
        return GraphRecursionRunError(final)
    return GraphRunError(final)


def workflow_failed(
    final: Dict[str, Any], message: str, error_type: str
) -> Dict[str, Any]:
    """``final`` turned into a ``workflow_failed`` event with this error."""
    return {
        **final,
        "type": "workflow_failed",
        "error": {"message": message, "error_type": error_type},
    }


def normalize_final(
    final: Dict[str, Any], *, workflow_id: Optional[str], thread_id: str
) -> Dict[str, Any]:
    """Turn whatever ended ``run_async`` into one terminal event.

    The result is a ``workflow_completed`` or ``workflow_failed`` event that
    names the run. The workflow reports some failures as a completed run
    (``status`` of ``error`` for a missing node or a failed condition,
    ``max_steps_reached``); a run can also end terminated, errored, or with no
    final event at all. None of those is a success.
    """
    kind = final.get("type")
    status = final.get("status")
    if not final:
        out = workflow_failed({}, NO_RESULT_MESSAGE, "NoResult")
    elif kind == "workflow_completed" and status in FAILED_STATUSES:
        out = workflow_failed(
            final,
            str(final.get("error") or f"Workflow ended with status {status!r}"),
            "WorkflowStatus",
        )
    elif kind == "workflow_error":
        out = workflow_failed(
            final, str(final.get("error") or "Workflow error"), "WorkflowError"
        )
    elif kind == "workflow_terminated":
        out = workflow_failed(final, "Workflow was terminated", "WorkflowTerminated")
    else:
        out = dict(final)
    if workflow_id:
        out.setdefault("workflow_id", workflow_id)
    out.setdefault("thread_id", thread_id)
    return out


UNAVAILABLE_REASON = (
    "No chunks arrived for the listed nodes: their activities ran in another "
    "process (several replicas) or before this stream was opened. Streaming only "
    "works while the activities run in the process that serves the stream; the "
    "run's result is complete."
)


def stream_unavailable_event(
    final: Dict[str, Any], seen: Set[Tuple[int, str]], workflow_id: str
) -> Optional[StreamEvent]:
    """The ``stream_unavailable`` event for a finished run, if it applies.

    ``seen`` holds the ``(step, node)`` pairs whose ``updates`` chunk arrived;
    every node that runs here sends one when it finishes, so an executed node
    without one had its activity (or its last attempt) elsewhere, even when
    other nodes of its step streamed here. A final without ``executed`` is
    judged per step instead. A run
    that failed is just a failure, and one with no step count has nothing to
    report.
    """
    steps = final.get("steps")
    if final.get("type") != "workflow_completed" or not isinstance(steps, int):
        return None
    executed = final.get("executed")
    missing_nodes: List[Dict[str, Any]] = []
    if isinstance(executed, list):
        missing_nodes = [
            {"step": display_step(step), "node": node}
            for step, nodes in enumerate(executed)
            for node in nodes
            if (step, node) not in seen
        ]
        missing = sorted({m["step"] for m in missing_nodes})
    else:
        seen_steps = {step for step, _ in seen}
        missing = [
            display_step(step) for step in range(steps) if step not in seen_steps
        ]
    if not missing:
        return None
    return StreamEvent(
        workflow_id=workflow_id,
        step=missing[0] - 1,
        node="",
        attempt_id="",
        seq=0,
        ns=(),
        mode=STREAM_UNAVAILABLE,
        data={
            "workflow_id": workflow_id,
            "reason": UNAVAILABLE_REASON,
            "missing_steps": missing,
            "missing_nodes": missing_nodes,
        },
    )
