# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""LangGraph stream-mode support for node activities.

Reuses what ``CompiledStateGraph.astream`` uses natively: LangGraph's
``StreamMessagesHandler`` (``messages``), ``Runtime.stream_writer``
(``custom``) and the ``langgraph_*`` metadata keys the Pregel loop sets on
every task. Chunks go to a :class:`StreamSink` wrapped in a ``StreamEvent``.
"""

import itertools
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.load import dumpd
from langchain_core.load.serializable import Serializable

from diagrid.agent.core.workflow.stream import (
    STREAM_UNAVAILABLE,
    StreamEvent,
    StreamSink,
)

MESSAGES = "messages"
UPDATES = "updates"
CUSTOM = "custom"
SUPPORTED_MODES = (MESSAGES, UPDATES, CUSTOM)

# Configurable keys LangChain never copies into metadata (they are secrets).
_UNTRACED_CONFIGURABLE = frozenset(("api_key",))

# Set in the workflow config of a run someone is streaming, so an activity that
# finds no sink to write to knows that is a problem and not just a plain run.
STREAM_RUN_KEY = "diagrid_streamed"


@dataclass(frozen=True)
class StreamControl:
    """An event about the stream itself rather than a chunk of the run.

    ``kind`` is ``node_reset``, ``stream_unavailable`` or ``stream_dropped``;
    ``data`` is its payload. ``astream`` hands these to its ``on_control``
    callback, and in list mode also yields them as ``(kind, data)``.
    """

    kind: str
    data: Dict[str, Any]


class NodeStreamer:
    """Streams one activity attempt of one node into a sink."""

    def __init__(
        self,
        sink: StreamSink,
        *,
        workflow_id: str,
        step: int,
        node: str,
        task_id: Any,
        thread_id: Optional[str] = None,
    ) -> None:
        self._sink = sink
        self._thread_id = thread_id
        self._workflow_id = workflow_id
        self._step = step
        self._node = node
        self._task_id = task_id
        self._attempt_id = uuid.uuid4().hex
        self._seq = itertools.count()
        # A node may emit from several threads (parallel tool calls); seq is
        # taken and the chunk handed over under one lock so they stay in order.
        self._emit_lock = threading.Lock()
        self._handler: Any = None
        self._run_id = uuid.uuid4()

    @property
    def metadata(self) -> Dict[str, Any]:
        """The ``langgraph_*`` keys the Pregel loop sets on a task's config."""
        return {
            # Pregel numbers the first node after START as step 1.
            "langgraph_step": self._step + 1,
            "langgraph_node": self._node,
            "langgraph_triggers": (f"branch:to:{self._node}",),
            "langgraph_path": ("__pregel_pull", self._node),
            "langgraph_checkpoint_ns": f"{self._node}:{self._task_id}",
            "checkpoint_ns": f"{self._node}:{self._task_id}",
        }

    def _emit(self, ns: Tuple[str, ...], mode: str, data: Any) -> None:
        with self._emit_lock:
            self._sink.put(
                StreamEvent(
                    workflow_id=self._workflow_id,
                    step=self._step,
                    node=self._node,
                    attempt_id=self._attempt_id,
                    seq=next(self._seq),
                    ns=tuple(ns),
                    mode=mode,
                    data=data,
                )
            )

    def _on_chunk(self, chunk: Tuple[Any, ...]) -> None:
        """Receive ``(ns, mode, payload)`` from ``StreamMessagesHandler``."""
        ns, mode, payload = chunk
        message, metadata = payload
        self._emit(ns, mode, (message, metadata))

    def _messages_handler(self) -> Any:
        if self._handler is None:
            from langgraph.pregel._messages import StreamMessagesHandler

            self._handler = StreamMessagesHandler(self._on_chunk, False)
        return self._handler

    def callbacks(self) -> List[Any]:
        return [self._messages_handler()]

    # The activity calls the node's bound runnable directly, which never
    # starts the chain run Pregel's handler keys node-output messages on
    # (StateGraph nodes are built with trace=False). Drive those two
    # callbacks by hand so ToolNode / hand-built messages stream too.
    def node_started(self, node_input: Any) -> None:
        self._messages_handler().on_chain_start(
            {},
            node_input,
            run_id=self._run_id,
            name=self._node,
            metadata=self.metadata,
            tags=[],
        )

    def node_finished(self, output: Any) -> None:
        self._messages_handler().on_chain_end(output, run_id=self._run_id)

    def node_failed(self, error: BaseException) -> None:
        """Close the chain run :meth:`node_started` opened.

        Without it the message handler keeps that run's state for good.
        """
        self._messages_handler().on_chain_error(error, run_id=self._run_id)

    def writer(self, chunk: Any) -> None:
        """``Runtime.stream_writer``: what ``get_stream_writer()`` returns."""
        self._emit((), CUSTOM, chunk)

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
            **(patched.get("metadata") or {}),
            **self.metadata,
        }
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

    ``messages``, ``updates`` and ``custom`` chunks can carry LangChain objects
    next to arbitrary values a node chose to return or write. ``dumpd`` turns
    the objects into the form ``langchain_core.load.load`` restores, so a client
    sees the same objects an in-process ``astream`` consumer does; everything
    else becomes plain data first (see :func:`_plain_data`). Control events go
    as they are.
    """
    if mode in (MESSAGES, UPDATES, CUSTOM):
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


def _failed(final: Dict[str, Any], message: str, error_type: str) -> Dict[str, Any]:
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
        out = _failed({}, NO_RESULT_MESSAGE, "NoResult")
    elif kind == "workflow_completed" and status in FAILED_STATUSES:
        out = _failed(
            final,
            str(final.get("error") or f"Workflow ended with status {status!r}"),
            "WorkflowStatus",
        )
    elif kind == "workflow_error":
        out = _failed(
            final, str(final.get("error") or "Workflow error"), "WorkflowError"
        )
    elif kind == "workflow_terminated":
        out = _failed(final, "Workflow was terminated", "WorkflowTerminated")
    else:
        out = dict(final)
    if workflow_id:
        out.setdefault("workflow_id", workflow_id)
    out.setdefault("thread_id", thread_id)
    return out


UNAVAILABLE_REASON = (
    "No chunks arrived for the listed steps: their node activities ran in "
    "another process (several replicas) or before this stream was opened. "
    "Streaming only works while the activities run in the process that serves "
    "the stream; the run's result is complete."
)


def stream_unavailable_event(
    final: Dict[str, Any], seen_steps: Set[int], workflow_id: str
) -> Optional[StreamEvent]:
    """The ``stream_unavailable`` event for a finished run, if it applies.

    Every node that runs here sends at least its ``updates`` chunk, so a step
    of a completed run from which nothing arrived had its activities elsewhere.
    A run that failed is just a failure, and one with no step count (or none
    run) has nothing to report.
    """
    steps = final.get("steps")
    if final.get("type") != "workflow_completed" or not isinstance(steps, int):
        return None
    missing = [step for step in range(steps) if step not in seen_steps]
    if not missing:
        return None
    return StreamEvent(
        workflow_id=workflow_id,
        step=missing[0],
        node="",
        attempt_id="",
        seq=0,
        ns=(),
        mode=STREAM_UNAVAILABLE,
        data={
            "workflow_id": workflow_id,
            "reason": UNAVAILABLE_REASON,
            "missing_steps": missing,
        },
    )
