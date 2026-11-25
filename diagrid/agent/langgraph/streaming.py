# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""LangGraph stream-mode support for node activities.

Reuses what ``CompiledStateGraph.astream`` uses natively: LangGraph's
``StreamMessagesHandler`` (``messages``), ``Runtime.stream_writer``
(``custom``) and the ``langgraph_*`` metadata keys the Pregel loop sets on
every task. Chunks go to a :class:`StreamSink` wrapped in a ``StreamEvent``.
"""

import itertools
import uuid
from typing import Any, Dict, List, Tuple

from langchain_core.load import dumpd

from diagrid.agent.core.workflow.stream import StreamEvent, StreamSink

MESSAGES = "messages"
UPDATES = "updates"
CUSTOM = "custom"
SUPPORTED_MODES = (MESSAGES, UPDATES, CUSTOM)


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
    ) -> None:
        self._sink = sink
        self._workflow_id = workflow_id
        self._step = step
        self._node = node
        self._task_id = task_id
        self._attempt_id = uuid.uuid4().hex
        self._seq = itertools.count()
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
        patched["metadata"] = {**(patched.get("metadata") or {}), **self.metadata}
        return patched


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


def to_wire(mode: str, data: Any) -> Any:
    """JSON-ready form of a chunk for a transport.

    ``messages`` and ``updates`` carry LangChain objects; ``dumpd`` turns them
    into the form ``langchain_core.load.load`` restores, so a client sees the
    same objects an in-process ``astream`` consumer does. Other chunks (the
    ``custom`` payload a node chose to write, control events) go as they are.
    """
    if mode in (MESSAGES, UPDATES):
        return dumpd(data)
    return data
