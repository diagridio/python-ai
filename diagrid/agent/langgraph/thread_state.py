# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A thread's state, carried from one run to the next.

With a checkpointer, LangGraph starts a run on a thread from the state the
thread's last run ended with and applies the new input to it through the
channels' reducers. The workflow does the same: an activity loads the thread's
last state before the first step, the workflow applies the input to it, and an
activity saves the state the run ended with. Workflow code never touches the
store, so a replay reads both from the history.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Protocol

from .models import ChannelState
from .state import DaprMemoryCheckpointer

logger = logging.getLogger(__name__)

#: An instance started before threads carried state replays without the load and
#: save activities: the patch is what tells the two apart.
THREAD_STATE_PATCH = "langgraph-thread-state"
LOAD_THREAD_ACTIVITY = "load_thread_state"
SAVE_THREAD_ACTIVITY = "save_thread_state"


class ThreadStore(Protocol):
    """Where a thread's last channel state is kept (``ChannelState`` dicts)."""

    def load(self, thread_id: str) -> Optional[Dict[str, Any]]: ...

    def save(self, checkpoint: "ThreadCheckpoint") -> None: ...


@dataclass(frozen=True)
class ThreadCheckpoint:
    """The state a run on a thread ended with, as the save activity gets it."""

    thread_id: str
    #: The workflow instance that produced it.
    checkpoint_id: str
    channel_state: Dict[str, Any]
    steps: int = 0
    status: str = "completed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "checkpoint_id": self.checkpoint_id,
            "channel_state": self.channel_state,
            "steps": self.steps,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreadCheckpoint":
        return cls(
            thread_id=str(data["thread_id"]),
            checkpoint_id=str(data["checkpoint_id"]),
            channel_state=dict(data.get("channel_state") or {}),
            steps=int(data.get("steps") or 0),
            status=str(data.get("status") or "completed"),
        )


class DaprThreadStore:
    """A thread's state in a Dapr state store, through ``DaprMemoryCheckpointer``."""

    def __init__(self, checkpointer: DaprMemoryCheckpointer) -> None:
        self._checkpointer = checkpointer

    def load(self, thread_id: str) -> Optional[Dict[str, Any]]:
        data = self._checkpointer.load_checkpoint(thread_id)
        if not data:
            return None
        return ChannelState(
            values=dict(data.get("channel_values") or {}),
            versions=dict(data.get("channel_versions") or {}),
        ).to_dict()

    def save(self, checkpoint: ThreadCheckpoint) -> None:
        state = ChannelState.from_dict(checkpoint.channel_state)
        self._checkpointer.save_checkpoint(
            thread_id=checkpoint.thread_id,
            checkpoint_id=checkpoint.checkpoint_id,
            channel_values=state.values,
            channel_versions=state.versions,
            metadata={"steps": checkpoint.steps, "status": checkpoint.status},
        )


def resolve_thread_store(
    checkpointer: Any, graph: Any, state_store: Any
) -> Optional[ThreadStore]:
    """The store a runner keeps its threads' state in, or ``None`` for none.

    The runner's own ``checkpointer`` first, then the one the graph was
    compiled with, then the app's Dapr state store (``agent-memory``, which a
    Catalyst app has). Without any of them a thread id is only a label.

    A graph compiled with an in-memory saver (``MemorySaver``, as LangGraph's
    examples do) keeps threads in one process, which breaks across replicas
    and restarts: the app's Dapr state store is used instead when there is
    one, and either way a warning says so.
    """
    from .saver_store import SaverThreadStore, is_in_memory_saver, is_saver

    if checkpointer is not None:
        if isinstance(checkpointer, DaprMemoryCheckpointer):
            return DaprThreadStore(checkpointer)
        if is_saver(checkpointer):
            if is_in_memory_saver(checkpointer):
                logger.warning(_IN_MEMORY_WARNING, type(checkpointer).__name__)
            return SaverThreadStore(checkpointer, _state_channels(graph))
        raise TypeError(
            "checkpointer must be a DaprMemoryCheckpointer or a LangGraph "
            f"checkpoint saver, not {type(checkpointer).__name__}"
        )
    own = getattr(graph, "checkpointer", None)
    if is_in_memory_saver(own) and state_store is not None:
        logger.warning(
            "The graph was compiled with %s, which keeps threads in this process "
            "only; the runner keeps them in the app's Dapr state store %r "
            "instead, so they survive restarts and work across replicas",
            type(own).__name__,
            getattr(state_store, "store_name", None),
        )
        return DaprThreadStore(DaprMemoryCheckpointer(state_store=state_store))
    if is_saver(own):
        if is_in_memory_saver(own):
            logger.warning(_IN_MEMORY_WARNING, type(own).__name__)
        return SaverThreadStore(own, _state_channels(graph))
    if state_store is not None:
        return DaprThreadStore(DaprMemoryCheckpointer(state_store=state_store))
    return None


_IN_MEMORY_WARNING = (
    "Threads are kept in %s, in this process only: a thread's state is lost on "
    "a restart and not seen by another replica. Give the app a Dapr state "
    "store (Catalyst apps have one) or a durable checkpointer"
)


def _state_channels(graph: Any) -> List[str]:
    channels = getattr(getattr(graph, "builder", None), "channels", None) or {}
    return list(channels)


def make_load_activity(
    store: Optional[ThreadStore],
) -> Callable[[Any, Dict[str, Any]], Optional[Dict[str, Any]]]:
    """The activity that reads a thread's last state (``None`` for a new thread)."""

    def load_thread_state(
        ctx: Any, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if store is None:
            logger.warning("A run asked for its thread's state but no store is set")
            return None
        return store.load(str(input_data["thread_id"]))

    return load_thread_state


def make_save_activity(
    store: Optional[ThreadStore],
) -> Callable[[Any, Dict[str, Any]], None]:
    """The activity that keeps the state a run ended with as its thread's."""

    def save_thread_state(ctx: Any, input_data: Dict[str, Any]) -> None:
        if store is None:
            logger.warning("A run asked to save its thread's state but no store is set")
            return
        store.save(ThreadCheckpoint.from_dict(input_data))

    return save_thread_state


def continue_thread(loaded: ChannelState, incoming: ChannelState) -> ChannelState:
    """The thread's state with the run's input applied to it.

    Each input channel is written through its reducer, as LangGraph applies a
    run's input to a thread's state: ``add_messages`` appends, a plain channel
    is replaced. Deterministic, so it may run in workflow code.
    """
    from .workflow import _apply_write

    state = ChannelState(
        values=copy.deepcopy(loaded.values), versions=dict(loaded.versions)
    )
    for channel in incoming.updated_channels:
        if channel in incoming.values:
            _apply_write(state, channel, incoming.values[channel])
    state.updated_channels = list(incoming.updated_channels)
    return state


def load_thread(
    ctx: Any, thread_id: str, incoming: ChannelState, retry_policy: Any
) -> Generator[Any, Any, ChannelState]:
    """Workflow step: the run's starting state, from its thread's and its input."""
    loaded = yield ctx.call_activity(
        LOAD_THREAD_ACTIVITY,
        input={"thread_id": thread_id},
        retry_policy=retry_policy,
    )
    if not loaded:
        return incoming
    return continue_thread(ChannelState.from_dict(loaded), incoming)


def save_thread(
    ctx: Any, thread_id: str, output: Dict[str, Any], retry_policy: Any
) -> Generator[Any, Any, None]:
    """Workflow step: keep the state the run ended with as its thread's."""
    if not output.get("channel_state"):
        return
    checkpoint = ThreadCheckpoint(
        thread_id=thread_id,
        checkpoint_id=str(ctx.instance_id),
        channel_state=output["channel_state"],
        steps=int(output.get("steps") or 0),
        status=str(output.get("status") or "completed"),
    )
    yield ctx.call_activity(
        SAVE_THREAD_ACTIVITY,
        input=checkpoint.to_dict(),
        retry_policy=retry_policy,
    )
