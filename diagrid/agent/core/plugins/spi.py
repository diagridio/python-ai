# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Service provider interface for lifecycle plugins.

Defines the ``LifecycleEvent`` enum naming the points an agent dispatches
into the plugin chain, and the ``Plugin`` Protocol that every plugin
implements.
"""

from __future__ import annotations

from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    FrozenSet,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .context import LifecycleContext


FailureMode = Literal["closed", "open"]
"""How the registry treats an exception raised from ``Plugin.on_event``."""

FAILURE_MODE_CLOSED: Final[FailureMode] = "closed"
FAILURE_MODE_OPEN: Final[FailureMode] = "open"


class LifecycleEvent(str, Enum):
    """Lifecycle points an agent dispatches into the plugin chain.

    Values match the event names passed to ``LifecycleDispatcher.dispatch``
    so ``LifecycleEvent(event_name)`` round-trips a dispatched name back to
    its enum member.
    """

    BEFORE_AGENT_INVOKE = "BEFORE_AGENT_INVOKE"
    BEFORE_TOOL_CALL = "BEFORE_TOOL_CALL"
    BEFORE_LLM_CALL = "BEFORE_LLM_CALL"
    BEFORE_APPROVAL_DECISION = "BEFORE_APPROVAL_DECISION"
    AFTER_TOOL_CALL = "AFTER_TOOL_CALL"
    AFTER_LLM_CALL = "AFTER_LLM_CALL"
    AFTER_AGENT_INVOKE = "AFTER_AGENT_INVOKE"


@runtime_checkable
class Plugin(Protocol):
    """A single link in the lifecycle plugin chain.

    Plugins are ordered by ``priority`` (lower runs first) and invoked for
    the events they declare in ``capabilities``.
    A plugin with no declared capabilities is invoked for every event.

    ``failure_mode`` selects how the registry treats an exception raised
    from ``on_event``: ``"closed"`` denies the request (the security
    default), ``"open"`` logs and proceeds (the observability default).
    """

    name: str
    priority: int
    capabilities: Optional[FrozenSet[LifecycleEvent]]
    failure_mode: FailureMode

    def configure(self, agent: Any) -> None:
        """Bind the plugin to an agent before any events are dispatched."""
        ...

    def on_event(self, ctx: "LifecycleContext") -> Any:
        """Handle a lifecycle event.

        Returns a ``HookDecision`` to influence the step, or ``None`` to
        proceed.
        May be sync or async; the registry awaits coroutines and runs sync
        implementations off the event loop.
        """
        ...
