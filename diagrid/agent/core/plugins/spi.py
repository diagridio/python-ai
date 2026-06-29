# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Core SPI types for the plugin system.

The ``Plugin`` Protocol and the ``HookDecision`` union together define
what every plugin implements.
Concrete plugins (OAuth, HITL, observability hooks) live in sibling
packages and import these types.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    from diagrid.agent.core.plugins.context import LifecycleContext

# dapr-agents owns the decision taxonomy so the two codebases share a single
# definition. The local fallback below keeps the SPI importable on its own
# until a dapr-agents release ships the ``hooks`` module. Catch only
# ModuleNotFoundError so a genuine import error inside an existing module
# surfaces instead of being masked by the fallback.
if TYPE_CHECKING:
    from dapr_agents.hooks import (
        Deny,
        HookDecision,
        Mutate,
        Proceed,
        RequireApproval,
        Skip,
    )
else:
    try:
        from dapr_agents.hooks import (
            Deny,
            HookDecision,
            Mutate,
            Proceed,
            RequireApproval,
            Skip,
        )
    except ModuleNotFoundError:
        # Name-only placeholders; the real shapes arrive with the import above.
        class Proceed:
            pass

        class Skip:
            pass

        class Mutate:
            pass

        class RequireApproval:
            pass

        class Deny:
            pass

        HookDecision = Union[Proceed, Skip, Mutate, RequireApproval, Deny]


class LifecycleEvent(StrEnum):
    """Lifecycle events fired by dapr-agents and python-ai's framework runners.

    Plugins declare which events they care about through their
    ``capabilities`` attribute.
    """

    BEFORE_AGENT_INVOKE = "BEFORE_AGENT_INVOKE"
    AFTER_AGENT_INVOKE = "AFTER_AGENT_INVOKE"
    BEFORE_LLM_CALL = "BEFORE_LLM_CALL"
    AFTER_LLM_CALL = "AFTER_LLM_CALL"
    BEFORE_TOOL_CALL = "BEFORE_TOOL_CALL"
    AFTER_TOOL_CALL = "AFTER_TOOL_CALL"
    # Derived from BEFORE_TOOL_CALL when the target resolves to an MCP server.
    BEFORE_MCP_CALL = "BEFORE_MCP_CALL"
    BEFORE_APPROVAL_DECISION = "BEFORE_APPROVAL_DECISION"


@runtime_checkable
class Plugin(Protocol):
    """Protocol every plugin implements.

    Plugins are composable units.
    A ``PluginRegistry`` holds a chain of them and dispatches lifecycle
    events in priority order.

    Attributes:
        name: Stable identifier used in audit events and debug logs.
        priority: Lower runs first; ties break by registration order.
            By convention security plugins (OAuth, HITL) use 10,
            OBO-related plugins use 50, and the legacy hooks adapter uses 100.
        capabilities: The set of events this plugin handles.
            Events outside the set short-circuit before reaching
            ``on_event``, which avoids a no-op call.
        failure_mode: ``"closed"`` turns an exception into a ``Deny`` and
            short-circuits the chain; ``"open"`` logs, emits a metric, and
            proceeds. Security plugins must be ``"closed"``; observability
            plugins may be ``"open"``.
    """

    name: str
    priority: int
    capabilities: frozenset[LifecycleEvent]
    failure_mode: str

    def configure(self, agent: Any) -> None:
        """Initialize per-agent state.

        Called once when the plugin is attached to an agent.
        This is where clients are opened and sub-handlers are registered.
        """
        ...

    async def on_event(self, ctx: "LifecycleContext") -> Optional[HookDecision]:
        """Handle one of this plugin's ``capabilities`` events.

        Return ``None`` or ``Proceed`` to continue the chain.
        Return ``Skip``, ``Mutate``, ``RequireApproval``, or ``Deny`` to
        alter the workflow following dapr-agents semantics.
        """
        ...


__all__ = [
    "Plugin",
    "LifecycleEvent",
    "HookDecision",
    "Proceed",
    "Skip",
    "Mutate",
    "RequireApproval",
    "Deny",
]
