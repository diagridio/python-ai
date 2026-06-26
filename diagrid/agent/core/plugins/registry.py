# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""PluginRegistry — the chain dispatcher for the plugin system.

This module scaffolds the class shape so other code can import and
reference it.
TODO(@sicoyle): The concrete dispatch, attach, and detach logic lands in a follow-on
change; here those methods raise ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from diagrid.agent.core.plugins.spi import Plugin


class PluginRegistry:
    """Holds an ordered chain of plugins and dispatches lifecycle events.

    Plugins run in ascending ``priority`` order, with registration order
    breaking ties.
    Once implemented, the first ``Deny``, ``Skip``, or ``RequireApproval``
    short-circuits the chain while ``Mutate`` decisions accumulate.

    The registry is meant to satisfy dapr-agents' ``LifecycleDispatcher``
    Protocol (``attach``, ``detach``, ``dispatch``) so it can be passed as
    ``DurableAgent(lifecycle_dispatcher=registry)``.
    """

    def __init__(self, plugins: Optional[Sequence[Plugin]] = None) -> None:
        # Stable sort by priority alone preserves registration order for
        # ties, which is the tie-break the chain relies on.
        self._plugins: List[Plugin] = sorted(
            plugins or [], key=lambda p: getattr(p, "priority", 100)
        )

    def attach(self, agent: Any) -> None:
        """Configure each plugin against ``agent``; called at init time."""
        raise NotImplementedError

    def detach(self) -> None:
        """Release per-plugin resources; called during cleanup."""
        raise NotImplementedError

    def dispatch(
        self,
        event_name: str,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run the plugin chain for a lifecycle event."""
        raise NotImplementedError


__all__ = ["PluginRegistry"]
