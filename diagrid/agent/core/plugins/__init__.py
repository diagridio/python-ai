# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Plugin SPI for Diagrid agents.

The plugin system lets observability, authentication, OBO, HITL, and
custom plugins intercept agent lifecycle events without modifying
dapr-agents or python-ai's framework runners directly.

Public API:
    - Plugin: Protocol all plugins implement
    - LifecycleEvent: enum of supported event names
    - HookDecision: union of decision types
      (Proceed / Skip / Mutate / RequireApproval / Deny)
    - LifecycleContext: per-event context object passed to plugins
    - PluginRegistry: chain dispatcher implementing dapr-agents'
      LifecycleDispatcher Protocol
"""

from .spi import (
    Plugin,
    LifecycleEvent,
    HookDecision,
    Proceed,
    Skip,
    Mutate,
    RequireApproval,
    Deny,
)
from .context import (
    LifecycleContext,
    CallerIdentity,
    CallTarget,
)
from .registry import PluginRegistry

__all__ = [
    "Plugin",
    "LifecycleEvent",
    "HookDecision",
    "Proceed",
    "Skip",
    "Mutate",
    "RequireApproval",
    "Deny",
    "LifecycleContext",
    "CallerIdentity",
    "CallTarget",
    "PluginRegistry",
]
