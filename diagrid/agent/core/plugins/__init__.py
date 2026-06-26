# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Lifecycle plugin chain for agents."""

from .context import CallerIdentity, CallTarget, LifecycleContext
from .registry import PluginRegistry, dispatch_plugin_event
from .spi import (
    FAILURE_MODE_CLOSED,
    FAILURE_MODE_OPEN,
    FailureMode,
    LifecycleEvent,
    Plugin,
)

__all__ = [
    "FAILURE_MODE_CLOSED",
    "FAILURE_MODE_OPEN",
    "FailureMode",
    "CallerIdentity",
    "CallTarget",
    "LifecycleContext",
    "LifecycleEvent",
    "Plugin",
    "PluginRegistry",
    "dispatch_plugin_event",
]
