# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

# TODO: Minimal scaffolding for the plugins SPI; replace with the full module.

from enum import StrEnum


class LifecycleEvent(StrEnum):
    """Lifecycle hook points at which plugins may run."""

    BEFORE_AGENT_INVOKE = "before_agent_invoke"
    AFTER_AGENT_INVOKE = "after_agent_invoke"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
