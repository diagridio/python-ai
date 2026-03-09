# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Shared utilities for pydantic-ai integration."""

from typing import Any


def get_pydantic_ai_tools(agent: Any) -> dict[str, Any]:
    """Get function tools dict from a pydantic-ai agent, supporting both old and new APIs.

    pydantic-ai v1.61.0+ stores tools in ``_function_toolset.tools``.
    Older versions use ``_function_tools`` directly on the agent.

    Args:
        agent: A pydantic-ai Agent instance.

    Returns:
        Dictionary mapping tool names to tool info objects.
    """
    toolset = getattr(agent, "_function_toolset", None)
    if toolset is not None:
        return getattr(toolset, "tools", {}) or {}
    return getattr(agent, "_function_tools", {}) or {}
