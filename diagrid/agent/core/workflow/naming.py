# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Agent name sanitization for workflow ID construction.

Mirrors the TitleCase normalization used by dapr-agents so that workflow
IDs match across frameworks (e.g. ``dapr.openai.CateringCoordinator.workflow``).
"""

import re


def _normalize_to_title_case(name: str) -> str:
    """Normalize a name to TitleCase format.

    Converts snake_case, kebab-case, and space-separated names to TitleCase.

    Examples::

        "get_user"        -> "GetUser"
        "get-user"        -> "GetUser"
        "get user"        -> "GetUser"
        "GetUser"         -> "GetUser"  (preserved)
        "GET_USER"        -> "GetUser"
        "UPPERCASE"       -> "Uppercase"
        "SamwiseGamgee"   -> "SamwiseGamgee"  (preserved)
    """
    if not name:
        return ""

    # All-uppercase → capitalize only first letter
    if name.isupper() and len(name) > 1:
        return name.capitalize()

    # Already TitleCase (no separators, starts with upper then lower)
    if re.match(r"^[A-Z][a-z]", name) and not re.search(r"[_\s-]", name):
        return name

    # Split on separators, capitalize each part, join without separators
    parts = re.split(r"[_\s-]+", name)
    return "".join(part.capitalize() for part in parts if part)


def sanitize_agent_name(name: str) -> str:
    """Sanitize an agent name for use in workflow IDs.

    Converts to TitleCase and removes characters that are invalid in
    OpenAI tool names (spaces, ``<``, ``|``, ``\\``, ``/``, ``>``).

    This matches ``sanitize_openai_tool_name`` in dapr-agents so that
    workflow IDs are consistent across frameworks.

    Examples::

        "catering-coordinator" -> "CateringCoordinator"
        "Samwise Gamgee"       -> "SamwiseGamgee"
        "agent<name>"          -> "Agentname"
        ""                     -> "unnamed_agent"
    """
    if not name:
        return "unnamed_agent"

    sanitized = _normalize_to_title_case(name)
    if not sanitized:
        return "unnamed_agent"

    # Remove invalid characters
    sanitized = re.sub(r"[<|\\/>]", "", sanitized)

    return sanitized or "unnamed_agent"
