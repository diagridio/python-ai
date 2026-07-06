# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Lifecycle context types passed into ``Plugin.on_event``.

A ``LifecycleContext`` is the input to every plugin invocation.
It carries the event name and the event-specific payload — caller
identity, target, headers, and metadata.
Plugins read what they need, optionally leave metadata for downstream
plugins, and return a ``HookDecision``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from diagrid.agent.core.plugins.spi import LifecycleEvent


@dataclass
class CallerIdentity:
    """Verified caller identity for the current request.

    Populated by the OAuth plugin on ``BEFORE_AGENT_INVOKE`` once the
    sidecar ``/auth/verify`` call returns.
    Downstream plugins read it through ``ctx.caller`` on later events in
    the same request lifecycle.

    Attributes:
        subject: Verified ``sub`` claim, typically an email or user ID.
        tenant: Tenant claim (e.g. ``org_id`` or ``tid``) per the IdP mapping.
        scopes: OAuth scopes carried by the caller's JWT.
        claims: Full verified claims, for plugins that need richer access.
        issuer_id: Identifier of the IdP that issued the JWT.
        is_agent: True when ``subject`` looks like an agent SPIFFE ID rather
            than a human user, letting plugins tell sub-agent calls apart
            from user calls.
    """

    subject: str
    tenant: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    claims: Dict[str, Any] = field(default_factory=dict)
    issuer_id: str = ""
    is_agent: bool = False


@dataclass
class CallTarget:
    """Target of an outbound call.

    Set for ``BEFORE_TOOL_CALL``, ``BEFORE_MCP_CALL``, and sub-agent
    dispatch.

    Attributes:
        kind: One of ``"tool"``, ``"mcp"``, ``"subagent"``, or ``"llm"``.
        name: Logical name, such as an MCP server or tool name.
        source: Origin of the tool definition — ``"local"``, ``"mcp"``,
            or ``"openapi"``.
        audience: Target audience for token-bound auth, when applicable.
        resource_indicators: RFC 8707 resource indicators.
    """

    kind: str
    name: str
    source: str = "local"
    audience: Optional[str] = None
    resource_indicators: frozenset[str] = field(default_factory=frozenset)


@dataclass
class LifecycleContext:
    """Per-event context passed into ``Plugin.on_event``.

    Attributes:
        event: Which lifecycle event this is.
        workflow_instance_id: Workflow instance ID, used for binding and audit.
        agent_identity: The agent's own identity (SPIFFE ID, AppID, name).
        caller: Verified caller identity, populated from
            ``BEFORE_AGENT_INVOKE`` onward.
        target: Outbound call target, populated for tool and MCP events.
        caller_headers: Headers from the inbound request; upstream plugins
            may scrub sensitive values.
        request_id: Request correlation ID.
        trace_id: W3C trace ID for correlation.
        span_id: W3C span ID for correlation.
        metadata: Plugin-to-plugin scratchpad, read and write.
        payload: Event-specific payload such as LLM messages or tool args.
    """

    event: LifecycleEvent
    workflow_instance_id: str
    agent_identity: Dict[str, str]
    caller: Optional[CallerIdentity] = None
    target: Optional[CallTarget] = None
    caller_headers: Dict[str, str] = field(default_factory=dict)
    request_id: str = ""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "LifecycleContext",
    "CallerIdentity",
    "CallTarget",
]
