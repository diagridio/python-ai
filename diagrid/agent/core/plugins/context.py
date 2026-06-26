# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

# TODO(AI-598): Minimal scaffolding for the plugins lifecycle context. Replace
# with the full module from the AI-598 PR once it lands. The caller_headers
# field is sourced from TriggerAction (AI-592); see the LifecycleContext test
# TODO in tests/agent/core/plugins/test_oauth_plugin.py.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional

from .spi import LifecycleEvent


@dataclass
class CallerIdentity:
    """Verified identity of the caller that triggered an agent invocation."""

    subject: str
    tenant: Optional[str] = None
    scopes: FrozenSet[str] = field(default_factory=frozenset)
    issuer_id: Optional[str] = None
    claims: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecycleContext:
    """
    Mutable context threaded through the plugin chain at each lifecycle event.

    Plugins read the inbound state (event, caller_headers) and may populate
    derived fields such as caller before the agent runs.
    """

    event: LifecycleEvent
    caller_headers: Dict[str, str] = field(default_factory=dict)
    caller: Optional[CallerIdentity] = None
