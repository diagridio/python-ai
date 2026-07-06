# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

# TODO: Minimal scaffolding for the plugins lifecycle context; replace with the
# full module.

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
    trigger_action: Optional[Any] = None
    caller: Optional[CallerIdentity] = None
