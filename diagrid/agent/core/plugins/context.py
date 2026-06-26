# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Context objects passed to plugins on each lifecycle event.

``LifecycleContext`` carries the dispatched event plus everything a plugin
needs to make a decision: who is calling (``CallerIdentity``), what is
being called (``CallTarget``), and the step payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .spi import LifecycleEvent


@dataclass
class CallerIdentity:
    """The authenticated principal on whose behalf a step runs.

    ``claims`` holds the verified token claims; ``token`` is the raw
    credential when a plugin needs to forward it.
    Neither is logged by the registry.
    """

    subject: Optional[str] = None
    scopes: Tuple[str, ...] = ()
    claims: Dict[str, Any] = field(default_factory=dict)
    token: Optional[str] = None


@dataclass
class CallTarget:
    """The step a lifecycle event is about to run.

    ``kind`` is ``"tool"``, ``"llm"``, or ``"agent"``; ``source`` records
    where a tool came from, such as ``"local"``, ``"mcp"``, or
    ``"openapi"``.
    """

    name: Optional[str] = None
    kind: Optional[str] = None
    source: Optional[str] = None


@dataclass
class LifecycleContext:
    """Everything a plugin sees for a single dispatched event.

    ``payload`` is the mutable step input plugins may rewrite via a
    ``Mutate`` decision.
    ``extra`` collects any context keys that do not map onto a named field,
    keeping the context forward-compatible with new dispatch sites.
    """

    event: LifecycleEvent
    caller: Optional[CallerIdentity] = None
    target: Optional[CallTarget] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    agent: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
