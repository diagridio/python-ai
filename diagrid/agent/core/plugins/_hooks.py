# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

# TODO: Local fallback for the hook-decision types until dapr-agents ships its
# `dapr_agents.hooks` module.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class HookDecision:
    """Base type for a plugin's decision at a lifecycle hook point."""


@dataclass
class Proceed(HookDecision):
    """Allow the lifecycle to continue to the next plugin / the agent."""


@dataclass
class Deny(HookDecision):
    """Halt the lifecycle and reject the request."""

    code: str
    details: Optional[Dict[str, Any]] = field(default=None)
