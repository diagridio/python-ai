# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Outbound user-token propagation via contextvar.

The ASGI middleware stores the raw inbound token; outbound MCP / sub-agent
calls read it and set ``X-Diagrid-User-Token`` on the request.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Dict, Optional

USER_TOKEN_HEADER = "X-Diagrid-User-Token"
BEARER_PREFIX = "Bearer "

_current_user_token: ContextVar[Optional[str]] = ContextVar(
    "_current_user_token", default=None
)


def set_current_token(raw_token: str) -> None:
    """Store the raw bearer token for the duration of this request."""
    _current_user_token.set(raw_token)


def clear_current_token() -> None:
    _current_user_token.set(None)


def current_user_token() -> Optional[str]:
    """Return the raw bearer token, or None outside an authenticated request."""
    return _current_user_token.get()


def outbound_identity_headers() -> Dict[str, str]:
    """Headers to attach on outbound MCP / sub-agent calls.

    Returns an empty dict when there is no inbound user context (scheduled,
    pub/sub, cron triggers) so the header is omitted entirely rather than
    sent empty.
    """
    token = _current_user_token.get()
    if not token:
        return {}
    return {USER_TOKEN_HEADER: f"{BEARER_PREFIX}{token}"}
