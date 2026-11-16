# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Conversation history for the Strands workflow runner.

History is keyed by runner name and session id. It lives in a Dapr state store
when one is available, otherwise in a dict inside the runner process.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Matches the Strands ``SlidingWindowConversationManager`` default.
DEFAULT_WINDOW_SIZE = 40


def history_key(runner_name: str, session_id: str) -> str:
    """Return the state key that holds one session's messages for one runner."""
    return f"strands-{runner_name}-{session_id}-messages"


def resolve_window_size(agent: Any) -> int:
    """Return how many messages to keep: the agent's sliding window, else 40."""
    try:
        from strands.agent.conversation_manager import (
            SlidingWindowConversationManager,
        )
    except ImportError:  # pragma: no cover - strands is required by this package
        return DEFAULT_WINDOW_SIZE

    manager = getattr(agent, "conversation_manager", None)
    if isinstance(manager, SlidingWindowConversationManager):
        return int(manager.window_size)
    return DEFAULT_WINDOW_SIZE


def _is_plain_user_message(message: Any) -> bool:
    """True for a user message that carries no tool result."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content") or []
    return not any(isinstance(b, dict) and "toolResult" in b for b in content)


def trim_history(messages: list[Any], window_size: int) -> list[Any]:
    """Keep the last ``window_size`` messages, starting at a clean user turn.

    After cutting, messages are dropped from the front until the first one is a
    user message without a ``toolResult``. Otherwise a tool result could be left
    without the tool call it answers, and the model provider rejects that.
    """
    kept = list(messages[-window_size:]) if window_size > 0 else []
    start = 0
    while start < len(kept) and not _is_plain_user_message(kept[start]):
        start += 1
    return kept[start:]


class HistoryStore:
    """Loads and saves session messages for one runner.

    Uses ``state_store`` (a ``DaprStateStore``) when given. Without one, history
    is kept in this process and a single warning says it will not survive a
    restart or be shared between replicas.
    """

    def __init__(self, runner_name: str, state_store: Optional[Any] = None) -> None:
        self._runner_name = runner_name
        self._state_store = state_store
        self._local: dict[str, list[Any]] = {}
        self._warned = False

    def _warn_in_process(self) -> None:
        if self._warned:
            return
        self._warned = True
        logger.warning(
            "No state store configured for runner %r: conversation history is "
            "kept in this process only and will not survive a restart or be "
            "shared between replicas. Add a Dapr state component named "
            "'agent-memory' (or pass state_store) for durable history.",
            self._runner_name,
        )

    def load(self, session_id: str) -> list[Any]:
        """Return the stored messages for ``session_id``, or an empty list."""
        key = history_key(self._runner_name, session_id)
        if self._state_store is not None:
            stored = self._state_store.get(key)
            return list(stored) if isinstance(stored, list) else []
        self._warn_in_process()
        return list(self._local.get(key, []))

    def save(self, session_id: str, messages: list[Any]) -> None:
        """Store ``messages`` as the full history for ``session_id``."""
        key = history_key(self._runner_name, session_id)
        if self._state_store is not None:
            self._state_store.save(key, messages)
            return
        self._warn_in_process()
        self._local[key] = list(messages)
