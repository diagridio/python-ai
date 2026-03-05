# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr Conversation API chat client and types."""

from .client import DaprChatClient, close_chat_clients, get_chat_client
from .types import (
    ChatMessage,
    ChatResponse,
    ChatRole,
    ChatToolCall,
    ChatToolDefinition,
)

__all__ = [
    "DaprChatClient",
    "close_chat_clients",
    "get_chat_client",
    "ChatMessage",
    "ChatResponse",
    "ChatRole",
    "ChatToolCall",
    "ChatToolDefinition",
]

# Conditionally export DaprChatModel if langchain-core is available
try:
    from .langchain_model import DaprChatModel

    __all__.append("DaprChatModel")
except ImportError:
    pass
