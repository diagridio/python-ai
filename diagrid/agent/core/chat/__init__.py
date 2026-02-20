# Copyright 2025 Diagrid Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dapr Conversation API chat client and types."""

from .client import DaprChatClient
from .types import (
    ChatMessage,
    ChatResponse,
    ChatRole,
    ChatToolCall,
    ChatToolDefinition,
)

__all__ = [
    "DaprChatClient",
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
