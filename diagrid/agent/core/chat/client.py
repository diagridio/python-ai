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

"""DaprChatClient - Routes LLM calls through the Dapr Conversation API."""

import logging
from typing import Any, Optional

from dapr.clients import DaprClient
from dapr.clients.grpc.conversation import (
    ConversationInputAlpha2,
    ConversationMessage,
    ConversationMessageContent,
    ConversationMessageOfAssistant,
    ConversationMessageOfSystem,
    ConversationMessageOfTool,
    ConversationMessageOfUser,
    ConversationToolCalls,
    ConversationToolCallsOfFunction,
    ConversationTools,
    ConversationToolsFunction,
)

from .types import (
    ChatMessage,
    ChatResponse,
    ChatRole,
    ChatToolCall,
    ChatToolDefinition,
)

logger = logging.getLogger(__name__)

_DEFAULT_COMPONENT_NAME = "llm-provider"


class DaprChatClient:
    """Client that routes LLM calls through the Dapr Conversation API.

    Holds a persistent DaprClient connection. Component name is resolved
    lazily on first use via Dapr metadata discovery.

    Args:
        component_name: Explicit Dapr conversation component name.
            If None, auto-resolved from sidecar metadata.
        temperature: Default temperature for LLM calls.
        metadata: Default metadata dict passed to every converse call.
    """

    def __init__(
        self,
        component_name: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        metadata: Optional[dict[str, str]] = None,
    ):
        self._dapr_client = DaprClient()
        self._explicit_component_name = component_name
        self._resolved_component_name: Optional[str] = None
        self._temperature = temperature
        self._metadata = metadata

    @property
    def component_name(self) -> str:
        """The resolved conversation component name."""
        if self._explicit_component_name:
            return self._explicit_component_name
        if self._resolved_component_name is None:
            self._resolved_component_name = self._resolve_component()
        return self._resolved_component_name

    def _resolve_component(self) -> str:
        """Auto-discover a conversation component from Dapr sidecar metadata.

        Priority:
        1. If "llm-provider" exists among conversation components, use it.
        2. If exactly one conversation component exists, use that.
        3. Fall back to "llm-provider" (Helm chart default).
        """
        try:
            meta = self._dapr_client.get_metadata()
            conversation_components = [
                c
                for c in meta.registered_components
                if c.type.startswith("conversation.")
            ]

            # Check for the well-known "llm-provider" component
            for c in conversation_components:
                if c.name == _DEFAULT_COMPONENT_NAME:
                    logger.info(
                        "Auto-detected conversation component: %s",
                        _DEFAULT_COMPONENT_NAME,
                    )
                    return _DEFAULT_COMPONENT_NAME

            # Single conversation component available
            if len(conversation_components) == 1:
                name = conversation_components[0].name
                logger.info("Auto-detected single conversation component: %s", name)
                return name

        except Exception as e:
            logger.debug("Failed to auto-detect conversation component: %s", e)

        logger.info("Using default conversation component: %s", _DEFAULT_COMPONENT_NAME)
        return _DEFAULT_COMPONENT_NAME

    def chat(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[ChatToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        context_id: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """Send a chat completion request through the Dapr Conversation API.

        Args:
            messages: Conversation messages.
            tools: Tool definitions available to the LLM.
            tool_choice: Tool choice strategy ('none', 'auto', 'required').
            context_id: Conversation context ID for multi-turn chats.
            temperature: Temperature override for this call.

        Returns:
            ChatResponse with the LLM's reply.
        """
        # Convert ChatMessage -> Dapr ConversationMessage
        dapr_messages = [self._to_dapr_message(msg) for msg in messages]

        # Wrap in ConversationInputAlpha2
        dapr_inputs = [ConversationInputAlpha2(messages=dapr_messages)]

        # Convert ChatToolDefinition -> ConversationTools
        dapr_tools: Optional[list[ConversationTools]] = None
        if tools:
            dapr_tools = [self._to_dapr_tool(t) for t in tools]

        temp = temperature if temperature is not None else self._temperature

        response = self._dapr_client.converse_alpha2(
            name=self.component_name,
            inputs=dapr_inputs,
            context_id=context_id,
            metadata=self._metadata,
            temperature=temp,
            tools=dapr_tools,
            tool_choice=tool_choice,
        )

        return self._parse_response(response)

    def close(self) -> None:
        """Close the persistent DaprClient connection."""
        self._dapr_client.close()

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dapr_message(msg: ChatMessage) -> ConversationMessage:
        """Convert a ChatMessage to a Dapr ConversationMessage."""
        if msg.role == ChatRole.SYSTEM:
            return ConversationMessage(
                of_system=ConversationMessageOfSystem(
                    content=[ConversationMessageContent(text=msg.content or "")]
                )
            )
        elif msg.role == ChatRole.USER:
            return ConversationMessage(
                of_user=ConversationMessageOfUser(
                    content=[ConversationMessageContent(text=msg.content or "")]
                )
            )
        elif msg.role == ChatRole.ASSISTANT:
            tool_calls = [
                ConversationToolCalls(
                    id=tc.id,
                    function=ConversationToolCallsOfFunction(
                        name=tc.name,
                        arguments=tc.arguments,
                    ),
                )
                for tc in msg.tool_calls
            ]
            content = (
                [ConversationMessageContent(text=msg.content)] if msg.content else []
            )
            return ConversationMessage(
                of_assistant=ConversationMessageOfAssistant(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        elif msg.role == ChatRole.TOOL:
            return ConversationMessage(
                of_tool=ConversationMessageOfTool(
                    tool_id=msg.tool_call_id,
                    name=msg.name or "",
                    content=[ConversationMessageContent(text=msg.content or "")],
                )
            )
        else:
            raise ValueError(f"Unknown ChatRole: {msg.role}")

    @staticmethod
    def _to_dapr_tool(tool_def: ChatToolDefinition) -> ConversationTools:
        """Convert a ChatToolDefinition to a Dapr ConversationTools."""
        return ConversationTools(
            function=ConversationToolsFunction(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters,
            )
        )

    @staticmethod
    def _parse_response(response: Any) -> ChatResponse:
        """Parse a Dapr ConversationResponseAlpha2 into a ChatResponse."""
        context_id = response.context_id

        if not response.outputs:
            return ChatResponse(context_id=context_id)

        # Take the first output's first choice
        first_output = response.outputs[0]
        if not first_output.choices:
            return ChatResponse(context_id=context_id)

        choice = first_output.choices[0]
        message = choice.message

        # Extract tool calls
        tool_calls = []
        for tc in message.tool_calls:
            if tc.function:
                tool_calls.append(
                    ChatToolCall(
                        id=tc.id or "",
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

        return ChatResponse(
            content=message.content if message.content else None,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            context_id=context_id,
        )


_chat_client_cache: dict[str, DaprChatClient] = {}


def get_chat_client(component_name: str | None = None) -> DaprChatClient:
    """Get or create a persistent DaprChatClient for the given component.

    Reuses existing clients to avoid gRPC channel churn that conflicts
    with OTEL instrumentation.
    """
    key = component_name or ""
    if key not in _chat_client_cache:
        _chat_client_cache[key] = DaprChatClient(component_name=component_name)
    return _chat_client_cache[key]


def close_chat_clients() -> None:
    """Close all cached chat clients. Called during shutdown."""
    for client in _chat_client_cache.values():
        client.close()
    _chat_client_cache.clear()
