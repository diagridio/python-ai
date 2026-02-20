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

"""DaprChatModel - LangChain BaseChatModel backed by the Dapr Conversation API.

Use this with LangGraph nodes in place of any LangChain chat model:

    from diagrid.agent.core.chat import DaprChatModel
    model = DaprChatModel(component_name="llm-provider")
"""

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall as LCToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from .client import DaprChatClient
from .types import ChatMessage, ChatRole, ChatToolCall, ChatToolDefinition

logger = logging.getLogger(__name__)


class DaprChatModel(BaseChatModel):
    """LangChain chat model that routes calls through the Dapr Conversation API.

    Example:
        ```python
        from diagrid.agent.core.chat import DaprChatModel

        model = DaprChatModel(component_name="llm-provider")
        result = model.invoke([HumanMessage("Hello!")])
        ```
    """

    component_name: Optional[str] = None
    temperature: Optional[float] = None
    metadata: Optional[dict[str, str]] = None
    _client: Optional[DaprChatClient] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "dapr-chat"

    @property
    def client(self) -> DaprChatClient:
        if self._client is None:
            self._client = DaprChatClient(
                component_name=self.component_name,
                temperature=self.temperature,
                metadata=self.metadata,
            )
        return self._client

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Convert LangChain messages to ChatMessage
        chat_messages = [self._to_chat_message(m) for m in messages]

        # Extract tool definitions from kwargs (set by bind_tools)
        tools: Optional[list[ChatToolDefinition]] = None
        if "tools" in kwargs:
            tools = [self._to_chat_tool_def(t) for t in kwargs["tools"]]

        tool_choice = kwargs.get("tool_choice")

        response = self.client.chat(
            messages=chat_messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        # Build AIMessage from response
        ai_msg = self._to_ai_message(response)

        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind tools to this model (LangChain convention)."""
        bind_kwargs: dict[str, Any] = {"tools": list(tools)}
        if tool_choice is not None:
            bind_kwargs["tool_choice"] = tool_choice
        return self.bind(**bind_kwargs, **kwargs)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_chat_message(msg: BaseMessage) -> ChatMessage:
        """Convert a LangChain BaseMessage to a ChatMessage."""
        # LangChain content can be str | list[str|dict]; coerce to str
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if isinstance(msg, SystemMessage):
            return ChatMessage(role=ChatRole.SYSTEM, content=content)
        elif isinstance(msg, HumanMessage):
            return ChatMessage(role=ChatRole.USER, content=content)
        elif isinstance(msg, AIMessage):
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append(
                    ChatToolCall(
                        id=tc["id"] or "",
                        name=tc["name"],
                        arguments=json.dumps(tc["args"]),
                    )
                )
            return ChatMessage(
                role=ChatRole.ASSISTANT,
                content=content if content else None,
                tool_calls=tool_calls,
            )
        elif isinstance(msg, ToolMessage):
            return ChatMessage(
                role=ChatRole.TOOL,
                content=content,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )
        else:
            return ChatMessage(role=ChatRole.USER, content=content)

    @staticmethod
    def _to_chat_tool_def(tool: Any) -> ChatToolDefinition:
        """Convert a LangChain tool schema to a ChatToolDefinition."""
        if isinstance(tool, dict):
            func = tool.get("function", tool)
            return ChatToolDefinition(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=func.get("parameters"),
            )
        # LangChain BaseTool
        name = getattr(tool, "name", "")
        description = getattr(tool, "description", "")
        parameters = None
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                parameters = tool.args_schema.model_json_schema()
            except Exception:
                pass
        return ChatToolDefinition(
            name=name, description=description, parameters=parameters
        )

    @staticmethod
    def _to_ai_message(response: Any) -> AIMessage:
        """Convert a ChatResponse to a LangChain AIMessage."""
        tool_calls = []
        for tc in response.tool_calls:
            try:
                args = json.loads(tc.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(LCToolCall(id=tc.id, name=tc.name, args=args))

        return AIMessage(
            content=response.content or "",
            tool_calls=tool_calls,
        )
