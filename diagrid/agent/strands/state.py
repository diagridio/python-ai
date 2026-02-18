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

"""Dapr State Session Manager - Persist Strands agent state to Dapr state stores.

This module provides a SessionManager implementation that uses Dapr state stores
for persisting agent conversation history and state.
"""

import json
import logging
from typing import Any, Optional, TYPE_CHECKING

from diagrid.agent.core import AgentRegistryMixin, find_agent_in_stack
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import (
    AfterInvocationEvent,
    BeforeInvocationEvent,
)
from strands.types.content import Message

if TYPE_CHECKING:
    from strands.agent import Agent

logger = logging.getLogger(__name__)


class DaprStateSessionManager(HookProvider, AgentRegistryMixin):
    """Session manager that persists agent state to Dapr state stores.

    This class implements the HookProvider protocol to automatically
    save and load agent state using Dapr's state management APIs.

    Features:
    - Automatic conversation history persistence
    - Custom state storage (agent.state)
    - Configurable state store name
    - Support for different consistency levels

    Example:
        ```python
        from diagrid.agent.strands import DaprStateSessionManager
        from strands import Agent

        # Create session manager
        session_manager = DaprStateSessionManager(
            store_name="agent-workflow",
            session_id="user-123-conversation",
        )

        # Create agent with session manager
        agent = Agent(
            model="us.amazon.nova-pro-v1:0",
            tools=[my_tool],
            hooks=[session_manager],
        )

        # Agent state will now persist across invocations
        result = agent("Hello!")
        ```

    Args:
        store_name: The Dapr state store name (default: "statestore")
        session_id: Unique identifier for the session
        auto_save: Whether to automatically save after each invocation (default: True)
        auto_load: Whether to automatically load on first invocation (default: True)
        registry_config: Optional registry configuration for metadata extraction
    """

    def __init__(
        self,
        store_name: str = "statestore",
        session_id: str | None = None,
        auto_save: bool = True,
        auto_load: bool = True,
        consistency: str = "strong",
        registry_config: Optional[Any] = None,
    ) -> None:
        """Initialize the Dapr state session manager.

        Args:
            store_name: The Dapr state store name
            session_id: Unique session identifier (required for persistence)
            auto_save: Automatically save after invocations
            auto_load: Automatically load on first invocation
            consistency: State consistency level ("strong" or "eventual")
            registry_config: Optional registry configuration for metadata extraction
        """
        self.store_name = store_name
        self.session_id = session_id
        self.auto_save = auto_save
        self.auto_load = auto_load
        self.consistency = consistency
        self.registry_config = registry_config

        self._dapr_client: Any = None
        self._loaded = False

    def _get_client(self) -> Any:
        """Get or create the Dapr client.

        Returns:
            The DaprClient instance

        Raises:
            ImportError: If dapr package is not installed
        """
        if self._dapr_client is None:
            try:
                from dapr.clients import DaprClient

                self._dapr_client = DaprClient()
            except ImportError:
                raise ImportError(
                    "dapr package is required. Install with: pip install dapr"
                )

        return self._dapr_client

    def _get_state_key(self, key_type: str) -> str:
        """Generate the state store key.

        Args:
            key_type: The type of state ("messages", "state", etc.)

        Returns:
            The full state store key
        """
        if not self.session_id:
            raise ValueError("session_id is required for state persistence")

        return f"strands-session-{self.session_id}-{key_type}"

    async def save_state(self, agent: "Agent") -> None:
        """Save agent state to Dapr state store.

        Args:
            agent: The Strands agent to save state from
        """
        if not self.session_id:
            logger.debug("No session_id - skipping state save")
            return

        client = self._get_client()

        # Serialize messages
        messages_key = self._get_state_key("messages")
        messages_data = json.dumps([self._serialize_message(m) for m in agent.messages])

        # Serialize agent state
        state_key = self._get_state_key("state")
        state_data = json.dumps(
            {k: v for k, v in agent.state.items()} if agent.state else {}  # type: ignore[attr-defined]
        )

        # Save to Dapr state store
        try:
            client.save_state(
                store_name=self.store_name,
                key=messages_key,
                value=messages_data,
            )

            client.save_state(
                store_name=self.store_name,
                key=state_key,
                value=state_data,
            )

            logger.debug(
                "session=%s | saved agent state (messages=%d)",
                self.session_id,
                len(agent.messages),
            )

        except Exception as e:
            logger.error(
                "session=%s | failed to save state: %s",
                self.session_id,
                str(e),
            )
            raise

    async def load_state(self, agent: "Agent") -> bool:
        """Load agent state from Dapr state store.

        Args:
            agent: The Strands agent to load state into

        Returns:
            True if state was loaded, False if no state exists
        """
        if not self.session_id:
            logger.debug("No session_id - skipping state load")
            return False

        client = self._get_client()

        try:
            # Load messages
            messages_key = self._get_state_key("messages")
            messages_response = client.get_state(
                store_name=self.store_name,
                key=messages_key,
            )

            if messages_response.data:
                messages_data = json.loads(messages_response.data.decode())
                agent.messages.clear()
                agent.messages.extend(
                    [self._deserialize_message(m) for m in messages_data]
                )

                logger.debug(
                    "session=%s | loaded %d messages",
                    self.session_id,
                    len(agent.messages),
                )

            # Load agent state
            state_key = self._get_state_key("state")
            state_response = client.get_state(
                store_name=self.store_name,
                key=state_key,
            )

            if state_response.data:
                state_data = json.loads(state_response.data.decode())
                for k, v in state_data.items():
                    agent.state[k] = v  # type: ignore[index]

                logger.debug(
                    "session=%s | loaded agent state",
                    self.session_id,
                )

            return bool(messages_response.data or state_response.data)

        except Exception as e:
            logger.error(
                "session=%s | failed to load state: %s",
                self.session_id,
                str(e),
            )
            return False

    async def delete_state(self) -> None:
        """Delete all state for this session."""
        if not self.session_id:
            return

        client = self._get_client()

        try:
            client.delete_state(
                store_name=self.store_name,
                key=self._get_state_key("messages"),
            )

            client.delete_state(
                store_name=self.store_name,
                key=self._get_state_key("state"),
            )

            logger.info("session=%s | deleted session state", self.session_id)

        except Exception as e:
            logger.error(
                "session=%s | failed to delete state: %s",
                self.session_id,
                str(e),
            )

    def _serialize_message(self, message: Message) -> dict:  # type: ignore[type-arg]
        """Serialize a message for storage.

        Args:
            message: The message to serialize

        Returns:
            Serialized message dict
        """
        # Messages are already dicts/TypedDicts, but may contain non-JSON types
        return dict(message)

    def _deserialize_message(self, data: dict) -> Message:  # type: ignore[type-arg]
        """Deserialize a message from storage.

        Args:
            data: The serialized message data

        Returns:
            Deserialized Message
        """
        return Message(role=data.get("role", "user"), content=data.get("content", []))

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks with the agent's hook registry.

        This implements the HookProvider protocol.

        Args:
            registry: The agent's hook registry
            **kwargs: Additional arguments
        """
        # Register metadata if agent is available in kwargs or can be found in stack
        agent = kwargs.get("agent")
        if agent:
            self._register_agent_metadata(
                agent=agent, framework="strands", registry=self.registry_config
            )
        else:
            # Try to find agent in stack
            stack_agent = find_agent_in_stack()
            if stack_agent:
                self._register_agent_metadata(
                    agent=stack_agent,
                    framework="strands",
                    registry=self.registry_config,
                )

        if self.auto_load:
            registry.add_callback(BeforeInvocationEvent, self._on_before_invocation)

        if self.auto_save:
            registry.add_callback(
                AfterInvocationEvent,
                self._on_after_invocation,
            )

    async def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Hook handler for before invocation - loads state.

        Args:
            event: The BeforeInvocationEvent
        """
        if not self._loaded and self.auto_load:
            await self.load_state(event.agent)
            self._loaded = True

    async def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Hook handler for after invocation - saves state.

        Args:
            event: The AfterInvocationEvent
        """
        if self.auto_save:
            await self.save_state(event.agent)


class DaprWorkflowStateManager:
    """State manager for use within Dapr workflows.

    This class provides state management specifically designed for use
    within Dapr workflow activities, using the workflow's built-in
    state management rather than the state store directly.

    Example:
        ```python
        @workflow_runtime.activity(name="stateful_tool")
        async def stateful_tool(ctx, input_data):
            state_manager = DaprWorkflowStateManager(ctx)

            # Get previous state
            counter = await state_manager.get("counter", default=0)

            # Update state
            counter += 1
            await state_manager.set("counter", counter)

            return {"count": counter}
        ```
    """

    def __init__(self, workflow_context: Any) -> None:
        """Initialize with workflow context.

        Args:
            workflow_context: The Dapr workflow context
        """
        self._ctx = workflow_context
        self._state: dict[str, Any] = {}

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a state value.

        Args:
            key: The state key
            default: Default value if key doesn't exist

        Returns:
            The state value or default
        """
        return self._state.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        """Set a state value.

        Args:
            key: The state key
            value: The value to set
        """
        self._state[key] = value

    async def delete(self, key: str) -> None:
        """Delete a state value.

        Args:
            key: The state key to delete
        """
        self._state.pop(key, None)

    @property
    def data(self) -> dict[str, Any]:
        """Get all state data."""
        return dict(self._state)
