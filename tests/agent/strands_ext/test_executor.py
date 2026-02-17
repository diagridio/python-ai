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

"""Tests for DaprWorkflowToolExecutor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from diagrid.agent.strands.executor import DaprWorkflowToolExecutor


class TestDaprWorkflowToolExecutor:
    """Tests for the DaprWorkflowToolExecutor class."""

    def test_init_defaults(self):
        """Test default initialization."""
        executor = DaprWorkflowToolExecutor()

        assert executor.activity_prefix == "strands_tool_"
        assert executor.concurrent is True
        assert executor.workflow_context_var is None
        assert executor._workflow_context is None

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        executor = DaprWorkflowToolExecutor(
            activity_prefix="custom_",
            concurrent=False,
            workflow_context_var="my_ctx",
        )

        assert executor.activity_prefix == "custom_"
        assert executor.concurrent is False
        assert executor.workflow_context_var == "my_ctx"

    def test_set_workflow_context(self):
        """Test setting workflow context."""
        executor = DaprWorkflowToolExecutor()
        mock_ctx = MagicMock()

        executor.set_workflow_context(mock_ctx)

        assert executor._workflow_context is mock_ctx
        assert executor.is_workflow_mode is True

    def test_clear_workflow_context(self):
        """Test clearing workflow context."""
        executor = DaprWorkflowToolExecutor()
        executor.set_workflow_context(MagicMock())

        executor.clear_workflow_context()

        assert executor._workflow_context is None
        assert executor.is_workflow_mode is False

    def test_is_workflow_mode(self):
        """Test workflow mode detection."""
        executor = DaprWorkflowToolExecutor()

        # Initially not in workflow mode
        assert executor.is_workflow_mode is False

        # After setting context
        executor.set_workflow_context(MagicMock())
        assert executor.is_workflow_mode is True

        # After clearing context
        executor.clear_workflow_context()
        assert executor.is_workflow_mode is False

    def test_get_activity_name(self):
        """Test activity name generation."""
        executor = DaprWorkflowToolExecutor(activity_prefix="test_")

        assert executor._get_activity_name("search") == "test_search"
        assert executor._get_activity_name("calculate") == "test_calculate"
        assert executor._get_activity_name("get_weather") == "test_get_weather"

    def test_get_activity_name_default_prefix(self):
        """Test activity name with default prefix."""
        executor = DaprWorkflowToolExecutor()

        assert executor._get_activity_name("my_tool") == "strands_tool_my_tool"


class TestDaprWorkflowToolExecutorAsync:
    """Async tests for DaprWorkflowToolExecutor."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.tool_registry = MagicMock()
        agent.tool_registry.registry = {}
        agent.tool_registry.dynamic_tools = {}
        agent.tool_registry.get_all_tool_specs = MagicMock(return_value=[])
        agent.messages = []
        agent.system_prompt = "Test prompt"
        agent.model = MagicMock()
        agent.hooks = MagicMock()
        agent.hooks.invoke_callbacks_async = AsyncMock(return_value=(MagicMock(), []))
        agent.event_loop_metrics = MagicMock()
        agent.trace_attributes = {}
        return agent

    @pytest.fixture
    def mock_tool_use(self):
        """Create a mock tool use."""
        return {
            "name": "test_tool",
            "toolUseId": "test-123",
            "input": {"arg": "value"},
        }

    @pytest.mark.asyncio
    async def test_execute_direct_mode(self, mock_agent, mock_tool_use):
        """Test execution in direct mode (no workflow context)."""
        executor = DaprWorkflowToolExecutor()

        # Patch _execute_direct to verify it's called in direct mode
        with patch.object(
            DaprWorkflowToolExecutor,
            "_execute_direct",
        ) as mock_execute_direct:
            mock_execute_direct.return_value = self._async_generator([])

            tool_results = []
            events = []
            async for event in executor._execute(
                agent=mock_agent,
                tool_uses=[mock_tool_use],
                tool_results=tool_results,
                cycle_trace=MagicMock(),
                cycle_span=MagicMock(),
                invocation_state={},
            ):
                events.append(event)

            mock_execute_direct.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_workflow_mode_activity_call(self, mock_agent, mock_tool_use):
        """Test that workflow mode calls activities."""
        executor = DaprWorkflowToolExecutor()

        # Set up mock workflow context
        mock_ctx = MagicMock()
        mock_ctx.call_activity = AsyncMock(
            return_value={"status": "success", "content": [{"text": "result"}]}
        )
        executor.set_workflow_context(mock_ctx)

        tool_results = []
        events = []
        async for event in executor._execute(
            agent=mock_agent,
            tool_uses=[mock_tool_use],
            tool_results=tool_results,
            cycle_trace=MagicMock(),
            cycle_span=MagicMock(),
            invocation_state={},
        ):
            events.append(event)

        # Verify activity was called
        mock_ctx.call_activity.assert_called_once()
        call_args = mock_ctx.call_activity.call_args

        # Check activity name
        assert call_args[0][0] == "strands_tool_test_tool"

        # Check activity input
        activity_input = call_args[1]["input"]
        assert activity_input["tool_name"] == "test_tool"
        assert activity_input["tool_use"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_execute_workflow_mode_activity_error(
        self, mock_agent, mock_tool_use
    ):
        """Test error handling when activity fails."""
        executor = DaprWorkflowToolExecutor()

        # Set up mock workflow context that raises an error
        mock_ctx = MagicMock()
        mock_ctx.call_activity = AsyncMock(side_effect=Exception("Activity failed"))
        executor.set_workflow_context(mock_ctx)

        tool_results = []
        events = []
        async for event in executor._execute(
            agent=mock_agent,
            tool_uses=[mock_tool_use],
            tool_results=tool_results,
            cycle_trace=MagicMock(),
            cycle_span=MagicMock(),
            invocation_state={},
        ):
            events.append(event)

        # Should have one result event
        assert len(events) == 1

        # Should have error status
        assert len(tool_results) == 1
        assert tool_results[0]["status"] == "error"
        assert "Activity failed" in tool_results[0]["content"][0]["text"]

    @staticmethod
    async def _async_generator(items):
        """Create an async generator from a list."""
        for item in items:
            yield item


class TestExecutorIntegration:
    """Integration tests for the executor."""

    @pytest.mark.asyncio
    async def test_concurrent_mode_multiple_tools(self):
        """Test concurrent execution of multiple tools."""
        executor = DaprWorkflowToolExecutor(concurrent=True)

        # Create mock context with activity tracking
        activity_calls = []

        async def mock_call_activity(name, input):
            activity_calls.append(name)
            await asyncio.sleep(0.01)  # Simulate work
            return {
                "status": "success",
                "content": [{"text": f"result from {name}"}],
            }

        mock_ctx = MagicMock()
        mock_ctx.call_activity = mock_call_activity
        executor.set_workflow_context(mock_ctx)

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.tool_registry = MagicMock()
        mock_agent.tool_registry.registry = {}
        mock_agent.tool_registry.dynamic_tools = {}

        # Multiple tool uses
        tool_uses = [
            {"name": "tool1", "toolUseId": "1", "input": {}},
            {"name": "tool2", "toolUseId": "2", "input": {}},
            {"name": "tool3", "toolUseId": "3", "input": {}},
        ]

        tool_results = []
        async for _ in executor._execute(
            agent=mock_agent,
            tool_uses=tool_uses,
            tool_results=tool_results,
            cycle_trace=MagicMock(),
            cycle_span=MagicMock(),
            invocation_state={},
        ):
            pass

        # All tools should have been called
        assert len(activity_calls) == 3
        assert "strands_tool_tool1" in activity_calls
        assert "strands_tool_tool2" in activity_calls
        assert "strands_tool_tool3" in activity_calls

    @pytest.mark.asyncio
    async def test_sequential_mode_multiple_tools(self):
        """Test sequential execution of multiple tools."""
        executor = DaprWorkflowToolExecutor(concurrent=False)

        # Track call order
        call_order = []

        async def mock_call_activity(name, input):
            call_order.append(name)
            return {"status": "success", "content": [{"text": "ok"}]}

        mock_ctx = MagicMock()
        mock_ctx.call_activity = mock_call_activity
        executor.set_workflow_context(mock_ctx)

        mock_agent = MagicMock()
        mock_agent.tool_registry = MagicMock()
        mock_agent.tool_registry.registry = {}
        mock_agent.tool_registry.dynamic_tools = {}

        tool_uses = [
            {"name": "first", "toolUseId": "1", "input": {}},
            {"name": "second", "toolUseId": "2", "input": {}},
        ]

        tool_results = []
        async for _ in executor._execute(
            agent=mock_agent,
            tool_uses=tool_uses,
            tool_results=tool_results,
            cycle_trace=MagicMock(),
            cycle_span=MagicMock(),
            invocation_state={},
        ):
            pass

        # Verify sequential order
        assert call_order == ["strands_tool_first", "strands_tool_second"]
