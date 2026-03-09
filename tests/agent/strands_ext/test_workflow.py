# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for DaprAgentWorkflow."""

from unittest.mock import MagicMock

import pytest

from diagrid.agent.strands.workflow import (
    DaprAgentWorkflow,
    WorkflowInput,
    WorkflowOutput,
    dapr_agent_workflow,
)


class TestWorkflowInput:
    """Tests for WorkflowInput dataclass."""

    def test_defaults(self):
        """Test default values."""
        input_data = WorkflowInput(task="Hello")

        assert input_data.task == "Hello"
        assert input_data.conversation_id is None
        assert input_data.metadata == {}

    def test_full_init(self):
        """Test full initialization."""
        input_data = WorkflowInput(
            task="Test prompt",
            conversation_id="conv-123",
            metadata={"key": "value"},
        )

        assert input_data.task == "Test prompt"
        assert input_data.conversation_id == "conv-123"
        assert input_data.metadata == {"key": "value"}


class TestWorkflowOutput:
    """Tests for WorkflowOutput dataclass."""

    def test_defaults(self):
        """Test default values."""
        output = WorkflowOutput(result="Hello")

        assert output.result == "Hello"
        assert output.tool_calls == []
        assert output.conversation_id is None
        assert output.metadata == {}

    def test_full_init(self):
        """Test full initialization."""
        output = WorkflowOutput(
            result="Response",
            tool_calls=[{"tool_name": "search"}],
            conversation_id="conv-123",
            metadata={"duration": 1.5},
        )

        assert output.result == "Response"
        assert len(output.tool_calls) == 1
        assert output.conversation_id == "conv-123"
        assert output.metadata == {"duration": 1.5}


class TestDaprAgentWorkflow:
    """Tests for DaprAgentWorkflow class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock Strands agent."""
        agent = MagicMock()
        agent.tool_registry = MagicMock()
        agent.tool_registry.registry = {"tool1": MagicMock(), "tool2": MagicMock()}
        agent.tool_executor = MagicMock()
        agent.messages = []
        return agent

    def test_init_defaults(self, mock_agent):
        """Test initialization with name."""
        workflow = DaprAgentWorkflow(agent=mock_agent, name="my_agent")

        assert workflow.agent is mock_agent
        assert workflow.workflow_name == "dapr.strands.my_agent.workflow"

    def test_init_custom_name(self, mock_agent):
        """Test initialization with custom name."""
        workflow = DaprAgentWorkflow(
            agent=mock_agent,
            name="test_workflow",
        )

        assert workflow.workflow_name == "dapr.strands.test_workflow.workflow"

    def test_register(self, mock_agent):
        """Test that register creates workflow and activity."""
        workflow = DaprAgentWorkflow(agent=mock_agent, name="my_agent")

        mock_runtime = MagicMock()

        workflow.register(mock_runtime)

        mock_runtime.register_workflow.assert_called_once()
        mock_runtime.register_activity.assert_called_once()


class TestDaprAgentWorkflowDecorator:
    """Tests for the dapr_agent_workflow decorator."""

    def test_decorator_creates_workflow(self):
        """Test that decorator creates a DaprAgentWorkflow."""
        mock_agent = MagicMock()
        mock_agent.tool_registry = MagicMock()
        mock_agent.tool_registry.registry = {}

        @dapr_agent_workflow(name="decorated_workflow")
        def create_agent():
            return mock_agent

        workflow = create_agent()

        assert isinstance(workflow, DaprAgentWorkflow)
        assert workflow.workflow_name == "dapr.strands.decorated_workflow.workflow"

    def test_decorator_passes_args(self):
        """Test that decorator passes arguments to factory."""
        mock_agent = MagicMock()
        mock_agent.tool_registry = MagicMock()
        mock_agent.tool_registry.registry = {}

        received_args = []

        @dapr_agent_workflow(name="test_agent")
        def create_agent(param1, param2=None):
            received_args.extend([param1, param2])
            return mock_agent

        create_agent("arg1", param2="arg2")

        assert received_args == ["arg1", "arg2"]
