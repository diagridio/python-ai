# -*- coding: utf-8 -*-

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1


import unittest
from unittest import mock

import pytest

pytest.importorskip("langgraph.pregel", reason="langgraph not installed")

from diagrid.agent.core.metadata.mapping.langgraph import LangGraphMapper  # noqa: E402


class MockCheckpointer:
    """Mock DaprCheckpointer for testing."""

    def __init__(self, state_store_name="test-store"):
        self.state_store_name = state_store_name


class MockTool:
    """Mock langchain-style tool for testing."""

    def __init__(self, name, description="", args=None):
        self.name = name
        self.description = description
        self.args = args or {}

    def __call__(self, *a, **kw):
        pass


class MockCompiledStateGraph:
    """Mock CompiledStateGraph for testing."""

    def __init__(
        self,
        name="test-graph",
        checkpointer=None,
        nodes=None,
    ):
        self._name = name
        self.checkpointer = checkpointer
        self.nodes = nodes or {}

    def get_name(self):
        return self._name


class LangGraphMapperTest(unittest.TestCase):
    """Tests for LangGraphMapper metadata extraction."""

    def test_mapper_instantiation(self):
        """Test that LangGraphMapper can be instantiated."""
        mapper = LangGraphMapper()
        self.assertIsNotNone(mapper)

    @mock.patch("diagrid.agent.core.metadata.mapping.langgraph.PregelNode")
    def test_basic_metadata_extraction(self, mock_pregel_node):
        """Test basic metadata extraction from a mock graph."""
        checkpointer = MockCheckpointer(state_store_name="my-store")
        graph = MockCompiledStateGraph(
            name="my-graph",
            checkpointer=checkpointer,
            nodes={"__start__": None},
        )

        mapper = LangGraphMapper()
        metadata = mapper.map_agent_metadata(graph, schema_version="1.0.0")

        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.agent.type, "MockCompiledStateGraph")
        self.assertEqual(metadata.name, "my-graph")
        assert metadata.memory is not None
        assert metadata.memory.short_term is not None
        self.assertEqual(metadata.memory.short_term.type, "DaprCheckpointer")
        self.assertEqual(metadata.memory.short_term.resource_name, "my-store")

    @mock.patch("diagrid.agent.core.metadata.mapping.langgraph.PregelNode")
    def test_metadata_without_checkpointer(self, mock_pregel_node):
        """Test metadata extraction without a checkpointer."""
        graph = MockCompiledStateGraph(
            name="no-checkpointer-graph",
            checkpointer=None,
            nodes={},
        )

        mapper = LangGraphMapper()
        metadata = mapper.map_agent_metadata(graph, schema_version="1.0.0")

        assert metadata.memory is not None
        assert metadata.memory.short_term is not None
        self.assertIsNone(metadata.memory.short_term.resource_name)

    @mock.patch("diagrid.agent.core.metadata.mapping.langgraph.PregelNode")
    def test_metadata_agent_role_defaults(self, mock_pregel_node):
        """Test agent metadata default values."""
        graph = MockCompiledStateGraph(name="test")

        mapper = LangGraphMapper()
        metadata = mapper.map_agent_metadata(graph, schema_version="1.0.0")

        self.assertEqual(metadata.agent.role, "Assistant")
        self.assertFalse(metadata.agent.orchestrator)
        self.assertEqual(metadata.agent.appid, "")

    @mock.patch("diagrid.agent.core.metadata.mapping.langgraph.PregelNode")
    def test_metadata_llm_defaults(self, mock_pregel_node):
        """Test LLM metadata defaults when no LLM is detected."""
        graph = MockCompiledStateGraph(name="test")

        mapper = LangGraphMapper()
        metadata = mapper.map_agent_metadata(graph, schema_version="1.0.0")

        assert metadata.llm is not None
        self.assertEqual(metadata.llm.client, "")
        self.assertEqual(metadata.llm.provider, "unknown")
        self.assertEqual(metadata.llm.model, "unknown")

    @mock.patch("diagrid.agent.core.metadata.mapping.langgraph.PregelNode")
    def test_metadata_pubsub_defaults(self, mock_pregel_node):
        """Test PubSub metadata defaults."""
        graph = MockCompiledStateGraph(name="test")

        mapper = LangGraphMapper()
        metadata = mapper.map_agent_metadata(graph, schema_version="1.0.0")

        assert metadata.pubsub is not None
        self.assertEqual(metadata.pubsub.resource_name, "")
        self.assertIsNone(metadata.pubsub.broadcast_topic)
        self.assertIsNone(metadata.pubsub.agent_topic)

    @mock.patch("diagrid.agent.core.metadata.mapping.langgraph.PregelNode")
    def test_metadata_registered_at_is_set(self, mock_pregel_node):
        """Test registered_at timestamp is set."""
        graph = MockCompiledStateGraph(name="test")

        mapper = LangGraphMapper()
        metadata = mapper.map_agent_metadata(graph, schema_version="1.0.0")

        self.assertIsNotNone(metadata.registered_at)
        self.assertIn("T", metadata.registered_at)

    @mock.patch("diagrid.agent.core.metadata.mapping.langgraph.PregelNode")
    def test_system_prompt_populates_instructions(self, mock_pregel_node):
        """Test that a discovered system prompt populates instructions and system_prompt."""
        graph = MockCompiledStateGraph(name="test")

        mapper = LangGraphMapper()

        prompt_text = "You are a helpful assistant."
        with mock.patch.object(
            mapper,
            "map_agent_metadata",
            wraps=mapper.map_agent_metadata,
        ):
            metadata = mapper.map_agent_metadata(graph, schema_version="1.0.0")

        self.assertEqual(metadata.agent.instructions, [])
        self.assertEqual(metadata.agent.system_prompt, "")

    @mock.patch("diagrid.agent.core.metadata.mapping.langgraph.PregelNode")
    def test_runner_hints_role_goal(self, mock_pregel_node):
        """Test that _diagrid_role and _diagrid_goal hints are used."""
        graph = MockCompiledStateGraph(name="test")
        graph._diagrid_role = "Banker worker"
        graph._diagrid_goal = "Process credit tasks"

        mapper = LangGraphMapper()
        metadata = mapper.map_agent_metadata(graph, schema_version="1.0.0")

        self.assertEqual(metadata.agent.role, "Banker worker")
        self.assertEqual(metadata.agent.goal, "Process credit tasks")

    @mock.patch("diagrid.agent.core.metadata.mapping.langgraph.PregelNode")
    def test_max_iterations_from_hint(self, mock_pregel_node):
        """Test that _diagrid_max_steps hint is used for max_iterations."""
        graph = MockCompiledStateGraph(name="test")
        graph._diagrid_max_steps = 50

        mapper = LangGraphMapper()
        metadata = mapper.map_agent_metadata(graph, schema_version="1.0.0")

        self.assertEqual(metadata.agent.max_iterations, 50)

    @mock.patch("diagrid.agent.core.metadata.mapping.langgraph.PregelNode")
    def test_max_iterations_default(self, mock_pregel_node):
        """Test that max_iterations defaults to 1 without hint."""
        graph = MockCompiledStateGraph(name="test")

        mapper = LangGraphMapper()
        metadata = mapper.map_agent_metadata(graph, schema_version="1.0.0")

        self.assertEqual(metadata.agent.max_iterations, 1)

    def test_collect_tools_from_list(self):
        """Test _collect_tools_from_list extracts tool metadata."""
        tool_a = MockTool(
            "get_balance", "Look up balance", {"customer_id": {"type": "integer"}}
        )
        tool_b = MockTool("credit_account", "Credit an account")

        tools: list = []
        seen: set = set()
        LangGraphMapper._collect_tools_from_list([tool_a, tool_b], tools, seen)

        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["name"], "get_balance")
        self.assertEqual(tools[0]["description"], "Look up balance")
        self.assertIn("customer_id", tools[0]["args"])
        self.assertEqual(tools[1]["name"], "credit_account")
        self.assertEqual({"get_balance", "credit_account"}, seen)

    def test_collect_tools_deduplicates(self):
        """Test that _collect_tools_from_list skips already-seen tools."""
        tool = MockTool("get_balance", "Look up balance")

        tools: list = []
        seen: set = {"get_balance"}
        LangGraphMapper._collect_tools_from_list([tool], tools, seen)

        self.assertEqual(len(tools), 0)

    def test_collect_tools_ignores_non_tools(self):
        """Test that _collect_tools_from_list skips non-tool lists."""
        tools: list = []
        seen: set = set()
        LangGraphMapper._collect_tools_from_list(["a", "b"], tools, seen)
        self.assertEqual(len(tools), 0)


class LangGraphProviderExtractionTest(unittest.TestCase):
    """Tests for LLM provider extraction."""

    def test_extract_openai_provider(self):
        """Test OpenAI provider extraction."""
        mapper = LangGraphMapper()
        self.assertEqual(mapper._extract_provider("langchain_openai.chat"), "openai")

    def test_extract_azure_provider(self):
        """Test Azure OpenAI provider extraction."""
        mapper = LangGraphMapper()
        self.assertEqual(
            mapper._extract_provider("langchain.azure_openai"), "azure_openai"
        )

    def test_extract_anthropic_provider(self):
        """Test Anthropic provider extraction."""
        mapper = LangGraphMapper()
        self.assertEqual(
            mapper._extract_provider("langchain_anthropic.chat"), "anthropic"
        )

    def test_extract_ollama_provider(self):
        """Test Ollama provider extraction."""
        mapper = LangGraphMapper()
        self.assertEqual(mapper._extract_provider("langchain_ollama.llms"), "ollama")

    def test_extract_google_provider(self):
        """Test Google provider extraction."""
        mapper = LangGraphMapper()
        self.assertEqual(mapper._extract_provider("langchain_google.genai"), "google")

    def test_extract_gemini_provider(self):
        """Test Gemini provider extraction."""
        mapper = LangGraphMapper()
        self.assertEqual(mapper._extract_provider("langchain_gemini.chat"), "google")

    def test_extract_cohere_provider(self):
        """Test Cohere provider extraction."""
        mapper = LangGraphMapper()
        self.assertEqual(mapper._extract_provider("langchain_cohere.chat"), "cohere")

    def test_extract_unknown_provider(self):
        """Test unknown provider returns 'unknown'."""
        mapper = LangGraphMapper()
        self.assertEqual(mapper._extract_provider("some.unknown.module"), "unknown")


if __name__ == "__main__":
    unittest.main()
