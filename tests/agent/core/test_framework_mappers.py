# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import unittest
from unittest import mock
from diagrid.agent.core.metadata.mapping.crewai import CrewAIMapper
from diagrid.agent.core.metadata.mapping.adk import ADKMapper
from diagrid.agent.core.metadata.mapping.openai import OpenAIAgentsMapper
from diagrid.agent.core.metadata.mapping.langchain import LangChainMapper
from diagrid.agent.core.metadata.mapping.smolagents import SmolagentsMapper
from diagrid.agent.core.types import SupportedFrameworks


class MockTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description


class CrewAIMapperTest(unittest.TestCase):
    def test_crewai_mapper(self):
        agent = mock.MagicMock()
        agent.role = "Researcher"
        agent.goal = "Find information"
        agent.backstory = "Expert researcher"
        agent.tools = [MockTool("search", "search the web")]
        agent.llm = "gpt-4o"
        agent.max_iter = 10

        mapper = CrewAIMapper()
        metadata = mapper.map_agent_metadata(agent, "1.0.0")

        self.assertEqual(metadata.agent.role, "Researcher")
        self.assertEqual(metadata.agent.goal, "Find information")
        self.assertEqual(metadata.agent.instructions, ["Expert researcher"])
        self.assertEqual(len(metadata.tools), 1)
        self.assertEqual(metadata.tools[0].name, "search")
        self.assertEqual(metadata.llm.model, "gpt-4o")
        self.assertEqual(metadata.agent.framework, SupportedFrameworks.CREWAI)


class ADKMapperTest(unittest.TestCase):
    def test_adk_mapper(self):
        agent = mock.MagicMock()
        agent.name = "adk-agent"
        agent.model = "gemini-2.0-flash"
        agent.instruction = "Be helpful"
        agent.tools = [MockTool("calc", "calculator")]

        mapper = ADKMapper()
        metadata = mapper.map_agent_metadata(agent, "1.0.0")

        self.assertEqual(metadata.agent.role, "adk-agent")
        self.assertEqual(metadata.agent.goal, "Be helpful")
        self.assertEqual(len(metadata.tools), 1)
        self.assertEqual(metadata.llm.model, "gemini-2.0-flash")
        self.assertEqual(metadata.agent.framework, SupportedFrameworks.ADK)


class OpenAIAgentsMapperTest(unittest.TestCase):
    def test_openai_mapper(self):
        agent = mock.MagicMock()
        agent.name = "openai-agent"
        agent.model = "gpt-4"
        agent.instructions = "Help the user"
        agent.tools = [MockTool("weather", "check weather")]

        mapper = OpenAIAgentsMapper()
        metadata = mapper.map_agent_metadata(agent, "1.0.0")

        self.assertEqual(metadata.agent.role, "openai-agent")
        self.assertEqual(metadata.agent.goal, "Help the user")
        self.assertEqual(len(metadata.tools), 1)
        self.assertEqual(metadata.llm.model, "gpt-4")
        self.assertEqual(metadata.agent.framework, SupportedFrameworks.OPENAI)


class LangChainMapperTest(unittest.TestCase):
    def test_langchain_mapper(self):
        # LangChain has no native "Agent" object - the runner itself carries
        # the configuration (see LangChainMapper docstring), so we mock the
        # runner's private attributes directly.
        runner = mock.MagicMock()
        runner._name = "langchain-agent"
        runner._model = mock.MagicMock()
        runner._model.model_name = "gpt-4o-mini"
        runner._system_prompt = "Be helpful"
        runner._max_iterations = 25
        runner._tools = [MockTool("search", "search the web")]

        mapper = LangChainMapper()
        metadata = mapper.map_agent_metadata(runner, "1.0.0")

        self.assertEqual(metadata.agent.role, "langchain-agent")
        self.assertEqual(metadata.agent.goal, "Be helpful")
        self.assertEqual(len(metadata.tools), 1)
        self.assertEqual(metadata.tools[0].name, "search")
        self.assertEqual(metadata.llm.model, "gpt-4o-mini")
        self.assertEqual(metadata.agent.framework, SupportedFrameworks.LANGCHAIN)


class SmolagentsMapperTest(unittest.TestCase):
    def test_smolagents_mapper(self):
        agent = mock.MagicMock()
        agent.name = "smol-agent"
        agent.system_prompt = "Be helpful"
        agent.max_steps = 20
        agent.tools = {
            "search": MockTool("search", "search the web"),
            # Auto-injected by smolagents on every agent - must be excluded
            # from the user-facing tool list.
            "final_answer": MockTool("final_answer", "Provide the final answer"),
        }
        agent.model = mock.MagicMock()
        agent.model.model_id = "gpt-4o-mini"

        mapper = SmolagentsMapper()
        metadata = mapper.map_agent_metadata(agent, "1.0.0")

        self.assertEqual(metadata.agent.role, "smol-agent")
        self.assertEqual(metadata.agent.goal, "Be helpful")
        self.assertEqual(len(metadata.tools), 1)
        self.assertEqual(metadata.tools[0].name, "search")
        self.assertEqual(metadata.llm.model, "gpt-4o-mini")
        self.assertEqual(metadata.agent.framework, SupportedFrameworks.SMOLAGENTS)


if __name__ == "__main__":
    unittest.main()
