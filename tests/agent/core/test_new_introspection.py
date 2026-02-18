import unittest
from unittest import mock
from diagrid.agent.core.metadata.introspection import (
    detect_framework,
    find_agent_in_stack,
)


class Agent:
    """Mock CrewAI Agent."""

    __module__ = "crewai.agent"


class LlmAgent:
    """Mock ADK LlmAgent."""

    __module__ = "google.adk.agents.llm_agent"


class OpenAIAgent:
    """Mock OpenAI Agent."""

    __module__ = "agents.agent"
    # We'll use a trick to make type(obj).__name__ == "Agent"
    # by using a class named Agent in the agents module.


class DaprWorkflowAgentRunner:
    """Mock Runner."""

    __module__ = "diagrid.agent.crewai.runner"

    def __init__(self, agent):
        self._agent = agent


class IntrospectionTest(unittest.TestCase):
    def test_detect_framework_crewai(self):
        agent = Agent()
        # Ensure it's the right name
        self.assertEqual(type(agent).__name__, "Agent")
        self.assertEqual(detect_framework(agent), "crewai")

    def test_detect_framework_adk(self):
        agent = LlmAgent()
        self.assertEqual(type(agent).__name__, "LlmAgent")
        self.assertEqual(detect_framework(agent), "adk")

    def test_detect_framework_openai(self):
        # We need a class named "Agent" in a module with "agents"
        class Agent:
            __module__ = "agents.agent"

        agent = Agent()
        self.assertEqual(detect_framework(agent), "openai")

    def test_find_agent_in_stack_crewai_runner(self):
        agent = Agent()
        runner = DaprWorkflowAgentRunner(agent)

        with mock.patch("inspect.stack") as mock_stack:
            mock_frame = mock.MagicMock()
            mock_frame.frame.f_locals = {"self": runner}
            mock_stack.return_value = [mock_frame]

            result = find_agent_in_stack()
            self.assertEqual(result, agent)


if __name__ == "__main__":
    unittest.main()
