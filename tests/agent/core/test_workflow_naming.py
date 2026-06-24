# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for agent name sanitization in workflow naming."""

import pytest

from diagrid.agent.core.types import SupportedFrameworks
from diagrid.agent.core.workflow.naming import build_workflow_name, sanitize_agent_name


class TestSanitizeAgentName:
    """sanitize_agent_name should produce TitleCase names matching dapr-agents."""

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("catering-coordinator", "CateringCoordinator"),
            ("decoration-planner", "DecorationPlanner"),
            ("schedule-planner", "SchedulePlanner"),
            ("venue-scout", "VenueScout"),
            ("strands-default", "StrandsDefault"),
            ("my-agent", "MyAgent"),
        ],
    )
    def test_kebab_case_to_title_case(self, input_name, expected):
        assert sanitize_agent_name(input_name) == expected

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("get_user", "GetUser"),
            ("valid_name", "ValidName"),
        ],
    )
    def test_snake_case_to_title_case(self, input_name, expected):
        assert sanitize_agent_name(input_name) == expected

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("Samwise Gamgee", "SamwiseGamgee"),
            ("tool  name", "ToolName"),
        ],
    )
    def test_spaces_to_title_case(self, input_name, expected):
        assert sanitize_agent_name(input_name) == expected

    def test_already_title_case_preserved(self):
        assert sanitize_agent_name("SamwiseGamgee") == "SamwiseGamgee"
        assert sanitize_agent_name("GetUser") == "GetUser"

    def test_all_uppercase(self):
        assert sanitize_agent_name("UPPERCASE") == "Uppercase"

    def test_special_chars_removed(self):
        assert sanitize_agent_name("agent<name>") == "Agentname"
        assert sanitize_agent_name("tool|name") == "Toolname"
        assert sanitize_agent_name("tool\\name") == "Toolname"
        assert sanitize_agent_name("tool/name") == "Toolname"

    def test_empty_string(self):
        assert sanitize_agent_name("") == "unnamed_agent"

    def test_simple_names(self):
        assert sanitize_agent_name("sam") == "Sam"
        assert sanitize_agent_name("frodo") == "Frodo"


class TestBuildWorkflowName:
    """build_workflow_name should produce ``dapr.<framework>.<TitleCase>.workflow``."""

    @pytest.mark.parametrize(
        "framework,name,expected",
        [
            (
                "LangGraph",
                "catering-coordinator",
                "dapr.langgraph.CateringCoordinator.workflow",
            ),
            ("strands", "my-agent", "dapr.strands.MyAgent.workflow"),
            ("OpenAI", "get_user", "dapr.openai.GetUser.workflow"),
            ("crewai", "Samwise Gamgee", "dapr.crewai.SamwiseGamgee.workflow"),
        ],
    )
    def test_string_framework(self, framework, name, expected):
        assert build_workflow_name(framework, name) == expected

    def test_accepts_supported_frameworks_enum(self):
        # StrEnum members must lower-case to the same prefix as plain strings.
        assert (
            build_workflow_name(SupportedFrameworks.OPENAI, "catering-coordinator")
            == "dapr.openai.CateringCoordinator.workflow"
        )
        assert (
            build_workflow_name(SupportedFrameworks.LANGGRAPH, "my-agent")
            == "dapr.langgraph.MyAgent.workflow"
        )

    def test_empty_name_falls_back(self):
        assert (
            build_workflow_name("strands", "") == "dapr.strands.unnamed_agent.workflow"
        )
