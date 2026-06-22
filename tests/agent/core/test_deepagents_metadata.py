# -*- coding: utf-8 -*-

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Unit tests for ``DeepAgentsMapper`` metadata extraction.

The real-graph test builds a real compiled graph with
``langchain.agents.create_agent`` (the primitive ``deepagents.create_deep_agent``
compiles down to) but never invokes it, so no network or API key is required (a
dummy key is set only to satisfy chat-model construction). The remaining tests
exercise the mapper's helpers and fallbacks with lightweight fakes.
"""

import os
import sys
import types
import unittest
from pathlib import Path

import pytest

from diagrid.agent.core.metadata.mapping.deepagents import DeepAgentsMapper
from diagrid.agent.core.types import SupportedFrameworks


def _repair_langgraph_namespace() -> None:
    """Point ``langgraph.__path__`` at the real site-packages package.

    Under pytest's prepend import mode, ``tests/agent`` (which has no
    ``__init__.py``) lands on ``sys.path`` and its ``langgraph`` subpackage
    shadows the installed one, breaking ``import langgraph._internal`` — and
    therefore ``langchain.agents``. This mirrors the workaround already used by
    ``tests/agent/deepagents/conftest.py`` so the real-graph test can build a
    compiled graph instead of being skipped. Called from ``setUpClass`` (after
    collection), so it never affects how other test modules are imported.
    """
    site_lg = next(
        (
            str(Path(sp) / "langgraph")
            for sp in sys.path
            if "site-packages" in sp
            and (Path(sp) / "langgraph" / "constants.py").exists()
        ),
        None,
    )
    if not site_lg:
        return
    mod = sys.modules.get("langgraph")
    if mod is not None:
        path = getattr(mod, "__path__", None)
        if path is None:
            mod.__path__ = [site_lg]  # type: ignore[attr-defined]
        elif site_lg not in list(path):
            path.insert(0, site_lg)
    else:
        synthetic = types.ModuleType("langgraph")
        synthetic.__path__ = [site_lg]  # type: ignore[attr-defined]
        synthetic.__package__ = "langgraph"
        sys.modules["langgraph"] = synthetic


# ---------------------------------------------------------------------------
# Lightweight fakes for helper-level tests (no real graph required)
# ---------------------------------------------------------------------------


class _FakeTool:
    def __init__(self, description: str):
        self.description = description


class _FakeToolNode:
    """Stand-in for a LangGraph ToolNode (exposes ``_tools_by_name``)."""

    def __init__(self, tools_by_name):
        self._tools_by_name = tools_by_name


class _FakeNodeSpec:
    """Stand-in for a PregelNode (exposes ``bound``)."""

    def __init__(self, bound):
        self.bound = bound


class _FakeModel:
    """Minimal chat-model stand-in for the attribute-reading helpers."""

    def __init__(self, model_name="gpt-4o-mini", provider="openai", base_url=None):
        self.model_name = model_name
        self._provider = provider
        self.base_url = base_url

    def _get_ls_params(self):
        return {"ls_provider": self._provider}


class _FakeCheckpointer:
    def __init__(self, store_name="checkpoint-store"):
        self.store_name = store_name


class _FakeGraph:
    """Stand-in for a CompiledStateGraph (``vars()`` exposes nodes/checkpointer)."""

    def __init__(self, nodes=None, checkpointer=None, name="fake-graph"):
        self.nodes = nodes or {}
        self.checkpointer = checkpointer
        self._name = name

    def get_name(self):
        return self._name


class DeepAgentsMapperHelperTest(unittest.TestCase):
    """Tests for the pure extraction helpers."""

    def test_extract_tools_from_tool_node(self):
        nodes = {
            "__start__": _FakeNodeSpec(bound=None),
            "model": _FakeNodeSpec(bound=object()),
            "tools": _FakeNodeSpec(
                bound=_FakeToolNode(
                    {
                        "write_todos": _FakeTool("manage a todo list"),
                        "get_weather": _FakeTool("get the weather"),
                    }
                )
            ),
        }
        tools = DeepAgentsMapper._extract_tools(nodes)
        names = {t["name"] for t in tools}
        self.assertEqual(names, {"write_todos", "get_weather"})
        self.assertEqual(tools[0]["description"], "manage a todo list")

    def test_extract_tools_handles_no_tool_node(self):
        nodes = {"model": _FakeNodeSpec(bound=object())}
        self.assertEqual(DeepAgentsMapper._extract_tools(nodes), [])

    def test_build_llm_metadata_uses_ls_provider(self):
        mapper = DeepAgentsMapper()
        meta = mapper._build_llm_metadata(_FakeModel(model_name="gpt-4o-mini"))
        self.assertEqual(meta["client"], "_FakeModel")
        self.assertEqual(meta["provider"], "openai")
        self.assertEqual(meta["model"], "gpt-4o-mini")
        self.assertIsNone(meta["base_url"])

    def test_build_llm_metadata_stringifies_non_str_base_url(self):
        mapper = DeepAgentsMapper()
        meta = mapper._build_llm_metadata(
            _FakeModel(base_url=types.SimpleNamespace(__str__=lambda self: "http://x"))
        )
        self.assertIsInstance(meta["base_url"], str)

    def test_provider_from_model_reads_ls_params(self):
        self.assertEqual(
            DeepAgentsMapper._provider_from_model(_FakeModel(provider="anthropic")),
            "anthropic",
        )

    def test_provider_from_model_returns_none_without_ls_params(self):
        # Object with no ``_get_ls_params`` -> AttributeError is swallowed.
        self.assertIsNone(DeepAgentsMapper._provider_from_model(object()))

    def test_build_llm_metadata_falls_back_to_module_when_no_ls_params(self):
        mapper = DeepAgentsMapper()
        # No ``_get_ls_params`` and module name is this test module -> unknown.
        meta = mapper._build_llm_metadata(
            types.SimpleNamespace(model_name="m", base_url=None)
        )
        self.assertEqual(meta["provider"], "unknown")

    def test_extract_prompt_text_from_string(self):
        msg = types.SimpleNamespace(content="hello world")
        self.assertEqual(DeepAgentsMapper._extract_prompt_text(msg), "hello world")

    def test_extract_prompt_text_from_block_list(self):
        msg = types.SimpleNamespace(
            content=[{"type": "text", "text": "a"}, "b", {"image": "x"}]
        )
        self.assertEqual(DeepAgentsMapper._extract_prompt_text(msg), "a\nb")

    def test_extract_prompt_text_handles_none(self):
        self.assertIsNone(DeepAgentsMapper._extract_prompt_text(None))

    def test_first_line_skips_blank_lines(self):
        self.assertEqual(
            DeepAgentsMapper._first_line("\n\n  goal here \nmore"), "goal here"
        )

    def test_first_line_handles_empty(self):
        self.assertIsNone(DeepAgentsMapper._first_line(""))
        self.assertIsNone(DeepAgentsMapper._first_line(None))


class DeepAgentsMapperFallbackTest(unittest.TestCase):
    """Tests for ``map_agent_metadata`` when the model cannot be extracted."""

    def test_unknown_llm_when_no_model_in_closure(self):
        graph = _FakeGraph(
            nodes={
                "tools": _FakeNodeSpec(bound=_FakeToolNode({"ls": _FakeTool("list")}))
            },
            checkpointer=_FakeCheckpointer(store_name="my-store"),
        )
        md = DeepAgentsMapper().map_agent_metadata(graph, schema_version="1.0.0")

        self.assertEqual(md.agent.framework, SupportedFrameworks.DEEPAGENTS)
        self.assertEqual(md.llm.client, "")
        self.assertEqual(md.llm.provider, "unknown")
        self.assertEqual(md.llm.model, "unknown")
        # Tools are still extracted from the ToolNode.
        self.assertEqual([t.name for t in md.tools], ["ls"])
        # Memory resource comes from the checkpointer.
        self.assertEqual(md.memory.short_term.resource_name, "my-store")

    def test_name_derived_when_not_provided(self):
        graph = _FakeGraph(nodes={}, name="My Agent")
        md = DeepAgentsMapper().map_agent_metadata(graph, schema_version="1.0.0")
        self.assertEqual(md.name, "deepagents-my-agent")

    def test_explicit_name_and_hints_win(self):
        graph = _FakeGraph(nodes={})
        graph._diagrid_role = "Planner"  # type: ignore[attr-defined]
        graph._diagrid_goal = "Plan things"  # type: ignore[attr-defined]
        md = DeepAgentsMapper().map_agent_metadata(
            graph, schema_version="1.0.0", name="explicit-name"
        )
        self.assertEqual(md.name, "explicit-name")
        self.assertEqual(md.agent.role, "Planner")
        self.assertEqual(md.agent.goal, "Plan things")

    def test_registered_at_is_iso_timestamp(self):
        md = DeepAgentsMapper().map_agent_metadata(_FakeGraph(), schema_version="1.0.0")
        self.assertIsNotNone(md.registered_at)
        self.assertIn("T", md.registered_at)


class DeepAgentsProviderExtractionTest(unittest.TestCase):
    """Module-name provider extraction inherited from ``BaseAgentMapper``."""

    def test_extract_openai_provider(self):
        self.assertEqual(
            DeepAgentsMapper()._extract_provider("langchain_openai.chat_models.base"),
            "openai",
        )

    def test_extract_anthropic_provider(self):
        self.assertEqual(
            DeepAgentsMapper()._extract_provider("langchain_anthropic.chat"),
            "anthropic",
        )

    def test_extract_unknown_provider(self):
        self.assertEqual(
            DeepAgentsMapper()._extract_provider("some.unknown.module"), "unknown"
        )


class DeepAgentsMapperRealGraphTest(unittest.TestCase):
    """End-to-end closure extraction against a real compiled agent graph.

    Built with ``langchain.agents.create_agent`` — the exact primitive that
    ``deepagents.create_deep_agent`` compiles down to (same ``model``-node
    closure holding the chat model + system message, same ``tools`` ToolNode).
    This is the structure the mapper's novel closure extraction targets, and it
    sidesteps the ``tests/agent/deepagents`` package that shadows the real
    ``deepagents`` import on ``sys.path`` during collection. Deep Agent built-in
    tools (a superset in the same ToolNode) are covered by
    ``DeepAgentsMapperHelperTest.test_extract_tools_from_tool_node``.

    The graph is constructed but never invoked, so no network call is made.
    """

    @classmethod
    def setUpClass(cls):
        pytest.importorskip("langchain_openai", reason="langchain_openai not installed")
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-not-used")

        # Repair the langgraph namespace shadow so the real graph builder imports.
        _repair_langgraph_namespace()
        try:
            from langchain.agents import create_agent
        except ImportError as exc:  # pragma: no cover - defensive
            raise unittest.SkipTest(
                "langchain.agents unavailable; langgraph namespace shadow could "
                f"not be repaired in this run mode: {exc}"
            )

        from langchain_core.tools import tool

        @tool
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return "sunny"

        cls.agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather],
            system_prompt="You are a helpful research assistant.",
        )

    def test_extracts_llm_metadata(self):
        md = DeepAgentsMapper().map_agent_metadata(self.agent, schema_version="edge")
        self.assertEqual(md.agent.framework, SupportedFrameworks.DEEPAGENTS)
        self.assertEqual(md.llm.client, "ChatOpenAI")
        self.assertEqual(md.llm.provider, "openai")
        self.assertEqual(md.llm.model, "gpt-4o-mini")
        self.assertEqual(md.llm.api, "chat")

    def test_extracts_system_prompt_and_goal(self):
        md = DeepAgentsMapper().map_agent_metadata(self.agent, schema_version="edge")
        self.assertTrue(md.agent.system_prompt.startswith("You are a helpful research"))
        self.assertEqual(md.agent.goal, "You are a helpful research assistant.")

    def test_extracts_user_tool_from_tool_node(self):
        md = DeepAgentsMapper().map_agent_metadata(self.agent, schema_version="edge")
        self.assertIn("get_weather", {t.name for t in md.tools})

    def test_name_uses_explicit_runner_name(self):
        md = DeepAgentsMapper().map_agent_metadata(
            self.agent, schema_version="edge", name="deep-agent"
        )
        self.assertEqual(md.name, "deep-agent")


if __name__ == "__main__":
    unittest.main()
