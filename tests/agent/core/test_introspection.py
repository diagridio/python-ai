# -*- coding: utf-8 -*-

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1


import unittest

from diagrid.agent.core.metadata.introspection import detect_framework


class DetectFrameworkTest(unittest.TestCase):
    """Tests for detect_framework function."""

    def test_detect_langgraph_by_class_name(self):
        """Test detection of LangGraph by CompiledStateGraph class name."""

        class CompiledStateGraph:
            pass

        agent = CompiledStateGraph()
        result = detect_framework(agent)
        self.assertEqual(result, "langgraph")

    def test_detect_langgraph_by_module(self):
        """Test detection of LangGraph by module path."""

        class MockGraph:
            pass

        MockGraph.__module__ = "langgraph.graph.state"
        agent = MockGraph()
        result = detect_framework(agent)
        self.assertEqual(result, "langgraph")

    def test_detect_strands_by_class_name(self):
        """Test detection of Strands by DaprSessionManager class name."""

        class DaprSessionManager:
            pass

        agent = DaprSessionManager()
        result = detect_framework(agent)
        self.assertEqual(result, "strands")

    def test_detect_strands_by_module(self):
        """Test detection of Strands by module path."""

        class MockSessionManager:
            pass

        # Use actual type name that detection looks for
        MockSessionManager.__module__ = "dapr.ext.strands"
        MockSessionManager.__name__ = "DaprSessionManager"
        agent = MockSessionManager()
        result = detect_framework(agent)
        self.assertEqual(result, "strands")

    def test_detect_unknown_framework(self):
        """Test detection returns None for unknown frameworks."""

        class UnknownAgent:
            pass

        UnknownAgent.__module__ = "some.unknown.module"
        agent = UnknownAgent()
        result = detect_framework(agent)
        self.assertIsNone(result)

    def test_detect_builtin_object(self):
        """Test detection returns None for builtin objects."""
        result = detect_framework("string")
        self.assertIsNone(result)

        result = detect_framework(42)
        self.assertIsNone(result)

        result = detect_framework([1, 2, 3])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
