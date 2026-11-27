# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The names a streaming caller needs are importable, and every control event
is documented where a consumer looks."""

import unittest
from pathlib import Path

from langgraph.graph import END, START, MessagesState, StateGraph

from diagrid.agent.core.workflow.stream import (
    CONTROL_MODES,
    STREAM_DROPPED,
    StreamSinkRegistry,
)
from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner
from diagrid.agent.langgraph import DaprWorkflowGraphRunner, streaming

from .test_runner import _RunnerTestCase


def _compiled():
    graph = StateGraph(MessagesState)
    graph.add_node("n", lambda state: {})
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return graph.compile()


class TestExports(unittest.TestCase):
    def test_streaming_names_are_exported(self):
        import diagrid.agent.langgraph as pkg

        for name in ("GraphRunError", "StreamControl", "StreamSinkRegistry"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(pkg, name), name)
                self.assertIn(name, pkg.__all__)

    def test_exported_registry_is_the_one_runners_take(self):
        import diagrid.agent.langgraph as pkg

        self.assertIs(pkg.StreamSinkRegistry, StreamSinkRegistry)


class TestServeDocstring(unittest.TestCase):
    def test_graph_runner_serve_documents_streaming_and_its_limit(self):
        doc = DaprWorkflowGraphRunner.serve.__doc__ or ""
        self.assertIn('"stream": true', doc)
        self.assertIn("single replica", doc.lower())


class TestDeepAgentRunner(_RunnerTestCase):
    def test_stream_registry_is_forwarded(self):
        registry = StreamSinkRegistry()
        runner = DaprWorkflowDeepAgentRunner(
            _compiled(), name="deep-reg", stream_registry=registry
        )
        self.assertIs(runner.stream_registry, registry)


README = Path(streaming.__file__).with_name("README.md")


KINDS = (*CONTROL_MODES, STREAM_DROPPED)


class TestControlKindsAreDocumented(unittest.TestCase):
    def test_the_stream_control_docstring_lists_every_kind(self):
        for kind in KINDS:
            self.assertIn(kind, streaming.StreamControl.__doc__ or "", kind)

    def test_the_readme_table_lists_every_kind(self):
        table = [
            line for line in README.read_text().splitlines() if line.startswith("| `")
        ]
        for kind in KINDS:
            self.assertTrue(any(line.startswith(f"| `{kind}`") for line in table), kind)

    def test_workflow_started_is_said_to_be_on_control_only(self):
        doc = " ".join((streaming.StreamControl.__doc__ or "").split())
        self.assertIn("only ever given to ``on_control`` and never yielded", doc)


class TestSendTaskNameIsDocumentedWhereItIsSeen(unittest.TestCase):
    """``executed`` never reaches a reader; ``missing_nodes`` is where ``node#k`` does."""

    def _lines(self):
        return README.read_text().splitlines()

    def test_the_stream_unavailable_row_names_a_send_task(self):
        row = next(
            line for line in self._lines() if line.startswith("| `stream_unavailable`")
        )
        self.assertIn("node#k", row)

    def test_the_send_bullet_does_not_point_at_the_hidden_executed_key(self):
        text = README.read_text()
        start = text.index("- **`Send` tasks.**")
        bullet = text[start : text.index("\n- ", start + 1)]
        self.assertIn("missing_nodes", bullet)
        self.assertNotIn("In `executed`", bullet)
