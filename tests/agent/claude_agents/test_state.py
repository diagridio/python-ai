# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the Claude Agents DaprMemoryStore wrapper."""

from unittest import TestCase, mock

from diagrid.agent.claude_agents.state import DaprMemoryStore


class TestDaprMemoryStore(TestCase):
    def _make_store(self):
        state_store = mock.MagicMock()
        return DaprMemoryStore(state_store=state_store), state_store

    def test_save_memory(self):
        store, mock_ss = self._make_store()
        store.save_memory("sess-1", {"messages": ["hi"], "context_id": "c1"})

        mock_ss.save.assert_called_once_with(
            "claude-sess-1-memory",
            {"messages": ["hi"], "context_id": "c1"},
        )

    def test_load_memory_returns_data(self):
        store, mock_ss = self._make_store()
        mock_ss.get.return_value = {"messages": ["hi"]}

        result = store.load_memory("sess-1")

        mock_ss.get.assert_called_once_with("claude-sess-1-memory")
        self.assertEqual(result, {"messages": ["hi"]})

    def test_load_memory_not_found(self):
        store, mock_ss = self._make_store()
        mock_ss.get.return_value = None

        result = store.load_memory("sess-missing")
        self.assertIsNone(result)

    def test_delete_memory(self):
        store, mock_ss = self._make_store()
        store.delete_memory("sess-1")

        mock_ss.delete.assert_called_once_with("claude-sess-1-memory")

    def test_close(self):
        store, mock_ss = self._make_store()
        store.close()
        mock_ss.close.assert_called_once()

    def test_key_namespace_is_isolated_from_other_frameworks(self):
        """Claude store keys must be prefixed ``claude-`` so they cannot
        collide with the ``openai-`` / ``strands-`` keys produced by the
        sibling memory stores sharing a state store."""
        store, mock_ss = self._make_store()
        store.save_memory("sess-1", {})
        saved_key = mock_ss.save.call_args[0][0]
        self.assertTrue(saved_key.startswith("claude-"))
