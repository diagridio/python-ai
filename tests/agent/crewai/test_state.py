# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for CrewAI DaprMemoryStore."""

from unittest import TestCase, mock

from diagrid.agent.crewai.state import DaprMemoryStore


class TestDaprMemoryStore(TestCase):
    def _make_store(self):
        state_store = mock.MagicMock()
        return DaprMemoryStore(state_store=state_store), state_store

    def test_save_memory(self):
        store, mock_ss = self._make_store()
        store.save_memory("sess-1", {"messages": ["hi"], "task_output": "done"})

        mock_ss.save.assert_called_once_with(
            "crewai-sess-1-memory",
            {"messages": ["hi"], "task_output": "done"},
        )

    def test_load_memory(self):
        store, mock_ss = self._make_store()
        mock_ss.get.return_value = {"messages": ["hi"]}

        result = store.load_memory("sess-1")

        mock_ss.get.assert_called_once_with("crewai-sess-1-memory")
        self.assertEqual(result, {"messages": ["hi"]})

    def test_load_memory_not_found(self):
        store, mock_ss = self._make_store()
        mock_ss.get.return_value = None

        result = store.load_memory("sess-missing")
        self.assertIsNone(result)

    def test_delete_memory(self):
        store, mock_ss = self._make_store()
        store.delete_memory("sess-1")

        mock_ss.delete.assert_called_once_with("crewai-sess-1-memory")

    def test_close(self):
        store, mock_ss = self._make_store()
        store.close()
        mock_ss.close.assert_called_once()
