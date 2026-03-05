# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for ADK DaprSessionStore."""

from unittest import TestCase, mock

from diagrid.agent.adk.state import DaprSessionStore


class TestDaprSessionStore(TestCase):
    def _make_store(self):
        state_store = mock.MagicMock()
        return DaprSessionStore(state_store=state_store), state_store

    def test_save_session(self):
        store, mock_ss = self._make_store()
        store.save_session("sess-1", {"messages": ["hi"], "user_id": "u1"})

        mock_ss.save.assert_called_once_with(
            "adk-sess-1-state",
            {"messages": ["hi"], "user_id": "u1"},
        )

    def test_load_session(self):
        store, mock_ss = self._make_store()
        mock_ss.get.return_value = {"messages": ["hi"]}

        result = store.load_session("sess-1")

        mock_ss.get.assert_called_once_with("adk-sess-1-state")
        self.assertEqual(result, {"messages": ["hi"]})

    def test_load_session_not_found(self):
        store, mock_ss = self._make_store()
        mock_ss.get.return_value = None

        result = store.load_session("sess-missing")
        self.assertIsNone(result)

    def test_delete_session(self):
        store, mock_ss = self._make_store()
        store.delete_session("sess-1")

        mock_ss.delete.assert_called_once_with("adk-sess-1-state")

    def test_close(self):
        store, mock_ss = self._make_store()
        store.close()
        mock_ss.close.assert_called_once()
