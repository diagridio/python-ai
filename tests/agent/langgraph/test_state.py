# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for LangGraph DaprMemoryCheckpointer."""

from unittest import TestCase, mock

from diagrid.agent.langgraph.state import DaprMemoryCheckpointer


class TestDaprMemoryCheckpointer(TestCase):
    def _make_checkpointer(self):
        store = mock.MagicMock()
        return DaprMemoryCheckpointer(state_store=store), store

    def test_save_checkpoint(self):
        cp, store = self._make_checkpointer()
        store.get.return_value = None  # No existing index

        cp.save_checkpoint(
            thread_id="t1",
            checkpoint_id="cp1",
            channel_values={"messages": ["hi"]},
            channel_versions={"messages": 1},
        )

        # Should save the checkpoint
        store.save.assert_any_call(
            "langgraph-t1-checkpoint-cp1",
            {
                "thread_id": "t1",
                "checkpoint_id": "cp1",
                "channel_values": {"messages": ["hi"]},
                "channel_versions": {"messages": 1},
                "metadata": {},
            },
        )
        # Should save the index
        store.save.assert_any_call(
            "langgraph-t1-index",
            {"checkpoints": ["cp1"], "latest": "cp1"},
        )

    def test_load_checkpoint_by_id(self):
        cp, store = self._make_checkpointer()
        store.get.return_value = {"channel_values": {"messages": ["hi"]}}

        result = cp.load_checkpoint("t1", "cp1")

        store.get.assert_called_with("langgraph-t1-checkpoint-cp1")
        self.assertEqual(result, {"channel_values": {"messages": ["hi"]}})

    def test_load_latest_checkpoint(self):
        cp, store = self._make_checkpointer()
        store.get.side_effect = [
            {"checkpoints": ["cp1", "cp2"], "latest": "cp2"},  # index
            {"channel_values": {"messages": ["latest"]}},  # checkpoint
        ]

        result = cp.load_checkpoint("t1")

        self.assertEqual(result, {"channel_values": {"messages": ["latest"]}})

    def test_load_checkpoint_no_index(self):
        cp, store = self._make_checkpointer()
        store.get.return_value = None

        result = cp.load_checkpoint("t1")
        self.assertIsNone(result)

    def test_list_checkpoints(self):
        cp, store = self._make_checkpointer()
        store.get.return_value = {
            "checkpoints": ["cp1", "cp2"],
            "latest": "cp2",
        }

        result = cp.list_checkpoints("t1")
        self.assertEqual(result, ["cp1", "cp2"])

    def test_list_checkpoints_empty(self):
        cp, store = self._make_checkpointer()
        store.get.return_value = None

        result = cp.list_checkpoints("t1")
        self.assertEqual(result, [])

    def test_delete_checkpoint(self):
        cp, store = self._make_checkpointer()
        store.get.return_value = {
            "checkpoints": ["cp1", "cp2"],
            "latest": "cp2",
        }

        cp.delete_checkpoint("t1", "cp2")

        store.delete.assert_called_once_with("langgraph-t1-checkpoint-cp2")
        # Should update index
        store.save.assert_called_with(
            "langgraph-t1-index",
            {"checkpoints": ["cp1"], "latest": "cp1"},
        )

    def test_close(self):
        cp, store = self._make_checkpointer()
        cp.close()
        store.close.assert_called_once()
