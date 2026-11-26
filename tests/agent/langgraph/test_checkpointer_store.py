# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""How ``DaprMemoryCheckpointer`` keeps a thread's checkpoints in the store.

Checkpoints and the thread's index expire (TTL), a thread keeps only its last
few checkpoints, the index is written with an etag so concurrent runs on one
thread lose no entry, and checkpoints written under the old key format are
still read.
"""

import threading
import unittest
from typing import Any

from diagrid.agent.langgraph.state import DaprMemoryCheckpointer
from diagrid.agent.langgraph.thread_state import DaprThreadStore

from .thread_helpers import MemoryStateStore


def _save(cp: DaprMemoryCheckpointer, thread: str, checkpoint: str) -> None:
    cp.save_checkpoint(
        thread_id=thread,
        checkpoint_id=checkpoint,
        channel_values={"messages": [checkpoint]},
        channel_versions={"messages": 1},
    )


class TestExpiryAndPruning(unittest.TestCase):
    def test_checkpoints_and_the_index_expire(self):
        store = MemoryStateStore()
        cp = DaprMemoryCheckpointer(state_store=store, ttl_s=600)  # type: ignore[arg-type]
        _save(cp, "t1", "cp1")
        for key in ("langgraph:t1:checkpoint:cp1", "langgraph:t1:index"):
            self.assertEqual(store.metadata[key], {"ttlInSeconds": "600"}, key)

    def test_the_default_ttl_is_set(self):
        store = MemoryStateStore()
        _save(DaprMemoryCheckpointer(state_store=store), "t1", "cp1")  # type: ignore[arg-type]
        self.assertIn("ttlInSeconds", store.metadata["langgraph:t1:index"])

    def test_a_thread_keeps_only_its_last_checkpoints(self):
        store = MemoryStateStore()
        cp = DaprMemoryCheckpointer(state_store=store, keep_checkpoints=2)  # type: ignore[arg-type]
        for n in range(5):
            _save(cp, "t1", f"cp{n}")
        self.assertEqual(cp.list_checkpoints("t1"), ["cp3", "cp4"])
        kept = sorted(k for k in store.data if ":checkpoint:" in k)
        self.assertEqual(
            kept, ["langgraph:t1:checkpoint:cp3", "langgraph:t1:checkpoint:cp4"]
        )
        self.assertEqual(
            cp.load_checkpoint("t1")["channel_values"], {"messages": ["cp4"]}
        )

    def test_keep_checkpoints_must_be_at_least_two(self):
        # with one, a concurrent save deletes the checkpoint a load was reading
        for keep in (0, 1):
            with self.assertRaises(ValueError):
                DaprMemoryCheckpointer(
                    state_store=MemoryStateStore(), keep_checkpoints=keep
                )  # type: ignore[arg-type]

    def test_a_checkpoint_pruned_under_a_load_is_read_from_a_fresh_index(self):
        store = MemoryStateStore()
        cp = DaprMemoryCheckpointer(state_store=store)  # type: ignore[arg-type]
        other = DaprMemoryCheckpointer(state_store=store)  # type: ignore[arg-type]
        _save(cp, "t1", "cp1")
        real_get = store.get
        raced = []

        def racing_get(key: str) -> Any:
            found = real_get(key)
            if key.endswith(":index") and not raced:
                # two saves land between the index read and the checkpoint read
                raced.append(key)
                _save(other, "t1", "cp2")
                _save(other, "t1", "cp3")
            return found

        store.get = racing_get  # type: ignore[method-assign]
        loaded = cp.load_checkpoint("t1")
        self.assertEqual(loaded["channel_values"], {"messages": ["cp3"]})


class TestConcurrentRuns(unittest.TestCase):
    def test_a_concurrent_index_write_is_not_lost(self):
        store = MemoryStateStore()
        cp = DaprMemoryCheckpointer(state_store=store, keep_checkpoints=10)  # type: ignore[arg-type]
        other = DaprMemoryCheckpointer(state_store=store, keep_checkpoints=10)  # type: ignore[arg-type]
        fired = []

        def race(key: str) -> None:
            if key.endswith(":index") and not fired:
                fired.append(key)
                _save(other, "t1", "other")

        store.between = race
        _save(cp, "t1", "mine")
        self.assertEqual(sorted(cp.list_checkpoints("t1")), ["mine", "other"])

    def test_many_threads_saving_at_once_keep_every_entry(self):
        store = MemoryStateStore()
        cp = DaprMemoryCheckpointer(state_store=store, keep_checkpoints=50)  # type: ignore[arg-type]
        workers = [
            threading.Thread(target=_save, args=(cp, "t1", f"cp{n}")) for n in range(20)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        self.assertEqual(len(cp.list_checkpoints("t1")), 20)


class TestOldKeyFormat(unittest.TestCase):
    def _old_store(self) -> Any:
        store = MemoryStateStore()
        store.data["langgraph-t1-index"] = {"checkpoints": ["cp1"], "latest": "cp1"}
        store.data["langgraph-t1-checkpoint-cp1"] = {
            "thread_id": "t1",
            "checkpoint_id": "cp1",
            "channel_values": {"messages": ["old"]},
            "channel_versions": {"messages": 1},
            "metadata": {},
        }
        return store

    def test_a_checkpoint_saved_under_the_old_keys_is_read(self):
        cp = DaprMemoryCheckpointer(state_store=self._old_store())
        self.assertEqual(
            cp.load_checkpoint("t1")["channel_values"], {"messages": ["old"]}
        )
        self.assertEqual(cp.load_checkpoint("t1", "cp1")["checkpoint_id"], "cp1")
        self.assertEqual(cp.list_checkpoints("t1"), ["cp1"])

    def test_a_thread_continues_from_its_old_checkpoint(self):
        store = self._old_store()
        threads = DaprThreadStore(DaprMemoryCheckpointer(state_store=store))
        loaded = threads.load("t1")
        self.assertEqual(loaded["values"], {"messages": ["old"]})

    def test_a_new_save_takes_over_from_the_old_keys(self):
        cp = DaprMemoryCheckpointer(state_store=self._old_store())
        _save(cp, "t1", "cp2")
        self.assertEqual(
            cp.load_checkpoint("t1")["channel_values"], {"messages": ["cp2"]}
        )


class _RefusingStore(MemoryStateStore):
    """Refuses keys as Dapr does: ``||`` is the actor separator; ``broken`` fails."""

    def __init__(self, broken: str = "") -> None:
        super().__init__()
        self.broken = broken

    def get(self, key: str) -> Any:
        if "||" in key or (self.broken and key == self.broken):
            raise RuntimeError(f"state store refused key {key!r}")
        return super().get(key)


class TestThreadIdsTheOldKeysCannotHold(unittest.TestCase):
    def test_a_thread_id_with_the_actor_separator_never_reads_the_old_keys(self):
        cp = DaprMemoryCheckpointer(state_store=_RefusingStore())  # type: ignore[arg-type]
        self.assertIsNone(cp.load_checkpoint("a||b"))
        self.assertEqual(cp.list_checkpoints("a||b"), [])
        _save(cp, "a||b", "cp1")
        self.assertEqual(
            cp.load_checkpoint("a||b")["channel_values"], {"messages": ["cp1"]}
        )

    def test_an_old_index_that_cannot_be_read_is_skipped(self):
        store = _RefusingStore(broken="langgraph-t1-index")
        cp = DaprMemoryCheckpointer(state_store=store)  # type: ignore[arg-type]
        with self.assertLogs("diagrid.agent.langgraph.state", "WARNING"):
            self.assertIsNone(cp.load_checkpoint("t1"))
