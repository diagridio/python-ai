# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The state-store event log: what an activity writes and what any replica reads back."""

import time
import unittest
from typing import Any, List

from diagrid.agent.core.workflow.stream import (
    BRANCH_DONE,
    UPDATES,
    StreamEvent,
)
from diagrid.agent.core.workflow.stream_log import (
    LogReader,
    LogWriter,
    StateLogStreamConfig,
    UnitWant,
    batch_key,
    escape_key_part,
    head_key,
    wire_event,
)

from .log_store_fake import FakeLogStore


def _config(**kw: Any) -> StateLogStreamConfig:
    base = dict(flush_interval_s=30.0, flush_max_chunks=1000)
    base.update(kw)
    return StateLogStreamConfig(**base)


def _event(
    node: str, seq: int, mode: str = "messages", attempt: str = "att"
) -> StreamEvent:
    return StreamEvent("wf:1", 0, node, attempt, seq, (), mode, f"{node}-{seq}")


def _writer(store: FakeLogStore, unit: str = "research#0", **kw: Any) -> LogWriter:
    return LogWriter(store, _config(**kw), "wf:1", 0, unit, encode=lambda e: e.data)


def _datas(chunk: Any) -> List[Any]:
    return [e["data"] for e in chunk.events]


class TestKeys(unittest.TestCase):
    def test_parts_are_escaped_and_never_use_the_actor_separator(self):
        for part in ("research#0", "review:branch:route", "a/b", "x||y", "a b"):
            escaped = escape_key_part(part)
            for bad in (":", "#", "/", "|"):
                self.assertNotIn(bad, escaped)
        key = batch_key("lgstream", "wf:1", 3, "review:branch:route", "abc", 2)
        self.assertNotIn("||", key)
        self.assertEqual(key.count(":"), 5)
        self.assertTrue(head_key("lgstream", "wf:1", 3, "n").endswith(":head"))

    def test_distinct_units_never_share_a_key(self):
        self.assertNotEqual(head_key("p", "w", 1, "a:b"), head_key("p", "w:1", 1, "b"))


class TestWriter(unittest.TestCase):
    def test_flushes_on_the_chunk_count(self):
        store = FakeLogStore()
        writer = _writer(store, flush_max_chunks=3)
        for seq in range(3):
            writer.put(_event("research#0", seq))
        self.assertEqual(len(store.saves), 1)
        writer.finish(done=True)

    def test_flushes_on_the_interval(self):
        store = FakeLogStore()
        writer = _writer(store, flush_interval_s=0.02)
        writer.put(_event("research#0", 0))
        deadline = time.monotonic() + 5
        while not store.saves and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(store.saves), 1)
        writer.finish(done=True)

    def test_the_last_flush_runs_before_finish_returns(self):
        store = FakeLogStore()
        writer = _writer(store)
        writer.put(_event("research#0", 0))
        self.assertEqual(store.saves, [])
        writer.finish(done=True)
        head = store.get(head_key("lgstream", "wf:1", 0, "research#0"))
        self.assertEqual(
            head, {"attempt": writer.attempt, "batches": 1, "events": 1, "done": True}
        )

    def test_every_save_carries_a_ttl_and_heads_outlive_batches(self):
        store = FakeLogStore()
        writer = _writer(store, event_ttl_s=100, record_ttl_factor=24)
        writer.put(_event("research#0", 0))
        writer.finish(done=True)
        batch = batch_key("lgstream", "wf:1", 0, "research#0", writer.attempt, 0)
        head = head_key("lgstream", "wf:1", 0, "research#0")
        self.assertEqual(store.metadata[batch], {"ttlInSeconds": "100"})
        self.assertEqual(store.metadata[head], {"ttlInSeconds": "2400"})
        self.assertEqual(store.saves, [[batch, head]])

    def test_a_write_error_never_raises_and_is_retried(self):
        store = FakeLogStore()
        writer = _writer(store, flush_max_chunks=1)
        store.fail_saves = 1
        writer.put(_event("research#0", 0))  # the save fails: logged, counted
        self.assertEqual(writer.write_failures, 1)
        writer.put(_event("research#0", 1))
        writer.finish(done=True)
        chunk = LogReader(store, _config(), "wf:1").read([UnitWant(0, "research#0")])[0]
        self.assertEqual(_datas(chunk), ["research#0-0", "research#0-1"])

    def test_a_transient_error_at_the_last_flush_is_retried(self):
        store = FakeLogStore()
        writer = _writer(store, final_flush_backoff_s=0.001)
        writer.put(_event("research#0", 0, UPDATES))
        store.fail_saves = 2  # two errors in a row at the final flush
        writer.finish(done=True)
        head = store.get(head_key("lgstream", "wf:1", 0, "research#0"))
        self.assertTrue(head["done"])
        chunk = LogReader(store, _config(), "wf:1").read([UnitWant(0, "research#0")])[0]
        self.assertEqual(_datas(chunk), ["research#0-0"])
        self.assertEqual(writer.write_failures, 2)

    def test_a_last_flush_that_keeps_failing_leaves_the_unit_not_done(self):
        store = FakeLogStore()
        writer = _writer(store, final_flush_attempts=3, final_flush_backoff_s=0.001)
        writer.put(_event("research#0", 0))
        store.fail_saves = 100
        writer.finish(done=True)
        self.assertIsNone(store.get(head_key("lgstream", "wf:1", 0, "research#0")))
        self.assertEqual(writer.write_failures, 3)

    def test_a_full_buffer_never_drops_internal_markers(self):
        store = FakeLogStore()
        writer = _writer(store, max_buffered=5, flush_max_chunks=1)
        store.fail_saves = 10_000  # the store is down for the whole node
        for seq in range(20):
            writer.put(_event("research#0", seq))
        writer.put(_event("research#0", 20, UPDATES))
        writer.put(_event("review:branch:route", 0, BRANCH_DONE))
        store.fail_saves = 0
        writer.finish(done=True)
        chunk = LogReader(store, _config(), "wf:1").read([UnitWant(0, "research#0")])[0]
        modes = [e["mode"] for e in chunk.events]
        self.assertIn(UPDATES, modes)
        self.assertIn(BRANCH_DONE, modes)
        self.assertLessEqual(modes.count("messages"), 5)
        self.assertGreater(writer.dropped, 0)

    def test_a_new_attempt_takes_over_the_unit(self):
        store = FakeLogStore()
        first = _writer(store)
        first.put(_event("research#0", 0, attempt="a1"))
        first.finish(done=False)  # the node raised: Dapr retries it
        second = _writer(store)
        second.put(_event("research#0", 0, attempt="a2"))
        second.finish(done=True)
        head = store.get(head_key("lgstream", "wf:1", 0, "research#0"))
        self.assertEqual(head["attempt"], second.attempt)

    def test_a_done_unit_blocks_a_late_attempt(self):
        store = FakeLogStore()
        first = _writer(store)
        first.put(_event("research#0", 0))
        first.finish(done=True)
        late = _writer(store)
        late.put(_event("research#0", 0))
        late.finish(done=True)
        head = store.get(head_key("lgstream", "wf:1", 0, "research#0"))
        self.assertEqual(head["attempt"], first.attempt)

    def test_events_keep_their_identity_on_the_wire(self):
        event = StreamEvent(
            "wf",
            2,
            "research#1",
            "att",
            4,
            ("x",),
            "messages",
            "d",
            display_node="research",
            path=("__pregel_push", 1, False),
            branch=None,
        )
        wire = wire_event(event, lambda e: e.data)
        self.assertEqual(wire["node"], "research#1")
        self.assertEqual(wire["display_node"], "research")
        self.assertEqual(wire["path"], ["__pregel_push", 1, False])
        self.assertEqual(wire["step"], 2)


class TestReader(unittest.TestCase):
    def _written(self, store: FakeLogStore, n: int, flush: int = 2) -> LogWriter:
        writer = _writer(store, flush_max_chunks=flush)
        for seq in range(n):
            writer.put(_event("research#0", seq))
        writer.finish(done=True)
        return writer

    def test_reads_a_unit_in_order(self):
        store = FakeLogStore()
        writer = self._written(store, 5)
        chunk = LogReader(store, _config(), "wf:1").read([UnitWant(0, "research#0")])[0]
        self.assertEqual(_datas(chunk), [f"research#0-{i}" for i in range(5)])
        self.assertEqual(chunk.attempt, writer.attempt)
        self.assertTrue(chunk.done)
        self.assertFalse(chunk.reset)
        self.assertFalse(chunk.purged)
        self.assertEqual(chunk.batches, 3)

    def test_reads_only_what_is_new(self):
        store = FakeLogStore()
        writer = self._written(store, 5)
        reader = LogReader(store, _config(), "wf:1")
        chunk = reader.read(
            [UnitWant(0, "research#0", attempt=writer.attempt, from_batch=2)]
        )[0]
        self.assertEqual(_datas(chunk), ["research#0-4"])

    def test_resumes_inside_a_batch(self):
        store = FakeLogStore()
        writer = self._written(store, 5)
        reader = LogReader(store, _config(), "wf:1")
        chunk = reader.read(
            [UnitWant(0, "research#0", attempt=writer.attempt[:8], skip_events=3)]
        )[0]
        self.assertEqual(_datas(chunk), ["research#0-3", "research#0-4"])
        self.assertEqual(chunk.start, 3)

    def test_an_attempt_that_changed_is_read_from_its_start(self):
        store = FakeLogStore()
        self._written(store, 3)
        reader = LogReader(store, _config(), "wf:1")
        chunk = reader.read(
            [UnitWant(0, "research#0", attempt="oldattem", skip_events=2)]
        )[0]
        self.assertTrue(chunk.reset)
        self.assertEqual(len(chunk.events), 3)
        self.assertEqual(chunk.start, 0)

    def test_a_unit_with_no_head_has_nothing(self):
        chunk = LogReader(FakeLogStore(), _config(), "wf:1").read(
            [UnitWant(0, "nobody")]
        )[0]
        self.assertIsNone(chunk.attempt)
        self.assertEqual(chunk.events, [])

    def test_a_missing_batch_inside_the_count_is_purged(self):
        store = FakeLogStore()
        writer = self._written(store, 5)
        store.delete(batch_key("lgstream", "wf:1", 0, "research#0", writer.attempt, 1))
        chunk = LogReader(store, _config(), "wf:1").read([UnitWant(0, "research#0")])[0]
        self.assertTrue(chunk.purged)
        self.assertEqual(chunk.events, [])

    def test_a_store_that_fails_raises(self):
        store = FakeLogStore()
        self._written(store, 1)
        store.fail_gets = 1
        with self.assertRaises(ConnectionError):
            LogReader(store, _config(), "wf:1").read([UnitWant(0, "research#0")])

    def test_one_round_trip_for_heads_and_one_for_batches(self):
        store = FakeLogStore()
        self._written(store, 5)
        calls: List[Any] = []
        real = store.get_bulk
        store.get_bulk = lambda keys: calls.append(list(keys)) or real(keys)  # type: ignore[method-assign]
        LogReader(store, _config(), "wf:1").read(
            [UnitWant(0, "research#0"), UnitWant(0, "other")]
        )
        self.assertEqual(len(calls), 2)
