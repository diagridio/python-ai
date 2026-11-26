# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The SSE id of a stream read from the event log: a cursor any replica can resume."""

import unittest

from diagrid.agent.core.workflow.stream_cursor import (
    END_ID,
    START_ID,
    StreamCursor,
    encode_cursor,
    parse_cursor,
)


class TestStreamCursor(unittest.TestCase):
    def test_round_trip(self):
        cursor = StreamCursor(
            base=3,
            positions=(
                (3, "research#0", "a1b2c3d4", 7),
                (3, "review:branch:route", "ffff0000", 2),
            ),
        )
        text = encode_cursor(cursor)
        self.assertTrue(text.startswith("v1.3."))
        self.assertEqual(parse_cursor(text), cursor)
        self.assertEqual(cursor.position(3, "research#0"), ("a1b2c3d4", 7))
        self.assertIsNone(cursor.position(3, "other"))

    def test_negative_base_round_trips(self):
        cursor = StreamCursor(
            base=-1, positions=((-1, "__start__:branch:c", "abcd1234", 1),)
        )
        self.assertEqual(parse_cursor(encode_cursor(cursor)), cursor)

    def test_fixed_ids(self):
        self.assertEqual(parse_cursor(START_ID), StreamCursor())
        self.assertTrue(parse_cursor(END_ID).at_end)
        self.assertEqual(encode_cursor(StreamCursor()), START_ID)
        self.assertEqual(encode_cursor(StreamCursor(at_end=True)), END_ID)

    def test_missing_or_junk_replays_from_the_start(self):
        for junk in (
            None,
            "",
            "17",
            "v1",
            "v1.x.y",
            "v2.1.abc",
            "v1.1.!!!",
            "v1.1.e30",
        ):
            with self.subTest(junk=junk):
                self.assertEqual(parse_cursor(junk), StreamCursor())

    def test_the_start_skips_nothing(self):
        start = StreamCursor()
        self.assertFalse(start.skips_step(-1))
        self.assertFalse(start.skips_step(0))
        later = StreamCursor(base=2)
        self.assertTrue(later.skips_step(1))
        self.assertFalse(later.skips_step(2))

    def test_the_id_holds_only_open_steps(self):
        cursor = StreamCursor(base=5, positions=((5, "a", "12345678", 1),))
        # small: grows with one step's parallelism, not with the run
        self.assertLess(len(encode_cursor(cursor)), 80)
