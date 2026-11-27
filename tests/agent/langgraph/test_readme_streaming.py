# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The streaming README says how to reconnect, which transport is used, and how to resume."""

import unittest
from pathlib import Path

from diagrid.agent.langgraph import streaming

README = Path(streaming.__file__).with_name("README.md")


class TestStreamingReadme(unittest.TestCase):
    def setUp(self) -> None:
        self.text = README.read_text()
        self.flat = " ".join(self.text.split())

    def _section(self, heading: str) -> str:
        start = self.text.index(heading)
        end = self.text.find("\n#", start + len(heading))
        return self.text[start : end if end != -1 else None]

    def test_reconnect_replaces_the_poll_limit(self):
        self.assertNotIn("A dropped connection cannot be resumed", self.text)
        section = " ".join(self._section("#### Reconnecting").split())
        for needed in (
            "GET /agent/run/{workflow_id}/stream",
            "Last-Event-ID",
            "last_event_id",
            "astream_resume",
            "404",
            "transport_not_replayable",
        ):
            self.assertIn(needed, section, needed)

    def test_the_transport_choice_is_documented(self):
        section = " ".join(self._section("#### Streaming across replicas").split())
        for needed in (
            "StateLogTransport",
            "InProcessTransport",
            "state store",
            "stream_transport",
            "write_failed",
        ):
            self.assertIn(needed, section, needed)
        self.assertNotIn("Delivery across replicas is not implemented yet", self.text)

    def test_sse_ids_are_documented(self):
        for needed in ("v1.start", "v1.end", "cursor"):
            self.assertIn(needed, self.flat, needed)

    def test_interrupt_and_resume_are_documented(self):
        section = " ".join(self._section("### Interrupts").split())
        for needed in (
            "interrupt(",
            "Command(resume=",
            "POST /agent/run/{workflow_id}/resume",
            "workflow_interrupted",
            "__interrupt__",
            "409",
        ):
            self.assertIn(needed, section, needed)

    def test_the_proxy_wording_is_kept(self):
        self.assertIn("#### Behind a proxy", self.text)
        self.assertNotIn("behind a load balancer", self.text.lower())

    def test_the_recorded_modes_are_documented(self):
        for needed in (
            "A streamed run records only what it was asked for.",
            "A mode the run did not record",
            "A resumed run keeps the stream modes it was started with",
        ):
            self.assertIn(needed, self.flat, needed)
