# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the polling event tape backed by Dapr state store."""

import json
from types import SimpleNamespace
from unittest import mock

from diagrid.agent.holmesgpt import event_log


def _make_fake_client(saved: dict, bulk_get_handler=None):
    """Build a DaprClient stand-in that records save_state and serves bulk reads."""

    fake = mock.MagicMock()
    fake.__enter__ = mock.MagicMock(return_value=fake)
    fake.__exit__ = mock.MagicMock(return_value=False)

    def _save_state(*, store_name, key, value, state_metadata=None):
        saved[key] = (value, state_metadata)

    fake.save_state.side_effect = _save_state

    if bulk_get_handler is not None:
        fake.get_bulk_state.side_effect = bulk_get_handler

    return fake


def test_save_record_writes_to_state_store_with_ttl():
    saved: dict[str, tuple] = {}
    fake = _make_fake_client(saved)

    with mock.patch("diagrid.agent.holmesgpt.event_log.DaprClient", return_value=fake):
        event_log.save_record(
            instance_id="wf-1",
            seq=3,
            event="iteration_started",
            data={"message_count": 5},
            store_name="my-store",
            key_prefix="holmes.stream",
            ttl_seconds=120,
        )

    expected_key = "holmes.stream:wf-1:0000000003"
    assert expected_key in saved
    value, meta = saved[expected_key]
    payload = json.loads(value)
    assert payload["seq"] == 3
    assert payload["event"] == "iteration_started"
    assert payload["data"] == {"message_count": 5}
    assert meta == {"ttlInSeconds": "120"}


def test_save_record_omits_ttl_when_disabled():
    saved: dict[str, tuple] = {}
    fake = _make_fake_client(saved)

    with mock.patch("diagrid.agent.holmesgpt.event_log.DaprClient", return_value=fake):
        event_log.save_record(
            instance_id="wf-1",
            seq=1,
            event="x",
            data={},
            store_name="s",
            key_prefix="p",
            ttl_seconds=0,
        )

    _, meta = next(iter(saved.values()))
    assert meta is None


def test_read_after_returns_dense_prefix_and_stops_at_first_gap():
    """Tape readers must walk in seq order and stop at the first missing key."""
    seqs_available = {1, 2, 3, 5}  # gap at 4
    prefix = "holmes.stream"
    instance_id = "wf-1"

    def _bulk_get(*, store_name, keys):
        items = []
        for key in keys:
            seq = int(key.split(":")[-1])
            if seq in seqs_available:
                payload = json.dumps(
                    {"seq": seq, "ts": 1.0, "event": "x", "data": {"i": seq}}
                ).encode()
                items.append(SimpleNamespace(key=key, data=payload))
            else:
                items.append(SimpleNamespace(key=key, data=b""))
        return SimpleNamespace(items=items)

    fake = _make_fake_client({}, bulk_get_handler=_bulk_get)
    with mock.patch("diagrid.agent.holmesgpt.event_log.DaprClient", return_value=fake):
        events = event_log.read_after(
            instance_id=instance_id,
            since_seq=0,
            limit=10,
            store_name="s",
            key_prefix=prefix,
        )

    assert [e["seq"] for e in events] == [1, 2, 3]
    assert events[0]["data"] == {"i": 1}


def test_read_after_respects_since_seq():
    requested_keys: list[list[str]] = []

    def _bulk_get(*, store_name, keys):
        requested_keys.append(list(keys))
        return SimpleNamespace(items=[SimpleNamespace(key=k, data=b"") for k in keys])

    fake = _make_fake_client({}, bulk_get_handler=_bulk_get)
    with mock.patch("diagrid.agent.holmesgpt.event_log.DaprClient", return_value=fake):
        event_log.read_after(
            instance_id="wf-1",
            since_seq=5,
            limit=3,
            store_name="s",
            key_prefix="p",
        )

    assert requested_keys[0] == [
        "p:wf-1:0000000006",
        "p:wf-1:0000000007",
        "p:wf-1:0000000008",
    ]


def test_read_after_skips_malformed_entries():
    """Malformed JSON in a tape entry should stop the read, not raise."""
    items = [
        SimpleNamespace(
            key="p:wf-1:0000000001",
            data=json.dumps({"seq": 1, "event": "ok", "data": {}}).encode(),
        ),
        SimpleNamespace(key="p:wf-1:0000000002", data=b"not-json"),
    ]
    fake = _make_fake_client(
        {}, bulk_get_handler=lambda **_: SimpleNamespace(items=items)
    )
    with mock.patch("diagrid.agent.holmesgpt.event_log.DaprClient", return_value=fake):
        events = event_log.read_after(
            instance_id="wf-1",
            since_seq=0,
            limit=2,
            store_name="s",
            key_prefix="p",
        )

    assert len(events) == 1
    assert events[0]["seq"] == 1
