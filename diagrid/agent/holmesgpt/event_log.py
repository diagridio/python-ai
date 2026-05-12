# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Per-instance event tape backed by a Dapr state store.

The investigation workflow allocates a monotonic ``seq`` for each emitted
event and hands the seq to activities as part of their input. Activities
write events to keys of the form ``{key_prefix}:{instance_id}:{seq:010d}``
with optional TTL. Readers (typically an SSE handler) pull events with
``read_after`` using bulk reads — the first missing key marks the end of
the currently-available tape.

The store name and key prefix are runtime-configurable so that callers
can collocate the event tape with any state store component they already
operate (Redis, Postgres, Cosmos, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from dapr.clients import DaprClient

logger = logging.getLogger(__name__)

# By default we collocate the polling tape on Dapr's actor state store
# (the one configured for workflow durability) so the two components
# share a single failure domain — if the workflow can make progress, the
# tape can record progress. ``statestore`` is the conventional name from
# ``dapr init``; override via the ``HOLMES_EVENTS_STORE`` env var or the
# ``DaprWorkflowHolmesRunner(events_store_name=...)`` argument.
DEFAULT_STORE_NAME = os.environ.get("HOLMES_EVENTS_STORE", "statestore")
DEFAULT_KEY_PREFIX = os.environ.get("HOLMES_EVENTS_PREFIX", "holmes.stream")
DEFAULT_TTL_SECONDS = int(os.environ.get("HOLMES_EVENTS_TTL", "86400"))


def _key(prefix: str, instance_id: str, seq: int) -> str:
    return f"{prefix}:{instance_id}:{seq:010d}"


def save_record(
    instance_id: str,
    seq: int,
    event: str,
    data: Dict[str, Any],
    *,
    store_name: str = DEFAULT_STORE_NAME,
    key_prefix: str = DEFAULT_KEY_PREFIX,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """Append a single event record to the per-instance tape.

    Idempotent on (instance_id, seq): replays write the same key with the
    same value, which is a no-op as far as readers are concerned.

    **Failure semantics**: this function deliberately does NOT swallow
    state-store errors. If the underlying ``save_state`` raises (Dapr
    unreachable, store down, transient network blip, …), the exception
    propagates through the calling activity and Dapr's retry policy
    re-runs the activity. Because seq allocation is deterministic in the
    workflow, retries write the same key with the same value — idempotent.
    """
    payload = json.dumps(
        {
            "seq": seq,
            "ts": time.time(),
            "event": event,
            "data": data,
        }
    )
    options: Optional[Dict[str, str]] = None
    if ttl_seconds > 0:
        options = {"ttlInSeconds": str(ttl_seconds)}

    with DaprClient() as client:
        client.save_state(
            store_name=store_name,
            key=_key(key_prefix, instance_id, seq),
            value=payload,
            state_metadata=options,
        )
    logger.debug(
        "save_record store=%s instance=%s seq=%d event=%s",
        store_name,
        instance_id,
        seq,
        event,
    )


def read_after(
    instance_id: str,
    since_seq: int = 0,
    limit: int = 64,
    *,
    store_name: str = DEFAULT_STORE_NAME,
    key_prefix: str = DEFAULT_KEY_PREFIX,
) -> List[Dict[str, Any]]:
    """Read events with ``seq > since_seq``, in seq order, until first gap.

    Because the workflow allocates seqs densely and writes are idempotent,
    encountering a missing key safely indicates the end of currently-
    available events for this instance.
    """
    keys = [_key(key_prefix, instance_id, since_seq + i + 1) for i in range(limit)]
    with DaprClient() as client:
        items = client.get_bulk_state(store_name=store_name, keys=keys)

    out: List[Dict[str, Any]] = []
    # Items may come back in arbitrary order; index by key to walk in seq order.
    # The Dapr SDK exposes ``data`` as ``bytes | str`` depending on the build.
    by_key: Dict[str, "bytes | str"] = {}
    for it in items.items:
        if it.data:
            by_key[it.key] = it.data

    for key in keys:
        raw = by_key.get(key)
        if not raw:
            break
        try:
            out.append(json.loads(raw))
        except (ValueError, TypeError) as e:
            logger.warning("Skipping malformed event at key=%s err=%s", key, e)
            break
    return out
