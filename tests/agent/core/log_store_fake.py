# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""An in-memory state store with the bulk API the stream event log uses."""

import copy
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Item = Tuple[str, Any, Dict[str, str]]


class FakeLogStore:
    """Keys to JSON values, with the metadata each save carried.

    ``fail_saves`` / ``fail_gets`` make the next calls raise, as a store that
    is down would.

    A Dapr bulk save is neither atomic nor ordered (components-contrib runs
    the sets in parallel), and these simulate that: ``bulk_order`` lands a
    save's items one by one in the order it returns, ``after_item`` runs after
    each item landed (a reader racing the save), and ``partial_saves`` makes
    the next save land only that many of its items, then raise.
    """

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, str]] = {}
        self.saves: List[List[str]] = []
        self.fail_saves = 0
        self.fail_gets = 0
        self.bulk_order: Optional[Callable[[List[Item]], List[Item]]] = None
        self.after_item: Optional[Callable[[str], None]] = None
        self.partial_saves: Optional[int] = None
        self._lock = threading.RLock()

    def save_bulk(self, items: Sequence[Item]) -> None:
        with self._lock:
            if self.fail_saves:
                self.fail_saves -= 1
                raise ConnectionError("store down")
            self.saves.append([key for key, _, _ in items])
            ordered = list(items)
            if self.bulk_order is not None:
                ordered = self.bulk_order(ordered)
            landing, self.partial_saves = self.partial_saves, None
        for n, (key, value, metadata) in enumerate(ordered):
            if landing is not None and n >= landing:
                raise ConnectionError(f"bulk save: {key!r} failed")
            with self._lock:
                # crosses JSON, as a real store does
                self.data[key] = json.loads(json.dumps(value))
                self.metadata[key] = dict(metadata)
            if self.after_item is not None:
                self.after_item(key)

    def get_bulk(self, keys: Sequence[str]) -> Dict[str, Any]:
        with self._lock:
            if self.fail_gets:
                self.fail_gets -= 1
                raise ConnectionError("store down")
            return {k: copy.deepcopy(self.data[k]) for k in keys if k in self.data}

    def delete(self, key: str) -> None:
        with self._lock:
            self.data.pop(key, None)

    def keys_like(self, part: str) -> List[str]:
        return sorted(k for k in self.data if part in k)

    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)
