# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""An in-memory state store with the bulk API the stream event log uses."""

import copy
import json
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple


class FakeLogStore:
    """Keys to JSON values, with the metadata each save carried.

    ``fail_saves`` / ``fail_gets`` make the next calls raise, as a store that
    is down would.
    """

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, str]] = {}
        self.saves: List[List[str]] = []
        self.fail_saves = 0
        self.fail_gets = 0
        self._lock = threading.Lock()

    def save_bulk(self, items: Sequence[Tuple[str, Any, Dict[str, str]]]) -> None:
        with self._lock:
            if self.fail_saves:
                self.fail_saves -= 1
                raise ConnectionError("store down")
            self.saves.append([key for key, _, _ in items])
            for key, value, metadata in items:
                # crosses JSON, as a real store does
                self.data[key] = json.loads(json.dumps(value))
                self.metadata[key] = dict(metadata)

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
