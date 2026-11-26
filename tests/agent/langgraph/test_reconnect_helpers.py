# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Reading SSE bodies in the reconnect and interrupt tests."""

import json
from typing import Any, List, Tuple


def frames(text: str) -> List[Tuple[str, str, Any]]:
    """``(id, event, data)`` of each SSE frame (comments left out)."""
    out = []
    for frame in text.strip().split("\n\n"):
        if frame.startswith(":") or not frame:
            continue
        fields = dict(line.split(": ", 1) for line in frame.split("\n"))
        out.append((fields.get("id"), fields["event"], json.loads(fields["data"])))
    return out
