# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""How a state value is stored in the workflow's channel state.

The channel state crosses JSON (it is a workflow input and an activity result),
yet a node expects the types it wrote: a message object in a list, a ``set``, a
``datetime``. A value that is plain JSON is stored as it is. Anything else is
stored as the checkpoint serializer writes it (LangGraph's own
``JsonPlusSerializer``), wrapped as ``{"__typed__": [kind, base64]}``, and is
decoded when a node, a router or a reducer reads it.
"""

import base64
import math
from typing import Any

TYPED = "__typed__"


def is_plain_json(value: Any) -> bool:
    """Whether ``value`` is what JSON gives back unchanged (no tuple, set, object)."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(is_plain_json(v) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and k != TYPED and is_plain_json(v)
            for k, v in value.items()
        )
    return False


def is_encoded(value: Any) -> bool:
    """Whether ``value`` is a typed envelope made by :func:`encode_value`."""
    return isinstance(value, dict) and len(value) == 1 and TYPED in value


def encode_value(value: Any) -> Any:
    """``value`` as JSON: itself when plain, else a typed envelope.

    A value the serializer cannot write is left to the caller's fallback: the
    raw value comes back (and is made JSON by it).
    """
    if is_plain_json(value):
        return value
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        kind, data = JsonPlusSerializer().dumps_typed(value)
    except Exception:
        return value
    return {TYPED: [kind, base64.b64encode(data).decode("ascii")]}


def decode_value(value: Any) -> Any:
    """What :func:`encode_value` was given (anything else is returned as is)."""
    if not is_encoded(value):
        return value
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    kind, data = value[TYPED]
    return JsonPlusSerializer().loads_typed((kind, base64.b64decode(data)))


def coerce_to_default(value: Any, default: Any) -> Any:
    """``value`` as the type of the channel's ``default`` when JSON blurred them.

    A tuple comes back from the serializer as a list, so a ``tuple`` channel is
    given its tuple again (a ``set`` or ``frozenset`` survives, but a list that
    was stored for one is turned back too).
    """
    if type(default) in (tuple, set, frozenset) and isinstance(
        value, (list, tuple, set, frozenset)
    ):
        return type(default)(value)
    return value
