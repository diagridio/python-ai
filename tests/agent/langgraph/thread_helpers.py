# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Helpers for the thread continuation tests: an in-memory state store and graphs."""

import copy
import json
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, MessagesState, StateGraph

from .streaming_helpers import _StreamingModel


class MemoryStateStore:
    """A ``DaprStateStore`` stand-in: values cross JSON as they do over gRPC.

    ``guard`` is called on every access, so a test can check where the store
    is touched from.
    """

    def __init__(self, store_name: str = "agent-memory", **_: Any) -> None:
        self.store_name = store_name
        self.data: Dict[str, Any] = {}
        self.guard: Optional[Callable[[], None]] = None

    def _check(self) -> None:
        if self.guard is not None:
            self.guard()

    def save(self, key: str, value: Any, metadata: Any = None) -> None:
        self._check()
        self.data[key] = json.loads(json.dumps(value))

    def get(self, key: str) -> Any:
        self._check()
        return copy.deepcopy(self.data.get(key))

    def delete(self, key: str) -> None:
        self._check()
        self.data.pop(key, None)

    def close(self) -> None:
        pass


def chat_graph(checkpointer: Any = None) -> Any:
    """One chat node that answers every turn: the messages accumulate per thread."""
    model = _StreamingModel()

    def chat(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer)


def turn(text: str) -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def shape(messages: Any) -> list:
    """``(type, content)`` of each message, dict or object: ids differ by design."""
    out = []
    for m in messages:
        if isinstance(m, dict):
            out.append((m.get("type"), m.get("content")))
        else:
            out.append((m.type, m.content))
    return out
