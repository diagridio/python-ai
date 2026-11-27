# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A ``thread_id`` carries the graph's state from one run to the next.

With a checkpointer (the runner's, the graph's own, or by default the Dapr
state store the app has), the second run on a thread starts from the state
the first one ended with, as native LangGraph does with a checkpointer. The
thread's state is read and written by activities, never by workflow code.
"""

from typing import Any, List
from unittest import mock

from langgraph.checkpoint.memory import MemorySaver

from diagrid.agent.core.discovery import DiscoveredComponents
from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.state import DaprMemoryCheckpointer

from .golden_harness import _FakeDaprTest, _FakeWorkflowClient
from .streaming_helpers import _collect
from .thread_helpers import MemoryStateStore, chat_graph, shape, turn

HI = ("human", "hi")
AGAIN = ("human", "again")
ANSWER = ("ai", "Hello world")


def _runner(graph: Any, name: str, **kwargs: Any) -> DaprWorkflowGraphRunner:
    runner = DaprWorkflowGraphRunner(graph=graph, name=name, **kwargs)
    runner._started = True
    runner._workflow_client = _FakeWorkflowClient(runner)  # type: ignore[assignment]
    return runner


def _invoke(runner: DaprWorkflowGraphRunner, text: str, thread: str) -> Any:
    return runner.invoke(turn(text), thread_id=thread, poll_interval=0.01, timeout=10)


class TestThreadContinuation(_FakeDaprTest):
    def _dapr_runner(self, name: str) -> Any:
        store = MemoryStateStore()
        checkpointer = DaprMemoryCheckpointer(state_store=store)  # type: ignore[arg-type]
        return _runner(chat_graph(), name, checkpointer=checkpointer), store

    def test_two_invokes_on_a_thread_accumulate_messages(self):
        runner, _ = self._dapr_runner("thread-invoke")
        _invoke(runner, "hi", "t1")
        out = _invoke(runner, "again", "t1")
        self.assertEqual(shape(out["messages"]), [HI, ANSWER, AGAIN, ANSWER])

    async def test_two_streams_on_a_thread_accumulate_messages(self):
        runner, _ = self._dapr_runner("thread-astream")
        for text in ("hi", "again"):
            values = await _collect(
                runner.astream(
                    turn(text), thread_id="t1", stream_mode="values", poll_interval=0.01
                )
            )
        self.assertEqual(shape(values[-1]["messages"]), [HI, ANSWER, AGAIN, ANSWER])
        # the first values chunk is the thread's state with the new input, as native
        self.assertEqual(shape(values[0]["messages"]), [HI, ANSWER, AGAIN])

    def test_threads_do_not_share_state(self):
        runner, _ = self._dapr_runner("thread-apart")
        _invoke(runner, "hi", "t1")
        out = _invoke(runner, "again", "t2")
        self.assertEqual(shape(out["messages"]), [AGAIN, ANSWER])

    def test_the_store_keys_use_colons_and_escape_the_thread_id(self):
        runner, store = self._dapr_runner("thread-keys")
        _invoke(runner, "hi", "a:b|c#d")
        self.assertTrue(store.data)
        for key in store.data:
            self.assertTrue(key.startswith("langgraph:a%3Ab%7Cc%23d:"), key)
            self.assertNotIn("||", key)

    def test_without_a_checkpointer_or_store_a_thread_is_a_label(self):
        runner = _runner(chat_graph(), "thread-none")
        _invoke(runner, "hi", "t1")
        out = _invoke(runner, "again", "t1")
        self.assertEqual(shape(out["messages"]), [AGAIN, ANSWER])

    def test_a_graph_compiled_with_a_checkpointer_continues_its_threads(self):
        runner = _runner(chat_graph(MemorySaver()), "thread-saver")
        _invoke(runner, "hi", "t1")
        out = _invoke(runner, "again", "t1")
        self.assertEqual(shape(out["messages"]), [HI, ANSWER, AGAIN, ANSWER])

    def test_the_apps_state_store_is_used_by_default(self):
        found = DiscoveredComponents(memory_store_name="agent-memory")
        with (
            mock.patch(
                "diagrid.agent.core.workflow.runner.discover_components",
                return_value=found,
            ),
            mock.patch("diagrid.agent.core.state.DaprStateStore", MemoryStateStore),
        ):
            runner = _runner(chat_graph(), "thread-default")
        _invoke(runner, "hi", "t1")
        out = _invoke(runner, "again", "t1")
        self.assertEqual(shape(out["messages"]), [HI, ANSWER, AGAIN, ANSWER])

    def test_matches_native_memory_saver_over_two_turns(self):
        native_graph = chat_graph(MemorySaver())
        config = {"configurable": {"thread_id": "t1"}}
        native_graph.invoke(turn("hi"), config)
        native = native_graph.invoke(turn("again"), config)

        runner, _ = self._dapr_runner("thread-golden")
        _invoke(runner, "hi", "t1")
        ours = _invoke(runner, "again", "t1")
        self.assertEqual(set(ours), set(native))
        self.assertEqual(shape(ours["messages"]), shape(native["messages"]))
        # every message kept its own id across the turns, and no id repeats
        ids: List[str] = [m["id"] for m in ours["messages"]]
        self.assertEqual(len(set(ids)), 4)
