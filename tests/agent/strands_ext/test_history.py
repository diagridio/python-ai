# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for conversation history in the Strands runner."""

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("strands")

from strands.agent.conversation_manager import (  # noqa: E402
    NullConversationManager,
    SlidingWindowConversationManager,
)

from diagrid.agent.strands.history import (  # noqa: E402
    DEFAULT_WINDOW_SIZE,
    HistoryStore,
    history_key,
    resolve_window_size,
    trim_history,
)
from diagrid.agent.strands.runner import DaprWorkflowAgentRunner  # noqa: E402


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"text": text}]}


def _assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"text": text}]}


def _tool_use(tool_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_id, "name": "t", "input": {}}}],
    }


def _tool_result(tool_id: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": tool_id,
                    "status": "success",
                    "content": [{"text": "ok"}],
                }
            }
        ],
    }


class FakeModel:
    """Replies with how many messages it received."""

    def __init__(self) -> None:
        self.seen: list[list[Any]] = []

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.seen.append(list(messages))
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": f"saw {len(messages)}"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


class FakeStateStore:
    store_name = "agent-memory"

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def save(self, key: str, value: Any, metadata: Any = None) -> None:
        self.data[key] = value

    def close(self) -> None:
        pass


def _make_runner(
    name: str = "chat",
    *,
    state_store: Any = None,
    conversation_manager: Any = None,
    model: Any = None,
) -> tuple[DaprWorkflowAgentRunner, dict[str, Any], FakeModel]:
    model = model or FakeModel()
    agent = SimpleNamespace(
        model=model,
        system_prompt="",
        tool_registry=SimpleNamespace(registry={}),
        conversation_manager=conversation_manager,
        messages=[],
    )
    registered: dict[str, Any] = {}
    runtime = MagicMock()
    runtime.register_workflow.side_effect = lambda fn, name=None: registered.update(
        workflow=fn
    )
    runtime.register_activity.side_effect = lambda fn, name=None: registered.update(
        {name: fn}
    )
    with (
        patch(
            "diagrid.agent.core.workflow.runner.WorkflowRuntime", return_value=runtime
        ),
        patch(
            "diagrid.agent.core.workflow.runner.discover_components",
            return_value=SimpleNamespace(runtime_conf=None, memory_store_name=None),
        ),
        patch.object(DaprWorkflowAgentRunner, "_register_agent_metadata"),
    ):
        runner = DaprWorkflowAgentRunner(
            agent=agent,
            name=name,
            state_store=state_store,  # type: ignore[arg-type]
        )
    return runner, registered, model


def _drive(registered: dict[str, Any], task: str, session_id: str) -> dict[str, Any]:
    """Run the workflow generator, executing each activity inline."""
    ctx = SimpleNamespace(call_activity=lambda fn, input=None: (fn, input))
    gen = registered["workflow"](ctx, {"task": task, "session_id": session_id})
    try:
        step = next(gen)
        while True:
            fn, activity_input = step
            step = gen.send(fn(None, activity_input))
    except StopIteration as done:
        return done.value


def _drive_all(runner: DaprWorkflowAgentRunner, registered: dict[str, Any], *calls):
    return [_drive(registered, task, sid)["result"] for task, sid in calls]


class TestTrimHistory:
    def test_keeps_everything_under_window(self):
        msgs = [_user("a"), _assistant("b")]
        assert trim_history(msgs, 10) == msgs

    def test_cuts_to_window_and_starts_on_user(self):
        msgs = [_user("1"), _assistant("1"), _user("2"), _assistant("2")]
        # Window of 3 starts on an assistant message, which is dropped.
        assert trim_history(msgs, 3) == [_user("2"), _assistant("2")]

    def test_drops_orphan_tool_result(self):
        msgs = [
            _user("q"),
            _tool_use("a"),
            _tool_result("a"),
            _assistant("done"),
            _user("next"),
            _assistant("ok"),
        ]
        # Window of 5 would start at the tool_use; the next clean user turn is "next".
        assert trim_history(msgs, 4) == [_user("next"), _assistant("ok")]
        trimmed = trim_history(msgs, 5)
        assert trimmed[0] == _user("next")

    def test_returns_empty_when_no_clean_user_message(self):
        msgs = [_assistant("x"), _tool_use("a"), _tool_result("a")]
        assert trim_history(msgs, 3) == []

    def test_does_not_mutate_input(self):
        msgs = [_user("1"), _assistant("1"), _user("2")]
        snapshot = list(msgs)
        trim_history(msgs, 1)
        assert msgs == snapshot


class TestWindowSize:
    def test_default_without_manager(self):
        assert resolve_window_size(SimpleNamespace()) == DEFAULT_WINDOW_SIZE

    def test_sliding_window_size_is_used(self):
        agent = SimpleNamespace(
            conversation_manager=SlidingWindowConversationManager(window_size=6)
        )
        assert resolve_window_size(agent) == 6

    def test_other_manager_uses_default(self):
        agent = SimpleNamespace(conversation_manager=NullConversationManager())
        assert resolve_window_size(agent) == DEFAULT_WINDOW_SIZE


class TestHistoryStore:
    def test_key_includes_runner_and_session(self):
        assert history_key("chat", "s1") == "strands-chat-s1-messages"

    def test_state_store_round_trip(self):
        store = FakeStateStore()
        history = HistoryStore("chat", store)
        assert history.load("s1") == []
        history.save("s1", [_user("hi")])
        assert store.data["strands-chat-s1-messages"] == [_user("hi")]
        assert history.load("s1") == [_user("hi")]

    def test_in_process_fallback_warns_once(self, caplog):
        history = HistoryStore("chat")
        with caplog.at_level(logging.WARNING):
            history.load("s1")
            history.save("s1", [_user("hi")])
            history.load("s1")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert history.load("s1") == [_user("hi")]

    def test_state_store_does_not_warn(self, caplog):
        history = HistoryStore("chat", FakeStateStore())
        with caplog.at_level(logging.WARNING):
            history.load("s1")
        assert not caplog.records


class TestRunnerHistory:
    def test_second_call_sees_first_turn(self):
        runner, registered, model = _make_runner()
        first, second = _drive_all(runner, registered, ("hi", "s1"), ("again", "s1"))
        assert first == "saw 1"
        # user, assistant, user
        assert second == "saw 3"

    def test_new_session_starts_empty(self):
        runner, registered, _ = _make_runner()
        results = _drive_all(runner, registered, ("a", "s1"), ("b", "s1"), ("c", "s2"))
        assert results == ["saw 1", "saw 3", "saw 1"]

    def test_history_survives_runner_restart_with_state_store(self):
        store = FakeStateStore()
        runner1, reg1, _ = _make_runner(state_store=store)
        _drive(reg1, "hi", "s1")
        runner2, reg2, _ = _make_runner(state_store=store)
        assert _drive(reg2, "again", "s1")["result"] == "saw 3"

    def test_runner_names_do_not_share_history(self):
        store = FakeStateStore()
        _, reg_a, _ = _make_runner("agent-a", state_store=store)
        _, reg_b, _ = _make_runner("agent-b", state_store=store)
        _drive(reg_a, "hi", "shared")
        assert _drive(reg_b, "hi", "shared")["result"] == "saw 1"
        assert set(store.data) == {
            "strands-agent-a-shared-messages",
            "strands-agent-b-shared-messages",
        }

    def test_history_is_trimmed_to_window(self):
        runner, registered, _ = _make_runner(
            conversation_manager=SlidingWindowConversationManager(window_size=4)
        )
        results = _drive_all(
            runner,
            registered,
            ("1", "s"),
            ("2", "s"),
            ("3", "s"),
            ("4", "s"),
        )
        # Saved history is capped at 4 messages, so the model sees at most 5.
        assert results == ["saw 1", "saw 3", "saw 5", "saw 5"]

    def test_agent_messages_are_not_touched(self):
        runner, registered, _ = _make_runner()
        _drive(registered, "hi", "s1")
        assert runner.agent.messages == []

    def test_failed_model_call_saves_nothing(self):
        class BrokenModel:
            async def stream(self, **kwargs):
                raise RuntimeError("boom")
                yield  # pragma: no cover

        store = FakeStateStore()
        _, registered, _ = _make_runner(state_store=store, model=BrokenModel())
        out = _drive(registered, "hi", "s1")
        assert out["result"].startswith("Error:")
        assert store.data == {}

    def test_session_id_is_in_workflow_input(self):
        runner, _, _ = _make_runner()
        runner._started = True
        client = MagicMock()
        runner._workflow_client = client
        runner._poll_workflow = MagicMock()  # type: ignore[method-assign]

        async def _empty(*args, **kwargs):
            return
            yield  # pragma: no cover

        runner._poll_workflow = _empty  # type: ignore[method-assign]

        import asyncio

        async def _run() -> None:
            async for _ in runner.run_async("task", "s-9", workflow_id="wf"):
                pass

        asyncio.run(_run())
        sent = client.schedule_new_workflow.call_args.kwargs["input"]
        assert sent == {"task": "task", "session_id": "s-9"}


class _ToolOnceModel:
    """Calls a tool on every new user text turn, then answers in text."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        last = messages[-1]
        is_tool_turn = any("toolResult" in b for b in last["content"])
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        if is_tool_turn:
            yield {"contentBlockDelta": {"delta": {"text": "done"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            return
        yield {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": f"id{len(messages)}", "name": "t"}}
            }
        }
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


class TestToolTurnsInHistory:
    def test_saved_history_has_no_orphan_tool_result(self):
        store = FakeStateStore()
        runner, registered, _ = _make_runner(
            state_store=store,
            conversation_manager=SlidingWindowConversationManager(window_size=6),
            model=_ToolOnceModel(),
        )

        class _Tool:
            tool_spec = {"name": "t", "description": "d", "inputSchema": {"json": {}}}

            async def stream(self, tool_use, invocation_state):
                yield {
                    "toolUseId": tool_use["toolUseId"],
                    "status": "success",
                    "content": [{"text": "ok"}],
                }

        runner.agent.tool_registry.registry["t"] = _Tool()
        for turn in ("one", "two", "three"):
            out = _drive(registered, turn, "s")
            assert out["tool_calls"], out

        saved = store.data["strands-chat-s-messages"]
        assert len(saved) <= 6
        assert saved[0]["role"] == "user"
        assert not any("toolResult" in b for b in saved[0]["content"])
        use_ids = {
            b["toolUse"]["toolUseId"]
            for m in saved
            for b in m["content"]
            if "toolUse" in b
        }
        result_ids = {
            b["toolResult"]["toolUseId"]
            for m in saved
            for b in m["content"]
            if "toolResult" in b
        }
        assert result_ids <= use_ids
