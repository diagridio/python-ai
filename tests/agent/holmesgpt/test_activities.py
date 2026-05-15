# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Activity-level tests: call_llm, invoke_tool, record_event.

Each activity is exercised in isolation with a stubbed ``HolmesRegistry``;
the activities' contract with the workflow (input/output shape, event
emissions) is verified.
"""

from __future__ import annotations

from diagrid.agent.holmesgpt.models import (
    LLMCallInput,
    LLMCallOutput,
    RecordEventInput,
    ToolCallInput,
    ToolCallOutput,
)
from diagrid.agent.holmesgpt.workflow import (
    call_llm_activity,
    invoke_tool_activity,
    record_event_activity,
)

from .conftest import StubTool, make_llm_response


# ---------------------------------------------------------------------------
# record_event_activity
# ---------------------------------------------------------------------------


def test_record_event_activity_writes_to_tape(patch_event_log_record):
    payload = RecordEventInput(
        instance_id="wf-1", seq=7, event="approval_required", data={"tcid": "abc"}
    ).model_dump()
    out = record_event_activity(None, payload)
    assert out == {"seq": 7, "event": "approval_required"}
    assert patch_event_log_record == [
        {
            "instance_id": "wf-1",
            "seq": 7,
            "event": "approval_required",
            "data": {"tcid": "abc"},
        }
    ]


# ---------------------------------------------------------------------------
# call_llm_activity
# ---------------------------------------------------------------------------


def test_call_llm_activity_returns_final_answer(
    install_stub_registry, patch_event_log_record
):
    install_stub_registry(
        completions=[
            make_llm_response(
                content="all good",
                tool_calls=None,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            )
        ]
    )
    payload = LLMCallInput(
        instance_id="wf-1",
        seq_base=10,
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        tool_choice="auto",
    ).model_dump()

    raw = call_llm_activity(None, payload)
    out = LLMCallOutput.model_validate(raw)
    assert out.assistant_message["content"] == "all good"
    assert out.tool_calls == []
    assert out.finish_reason == "stop"
    assert out.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }

    # Activity emits exactly 3 tape events at seq_base, +1, +2.
    assert [e["event"] for e in patch_event_log_record] == [
        "iteration_started",
        "conversation_history_compaction_status",
        "iteration_completed",
    ]
    assert [e["seq"] for e in patch_event_log_record] == [10, 11, 12]


def test_call_llm_activity_parses_tool_calls(
    install_stub_registry, patch_event_log_record
):
    install_stub_registry(
        completions=[
            make_llm_response(
                content="calling tools",
                tool_calls=[
                    {"id": "t1", "name": "bash", "arguments": {"command": "ls"}},
                    {"id": "t2", "name": "bash", "arguments": {"command": "pwd"}},
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    payload = LLMCallInput(
        instance_id="wf-1",
        seq_base=20,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "bash"}}],
    ).model_dump()

    out = LLMCallOutput.model_validate(call_llm_activity(None, payload))
    assert len(out.tool_calls) == 2
    assert out.tool_calls[0] == {
        "id": "t1",
        "name": "bash",
        "arguments": {"command": "ls"},
    }
    assert out.tool_calls[1] == {
        "id": "t2",
        "name": "bash",
        "arguments": {"command": "pwd"},
    }
    assert out.finish_reason == "tool_calls"


def test_call_llm_activity_handles_unparseable_arguments(
    install_stub_registry, patch_event_log_record
):
    """If the LLM emits non-JSON arguments, we fall back to ``{"_raw": ...}``."""
    import types

    # Build a response with raw, broken JSON in the arguments field.
    tc = types.SimpleNamespace(
        id="t1",
        function=types.SimpleNamespace(name="bash", arguments="not-json{"),
    )
    msg = types.SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[tc],
        model_dump=lambda exclude_none=False: {"role": "assistant", "tool_calls": []},
    )
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=None,
        id="r",
    )
    install_stub_registry(completions=[response])
    payload = LLMCallInput(
        instance_id="wf-1",
        seq_base=1,
        messages=[],
        tools=[{"x": 1}],
    ).model_dump()

    out = LLMCallOutput.model_validate(call_llm_activity(None, payload))
    assert out.tool_calls == [
        {"id": "t1", "name": "bash", "arguments": {"_raw": "not-json{"}}
    ]


# ---------------------------------------------------------------------------
# invoke_tool_activity
# ---------------------------------------------------------------------------


def _build_tool_result(status, *, data=None, error=None, invocation=None):
    """Build a real HolmesGPT ``StructuredToolResult`` for fidelity."""
    from holmes.core.tools import StructuredToolResult, StructuredToolResultStatus

    status_enum = (
        StructuredToolResultStatus(status) if isinstance(status, str) else status
    )
    return StructuredToolResult(
        status=status_enum,
        data=data,
        error=error,
        invocation=invocation,
        elapsed_seconds=0.0,
    )


def test_invoke_tool_activity_success(install_stub_registry, patch_event_log_record):
    tool = StubTool(
        "bash",
        results=[
            _build_tool_result(
                "success", data="hello world", invocation="echo hello world"
            )
        ],
    )
    install_stub_registry(tools={"bash": tool})
    payload = ToolCallInput(
        instance_id="wf-1",
        seq_base=30,
        tool_call_id="tc-1",
        tool_name="bash",
        arguments={"command": "echo hello world"},
    ).model_dump()

    out = ToolCallOutput.model_validate(invoke_tool_activity(None, payload))
    assert out.status == "success"
    assert out.data_str == "hello world"
    assert out.invocation == "echo hello world"
    assert tool.invocations[0]["params"] == {"command": "echo hello world"}

    # Activity emits start + result at seq_base / seq_base+1.
    assert [e["event"] for e in patch_event_log_record] == [
        "start_tool_calling",
        "tool_calling_result",
    ]
    assert [e["seq"] for e in patch_event_log_record] == [30, 31]
    # The recorded tape preview is a substring of the full output.
    assert "hello world" in patch_event_log_record[1]["data"]["data_preview"]


def test_invoke_tool_activity_unknown_tool_returns_error(
    install_stub_registry, patch_event_log_record
):
    install_stub_registry(tools={})
    payload = ToolCallInput(
        instance_id="wf-1",
        seq_base=1,
        tool_call_id="tc-1",
        tool_name="doesnt_exist",
        arguments={},
    ).model_dump()
    out = ToolCallOutput.model_validate(invoke_tool_activity(None, payload))
    assert out.status == "error"
    assert "Unknown tool" in (out.error or "")


def test_invoke_tool_activity_surfaces_tool_exception(
    install_stub_registry, patch_event_log_record
):
    """Tool exceptions become error results, not workflow-failing exceptions."""

    class ExplodingTool:
        name = "bash"

        def invoke(self, *, params, context):
            raise RuntimeError("kaboom")

    install_stub_registry(tools={"bash": ExplodingTool()})
    payload = ToolCallInput(
        instance_id="wf-1",
        seq_base=1,
        tool_call_id="tc-1",
        tool_name="bash",
        arguments={},
    ).model_dump()
    out = ToolCallOutput.model_validate(invoke_tool_activity(None, payload))
    assert out.status == "error"
    assert "RuntimeError" in (out.error or "")
    assert "kaboom" in (out.error or "")


def test_invoke_tool_activity_propagates_approval_required(
    install_stub_registry, patch_event_log_record
):
    """An APPROVAL_REQUIRED result must round-trip its status to the workflow."""
    tool = StubTool(
        "bash",
        results=[_build_tool_result("approval_required", error="needs approval")],
    )
    install_stub_registry(tools={"bash": tool})
    payload = ToolCallInput(
        instance_id="wf-1",
        seq_base=1,
        tool_call_id="tc-1",
        tool_name="bash",
        arguments={"command": "rm -rf /"},
        user_approved=False,
    ).model_dump()
    out = ToolCallOutput.model_validate(invoke_tool_activity(None, payload))
    assert out.status == "approval_required"
    assert out.error == "needs approval"


def test_invoke_tool_activity_passes_user_approved_through(
    install_stub_registry, patch_event_log_record
):
    """The activity must build a ToolInvokeContext with user_approved set."""
    captured: dict = {}

    class CapturingTool:
        name = "bash"

        def invoke(self, *, params, context):
            captured["user_approved"] = context.user_approved
            captured["tool_call_id"] = context.tool_call_id
            captured["session_approved_prefixes"] = context.session_approved_prefixes
            return _build_tool_result("success", data="ok")

    install_stub_registry(tools={"bash": CapturingTool()})
    payload = ToolCallInput(
        instance_id="wf-1",
        seq_base=1,
        tool_call_id="tc-1",
        tool_name="bash",
        arguments={"command": "ls /"},
        user_approved=True,
        session_approved_prefixes=["ls"],
    ).model_dump()
    invoke_tool_activity(None, payload)
    assert captured["user_approved"] is True
    assert captured["tool_call_id"] == "tc-1"
    assert captured["session_approved_prefixes"] == ["ls"]
