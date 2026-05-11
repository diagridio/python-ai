# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Pydantic round-trip tests for the IO contracts between workflow and activities."""

from diagrid.agent.holmesgpt.models import (
    InvestigationInput,
    InvestigationOutput,
    LLMCallInput,
    LLMCallOutput,
    RecordEventInput,
    ToolCallInput,
    ToolCallOutput,
)


def test_investigation_input_round_trip():
    src = InvestigationInput(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "bash"}}],
        max_steps=5,
        response_format={"type": "json_object"},
        temperature=0.2,
        request_context={"headers": {"x-trace": "abc"}},
    )
    assert (
        InvestigationInput.model_validate(src.model_dump()).model_dump()
        == src.model_dump()
    )


def test_investigation_input_defaults():
    src = InvestigationInput(messages=[{"role": "user", "content": "hi"}])
    assert src.max_steps == 10
    assert src.tools is None
    assert src.response_format is None


def test_investigation_output_defaults():
    out = InvestigationOutput(reason="completed", total_iterations=3)
    dumped = out.model_dump()
    assert dumped["messages"] == []
    assert dumped["final"] is None
    assert dumped["total_iterations"] == 3


def test_llm_call_input_required_fields():
    src = LLMCallInput(instance_id="wf-1", seq_base=2, messages=[])
    assert src.tool_choice == "auto"
    assert src.tools is None


def test_llm_call_output_default_tool_calls_empty():
    out = LLMCallOutput(assistant_message={"role": "assistant", "content": "ok"})
    assert out.tool_calls == []
    assert out.usage == {}


def test_tool_call_input_carries_session_prefixes():
    src = ToolCallInput(
        instance_id="wf-1",
        seq_base=10,
        tool_call_id="tc-1",
        tool_name="bash",
        arguments={"command": "ls"},
        user_approved=True,
        session_approved_prefixes=["ls", "cat"],
    )
    assert src.user_approved is True
    assert "ls" in src.session_approved_prefixes


def test_tool_call_output_round_trip():
    src = ToolCallOutput(
        tool_call_id="tc-1",
        tool_name="bash",
        status="success",
        invocation="ls /tmp",
        data_str="file1\nfile2",
        error=None,
        elapsed_seconds=0.4,
        raw_result={"status": "success", "data": "file1\nfile2"},
    )
    assert (
        ToolCallOutput.model_validate(src.model_dump()).model_dump() == src.model_dump()
    )


def test_record_event_input_minimal():
    rec = RecordEventInput(instance_id="wf-1", seq=1, event="ai_answer_end")
    assert rec.data == {}
    assert rec.event == "ai_answer_end"


def test_models_drop_extra_fields():
    """Extra keys passed to Pydantic models should be ignored, not raise."""
    src = InvestigationInput.model_validate(
        {
            "messages": [],
            "extra_field": "should-be-dropped",
        }
    )
    assert "extra_field" not in src.model_dump()
