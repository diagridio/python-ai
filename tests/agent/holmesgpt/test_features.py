# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the in-loop housekeeping features ported from HolmesGPT's
``ToolCallingLLM.call_stream``:

- ``compact_if_necessary`` (compaction)
- ``spill_oversized_tool_result`` (spill)
- ``prevent_overly_repeated_tool_call`` (repeated-call safeguard)
- OpenTelemetry trace-context propagation across activity boundaries
- L1: ``event_log.record`` raises on Dapr-side failure
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from diagrid.agent.holmesgpt.models import (
    LLMCallInput,
    LLMCallOutput,
    ToolCallInput,
    ToolCallOutput,
)
from diagrid.agent.holmesgpt.workflow import (
    _attach_trace_context,
    _check_repeated_call_safeguard,
    _current_trace_carrier,
    _spill_if_needed,
    call_llm_activity,
    invoke_tool_activity,
    investigation_workflow,
)

from .conftest import StubTool, make_llm_response


def _build_tool_result(status, *, data=None, error=None):
    from holmes.core.tools import StructuredToolResult, StructuredToolResultStatus

    return StructuredToolResult(
        status=(
            StructuredToolResultStatus(status) if isinstance(status, str) else status
        ),
        data=data,
        error=error,
        elapsed_seconds=0.0,
    )


# ---------------------------------------------------------------------------
# L1: event_log.record raises on failure
# ---------------------------------------------------------------------------


def test_event_log_save_record_propagates_dapr_failures(monkeypatch):
    """A DaprClient save_state error must propagate up so Dapr retries the activity."""
    from diagrid.agent.holmesgpt import event_log as real_event_log

    class _ExplodingClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def save_state(self, **_):
            raise RuntimeError("redis: connection refused")

    monkeypatch.setattr(
        "diagrid.agent.holmesgpt.event_log.DaprClient",
        lambda: _ExplodingClient(),
    )
    with pytest.raises(RuntimeError, match="redis: connection refused"):
        real_event_log.save_record(
            instance_id="wf-1",
            seq=1,
            event="x",
            data={},
            store_name="s",
            key_prefix="p",
            ttl_seconds=0,
        )


# ---------------------------------------------------------------------------
# OTel: trace context propagation
# ---------------------------------------------------------------------------


def test_attach_trace_context_is_noop_for_empty_carrier():
    with _attach_trace_context(None) as parent:
        assert parent is None
    with _attach_trace_context({}) as parent:
        assert parent is None


def test_attach_trace_context_attaches_and_detaches_when_carrier_present():
    """When a carrier is supplied, attach should call propagate.extract +
    context.attach, and detach on exit."""
    from opentelemetry import context as otel_context
    from opentelemetry import propagate

    extracted_marker = object()
    attach_calls: list = []
    detach_calls: list = []

    with mock.patch.object(propagate, "extract", return_value=extracted_marker):
        with mock.patch.object(
            otel_context,
            "attach",
            side_effect=lambda c: attach_calls.append(c) or "tok",
        ):
            with mock.patch.object(
                otel_context, "detach", side_effect=lambda t: detach_calls.append(t)
            ):
                with _attach_trace_context({"traceparent": "abc"}) as parent:
                    assert parent is extracted_marker

    assert attach_calls == [extracted_marker]
    assert detach_calls == ["tok"]


def test_current_trace_carrier_returns_none_when_no_active_span():
    """Outside any span, the carrier should be empty/None."""
    # Default OTel state: no span is active in a unit test process.
    assert _current_trace_carrier() in (None, {})


def test_call_llm_activity_attaches_trace_context(
    install_stub_registry, patch_event_log_record
):
    """When trace_context is set on the input, the activity must call
    ``_attach_trace_context``. We only verify the call (not OTel internals)."""
    install_stub_registry(
        completions=[make_llm_response(content="ok", finish_reason="stop")]
    )
    seen_carriers: list = []

    import diagrid.agent.holmesgpt.workflow as wf_mod

    real = wf_mod._attach_trace_context

    @pytest.fixture(autouse=False)
    def _noop():
        yield

    def _spy(carrier):
        seen_carriers.append(carrier)
        return real(carrier)

    with mock.patch.object(wf_mod, "_attach_trace_context", side_effect=_spy):
        payload = LLMCallInput(
            instance_id="wf-1",
            seq_base=1,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            trace_context={"traceparent": "00-trace-span-01"},
        ).model_dump()
        call_llm_activity(None, payload)

    assert seen_carriers == [{"traceparent": "00-trace-span-01"}]


# ---------------------------------------------------------------------------
# Compaction: emitted status event + adopted messages
# ---------------------------------------------------------------------------


def test_call_llm_emits_compaction_status_when_compaction_not_triggered(
    install_stub_registry, patch_event_log_record
):
    """When the message tape is small (StubLLM reports 0 tokens), compaction
    is not triggered. The activity must still emit a status event so the
    tape stays dense."""
    install_stub_registry(
        completions=[make_llm_response(content="ok", finish_reason="stop")]
    )
    payload = LLMCallInput(
        instance_id="wf-1",
        seq_base=5,
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
    ).model_dump()
    out = LLMCallOutput.model_validate(call_llm_activity(None, payload))

    statuses = [
        e
        for e in patch_event_log_record
        if e["event"] == "conversation_history_compaction_status"
    ]
    assert len(statuses) == 1
    assert statuses[0]["data"]["compacted"] is False
    # The activity surfaces the messages it used (the original here).
    assert out.messages == [{"role": "user", "content": "hi"}]
    assert out.compaction == {"compacted": False}


def test_call_llm_adopts_compacted_messages(
    install_stub_registry, patch_event_log_record
):
    """If ``compact_if_necessary`` returns a smaller tape, the activity must
    use it for the LLM call and report it back in ``LLMCallOutput.messages``."""
    install_stub_registry(
        completions=[make_llm_response(content="ok", finish_reason="stop")]
    )
    original = [{"role": "user", "content": "long"}] * 5
    compacted = [{"role": "user", "content": "summary"}]

    fake_result = SimpleNamespace(
        messages=compacted,
        conversation_history_compacted=True,
        metadata={"initial_tokens": 9000, "compacted_tokens": 800},
        events=[],
    )

    with mock.patch(
        "diagrid.agent.holmesgpt.workflow.compact_if_necessary",
        return_value=fake_result,
    ):
        payload = LLMCallInput(
            instance_id="wf-1",
            seq_base=10,
            messages=original,
            tools=[],
        ).model_dump()
        out = LLMCallOutput.model_validate(call_llm_activity(None, payload))

    assert out.messages == compacted
    assert out.compaction == {
        "compacted": True,
        "before_tokens": 9000,
        "after_tokens": 800,
    }


# ---------------------------------------------------------------------------
# Spill: oversized tool results
# ---------------------------------------------------------------------------


def test_spill_if_needed_mutates_result_via_holmes_helper(install_stub_registry):
    reg = install_stub_registry(tools={})
    original = _build_tool_result("success", data="x" * 10)
    mutated_marker = _build_tool_result("error", error="spilled to disk")

    def _fake_spill(*, tool_call_result, llm, tool_results_dir):
        tool_call_result.result = mutated_marker

    with mock.patch(
        "diagrid.agent.holmesgpt.workflow.spill_oversized_tool_result",
        side_effect=_fake_spill,
    ):
        returned = _spill_if_needed(reg, "tc-1", "bash", original)
    assert returned is mutated_marker


def test_spill_if_needed_returns_original_on_exception(install_stub_registry):
    """If spill raises, the original result is preserved (we'd rather show
    the LLM a too-large output than silently lose it)."""
    reg = install_stub_registry(tools={})
    original = _build_tool_result("success", data="x")

    def _explode(*, tool_call_result, llm, tool_results_dir):
        raise RuntimeError("disk full")

    with mock.patch(
        "diagrid.agent.holmesgpt.workflow.spill_oversized_tool_result",
        side_effect=_explode,
    ):
        returned = _spill_if_needed(reg, "tc-1", "bash", original)
    assert returned is original


def test_invoke_tool_activity_runs_spill(install_stub_registry, patch_event_log_record):
    """End-to-end: invoke_tool_activity must pass its result through
    ``_spill_if_needed`` so oversized outputs get rewritten before being
    serialised into the tape and returned to the workflow."""
    tool = StubTool(
        "bash", results=[_build_tool_result("success", data="huge raw data")]
    )
    install_stub_registry(tools={"bash": tool})

    def _shrink(*, tool_call_result, llm, tool_results_dir):
        tool_call_result.result.data = "SPILLED"

    with mock.patch(
        "diagrid.agent.holmesgpt.workflow.spill_oversized_tool_result",
        side_effect=_shrink,
    ):
        payload = ToolCallInput(
            instance_id="wf-1",
            seq_base=1,
            tool_call_id="tc-1",
            tool_name="bash",
            arguments={"command": "x"},
        ).model_dump()
        out = ToolCallOutput.model_validate(invoke_tool_activity(None, payload))
    assert out.data_str == "SPILLED"


# ---------------------------------------------------------------------------
# Repeated-call safeguard
# ---------------------------------------------------------------------------


def test_repeated_call_safeguard_short_circuits_when_args_match():
    prev = [{"tool_name": "bash", "result": {"params": {"command": "ls"}}}]
    result = _check_repeated_call_safeguard(
        tool_name="bash",
        tool_params={"command": "ls"},
        previous_tool_calls=prev,
        user_approved=False,
    )
    assert result is not None  # HolmesGPT returned a StructuredToolResult
    assert "already been called" in (result.error or "")


def test_repeated_call_safeguard_lets_new_args_through():
    prev = [{"tool_name": "bash", "result": {"params": {"command": "ls"}}}]
    result = _check_repeated_call_safeguard(
        tool_name="bash",
        tool_params={"command": "uname"},
        previous_tool_calls=prev,
        user_approved=False,
    )
    assert result is None


def test_repeated_call_safeguard_bypassed_when_user_approved():
    prev = [{"tool_name": "bash", "result": {"params": {"command": "ls"}}}]
    result = _check_repeated_call_safeguard(
        tool_name="bash",
        tool_params={"command": "ls"},
        previous_tool_calls=prev,
        user_approved=True,
    )
    assert result is None


def test_invoke_tool_activity_short_circuits_repeated_call(
    install_stub_registry, patch_event_log_record
):
    tool = StubTool(
        "bash", results=[_build_tool_result("success", data="must-not-be-returned")]
    )
    install_stub_registry(tools={"bash": tool})
    payload = ToolCallInput(
        instance_id="wf-1",
        seq_base=1,
        tool_call_id="tc-1",
        tool_name="bash",
        arguments={"command": "ls"},
        user_approved=False,
        previous_tool_calls=[
            {"tool_name": "bash", "result": {"params": {"command": "ls"}}}
        ],
    ).model_dump()

    out = ToolCallOutput.model_validate(invoke_tool_activity(None, payload))
    assert out.status == "error"
    assert "already been called" in (out.error or "")
    # The actual tool must NOT have been invoked.
    assert tool.invocations == []


# ---------------------------------------------------------------------------
# Workflow integration: previous_tool_calls accumulation
# ---------------------------------------------------------------------------


def test_workflow_accumulates_previous_tool_calls_across_iterations(drive_workflow):
    """Each new iteration's tool invocations must see the previous
    iteration's tool calls in ``previous_tool_calls``."""
    captured_inputs: list = []

    def _capture_when_yielded(payload):
        captured_inputs.append(payload)

    tc1 = {"id": "tc-1", "name": "bash", "arguments": {"command": "ls"}}
    tc2 = {"id": "tc-2", "name": "bash", "arguments": {"command": "uname"}}

    def _success(tid, data):
        return {
            "tool_call_id": tid,
            "tool_name": "bash",
            "status": "success",
            "invocation": None,
            "data_str": data,
            "error": None,
            "elapsed_seconds": 0.01,
            "raw_result": {"status": "success", "params": {"command": data}},
        }

    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [],
        "max_steps": 4,
    }

    # Use the existing drive_workflow harness. We're going to inspect the
    # tool-input payloads via a side channel on the fake context: every
    # ``Activity`` sentinel captures ``input``, so the assertions check the
    # second iteration's ``previous_tool_calls`` field.
    # The simplest way: run the workflow ourselves so we can inspect inputs.
    from .conftest import FakeWorkflowContext, _Activity, _WhenAll

    ctx = FakeWorkflowContext("wf-1")
    # Replace when_all with our sentinel
    import diagrid.agent.holmesgpt.workflow as wf_mod

    with mock.patch.object(wf_mod, "when_all", lambda tasks: _WhenAll(list(tasks))):
        gen = investigation_workflow(ctx, payload)

        # iter 1: LLM yields one tool call
        y1 = next(gen)
        assert isinstance(y1, _Activity)  # call_llm
        # send LLM response with one tool call
        y2 = gen.send(
            {
                "assistant_message": {"role": "assistant"},
                "tool_calls": [tc1],
                "usage": {},
                "finish_reason": "tool_calls",
            }
        )
        # workflow yields when_all of 1 tool task
        assert isinstance(y2, _WhenAll)
        tc1_input = y2.tasks[0].input
        assert tc1_input["previous_tool_calls"] == []  # iter 1: empty history
        y3 = gen.send([_success("tc-1", "ls")])

        # iter 2: workflow yields call_llm again
        assert isinstance(y3, _Activity)
        y4 = gen.send(
            {
                "assistant_message": {"role": "assistant"},
                "tool_calls": [tc2],
                "usage": {},
                "finish_reason": "tool_calls",
            }
        )
        assert isinstance(y4, _WhenAll)
        tc2_input = y4.tasks[0].input
        # iter 2: previous_tool_calls contains the result from iter 1
        assert len(tc2_input["previous_tool_calls"]) == 1
        prev = tc2_input["previous_tool_calls"][0]
        assert prev["tool_name"] == "bash"
        assert prev["result"]["params"] == {"command": "ls"}
