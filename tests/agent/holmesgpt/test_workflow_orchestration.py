# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Drive ``investigation_workflow`` as a Python generator and verify its
control flow scenario-by-scenario.

These tests do not require Dapr, the LLM, or any tool. The workflow body is
exercised by sending scripted activity results into ``gen.send(...)`` while
a fixture replaces ``when_all`` with a sentinel builder.
"""

from __future__ import annotations

from diagrid.agent.holmesgpt.workflow import (
    _tool_result_message,
    call_llm_activity,
    investigation_workflow,
    invoke_tool_activity,
    record_event_activity,
)


# ---------------------------------------------------------------------------
# _tool_result_message helper
# ---------------------------------------------------------------------------


def test_tool_result_message_prefers_data_str():
    msg = _tool_result_message(
        {
            "tool_call_id": "tc-1",
            "tool_name": "bash",
            "data_str": "hello",
            "error": None,
        }
    )
    assert msg == {
        "role": "tool",
        "tool_call_id": "tc-1",
        "name": "bash",
        "content": "hello",
    }


def test_tool_result_message_falls_back_to_error_when_no_data():
    msg = _tool_result_message(
        {
            "tool_call_id": "tc-1",
            "tool_name": "bash",
            "data_str": None,
            "error": "permission denied",
        }
    )
    assert msg["content"] == "permission denied"


def test_tool_result_message_empty_when_nothing_present():
    msg = _tool_result_message(
        {"tool_call_id": "tc-1", "tool_name": "bash", "data_str": None}
    )
    assert msg["content"] == ""


# ---------------------------------------------------------------------------
# Workflow scenarios
# ---------------------------------------------------------------------------


def _llm_final_answer(content="all done"):
    return {
        "assistant_message": {"role": "assistant", "content": content},
        "tool_calls": [],
        "usage": {},
        "finish_reason": "stop",
        "response_id": "r-1",
    }


def _llm_calls_tools(tool_calls):
    return {
        "assistant_message": {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        },
        "tool_calls": tool_calls,
        "usage": {},
        "finish_reason": "tool_calls",
        "response_id": "r-1",
    }


def _tool_success(tool_call_id="tc-1", tool_name="bash", data="ok"):
    return {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "status": "success",
        "invocation": None,
        "data_str": data,
        "error": None,
        "elapsed_seconds": 0.01,
        "raw_result": {"status": "success", "data": data},
    }


def _tool_approval_required(tool_call_id="tc-1", tool_name="bash"):
    return {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "status": "approval_required",
        "invocation": None,
        "data_str": None,
        "error": "needs approval",
        "elapsed_seconds": 0.0,
        "raw_result": {"status": "approval_required"},
    }


def _tool_frontend_pause(tool_call_id="tc-1", tool_name="frontend_tool"):
    return {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "status": "frontend_pause",
        "invocation": None,
        "data_str": None,
        "error": None,
        "elapsed_seconds": 0.0,
        "raw_result": {"status": "frontend_pause"},
    }


def _payload(*, messages=None, max_steps=4):
    return {
        "messages": messages or [{"role": "user", "content": "hi"}],
        "tools": [],
        "max_steps": max_steps,
    }


# ----- happy path: no tool calls -----


def test_workflow_returns_immediately_when_llm_has_no_tool_calls(drive_workflow):
    result = drive_workflow(
        investigation_workflow,
        _payload(),
        [
            (("activity", call_llm_activity), _llm_final_answer("the answer is 42")),
            (("activity", record_event_activity), {"seq": 2, "event": "ai_answer_end"}),
        ],
    )
    assert result["reason"] == "completed"
    assert result["total_iterations"] == 1
    assert result["final"]["content"] == "the answer is 42"
    # Original user + assistant final
    assert len(result["messages"]) == 2
    assert result["messages"][-1]["content"] == "the answer is 42"


# ----- single tool path -----


def test_workflow_handles_single_tool_then_completes(drive_workflow):
    tc = {"id": "tc-1", "name": "bash", "arguments": {"command": "ls"}}
    result = drive_workflow(
        investigation_workflow,
        _payload(),
        [
            # Iteration 1: LLM picks one tool
            (("activity", call_llm_activity), _llm_calls_tools([tc])),
            # Parallel fan-out with 1 task
            (("when_all", 1), [_tool_success("tc-1", "bash", "file1\nfile2")]),
            # Iteration 2: LLM produces final answer
            (("activity", call_llm_activity), _llm_final_answer("there are 2 files")),
            (("activity", record_event_activity), {"seq": 7, "event": "ai_answer_end"}),
        ],
    )
    assert result["reason"] == "completed"
    assert result["total_iterations"] == 2
    # user + assistant(tool_calls) + tool result + assistant(final) = 4
    assert [m.get("role") for m in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result["messages"][2]["tool_call_id"] == "tc-1"
    assert result["messages"][2]["content"] == "file1\nfile2"


# ----- parallel fan-out -----


def test_workflow_fans_out_parallel_tool_calls(drive_workflow):
    tc1 = {"id": "tc-1", "name": "bash", "arguments": {"command": "date"}}
    tc2 = {"id": "tc-2", "name": "bash", "arguments": {"command": "uname"}}
    result = drive_workflow(
        investigation_workflow,
        _payload(),
        [
            (("activity", call_llm_activity), _llm_calls_tools([tc1, tc2])),
            (
                ("when_all", 2),
                [
                    _tool_success("tc-1", "bash", "now"),
                    _tool_success("tc-2", "bash", "darwin"),
                ],
            ),
            (("activity", call_llm_activity), _llm_final_answer("done")),
            (("activity", record_event_activity), {"seq": 9, "event": "ai_answer_end"}),
        ],
    )
    assert result["reason"] == "completed"
    # Both tool results in the message tape, in original order
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["tc-1", "tc-2"]
    assert tool_msgs[0]["content"] == "now"
    assert tool_msgs[1]["content"] == "darwin"


# ----- approval pause / resume -----


def test_workflow_pauses_on_approval_and_re_invokes_after_approval(drive_workflow):
    tc = {"id": "tc-1", "name": "bash", "arguments": {"command": "rm -rf /tmp/x"}}
    result = drive_workflow(
        investigation_workflow,
        _payload(),
        [
            (("activity", call_llm_activity), _llm_calls_tools([tc])),
            (("when_all", 1), [_tool_approval_required("tc-1", "bash")]),
            # Workflow emits an approval_required event
            (
                ("activity", record_event_activity),
                {"seq": 5, "event": "approval_required"},
            ),
            # …then waits for the decision
            (
                ("wait", "resume:tc-1"),
                {"approved": True, "session_approved_prefixes": ["rm"]},
            ),
            # …then re-invokes the tool with user_approved=True
            (
                ("activity", invoke_tool_activity),
                _tool_success("tc-1", "bash", "ok deleted"),
            ),
            # Next iteration: LLM produces final answer
            (("activity", call_llm_activity), _llm_final_answer("done")),
            (
                ("activity", record_event_activity),
                {"seq": 10, "event": "ai_answer_end"},
            ),
        ],
    )
    assert result["reason"] == "completed"
    # The tool message reflects the approved re-invocation's output
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "ok deleted"


def test_workflow_rejects_when_approval_decision_is_false(drive_workflow):
    tc = {"id": "tc-1", "name": "bash", "arguments": {"command": "rm"}}
    result = drive_workflow(
        investigation_workflow,
        _payload(),
        [
            (("activity", call_llm_activity), _llm_calls_tools([tc])),
            (("when_all", 1), [_tool_approval_required("tc-1", "bash")]),
            (
                ("activity", record_event_activity),
                {"seq": 5, "event": "approval_required"},
            ),
            (("wait", "resume:tc-1"), {"approved": False, "reason": "user said no"}),
            (("activity", call_llm_activity), _llm_final_answer("acknowledged")),
            (("activity", record_event_activity), {"seq": 9, "event": "ai_answer_end"}),
        ],
    )
    assert result["reason"] == "completed"
    # Tool result message contains the rejection reason as content
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "user said no"


# ----- frontend pause / resume -----


def test_workflow_resumes_frontend_pause_with_client_supplied_result(drive_workflow):
    tc = {"id": "tc-9", "name": "browser", "arguments": {}}
    result = drive_workflow(
        investigation_workflow,
        _payload(),
        [
            (("activity", call_llm_activity), _llm_calls_tools([tc])),
            (("when_all", 1), [_tool_frontend_pause("tc-9", "browser")]),
            (
                ("activity", record_event_activity),
                {"seq": 5, "event": "approval_required"},
            ),
            (
                ("wait", "resume:tc-9"),
                {"frontend_result": "the user clicked the button"},
            ),
            (("activity", call_llm_activity), _llm_final_answer("noted")),
            (("activity", record_event_activity), {"seq": 9, "event": "ai_answer_end"}),
        ],
    )
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "the user clicked the button"


# ----- max steps exit -----


def test_workflow_stops_at_max_steps_when_tools_keep_being_called(drive_workflow):
    tc = {"id": "tc-1", "name": "bash", "arguments": {}}
    # max_steps=2, and every LLM call wants another tool — workflow should
    # bail with reason="max_steps_reached"
    result = drive_workflow(
        investigation_workflow,
        _payload(max_steps=2),
        [
            (("activity", call_llm_activity), _llm_calls_tools([tc])),
            (("when_all", 1), [_tool_success("tc-1", "bash", "a")]),
            (("activity", call_llm_activity), _llm_calls_tools([tc])),
            (("when_all", 1), [_tool_success("tc-1", "bash", "b")]),
            (
                ("activity", record_event_activity),
                {"seq": 99, "event": "ai_answer_end"},
            ),
        ],
    )
    assert result["reason"] == "max_steps_reached"
    assert result["total_iterations"] == 2
    assert result["final"] is None
