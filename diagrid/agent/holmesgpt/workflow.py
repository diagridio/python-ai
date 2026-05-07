# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr Workflow + Activities for durable HolmesGPT investigations.

The workflow body re-implements HolmesGPT's agent loop using Dapr
primitives:

* Each LLM iteration becomes a single ``call_llm`` activity invocation.
* Each tool call becomes one ``invoke_tool`` activity, fanned out in
  parallel via ``when_all`` (matching HolmesGPT's
  ``ThreadPoolExecutor(16)`` baseline).
* Approval / frontend pauses suspend the workflow on
  ``wait_for_external_event``; resuming is just
  ``raise_workflow_event``.

Per-instance event tape keys are allocated deterministically inside the
workflow so that activity replays write idempotently and so that the
SSE handler can poll a dense, totally-ordered stream.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any, Dict, Generator, List

from dapr.ext.workflow import (
    DaprWorkflowContext,
    RetryPolicy,
    WorkflowActivityContext,
    when_all,
)

from .event_log import record as record_event_to_store
from .models import (
    InvestigationInput,
    InvestigationOutput,
    LLMCallInput,
    LLMCallOutput,
    RecordEventInput,
    ToolCallInput,
    ToolCallOutput,
)
from .registry import get_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activity event budgets — kept in sync with what each activity emits.
# Read by the workflow's seq allocator so that the per-instance event tape
# stays dense (no gaps that would stall a polling reader).
# ---------------------------------------------------------------------------

EVENTS_PER_LLM_CALL = 2  # iteration_started + iteration_completed
EVENTS_PER_TOOL_CALL = 2  # start_tool_calling + tool_calling_result
EVENTS_PER_RECORD = 1  # approval_required, ai_answer_end, error, ...


# ---------------------------------------------------------------------------
# Activity: record_event
# Used for events emitted by the workflow itself (not by call_llm/invoke_tool).
# ---------------------------------------------------------------------------


def record_event_activity(ctx: WorkflowActivityContext, payload: Dict[str, Any]) -> Dict[str, Any]:
    inp = RecordEventInput.model_validate(payload)
    record_event_to_store(
        instance_id=inp.instance_id,
        seq=inp.seq,
        event=inp.event,
        data=inp.data,
    )
    return {"seq": inp.seq, "event": inp.event}


# ---------------------------------------------------------------------------
# Activity: call_llm
# Wraps a single ``LLM.completion`` invocation and surfaces the assistant
# message + parsed tool calls back to the workflow.
# ---------------------------------------------------------------------------


def call_llm_activity(ctx: WorkflowActivityContext, payload: Dict[str, Any]) -> Dict[str, Any]:
    inp = LLMCallInput.model_validate(payload)
    reg = get_registry()

    record_event_to_store(
        instance_id=inp.instance_id,
        seq=inp.seq_base,
        event="iteration_started",
        data={"message_count": len(inp.messages)},
    )

    response = reg.llm.completion(
        messages=inp.messages,
        tools=inp.tools or [],
        tool_choice=inp.tool_choice if inp.tools else None,
        response_format=inp.response_format,
        temperature=inp.temperature,
        drop_params=True,
        stream=False,
    )

    msg = response.choices[0].message
    assistant_dict = msg.model_dump(exclude_none=True)

    parsed_tool_calls: List[Dict[str, Any]] = []
    raw_tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in raw_tool_calls:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except (ValueError, TypeError):
            args = {"_raw": tc.function.arguments}
        parsed_tool_calls.append(
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": args,
            }
        )

    usage: Dict[str, Any] = {}
    if hasattr(response, "usage") and response.usage is not None:
        try:
            usage = response.usage.model_dump()
        except AttributeError:
            usage = dict(response.usage)

    out = LLMCallOutput(
        assistant_message=assistant_dict,
        tool_calls=parsed_tool_calls,
        usage=usage,
        finish_reason=getattr(response.choices[0], "finish_reason", None),
        response_id=getattr(response, "id", None),
    )

    record_event_to_store(
        instance_id=inp.instance_id,
        seq=inp.seq_base + 1,
        event="iteration_completed",
        data={
            "content": assistant_dict.get("content"),
            "tool_call_count": len(parsed_tool_calls),
            "usage": usage,
            "finish_reason": out.finish_reason,
        },
    )

    return out.model_dump()


# ---------------------------------------------------------------------------
# Activity: invoke_tool
# One HolmesGPT tool invocation through the registry's ToolExecutor.
# ---------------------------------------------------------------------------


def invoke_tool_activity(ctx: WorkflowActivityContext, payload: Dict[str, Any]) -> Dict[str, Any]:
    from holmes.core.tools import (
        StructuredToolResult,
        StructuredToolResultStatus,
        ToolInvokeContext,
    )

    inp = ToolCallInput.model_validate(payload)
    reg = get_registry()

    record_event_to_store(
        instance_id=inp.instance_id,
        seq=inp.seq_base,
        event="start_tool_calling",
        data={
            "tool_call_id": inp.tool_call_id,
            "tool_name": inp.tool_name,
            "params": inp.arguments,
        },
    )

    tool = reg.tool_executor.get_tool_by_name(inp.tool_name, user_id=None)
    if tool is None:
        result = StructuredToolResult(
            status=StructuredToolResultStatus.ERROR,
            error=f"Unknown tool: {inp.tool_name}",
            params=inp.arguments,
        )
    else:
        invoke_ctx = ToolInvokeContext(
            llm=reg.llm,
            max_token_count=reg.llm.get_max_token_count_for_single_tool(),
            tool_call_id=inp.tool_call_id,
            tool_name=inp.tool_name,
            user_approved=inp.user_approved,
            session_approved_prefixes=inp.session_approved_prefixes,
            request_context=inp.request_context,
        )
        try:
            result = tool.invoke(params=inp.arguments, context=invoke_ctx)
        except Exception as e:  # noqa: BLE001 — propagate as tool error, not workflow failure
            logger.exception("Tool %s raised an exception", inp.tool_name)
            result = StructuredToolResult(
                status=StructuredToolResultStatus.ERROR,
                error=f"{type(e).__name__}: {e}",
                params=inp.arguments,
            )

    status_val = (
        result.status.value if hasattr(result.status, "value") else str(result.status)
    )

    out = ToolCallOutput(
        tool_call_id=inp.tool_call_id,
        tool_name=inp.tool_name,
        status=status_val,
        invocation=result.invocation,
        data_str=result.get_stringified_data() or None,
        error=result.error,
        elapsed_seconds=result.elapsed_seconds,
        raw_result=result.model_dump(mode="json"),
    )

    record_event_to_store(
        instance_id=inp.instance_id,
        seq=inp.seq_base + 1,
        event="tool_calling_result",
        data={
            "tool_call_id": inp.tool_call_id,
            "tool_name": inp.tool_name,
            "status": status_val,
            "elapsed_seconds": out.elapsed_seconds,
            "error": out.error,
            "data_preview": (out.data_str or "")[:512],
        },
    )

    return out.model_dump()


# ---------------------------------------------------------------------------
# Workflow: investigation_workflow
# ---------------------------------------------------------------------------


_LLM_RETRY = RetryPolicy(
    first_retry_interval=timedelta(seconds=2),
    max_number_of_attempts=4,
    backoff_coefficient=2.0,
    max_retry_interval=timedelta(seconds=30),
)

_TOOL_RETRY = RetryPolicy(
    first_retry_interval=timedelta(seconds=1),
    max_number_of_attempts=2,
    backoff_coefficient=2.0,
    max_retry_interval=timedelta(seconds=10),
)


def _tool_result_message(out: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a ToolCallOutput dict as an OpenAI-format ``role=tool`` message."""
    content = out.get("data_str") or out.get("error") or ""
    return {
        "role": "tool",
        "tool_call_id": out["tool_call_id"],
        "name": out["tool_name"],
        "content": content,
    }


def investigation_workflow(
    ctx: DaprWorkflowContext, payload: Dict[str, Any]
) -> Generator[Any, Any, Dict[str, Any]]:
    """Durable agent loop. See module docstring for the contract."""
    inp = InvestigationInput.model_validate(payload)
    instance_id = ctx.instance_id

    messages: List[Dict[str, Any]] = list(inp.messages)
    tools = inp.tools
    next_seq = 0

    def alloc(n: int) -> int:
        nonlocal next_seq
        base = next_seq + 1
        next_seq += n
        return base

    iteration = 0
    while iteration < inp.max_steps:
        iteration += 1

        llm_input = LLMCallInput(
            instance_id=instance_id,
            seq_base=alloc(EVENTS_PER_LLM_CALL),
            messages=messages,
            tools=tools if iteration < inp.max_steps else None,
            tool_choice="auto",
            response_format=inp.response_format,
            temperature=inp.temperature,
        )
        llm_out_dict = yield ctx.call_activity(
            call_llm_activity,
            input=llm_input.model_dump(),
            retry_policy=_LLM_RETRY,
        )
        llm_out = LLMCallOutput.model_validate(llm_out_dict)

        messages.append(llm_out.assistant_message)

        # Terminal: LLM returned a final answer (no tool calls).
        if not llm_out.tool_calls:
            yield ctx.call_activity(
                record_event_activity,
                input=RecordEventInput(
                    instance_id=instance_id,
                    seq=alloc(EVENTS_PER_RECORD),
                    event="ai_answer_end",
                    data={
                        "content": llm_out.assistant_message.get("content"),
                        "iterations": iteration,
                    },
                ).model_dump(),
            )
            return InvestigationOutput(
                final=llm_out.assistant_message,
                messages=messages,
                reason="completed",
                total_iterations=iteration,
            ).model_dump()

        # Fan out tool invocations in parallel, each with a pre-allocated seq base.
        tool_inputs: List[ToolCallInput] = []
        for tc in llm_out.tool_calls:
            tool_inputs.append(
                ToolCallInput(
                    instance_id=instance_id,
                    seq_base=alloc(EVENTS_PER_TOOL_CALL),
                    tool_call_id=tc["id"],
                    tool_name=tc["name"],
                    arguments=tc.get("arguments") or {},
                    user_approved=False,
                    session_approved_prefixes=[],
                    request_context=inp.request_context,
                )
            )

        tasks = [
            ctx.call_activity(
                invoke_tool_activity,
                input=ti.model_dump(),
                retry_policy=_TOOL_RETRY,
            )
            for ti in tool_inputs
        ]
        results: List[Dict[str, Any]] = yield when_all(tasks)

        # Resolve any pauses (approval_required / frontend_pause) before continuing.
        for i, r in enumerate(results):
            status = r.get("status")
            if status not in ("approval_required", "frontend_pause"):
                continue

            yield ctx.call_activity(
                record_event_activity,
                input=RecordEventInput(
                    instance_id=instance_id,
                    seq=alloc(EVENTS_PER_RECORD),
                    event="approval_required",
                    data={
                        "tool_call_id": r["tool_call_id"],
                        "tool_name": r["tool_name"],
                        "status": status,
                        "reason": r.get("error"),
                    },
                ).model_dump(),
            )

            decision: Dict[str, Any] = yield ctx.wait_for_external_event(
                f"resume:{r['tool_call_id']}"
            )

            if status == "frontend_pause":
                # Client executed the tool out of band; treat the supplied
                # payload as the final tool output.
                results[i] = {
                    **r,
                    "status": "success",
                    "data_str": decision.get("frontend_result", ""),
                }
                continue

            if not decision.get("approved", False):
                results[i] = {
                    **r,
                    "status": "error",
                    "error": decision.get("reason") or "Tool execution rejected by user",
                }
                continue

            retry_input = tool_inputs[i].model_copy(
                update={
                    "seq_base": alloc(EVENTS_PER_TOOL_CALL),
                    "user_approved": True,
                    "session_approved_prefixes": list(
                        decision.get("session_approved_prefixes") or []
                    ),
                }
            )
            results[i] = yield ctx.call_activity(
                invoke_tool_activity,
                input=retry_input.model_dump(),
                retry_policy=_TOOL_RETRY,
            )

        # Append all tool result messages so the LLM sees a well-formed
        # tool-result block on the next iteration.
        for r in results:
            messages.append(_tool_result_message(r))

    yield ctx.call_activity(
        record_event_activity,
        input=RecordEventInput(
            instance_id=instance_id,
            seq=alloc(EVENTS_PER_RECORD),
            event="ai_answer_end",
            data={"reason": "max_steps_reached", "iterations": iteration},
        ).model_dump(),
    )
    return InvestigationOutput(
        final=None,
        messages=messages,
        reason="max_steps_reached",
        total_iterations=iteration,
    ).model_dump()


# ---------------------------------------------------------------------------
# Registration helper used by the runner.
# ---------------------------------------------------------------------------


def register_workflow_components(workflow_runtime: Any, *, workflow_name: str) -> None:
    """Register the investigation workflow + activities on a runtime."""
    workflow_runtime.register_workflow(investigation_workflow, name=workflow_name)
    workflow_runtime.register_activity(call_llm_activity, name="holmes_call_llm")
    workflow_runtime.register_activity(invoke_tool_activity, name="holmes_invoke_tool")
    workflow_runtime.register_activity(record_event_activity, name="holmes_record_event")
