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

Each activity runs the same housekeeping HolmesGPT's ``call_stream`` runs
between iterations:

* ``compact_if_necessary`` — compresses message history when it approaches
  the model's context window.
* ``spill_oversized_tool_result`` — writes huge tool outputs to disk and
  leaves a pointer in the message.
* ``prevent_overly_repeated_tool_call`` — short-circuits identical
  duplicate tool calls.

Per-instance event tape keys are allocated deterministically inside the
workflow so that activity replays write idempotently and so that the
SSE handler can poll a dense, totally-ordered stream.

OpenTelemetry context is captured at scheduling time (in the runner) and
forwarded as a serialised propagator carrier on each activity input;
activities re-attach it so spans they emit inherit the caller's trace.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, Generator, List, Optional

from dapr.ext.workflow import (
    DaprWorkflowContext,
    RetryPolicy,
    WorkflowActivityContext,
    when_all,
)
from holmes.core.models import ToolCallResult
from holmes.core.safeguards import prevent_overly_repeated_tool_call
from holmes.core.tools import (
    StructuredToolResult,
    StructuredToolResultStatus,
    ToolInvokeContext,
)
from holmes.core.tools_utils.tool_context_window_limiter import (
    spill_oversized_tool_result,
)
from holmes.core.truncation.input_context_window_limiter import (
    CompactionInsufficientError,
    compact_if_necessary,
)
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from .event_log import save_record
from .models import (
    InvestigationInput,
    InvestigationOutput,
    LLMCallInput,
    LLMCallOutput,
    RecordEventInput,
    ToolCallInput,
    ToolCallOutput,
)
from .registry import HolmesRegistry, get_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activity event budgets — kept in sync with what each activity emits.
# Read by the workflow's seq allocator so that the per-instance event tape
# stays dense (no gaps that would stall a polling reader).
# ---------------------------------------------------------------------------

# call_llm emits exactly three tape events (always, in order):
#   seq_base   : iteration_started
#   seq_base+1 : conversation_history_compaction_status  (compacted=true|false)
#   seq_base+2 : iteration_completed
EVENTS_PER_LLM_CALL = 3
EVENTS_PER_TOOL_CALL = 2  # start_tool_calling + tool_calling_result
EVENTS_PER_RECORD = 1  # approval_required, ai_answer_end, error, ...


# ---------------------------------------------------------------------------
# OpenTelemetry helpers — attach the calling process's context to spans
# created inside the activity so traces span the workflow boundary.
# ---------------------------------------------------------------------------


@contextmanager
def _attach_trace_context(carrier: Optional[Dict[str, str]]):
    """Re-attach a propagator carrier as the current OTel context.

    No-op when no carrier is provided. The caller is responsible for
    creating spans within the ``with`` block; they'll inherit the parent
    from the carrier.
    """
    if not carrier:
        yield None
        return

    token = None
    try:
        parent = propagate.extract(carrier)
        token = otel_context.attach(parent)
        yield parent
    finally:
        if token is not None:
            try:
                otel_context.detach(token)
            except Exception:
                logger.debug("Failed to detach OTel context", exc_info=True)


def _current_trace_carrier() -> Optional[Dict[str, str]]:
    """Inject the current OTel context into a carrier dict (or ``None``)."""
    span = trace.get_current_span()
    if span is None or not span.get_span_context().is_valid:
        return None
    carrier: Dict[str, str] = {}
    propagate.inject(carrier)
    return carrier or None


# ---------------------------------------------------------------------------
# Activity: record_event
# Used for events emitted by the workflow itself (not by call_llm/invoke_tool).
# ---------------------------------------------------------------------------


def record_event_activity(
    ctx: WorkflowActivityContext, payload: Dict[str, Any]
) -> Dict[str, Any]:
    inp = RecordEventInput.model_validate(payload)
    with _attach_trace_context(inp.trace_context):
        save_record(
            instance_id=inp.instance_id,
            seq=inp.seq,
            event=inp.event,
            data=inp.data,
        )
    return {"seq": inp.seq, "event": inp.event}


# ---------------------------------------------------------------------------
# Activity: call_llm
# Wraps a single ``LLM.completion`` invocation; runs HolmesGPT's compaction
# pass first so long investigations don't blow the model's context window.
# ---------------------------------------------------------------------------


def _run_compaction(
    reg: HolmesRegistry,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
):
    """Call ``compact_if_necessary``. Returns ``(messages, status_dict)``.

    On success: returns possibly-compacted messages plus a status dict the
    workflow emits as a tape event.

    On compaction-insufficient: returns the original messages plus a status
    dict flagging the issue. We don't raise here — Dapr would otherwise
    retry the activity, which won't help (compaction is deterministic on
    inputs).
    """
    try:
        result = compact_if_necessary(llm=reg.llm, messages=messages, tools=tools)
    except CompactionInsufficientError as e:
        return messages, {
            "compacted": False,
            "reason": "compaction_insufficient",
            "message": str(e),
        }

    status: Dict[str, Any] = {
        "compacted": bool(result.conversation_history_compacted),
    }
    if result.conversation_history_compacted:
        try:
            status["before_tokens"] = result.metadata.get("initial_tokens")
            status["after_tokens"] = result.metadata.get("compacted_tokens")
        except AttributeError:
            pass
    return list(result.messages), status


def call_llm_activity(
    ctx: WorkflowActivityContext, payload: Dict[str, Any]
) -> Dict[str, Any]:
    inp = LLMCallInput.model_validate(payload)
    reg = get_registry()

    with _attach_trace_context(inp.trace_context):
        save_record(
            instance_id=inp.instance_id,
            seq=inp.seq_base,
            event="iteration_started",
            data={"message_count": len(inp.messages)},
        )

        compacted_messages, compaction_status = _run_compaction(
            reg, list(inp.messages), inp.tools
        )

        save_record(
            instance_id=inp.instance_id,
            seq=inp.seq_base + 1,
            event="conversation_history_compaction_status",
            data=compaction_status,
        )

        response = reg.llm.completion(
            messages=compacted_messages,
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
            messages=compacted_messages,
            compaction=compaction_status,
        )

        save_record(
            instance_id=inp.instance_id,
            seq=inp.seq_base + 2,
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
# Wraps a HolmesGPT tool invocation. Runs HolmesGPT's repeated-call
# safeguard and oversized-result spill before/after the actual ``invoke``.
# ---------------------------------------------------------------------------


def _spill_if_needed(
    reg: HolmesRegistry,
    tool_call_id: str,
    tool_name: str,
    result: StructuredToolResult,
) -> StructuredToolResult:
    """Run HolmesGPT's oversized-result spill. Returns the (possibly mutated)
    ``StructuredToolResult``.

    ``spill_oversized_tool_result`` mutates ``tool_call_result.result``
    in-place when the message is too large, replacing the data with a
    pointer to a saved file. We wrap our existing result in a
    ``ToolCallResult`` so the helper can do its work.
    """
    tcr = ToolCallResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        description="",
        result=result,
        toolset_name=None,
    )
    try:
        spill_oversized_tool_result(
            tool_call_result=tcr,
            llm=reg.llm,
            tool_results_dir=None,
        )
    except Exception:
        logger.exception("spill_oversized_tool_result failed; using original result")
        return result
    return tcr.result


def invoke_tool_activity(
    ctx: WorkflowActivityContext, payload: Dict[str, Any]
) -> Dict[str, Any]:
    inp = ToolCallInput.model_validate(payload)
    reg = get_registry()

    with _attach_trace_context(inp.trace_context):
        save_record(
            instance_id=inp.instance_id,
            seq=inp.seq_base,
            event="start_tool_calling",
            data={
                "tool_call_id": inp.tool_call_id,
                "tool_name": inp.tool_name,
                "params": inp.arguments,
            },
        )

        # Repeated-call safeguard short-circuits before any tool dispatch.
        safeguard_result = _check_repeated_call_safeguard(
            inp.tool_name, inp.arguments, inp.previous_tool_calls, inp.user_approved
        )
        if safeguard_result is not None:
            return _emit_tool_result(inp, safeguard_result)

        tool = reg.tool_executor.get_tool_by_name(inp.tool_name, user_id=None)
        if tool is None:
            return _emit_tool_result(
                inp,
                StructuredToolResult(
                    status=StructuredToolResultStatus.ERROR,
                    error=f"Unknown tool: {inp.tool_name}",
                    params=inp.arguments,
                ),
            )

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
        except Exception as e:  # noqa: BLE001 — tool.invoke is contracted to NOT raise; defend in depth
            logger.exception("Tool %s raised an exception", inp.tool_name)
            return _emit_tool_result(
                inp,
                StructuredToolResult(
                    status=StructuredToolResultStatus.ERROR,
                    error=f"{type(e).__name__}: {e}",
                    params=inp.arguments,
                ),
            )

        # Happy path: spill oversized successful results to disk so they
        # don't blow the message budget on the next LLM call.
        result = _spill_if_needed(reg, inp.tool_call_id, inp.tool_name, result)
        return _emit_tool_result(inp, result)


def _emit_tool_result(
    inp: ToolCallInput, result: StructuredToolResult
) -> Dict[str, Any]:
    """Emit the ``tool_calling_result`` tape event and serialise the
    activity's return value.

    Shared between the happy path, the unknown-tool path, the
    repeated-call-safeguard path, and the tool-exception path so the seq
    budget (``seq_base+1`` for the result event) is honoured no matter how
    we reach the end of the activity.
    """
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
    save_record(
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


def _check_repeated_call_safeguard(
    tool_name: str,
    tool_params: Dict[str, Any],
    previous_tool_calls: List[Dict[str, Any]],
    user_approved: bool,
) -> Optional[StructuredToolResult]:
    """Run HolmesGPT's repeated-tool-call safeguard.

    Returns a ``StructuredToolResult`` to short-circuit invocation, or
    ``None`` to proceed with the normal invocation path.
    """
    if user_approved:
        # User explicitly approved a re-run; let it through.
        return None
    try:
        return prevent_overly_repeated_tool_call(
            tool_name=tool_name,
            tool_params=tool_params,
            tool_calls=previous_tool_calls,
        )
    except Exception:
        logger.exception(
            "prevent_overly_repeated_tool_call failed; allowing the invocation"
        )
        return None


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


def _previous_tool_call_entry(out: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a ``ToolCallOutput`` dict into the shape
    ``prevent_overly_repeated_tool_call`` expects in ``tool_calls``."""
    return {
        "tool_name": out.get("tool_name"),
        "result": {"params": (out.get("raw_result") or {}).get("params")},
    }


def investigation_workflow(
    ctx: DaprWorkflowContext, payload: Dict[str, Any]
) -> Generator[Any, Any, Dict[str, Any]]:
    """Durable agent loop. See module docstring for the contract."""
    inp = InvestigationInput.model_validate(payload)
    instance_id = ctx.instance_id
    trace_context = inp.trace_context  # captured by the runner at schedule time

    messages: List[Dict[str, Any]] = list(inp.messages)
    tools = inp.tools
    previous_tool_calls: List[Dict[str, Any]] = []
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
            trace_context=trace_context,
        )
        llm_out_dict = yield ctx.call_activity(
            call_llm_activity,
            input=llm_input.model_dump(),
            retry_policy=_LLM_RETRY,
        )
        llm_out = LLMCallOutput.model_validate(llm_out_dict)

        # Adopt the compacted message tape (when compaction ran) so the next
        # iteration doesn't re-compact the same prefix every turn.
        if llm_out.messages is not None:
            messages = list(llm_out.messages)

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
                    trace_context=trace_context,
                ).model_dump(),
            )
            return InvestigationOutput(
                final=llm_out.assistant_message,
                messages=messages,
                reason="completed",
                total_iterations=iteration,
            ).model_dump()

        # Fan out tool invocations in parallel, each with a pre-allocated seq base.
        # ``previous_tool_calls`` is snapshotted here so all parallel
        # invocations see the same history (no race on the safeguard).
        snapshot_prev = list(previous_tool_calls)
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
                    previous_tool_calls=snapshot_prev,
                    trace_context=trace_context,
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
                    trace_context=trace_context,
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
                    "error": decision.get("reason")
                    or "Tool execution rejected by user",
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
        # tool-result block on the next iteration. Also record each result
        # in the cumulative history that drives the repeated-call safeguard.
        for r in results:
            messages.append(_tool_result_message(r))
            previous_tool_calls.append(_previous_tool_call_entry(r))

    yield ctx.call_activity(
        record_event_activity,
        input=RecordEventInput(
            instance_id=instance_id,
            seq=alloc(EVENTS_PER_RECORD),
            event="ai_answer_end",
            data={"reason": "max_steps_reached", "iterations": iteration},
            trace_context=trace_context,
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
    workflow_runtime.register_activity(
        record_event_activity, name="holmes_record_event"
    )
