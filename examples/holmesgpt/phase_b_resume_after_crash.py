# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Phase B of the crash+approval test.

A fresh process attaches to the same Dapr ``app-id`` after Phase A was
SIGKILLed. The Dapr workflow runtime rehydrates the parked workflow from
its event-sourced history in Redis. We then verify the workflow is still
RUNNING (not COMPLETED), raise the external event to approve the tool
call, and watch it run to completion.

Usage::

    uv run python phase_b_resume_after_crash.py <workflow_id> <tool_call_id>
"""

from __future__ import annotations

import json
import os
import sys
import time

from dapr.ext.workflow import WorkflowStatus

from diagrid.agent.holmesgpt import DaprWorkflowHolmesRunner


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: phase_b_resume_after_crash.py <workflow_id> <tool_call_id>")
        return 2
    workflow_id = sys.argv[1]
    tool_call_id = sys.argv[2]

    runner = DaprWorkflowHolmesRunner(
        name="holmes-resume",
        model=os.environ["MODEL"],
        toolset_tags=["core"],
        max_steps=4,
    )
    runner.start()
    assert runner._workflow_client is not None

    # 1) Verify the parked workflow is still alive — Dapr should rehydrate it
    #    from the actor state we wrote to Redis before the SIGKILL.
    state = runner._workflow_client.get_workflow_state(instance_id=workflow_id)
    if state is None:
        print(f"PHASE_B: ERROR workflow {workflow_id} not found")
        return 1
    print(
        f"PHASE_B: pre-resume status={state.runtime_status} "
        f"created={state.created_at} updated={state.last_updated_at}",
        flush=True,
    )
    if state.runtime_status != WorkflowStatus.RUNNING:
        print(
            f"PHASE_B: ERROR workflow is {state.runtime_status}, expected RUNNING",
            flush=True,
        )
        return 1

    # 2) Read what's already on the tape so the resume stream picks up cleanly.
    pre_events = runner.read_events_after(workflow_id, since_seq=0, limit=64)
    last_seq = pre_events[-1]["seq"] if pre_events else 0
    print(
        f"PHASE_B: tape has {len(pre_events)} events; last_seq={last_seq}", flush=True
    )

    # 3) Approve the parked tool call. This is what makes durability real:
    #    the workflow has been parked across a process+sidecar restart and
    #    only resumes when we send the external event.
    print(f"PHASE_B: approving tool_call_id={tool_call_id}", flush=True)
    runner.approve(workflow_id, tool_call_id, approved=True)

    # 4) Drain remaining events until the workflow completes.
    deadline = time.time() + 120
    while time.time() < deadline:
        events = runner.read_events_after(workflow_id, since_seq=last_seq, limit=64)
        for ev in events:
            last_seq = ev["seq"]
            preview = json.dumps(ev.get("data", {}), default=str)
            if len(preview) > 240:
                preview = preview[:240] + "…"
            print(
                f"PHASE_B: seq={ev['seq']:>3} event={ev['event']} {preview}",
                flush=True,
            )

        state = runner._workflow_client.get_workflow_state(instance_id=workflow_id)
        if state is None:
            print("PHASE_B: ERROR state vanished during resume")
            return 1
        if state.runtime_status == WorkflowStatus.COMPLETED:
            output = state.serialized_output
            if isinstance(output, str):
                output = json.loads(output) if output else {}
            final = (output or {}).get("final") or {}
            print("PHASE_B: WORKFLOW_COMPLETED", flush=True)
            print(
                f"PHASE_B: total_iterations={(output or {}).get('total_iterations')}",
                flush=True,
            )
            print(
                f"PHASE_B: final_answer={final.get('content', '(empty)')}", flush=True
            )
            return 0
        if state.runtime_status in (WorkflowStatus.FAILED, WorkflowStatus.TERMINATED):
            print(f"PHASE_B: ERROR workflow ended {state.runtime_status}")
            return 1
        time.sleep(0.3)

    print("PHASE_B: timed out", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
