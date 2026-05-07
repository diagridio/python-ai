# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Phase A of the crash+approval test.

Starts a runner, schedules an investigation that asks Holmes to run a bash
command outside the default allow list (``ls /tmp``), waits until the
``approval_required`` event lands on the tape, prints the workflow_id and
last seq, then sleeps forever — giving the orchestrator a stable victim
to SIGKILL.
"""

from __future__ import annotations

import os
import sys
import time
import uuid

from diagrid.agent.holmesgpt import (
    DaprWorkflowHolmesRunner,
    InvestigationInput,
    investigation_workflow,
)

QUESTION = (
    "Use the bash toolset to run the command `ls /tmp` (exactly that, no flags) "
    "and report what files or directories are listed. Don't run any other commands."
)


def main() -> int:
    workflow_id = (
        os.environ.get("HOLMES_WORKFLOW_ID") or f"resume-{uuid.uuid4().hex[:8]}"
    )

    runner = DaprWorkflowHolmesRunner(
        name="holmes-resume",
        model=os.environ["MODEL"],
        toolset_tags=["core"],
        max_steps=4,
    )
    runner.start()
    assert runner._workflow_client is not None and runner.registry is not None
    print(f"PHASE_A: workflow_id={workflow_id}", flush=True)

    wf_input = InvestigationInput(
        messages=[{"role": "user", "content": QUESTION}],
        tools=runner.registry.openai_tools,
        max_steps=4,
    )
    runner._workflow_client.schedule_new_workflow(
        workflow=investigation_workflow,
        input=wf_input.model_dump(),
        instance_id=workflow_id,
    )
    print("PHASE_A: workflow scheduled, polling for approval_required…", flush=True)

    seen_seq = 0
    deadline = time.time() + 120
    while time.time() < deadline:
        events = runner.read_events_after(workflow_id, since_seq=seen_seq, limit=64)
        for ev in events:
            seen_seq = ev["seq"]
            print(f"PHASE_A: seq={ev['seq']:>3} event={ev['event']}", flush=True)
            if ev["event"] == "approval_required":
                tcid = ev.get("data", {}).get("tool_call_id")
                print(
                    f"PHASE_A: APPROVAL_PARKED workflow_id={workflow_id} "
                    f"tool_call_id={tcid} last_seq={seen_seq}",
                    flush=True,
                )
                # Hold so the orchestrator can SIGKILL the daprd + this process.
                while True:
                    time.sleep(5)
        time.sleep(0.3)

    print("PHASE_A: timed out waiting for approval", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
