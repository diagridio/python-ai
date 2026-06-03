# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""End-to-end smoke test for the durable HolmesGPT integration.

This is intentionally NOT mocked. It:

* Builds a real ``HolmesRegistry`` against a real LLM.
* Schedules a ``investigation_workflow`` instance via the local Dapr sidecar.
* Polls the per-instance event tape (real Redis-backed Dapr state store).
* Resumes the workflow if a tool requires approval (auto-approve).
* Prints every event as it arrives, then prints the final answer.

Run with the Dapr sidecar attached (uses the default ``statestore``
component from ``dapr init`` — no extra components needed):

    dapr run --app-id holmes-e2e -- python examples/holmesgpt/e2e_test.py

Required env vars:

    MODEL                — e.g. ``anthropic/claude-sonnet-4-5-20250929``
    ANTHROPIC_API_KEY    — (or OPENAI_API_KEY etc., matching MODEL)

Optional:

    HOLMES_QUESTION      — overrides the default question
    HOLMES_MAX_STEPS     — overrides the default 5
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any, Dict

from diagrid.agent.holmesgpt import DaprWorkflowHolmesRunner


DEFAULT_QUESTION = (
    "Run `date -u` and `uname -a` on this machine using the bash toolset, "
    "then report the current UTC time and the OS kernel name in one short paragraph."
)


def _format_event(ev: Dict[str, Any]) -> str:
    if ev.get("type") == "event":
        seq = ev.get("seq", 0)
        name = ev.get("event", "?")
        data = ev.get("data", {})
        preview = json.dumps(data, default=str)
        if len(preview) > 240:
            preview = preview[:240] + "…"
        return f"  [seq={seq:>4}] {name}: {preview}"
    return f"  [meta] {ev.get('type')}: {json.dumps({k: v for k, v in ev.items() if k != 'type'}, default=str)[:240]}"


async def _auto_approve_pumper(
    runner: DaprWorkflowHolmesRunner,
    workflow_id: str,
    seen_approvals: set[str],
) -> None:
    """Watch the event tape and auto-approve any tool that asks for it."""
    seq = 0
    while True:
        events = runner.read_events_after(workflow_id, since_seq=seq, limit=64)
        for ev in events:
            seq = ev["seq"]
            if ev.get("event") == "approval_required":
                tcid = ev.get("data", {}).get("tool_call_id")
                if tcid and tcid not in seen_approvals:
                    print(f"  >>> auto-approving tool_call_id={tcid}")
                    runner.approve(workflow_id, tcid, approved=True)
                    seen_approvals.add(tcid)
        await asyncio.sleep(0.3)


async def main_async() -> int:
    model = os.environ.get("MODEL")
    if not model:
        print(
            "ERROR: MODEL env var is required (e.g. anthropic/claude-sonnet-4-5-20250929)"
        )
        return 2
    if "anthropic" in model.lower() and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is required for an Anthropic model")
        return 2
    if "openai" in model.lower() and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is required for an OpenAI model")
        return 2

    question = os.environ.get("HOLMES_QUESTION") or DEFAULT_QUESTION
    max_steps = int(os.environ.get("HOLMES_MAX_STEPS", "5"))
    workflow_id = f"holmes-e2e-{uuid.uuid4().hex[:8]}"

    print("=" * 72)
    print(" HolmesGPT × Dapr Workflow — end-to-end smoke test")
    print("=" * 72)
    print(f" model       : {model}")
    print(f" workflow_id : {workflow_id}")
    print(f" max_steps   : {max_steps}")
    print(f" question    : {question}")
    print("-" * 72)

    runner = DaprWorkflowHolmesRunner(
        name="holmes-e2e",
        model=model,
        toolset_tags=["core"],
        max_steps=max_steps,
    )
    runner.start()
    try:
        seen_approvals: set[str] = set()
        approver_task = asyncio.create_task(
            _auto_approve_pumper(runner, workflow_id, seen_approvals)
        )

        completed = False
        try:
            t0 = time.monotonic()
            async for ev in runner.run_async(
                messages=[{"role": "user", "content": question}],
                workflow_id=workflow_id,
                max_steps=max_steps,
            ):
                print(_format_event(ev))
                if ev.get("type") == "workflow_completed":
                    completed = True
                    print("-" * 72)
                    output = ev.get("output") or {}
                    final = (output.get("final") or {}).get("content")
                    print(f" total_iterations : {output.get('total_iterations')}")
                    print(f" reason           : {output.get('reason')}")
                    print(f" approvals_seen   : {len(seen_approvals)}")
                    print(f" wall_clock       : {time.monotonic() - t0:.1f}s")
                    print(" final answer:")
                    print(f" >>> {final or '(no content)'}")
                    break
                if ev.get("type") in ("workflow_failed", "workflow_terminated"):
                    print("-" * 72)
                    print(f" workflow ended unexpectedly: {ev}")
                    return 1
        finally:
            approver_task.cancel()

        return 0 if completed else 1
    finally:
        runner.shutdown()


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
