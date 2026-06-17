# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""End-to-end test using ``runner.ask`` — the durable equivalent of ``holmes ask``.

Unlike ``e2e_test.py`` which passes raw user messages, this version goes
through HolmesGPT's ``build_chat_messages`` so the LLM sees the full
system prompt (toolset instructions, skills, etc.) that ``holmes ask``
would produce.

Prints the rendered system prompt's length and a sample so you can
verify that prompt construction actually happened.
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


QUESTION = "Use the bash toolset to run `whoami` and report which user is running this process."


def _format_event(ev: Dict[str, Any]) -> str:
    if ev.get("type") == "event":
        seq = ev.get("seq", 0)
        name = ev.get("event", "?")
        preview = json.dumps(ev.get("data", {}), default=str)
        if len(preview) > 240:
            preview = preview[:240] + "…"
        return f"  [seq={seq:>4}] {name}: {preview}"
    return f"  [meta] {ev.get('type')}"


async def main_async() -> int:
    model = os.environ.get("MODEL")
    if not model:
        print("ERROR: MODEL env var is required")
        return 2

    workflow_id = f"holmes-ask-{uuid.uuid4().hex[:8]}"
    runner = DaprWorkflowHolmesRunner(
        name="holmes-ask",
        model=model,
        toolset_tags=["core"],
        max_steps=4,
    )
    runner.start()

    # Render the messages directly so we can inspect what build_chat_messages
    # produced (this is the same call ``ask()`` makes internally).
    messages = runner._build_messages(
        question=QUESTION,
        conversation_history=None,
        additional_system_prompt=None,
        skills=None,
        images=None,
        prompt_component_overrides=None,
        global_instructions=None,
    )

    print("=" * 72)
    print(" HolmesGPT × Dapr Workflow — ask() end-to-end test")
    print("=" * 72)
    print(f" model       : {model}")
    print(f" workflow_id : {workflow_id}")
    print(f" question    : {QUESTION}")
    print()
    print(f" rendered {len(messages)} messages via build_chat_messages")
    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            preview = json.dumps(content)[:300]
        elif isinstance(content, str):
            preview = content[:300]
        else:
            preview = repr(content)[:300]
        ln = (
            len(content)
            if isinstance(content, str)
            else len(json.dumps(content) if content else "")
        )
        print(f"   [{i}] role={role:<10} content_len={ln:>6}  preview: {preview!r}")
    print("-" * 72)

    try:
        t0 = time.monotonic()
        seen_approvals: set[str] = set()
        completed = False

        async for ev in runner.ask_async(
            QUESTION,
            workflow_id=workflow_id,
            max_steps=4,
        ):
            print(_format_event(ev))
            # Auto-approve any tool that asks (unlikely for whoami).
            if ev.get("event") == "approval_required":
                tcid = ev["data"]["tool_call_id"]
                if tcid not in seen_approvals:
                    print(f"  >>> auto-approving tool_call_id={tcid}")
                    runner.approve(workflow_id, tcid, approved=True)
                    seen_approvals.add(tcid)
            if ev.get("type") == "workflow_completed":
                completed = True
                output = ev.get("output") or {}
                final = (output.get("final") or {}).get("content")
                print("-" * 72)
                print(f" total_iterations : {output.get('total_iterations')}")
                print(f" wall_clock       : {time.monotonic() - t0:.1f}s")
                print(f" final_answer     : {final}")
                break
            if ev.get("type") in ("workflow_failed", "workflow_terminated"):
                print(f" workflow ended unexpectedly: {ev}")
                return 1

        return 0 if completed else 1
    finally:
        runner.shutdown()


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
