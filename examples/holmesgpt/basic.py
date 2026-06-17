# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Minimal example: run a single HolmesGPT investigation as a Dapr Workflow.

Run with the Dapr sidecar attached (no extra components beyond a stock
``dapr init`` are required — the integration uses the default ``statestore``
component for both workflow state and the polling event tape):

    dapr run --app-id holmes-cli -- python examples/holmesgpt/basic.py
"""

from diagrid.agent.holmesgpt import DaprWorkflowHolmesRunner


def main() -> None:
    runner = DaprWorkflowHolmesRunner(
        name="sre-agent",
        # config_path="~/.holmes/config.yaml",  # optional
        # model="anthropic/claude-sonnet-4-5-20250929",
        max_steps=10,
    )
    runner.start()
    try:
        # ``ask`` is the durable equivalent of ``holmes ask`` — HolmesGPT's
        # full system prompt (toolset instructions, skills, runbooks, etc.)
        # is rendered locally, then the agent loop runs as a Dapr workflow.
        result = runner.ask(
            "List the pods that are not Running in the cluster and explain why."
        )
        final = result.get("final") or {}
        print("=== Final answer ===")
        print(final.get("content", "(no content)"))
        print("=== Iterations ===", result.get("total_iterations"))
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
