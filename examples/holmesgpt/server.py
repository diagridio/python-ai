# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Run the durable HolmesGPT server with SSE streaming and approval endpoints.

Run with the Dapr sidecar attached:

    dapr run --app-id holmes-server \
             --app-port 5001 \
             -- python diagrid/agent/holmesgpt/examples/server.py

Then:

    # Schedule an investigation
    curl -sS -X POST http://localhost:5001/investigations \
        -H 'content-type: application/json' \
        -d '{"question": "Why is checkout-api crashlooping?"}'

    # Stream events for that workflow ID
    curl -N http://localhost:5001/investigations/<id>/stream

    # Approve a paused tool
    curl -sS -X POST http://localhost:5001/investigations/<id>/approve \
        -H 'content-type: application/json' \
        -d '{"tool_call_id": "<tc-id>", "approved": true}'
"""

from diagrid.agent.holmesgpt import DaprWorkflowHolmesRunner


def main() -> None:
    runner = DaprWorkflowHolmesRunner(
        name="sre-agent",
        max_steps=10,
    )
    runner.serve(port=5001)


if __name__ == "__main__":
    main()
