# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Diagrid Agent HolmesGPT — Durable execution of HolmesGPT investigations.

Wraps each LLM iteration and tool call of HolmesGPT's agent loop in a Dapr
Workflow activity, providing fault tolerance, durable approvals, and a
polling-based SSE stream — without modifying HolmesGPT itself.

Example:
    ```python
    from diagrid.agent.holmesgpt import DaprWorkflowHolmesRunner

    runner = DaprWorkflowHolmesRunner(
        name="sre-agent",
        config_path="~/.holmes/config.yaml",
        model="anthropic/claude-sonnet-4-5-20250929",
    )
    runner.start()

    result = runner.invoke(
        messages=[{"role": "user", "content": "Why is checkout-api crashlooping?"}],
        max_steps=10,
    )
    print(result["final"]["content"])

    runner.shutdown()
    ```

Requires HolmesGPT to be installed in the same environment::

    pip install holmesgpt
"""

from diagrid.agent.holmesgpt import event_log
from diagrid.agent.holmesgpt.models import (
    InvestigationInput,
    InvestigationOutput,
    LLMCallInput,
    LLMCallOutput,
    RecordEventInput,
    ToolCallInput,
    ToolCallOutput,
)
from diagrid.agent.holmesgpt.registry import (
    HolmesRegistry,
    get_registry,
    set_registry,
)
from diagrid.agent.holmesgpt.runner import DaprWorkflowHolmesRunner
from diagrid.agent.holmesgpt.version import __version__
from diagrid.agent.holmesgpt.workflow import (
    EVENTS_PER_LLM_CALL,
    EVENTS_PER_RECORD,
    EVENTS_PER_TOOL_CALL,
    call_llm_activity,
    investigation_workflow,
    invoke_tool_activity,
    record_event_activity,
    register_workflow_components,
)

__all__ = [
    # Main runner
    "DaprWorkflowHolmesRunner",
    # Registry
    "HolmesRegistry",
    "get_registry",
    "set_registry",
    # Models
    "InvestigationInput",
    "InvestigationOutput",
    "LLMCallInput",
    "LLMCallOutput",
    "RecordEventInput",
    "ToolCallInput",
    "ToolCallOutput",
    # Workflow components
    "investigation_workflow",
    "call_llm_activity",
    "invoke_tool_activity",
    "record_event_activity",
    "register_workflow_components",
    "EVENTS_PER_LLM_CALL",
    "EVENTS_PER_TOOL_CALL",
    "EVENTS_PER_RECORD",
    # Event tape (advanced)
    "event_log",
    # Version
    "__version__",
]
