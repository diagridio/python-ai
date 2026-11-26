# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""How a LangGraph run ended, as ``GET /agent/run/{workflow_id}`` reports it."""

import json
from typing import Any, Dict

from dapr.ext.workflow import WorkflowStatus

from .models import GraphWorkflowOutput


def completed_result(state: Any) -> Dict[str, Any]:
    """The result of a completed workflow: what a plain ``POST`` returns for it.

    ``output`` and ``steps`` as the run reported them, ``run_status`` (the
    graph's own ``completed``, ``error`` or ``max_steps_reached``; ``status``
    stays the workflow's runtime status) and ``error`` when there is one. What
    the workflow output holds for the stream only (``executed``, the channel
    state) is left out. Empty for a workflow that has not completed.
    """
    if state.runtime_status != WorkflowStatus.COMPLETED:
        return {}
    raw = state.serialized_output
    if not raw:
        return {}
    data = json.loads(raw) if isinstance(raw, str) else raw
    output = GraphWorkflowOutput.from_dict(data)
    return {
        "output": output.output,
        "steps": output.steps,
        "run_status": output.status,
        **({"error": output.error} if output.error else {}),
    }
