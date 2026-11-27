# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""How many steps a run may take.

Two things bound a run. LangGraph's ``recursion_limit`` (from the caller's
config, or bound to the graph with ``with_config``) counts steps as LangGraph
does: that many node steps run, and the run fails once it has taken them, even
if nothing more is pending. The runner's ``max_steps`` is its own guard: a graph
that needs exactly that many steps completes.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.runnables.config import DEFAULT_RECURSION_LIMIT


@dataclass(frozen=True)
class StepBound:
    """The step limit of one run.

    ``strict`` is set when ``max_steps`` is a ``recursion_limit``: the run fails
    after that many steps whether or not more is pending. Otherwise a run that
    needs exactly ``max_steps`` steps completes.
    """

    max_steps: int
    strict: bool = False


def step_bound(
    caller_config: Optional[Dict[str, Any]],
    graph_config: Optional[Dict[str, Any]],
    max_steps: int,
    *,
    max_steps_given: bool,
) -> StepBound:
    """The bound of a run started with ``caller_config`` on a graph bound to ``graph_config``.

    A ``recursion_limit`` the caller passes is the bound. One that only the
    graph supplies (``with_config``; ``create_agent`` and Deep Agents bind
    thousands) applies too, except that a ``max_steps`` the runner was given
    explicitly is a guard it can lower. With no limit at all, ``max_steps``
    bounds the run.
    """
    caller_limit = (caller_config or {}).get("recursion_limit")
    graph_limit = (graph_config or {}).get("recursion_limit")
    from_graph_only = graph_limit is not None and caller_limit in (
        None,
        DEFAULT_RECURSION_LIMIT,
    )
    limit = graph_limit if from_graph_only else caller_limit
    if not isinstance(limit, int):
        return StepBound(max_steps)
    if from_graph_only and max_steps_given and max_steps < limit:
        return StepBound(max_steps)
    return StepBound(limit, strict=True)
