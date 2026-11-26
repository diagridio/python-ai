# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The ``langgraph_*`` metadata the Pregel loop gives every task.

A node and its routers run under a config built from these keys whether the
run is streamed or not, as they do in native LangGraph; the stream handler
reads the same keys for its chunks.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from diagrid.agent.core.workflow.stream import display_step

if TYPE_CHECKING:
    from .streaming import StaticConfig

# Configurable keys LangChain never copies into metadata (they are secrets).
_UNTRACED_CONFIGURABLE = frozenset(("api_key",))


@dataclass(frozen=True)
class TaskIdentity:
    """Which task of the run a node (or a router of it) executes as."""

    step: int
    node: str
    task_id: Any
    #: Every trigger channel of the node, sorted, as Pregel lists them.
    triggers: Optional[Tuple[str, ...]] = None
    #: The task's ``langgraph_path`` when it is not the node's own (a ``Send``).
    path: Optional[Tuple[Any, ...]] = None

    @property
    def checkpoint_ns(self) -> str:
        return f"{self.node}:{self.task_id}"

    def metadata(self) -> Dict[str, Any]:
        """The ``langgraph_*`` keys Pregel sets on the task's config metadata."""
        return {
            # Pregel numbers the first node after START as step 1.
            "langgraph_step": display_step(self.step),
            "langgraph_node": self.node,
            "langgraph_triggers": self.triggers or (f"branch:to:{self.node}",),
            "langgraph_path": self.path or ("__pregel_pull", self.node),
            "langgraph_checkpoint_ns": self.checkpoint_ns,
        }

    def config_metadata(
        self,
        config: Dict[str, Any],
        static: "StaticConfig",
        thread_id: Optional[str],
    ) -> Dict[str, Any]:
        """The metadata a task's config carries, as native LangGraph builds it.

        The primitive configurable values, then the graph's, the caller's and
        the node's own metadata, then the task's ``langgraph_*`` keys.
        """
        return {
            **configurable_metadata(config, thread_id),
            **static.metadata(config.get("metadata")),
            **self.metadata(),
        }


def configurable_metadata(
    config: Dict[str, Any], thread_id: Optional[str]
) -> Dict[str, Any]:
    """What native LangGraph copies from ``configurable`` into metadata.

    ``thread_id`` and the other primitive, non-private configurable values,
    so a client keying on ``metadata["thread_id"]`` sees what it would from
    ``CompiledStateGraph.astream``.
    """
    derived = {
        key: value
        for key, value in (config.get("configurable") or {}).items()
        if not key.startswith("__")
        and isinstance(value, (str, int, float, bool))
        and key not in _UNTRACED_CONFIGURABLE
    }
    if thread_id:
        derived.setdefault("thread_id", thread_id)
    return derived
