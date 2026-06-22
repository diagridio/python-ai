# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Metadata mapper for LangChain Deep Agents.

Deep Agents (``deepagents.create_deep_agent``) compile to a LangGraph
``CompiledStateGraph`` assembled on top of LangChain's ``create_agent``. Unlike a
hand-built LangGraph graph, the chat model and assembled system prompt are
captured in the *closure* of the ``"model"`` node's ``RunnableCallable`` rather
than in the node function's module globals. :class:`LangGraphMapper` only scans
``func.__globals__`` for the LLM, so for a Deep Agent it reports
``model="unknown"`` / ``provider="unknown"`` and no system prompt.

This mapper reads the model node's closure directly (type-matching the cell
contents so it survives LangChain renaming its internals) and pulls the tool
list from the ``"tools"`` ``ToolNode``.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from diagrid.agent.core.metadata.mapping.base import BaseAgentMapper
from diagrid.agent.core.types import SupportedFrameworks
from dapr_agents import (
    AgentMetadata,
    AgentMetadataSchema,
    LLMMetadata,
    MemoryMetadata,
    PubSubMetadata,
    RegistryMetadata,
    ToolMetadata,
)
from dapr_agents.agents.configs import MemoryStoreMetadata

if TYPE_CHECKING:
    from dapr.ext.langgraph import DaprCheckpointer

logger = logging.getLogger(__name__)


class DeepAgentsMapper(BaseAgentMapper):
    """Map a compiled Deep Agent graph to the standard metadata schema."""

    def map_agent_metadata(
        self, agent: Any, schema_version: str, *, name: Optional[str] = None
    ) -> AgentMetadataSchema:
        introspected_vars: Dict[str, Any] = vars(agent)
        nodes: Dict[str, Any] = introspected_vars.get("nodes", {})

        tools = self._extract_tools(nodes)
        llm_metadata, system_prompt = self._extract_model_and_prompt(nodes)

        checkpointer: Optional["DaprCheckpointer"] = introspected_vars.get(
            "checkpointer", None
        )

        # Metadata hints set by DaprWorkflowGraphRunner / DaprWorkflowDeepAgentRunner
        custom_name = getattr(agent, "_diagrid_name", None)
        custom_role = getattr(agent, "_diagrid_role", None)
        custom_goal = getattr(agent, "_diagrid_goal", None)

        if name:
            full_name = name
        else:
            raw_name = custom_name or (
                agent.get_name() if hasattr(agent, "get_name") else ""
            )
            agent_id = raw_name.lower().replace(" ", "-") if raw_name else "agent"
            full_name = f"deepagents-{agent_id}"

        role = custom_role or "Assistant"
        goal = custom_goal or self._first_line(system_prompt) or ""

        return AgentMetadataSchema(
            version=schema_version,
            agent=AgentMetadata(
                appid="",
                type=type(agent).__name__,
                orchestrator=False,
                role=role,
                goal=goal,
                instructions=[],
                system_prompt=system_prompt or "",
                framework=SupportedFrameworks.DEEPAGENTS,
                max_iterations=1,
                tool_choice="auto",
                metadata=None,
            ),
            name=full_name,
            registered_at=datetime.now(timezone.utc).isoformat(),
            pubsub=PubSubMetadata(
                resource_name="",
                broadcast_topic=None,
                agent_topic=None,
            ),
            memory=MemoryMetadata(
                short_term=MemoryStoreMetadata(
                    type="DaprCheckpointer",
                    resource_name=checkpointer.store_name if checkpointer else None,  # type: ignore[union-attr]
                ),
            ),
            llm=LLMMetadata(
                client=llm_metadata.get("client", "") if llm_metadata else "",
                provider=llm_metadata.get("provider", "unknown")
                if llm_metadata
                else "unknown",
                api="chat",
                model=llm_metadata.get("model", "unknown")
                if llm_metadata
                else "unknown",
                resource_name=None,
                base_url=llm_metadata.get("base_url") if llm_metadata else None,
                azure_endpoint=None,
                azure_deployment=None,
                prompt_template=None,
            ),
            registry=RegistryMetadata(
                resource_name=None,
                name=None,
            ),
            tools=[
                ToolMetadata(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    args="",
                )
                for tool in tools
            ],
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tools(nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect tool metadata from the graph's ``ToolNode``.

        A Deep Agent exposes its built-in tools (``write_todos``, the filesystem
        tools, ``execute``, ``task``) plus any user tools through a single
        ``ToolNode`` whose ``_tools_by_name`` maps name -> tool.
        """
        tools: List[Dict[str, Any]] = []
        for node_name, node_spec in nodes.items():
            if node_name == "__start__":
                continue
            bound = getattr(node_spec, "bound", None)
            tools_by_name = getattr(bound, "_tools_by_name", None)
            if not tools_by_name:
                continue
            tools.extend(
                {
                    "name": tool_name,
                    "description": getattr(tool, "description", "") or "",
                }
                for tool_name, tool in tools_by_name.items()
            )
        return tools

    def _extract_model_and_prompt(
        self, nodes: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Extract LLM metadata and system prompt from the model node closure.

        The model node's ``RunnableCallable.func`` closes over the chat model
        (a ``BaseChatModel``) and the assembled ``SystemMessage``. We type-match
        on the closure cell contents rather than relying on free-variable names,
        so the mapper survives LangChain renaming its internals.
        """
        for node_name, node_spec in nodes.items():
            if node_name == "__start__":
                continue
            bound = getattr(node_spec, "bound", None)
            if bound is None:
                continue
            # Skip the ToolNode — only the model node's callable holds the model.
            if getattr(bound, "_tools_by_name", None) is not None:
                continue
            model, system_message = self._scan_closure(bound)
            if model is not None:
                return (
                    self._build_llm_metadata(model),
                    self._extract_prompt_text(system_message),
                )
        return None, None

    @staticmethod
    def _scan_closure(bound: Any) -> Tuple[Optional[Any], Optional[Any]]:
        """Return ``(chat_model, system_message)`` found in a callable's closure."""
        model: Optional[Any] = None
        system_message: Optional[Any] = None
        for fn in (getattr(bound, "func", None), getattr(bound, "afunc", None)):
            closure = getattr(fn, "__closure__", None)
            if not closure:
                continue
            for cell in closure:
                try:
                    value = cell.cell_contents
                except ValueError:
                    # Empty cell (unbound free variable) — skip.
                    continue
                if model is None and isinstance(value, BaseChatModel):
                    model = value
                elif system_message is None and isinstance(value, SystemMessage):
                    system_message = value
            if model is not None and system_message is not None:
                break
        return model, system_message

    def _build_llm_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract client / provider / model id / base_url from a chat model."""
        module = type(model).__module__
        model_id = getattr(model, "model_name", None) or getattr(model, "model", None)
        provider = self._provider_from_model(model) or self._extract_provider(module)

        base_url = getattr(model, "base_url", None)
        if base_url is not None and not isinstance(base_url, str):
            base_url = str(base_url)

        return {
            "client": type(model).__name__,
            "provider": provider,
            "model": model_id or "unknown",
            "base_url": base_url,
        }

    @staticmethod
    def _provider_from_model(model: Any) -> Optional[str]:
        """Resolve the provider via the model's LangSmith params (e.g. ``"openai"``).

        All major LangChain chat models override ``_get_ls_params`` to return a
        hardcoded ``ls_provider``. Falls back to ``None`` so the caller can
        derive the provider from the module name instead.
        """
        try:
            ls_params = model._get_ls_params()
        except Exception:  # noqa: BLE001 - defensive; module-name fallback follows
            return None
        provider = ls_params.get("ls_provider") if isinstance(ls_params, dict) else None
        return provider if isinstance(provider, str) and provider else None

    @staticmethod
    def _extract_prompt_text(system_message: Any) -> Optional[str]:
        """Return the system prompt text from a ``SystemMessage``.

        ``content`` is normally a ``str`` but can be a list of content blocks
        (e.g. when Anthropic prompt-cache markers are used).
        """
        if system_message is None:
            return None
        content = getattr(system_message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "\n".join(parts) if parts else None
        return None

    @staticmethod
    def _first_line(text: Optional[str]) -> Optional[str]:
        """Return the first non-empty line of ``text`` (used for the goal)."""
        if not text:
            return None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return None
