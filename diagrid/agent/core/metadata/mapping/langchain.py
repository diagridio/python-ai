# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from diagrid.agent.core.metadata.mapping.base import BaseAgentMapper
from diagrid.agent.core.types import SupportedFrameworks
from dapr_agents import (
    AgentMetadata,
    AgentMetadataSchema,
    LLMMetadata,
    MemoryMetadata,
    RegistryMetadata,
    ToolMetadata,
)
from dapr_agents.agents.configs import MemoryStoreMetadata

logger = logging.getLogger(__name__)


class LangChainMapper(BaseAgentMapper):
    """Mapper for LangChain agents.

    LangChain has no standalone native ``Agent`` object to introspect (its
    own ``create_agent`` compiles to a LangGraph graph) — the runner itself
    (``diagrid.agent.langchain.DaprWorkflowAgentRunner``) holds the
    configuration. This mapper reads metadata directly off the runner
    instance, the same way ``ClaudeAgentsMapper`` does.
    """

    def __init__(self) -> None:
        pass

    def map_agent_metadata(
        self, agent: Any, schema_version: str, *, name: Optional[str] = None
    ) -> AgentMetadataSchema:
        """Map a LangChain DaprWorkflowAgentRunner to AgentMetadataSchema.

        Args:
            agent: A ``DaprWorkflowAgentRunner`` instance from
                ``diagrid.agent.langchain``.
            schema_version: Version of the schema.
            name: Runner-provided canonical name.

        Returns:
            AgentMetadataSchema with extracted metadata.
        """
        agent_name = getattr(agent, "_name", None) or "agent"
        model = getattr(agent, "_model", None)
        system_prompt = getattr(agent, "_system_prompt", "") or ""
        max_iterations = getattr(agent, "_max_iterations", 25)
        tools = getattr(agent, "_tools", []) or []

        tools_metadata = []
        for tool in tools:
            tool_name = getattr(tool, "name", None) or type(tool).__name__
            tool_description = getattr(tool, "description", "") or ""
            tools_metadata.append(
                ToolMetadata(
                    name=str(tool_name),
                    description=str(tool_description),
                    args="",
                )
            )

        model_id = "unknown"
        provider = "unknown"
        if model is not None:
            model_id = (
                getattr(model, "model_name", None)
                or getattr(model, "model", None)
                or type(model).__name__
            )
            provider = self._extract_provider(type(model).__module__)
            if provider == "unknown":
                provider = type(model).__name__.replace("Chat", "").lower()

        llm_metadata = LLMMetadata(
            client=type(model).__name__ if model is not None else "unknown",
            provider=provider,
            api="chat",
            model=str(model_id),
        )

        return AgentMetadataSchema(
            version=schema_version,
            agent=AgentMetadata(
                appid="",
                type="Agent",
                orchestrator=False,
                role=str(agent_name),
                goal=str(system_prompt),
                instructions=[str(system_prompt)] if system_prompt else None,
                framework=SupportedFrameworks.LANGCHAIN,
                system_prompt=str(system_prompt) if system_prompt else None,
                tool_choice="auto" if tools_metadata else None,
                max_iterations=max_iterations,
                metadata={
                    "framework": "langchain",
                    "name": agent_name,
                    "model": str(model_id),
                },
            ),
            name=name or f"langchain-{agent_name}",
            registered_at=datetime.now(timezone.utc).isoformat(),
            pubsub=None,
            memory=MemoryMetadata(
                short_term=MemoryStoreMetadata(type="DaprWorkflow"),
            ),
            llm=llm_metadata,
            tools=tools_metadata,
            registry=RegistryMetadata(
                resource_name=None,
                name="default",
            ),
        )
