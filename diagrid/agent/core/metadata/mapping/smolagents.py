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

_FINAL_ANSWER_TOOL_NAME = "final_answer"


class SmolagentsMapper(BaseAgentMapper):
    """Mapper for smolagents ``ToolCallingAgent`` instances."""

    def __init__(self) -> None:
        pass

    def map_agent_metadata(
        self, agent: Any, schema_version: str, *, name: Optional[str] = None
    ) -> AgentMetadataSchema:
        """Map a smolagents ToolCallingAgent to AgentMetadataSchema.

        Args:
            agent: A ``smolagents.ToolCallingAgent`` instance.
            schema_version: Version of the schema.
            name: Runner-provided canonical name.

        Returns:
            AgentMetadataSchema with extracted metadata.
        """
        agent_name = getattr(agent, "name", None) or "agent"

        system_prompt: Optional[str] = None
        try:
            system_prompt = agent.system_prompt
        except Exception:
            logger.debug("Failed to read agent.system_prompt", exc_info=True)

        max_steps = getattr(agent, "max_steps", None)

        tools = getattr(agent, "tools", {}) or {}
        tools_metadata = []
        for tool_name, tool in tools.items():
            if tool_name == _FINAL_ANSWER_TOOL_NAME:
                # Auto-injected by smolagents on every agent; not part of the
                # user-declared tool set.
                continue
            tools_metadata.append(
                ToolMetadata(
                    name=str(tool_name),
                    description=str(getattr(tool, "description", "") or ""),
                    args=str(getattr(tool, "inputs", {}) or {}),
                )
            )

        model = getattr(agent, "model", None)
        model_id = "unknown"
        provider = "unknown"
        if model is not None:
            model_id = getattr(model, "model_id", None) or type(model).__name__
            provider = self._extract_provider(type(model).__module__)
            if provider == "unknown":
                provider = type(model).__name__.replace("Model", "").lower()

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
                type="ToolCallingAgent",
                orchestrator=False,
                role=str(agent_name),
                goal=str(system_prompt) if system_prompt else "",
                instructions=[str(system_prompt)] if system_prompt else None,
                framework=SupportedFrameworks.SMOLAGENTS,
                system_prompt=str(system_prompt) if system_prompt else None,
                tool_choice="auto" if tools_metadata else None,
                max_iterations=max_steps,
                metadata={
                    "framework": "smolagents",
                    "name": agent_name,
                    "model": str(model_id),
                },
            ),
            name=name or f"smolagents-{agent_name}",
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
