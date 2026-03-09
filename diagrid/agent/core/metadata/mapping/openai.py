# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import logging
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

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

if TYPE_CHECKING:
    from agents import Agent

logger = logging.getLogger(__name__)


class OpenAIAgentsMapper(BaseAgentMapper):
    def __init__(self) -> None:
        pass

    def map_agent_metadata(
        self, agent: Any, schema_version: str
    ) -> AgentMetadataSchema:
        """Map OpenAI Agents SDK Agent to AgentMetadataSchema.

        Args:
            agent: An agents.Agent instance
            schema_version: Version of the schema

        Returns:
            AgentMetadataSchema with extracted metadata
        """
        # Basic agent info
        name = getattr(agent, "name", "agent")
        model = getattr(agent, "model", "gpt-4o-mini")
        instructions = getattr(agent, "instructions", "") or ""

        # Tools
        tools_metadata = []
        tools = getattr(agent, "tools", []) or []
        for tool in tools:
            tool_name = getattr(tool, "name", None) or type(tool).__name__
            tool_description = getattr(tool, "description", "")

            tools_metadata.append(
                ToolMetadata(
                    name=str(tool_name),
                    description=str(tool_description),
                    args="",
                )
            )

        # LLM Metadata
        llm_metadata = LLMMetadata(
            client="openai_agents_sdk",
            provider="openai",
            api="chat",
            model=str(model),
        )

        return AgentMetadataSchema(
            version=schema_version,
            agent=AgentMetadata(
                appid="",
                type="Agent",
                orchestrator=False,
                role=str(name),
                goal=str(instructions),
                instructions=[str(instructions)] if instructions else None,
                framework=SupportedFrameworks.OPENAI,
                system_prompt=str(instructions) if instructions else None,
                tool_choice="auto" if tools_metadata else None,
                max_iterations=25,
                metadata={
                    "framework": "openai-agents",
                    "name": name,
                    "model": str(model),
                },
            ),
            name=f"openai-agents-{name}",
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
