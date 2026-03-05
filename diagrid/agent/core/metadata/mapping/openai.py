# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import logging
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from diagrid.agent.core.metadata.mapping.base import BaseAgentMapper
from diagrid.agent.core.types import (
    AgentMetadata,
    AgentMetadataSchema,
    LLMMetadata,
    MemoryMetadata,
    RegistryMetadata,
    SupportedFrameworks,
    ToolMetadata,
)

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
                    tool_name=str(tool_name),
                    tool_description=str(tool_description),
                    tool_args="",
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
            schema_version=schema_version,
            agent=AgentMetadata(
                appid="",
                type="Agent",
                orchestrator=False,
                role=str(name),
                goal=str(instructions),
                instructions=[str(instructions)] if instructions else None,
                framework=SupportedFrameworks.OPENAI,
                system_prompt=str(instructions) if instructions else None,
            ),
            name=f"openai-agents-{name}",
            registered_at=datetime.now(timezone.utc).isoformat(),
            pubsub=None,
            memory=MemoryMetadata(
                type="DaprWorkflow",
            ),
            llm=llm_metadata,
            tools=tools_metadata,
            tool_choice="auto" if tools_metadata else None,
            max_iterations=25,  # Default
            registry=RegistryMetadata(
                statestore=None,
                name="default",
            ),
            agent_metadata={
                "framework": "openai-agents",
                "name": name,
                "model": str(model),
            },
        )
