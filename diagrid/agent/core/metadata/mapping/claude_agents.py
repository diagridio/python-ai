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


class ClaudeAgentsMapper(BaseAgentMapper):
    """Mapper for Claude Agent SDK agents.

    The Claude integration does not expose a separate ``Agent`` object — the
    runner itself (``diagrid.agent.claude_agents.DaprWorkflowAgentRunner``)
    holds the configuration. This mapper reads metadata directly off the
    runner instance.
    """

    def __init__(self) -> None:
        pass

    def map_agent_metadata(
        self, agent: Any, schema_version: str, *, name: Optional[str] = None
    ) -> AgentMetadataSchema:
        """Map a Claude Agents DaprWorkflowAgentRunner to AgentMetadataSchema.

        Args:
            agent: A ``DaprWorkflowAgentRunner`` instance from
                ``diagrid.agent.claude_agents``.
            schema_version: Version of the schema.
            name: Runner-provided canonical name.

        Returns:
            AgentMetadataSchema with extracted metadata.
        """
        agent_name = getattr(agent, "_name", None) or "agent"
        model = getattr(agent, "_model", None) or "claude-sonnet-4-6"
        system_prompt = getattr(agent, "_system_prompt", "") or ""
        max_iterations = getattr(agent, "_max_iterations", 25)
        tools = getattr(agent, "_tools", []) or []

        tools_metadata = []
        for tool in tools:
            tool_name = (
                getattr(tool, "name", None)
                or getattr(tool, "__name__", None)
                or type(tool).__name__
            )
            tool_description = getattr(tool, "description", "") or ""
            tools_metadata.append(
                ToolMetadata(
                    name=str(tool_name),
                    description=str(tool_description),
                    args="",
                )
            )

        llm_metadata = LLMMetadata(
            client="claude_agent_sdk",
            provider="anthropic",
            api="messages",
            model=str(model),
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
                framework=SupportedFrameworks.CLAUDE_AGENTS,
                system_prompt=str(system_prompt) if system_prompt else None,
                tool_choice="auto" if tools_metadata else None,
                max_iterations=max_iterations,
                metadata={
                    "framework": "claude-agents",
                    "name": agent_name,
                    "model": str(model),
                },
            ),
            name=name or f"claude-agents-{agent_name}",
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
