# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING

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
    from crewai import Agent

logger = logging.getLogger(__name__)


class CrewAIMapper(BaseAgentMapper):
    def __init__(self) -> None:
        pass

    def map_agent_metadata(
        self, agent: Any, schema_version: str
    ) -> AgentMetadataSchema:
        """Map CrewAI Agent to AgentMetadataSchema.

        Args:
            agent: A crewai.Agent instance
            schema_version: Version of the schema

        Returns:
            AgentMetadataSchema with extracted metadata
        """
        # Basic agent info
        role = getattr(agent, "role", "")
        goal = getattr(agent, "goal", "")
        backstory = getattr(agent, "backstory", "")

        # Handle CrewAI's _NotSpecified sentinel
        if type(role).__name__ == "_NotSpecified":
            role = ""
        if type(goal).__name__ == "_NotSpecified":
            goal = ""
        if type(backstory).__name__ == "_NotSpecified":
            backstory = ""

        # LLM Model
        llm = getattr(agent, "llm", None)
        llm_metadata = None

        if llm:
            model = "gpt-4o-mini"  # Default
            if isinstance(llm, str):
                model = llm
            elif hasattr(llm, "model_name"):
                model = llm.model_name
            elif hasattr(llm, "model"):
                model = str(llm.model)

            provider = self._extract_provider(
                type(llm).__module__ if not isinstance(llm, str) else model
            )

            llm_metadata = LLMMetadata(
                client=type(llm).__name__ if not isinstance(llm, str) else "LiteLLM",
                provider=provider,
                api="chat",
                model=model,
            )

        # Tools
        tools_metadata = []
        tools = getattr(agent, "tools", []) or []
        for tool in tools:
            name = getattr(tool, "name", None) or type(tool).__name__
            description = getattr(tool, "description", "")

            tools_metadata.append(
                ToolMetadata(
                    tool_name=str(name),
                    tool_description=str(description),
                    tool_args="",
                )
            )

        agent_id = role.lower().replace(" ", "-") if role else "crewai-agent"
        full_name = f"crewai-{agent_id}"

        return AgentMetadataSchema(
            schema_version=schema_version,
            agent=AgentMetadata(
                appid="",
                type="CrewAI",
                orchestrator=False,
                role=str(role),
                goal=str(goal),
                instructions=[str(backstory)] if backstory else None,
                framework=SupportedFrameworks.CREWAI,
            ),
            name=full_name,
            registered_at=datetime.now(timezone.utc).isoformat(),
            pubsub=None,
            memory=MemoryMetadata(
                type="DaprWorkflow",
            ),
            llm=llm_metadata,
            tools=tools_metadata,
            tool_choice="auto" if tools_metadata else None,
            max_iterations=int(getattr(agent, "max_iter", 25))
            if hasattr(agent, "max_iter")
            else 25,
            registry=RegistryMetadata(
                statestore=None,
                name="default",
            ),
            agent_metadata={
                "framework": "crewai",
                "role": role,
                "goal": goal,
            },
        )
