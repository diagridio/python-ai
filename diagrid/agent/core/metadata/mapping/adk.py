import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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
    from google.adk.agents.llm_agent import LlmAgent

logger = logging.getLogger(__name__)


class ADKMapper(BaseAgentMapper):
    def __init__(self) -> None:
        pass

    def map_agent_metadata(
        self, agent: Any, schema_version: str
    ) -> AgentMetadataSchema:
        """Map Google ADK Agent to AgentMetadataSchema.

        Args:
            agent: A google.adk.agents.llm_agent.LlmAgent instance
            schema_version: Version of the schema

        Returns:
            AgentMetadataSchema with extracted metadata
        """
        # Basic agent info
        name = getattr(agent, "name", "agent")
        model = getattr(agent, "model", "gemini-2.0-flash")

        # System instruction
        system_instruction = None
        if hasattr(agent, "instruction"):
            system_instruction = getattr(agent, "instruction", None)
        elif hasattr(agent, "system_instruction"):
            system_instruction = getattr(agent, "system_instruction", None)

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
            client="google_genai",
            provider="google",
            api="chat",
            model=str(model),
        )

        return AgentMetadataSchema(
            version=schema_version,
            agent=AgentMetadata(
                appid="",
                type="LlmAgent",
                orchestrator=False,
                role=str(name),
                goal=str(system_instruction) if system_instruction else "",
                instructions=[str(system_instruction)] if system_instruction else None,
                framework=SupportedFrameworks.ADK,
                system_prompt=str(system_instruction) if system_instruction else None,
                tool_choice="auto" if tools_metadata else None,
                max_iterations=100,
                metadata={
                    "framework": "adk",
                    "name": name,
                    "model": str(model),
                },
            ),
            name=f"adk-{name}",
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
