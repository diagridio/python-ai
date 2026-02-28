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
    from pydantic_ai import Agent

logger = logging.getLogger(__name__)


class PydanticAIMapper(BaseAgentMapper):
    def __init__(self) -> None:
        pass

    def map_agent_metadata(
        self, agent: Any, schema_version: str
    ) -> AgentMetadataSchema:
        """Map Pydantic AI Agent to AgentMetadataSchema.

        Args:
            agent: A pydantic_ai.Agent instance
            schema_version: Version of the schema

        Returns:
            AgentMetadataSchema with extracted metadata
        """
        # Basic agent info
        name = getattr(agent, "name", None) or "pydantic-ai-agent"
        model = getattr(agent, "model", "gpt-4o-mini")

        # Extract system prompt
        system_prompt = ""
        system_prompts = getattr(agent, "_system_prompts", [])
        if system_prompts:
            prompt_parts = []
            for sp in system_prompts:
                if isinstance(sp, str):
                    prompt_parts.append(sp)
            system_prompt = "\n".join(prompt_parts)

        if not system_prompt:
            system_prompt = getattr(agent, "system_prompt", "") or ""
            if callable(system_prompt):
                try:
                    system_prompt = system_prompt()
                except Exception:
                    system_prompt = ""

        # Tools
        tools_metadata = []
        function_tools = getattr(agent, "_function_tools", {}) or {}
        for tool_name, tool_info in function_tools.items():
            tool_description = getattr(tool_info, "description", "")

            tools_metadata.append(
                ToolMetadata(
                    tool_name=str(tool_name),
                    tool_description=str(tool_description),
                    tool_args="",
                )
            )

        # LLM Metadata
        llm_metadata = LLMMetadata(
            client="pydantic_ai",
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
                goal=str(system_prompt),
                instructions=[str(system_prompt)] if system_prompt else None,
                framework=SupportedFrameworks.PYDANTIC_AI,
                system_prompt=str(system_prompt) if system_prompt else None,
            ),
            name=f"pydantic-ai-{name}",
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
                "framework": "pydantic-ai",
                "name": name,
                "model": str(model),
            },
        )
