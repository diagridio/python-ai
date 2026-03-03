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
        name = self._extract_name(agent)
        model_str, provider = self._extract_model_info(agent)
        system_prompt = self._extract_system_prompt(agent)
        tools_metadata = self._extract_tools(agent)

        llm_metadata = LLMMetadata(
            client="pydantic_ai",
            provider=provider,
            api="chat",
            model=model_str,
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
            max_iterations=25,
            registry=RegistryMetadata(
                statestore=None,
                name="default",
            ),
            agent_metadata={
                "framework": "pydantic-ai",
                "name": name,
                "model": model_str,
            },
        )

    def _extract_name(self, agent: Any) -> str:
        """Extract agent name."""
        return getattr(agent, "name", None) or "pydantic-ai-agent"

    def _extract_model_info(self, agent: Any) -> tuple[str, str]:
        """Extract model string and provider from the agent.

        Returns:
            Tuple of (model_str, provider).
        """
        model = getattr(agent, "model", None)
        if model is None:
            return ("unknown", "unknown")

        # If model is a string like "openai:gpt-4o-mini"
        if isinstance(model, str):
            if ":" in model:
                provider = model.split(":")[0]
                return (model, provider)
            return (model, "unknown")

        # Model is a pydantic-ai Model object
        model_str = getattr(model, "model_id", None) or getattr(
            model, "model_name", None
        )
        if model_str is None:
            model_str = str(model)

        provider = getattr(model, "system", "unknown") or "unknown"
        return (model_str, provider)

    def _extract_system_prompt(self, agent: Any) -> str:
        """Extract system prompt from the agent.

        Checks _system_prompts tuple first, then _instructions list.
        """
        # Check _system_prompts (tuple of static strings)
        system_prompts = getattr(agent, "_system_prompts", ())
        if system_prompts:
            parts = [sp for sp in system_prompts if isinstance(sp, str)]
            if parts:
                return "\n".join(parts)

        # Check _instructions (list of strings or callables)
        instructions = getattr(agent, "_instructions", [])
        if instructions:
            parts = [inst for inst in instructions if isinstance(inst, str)]
            if parts:
                return "\n".join(parts)

        return ""

    def _extract_tools(self, agent: Any) -> list[ToolMetadata]:
        """Extract tool metadata from the agent."""
        tools_metadata: list[ToolMetadata] = []

        # pydantic-ai v1.61.0+: tools live in _function_toolset.tools
        toolset = getattr(agent, "_function_toolset", None)
        if toolset is not None:
            function_tools = getattr(toolset, "tools", {}) or {}
        else:
            # Fallback for older pydantic-ai versions
            function_tools = getattr(agent, "_function_tools", {}) or {}

        for tool_name, tool_info in function_tools.items():
            tool_description = getattr(tool_info, "description", "") or ""

            # Extract parameter schema
            tool_args = ""
            func_schema = getattr(tool_info, "function_schema", None)
            if func_schema is not None:
                json_schema = getattr(func_schema, "json_schema", None)
                if json_schema is not None:
                    import json

                    try:
                        tool_args = json.dumps(json_schema)
                    except (TypeError, ValueError):
                        pass

            tools_metadata.append(
                ToolMetadata(
                    tool_name=str(tool_name),
                    tool_description=str(tool_description),
                    tool_args=tool_args,
                )
            )

        return tools_metadata
