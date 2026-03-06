# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import json
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
from diagrid.agent.pydantic_ai.utils import get_pydantic_ai_tools

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
                    name=str(tool_name),
                    description=str(tool_description),
                    args="",
                )
            )

        # LLM Metadata - extract provider from model string (e.g. "openai:gpt-4o")
        model_str = str(model)
        if ":" in model_str:
            provider = model_str.split(":")[0]
        else:
            provider = "openai"

        llm_metadata = LLMMetadata(
            client="pydantic_ai",
            provider=provider,
            api="chat",
            model=model_str,
        )

        return AgentMetadataSchema(
            version=schema_version,
            agent=AgentMetadata(
                appid="",
                type="Agent",
                orchestrator=False,
                role=str(name),
                goal=str(system_prompt),
                instructions=[str(system_prompt)] if system_prompt else None,
                framework=SupportedFrameworks.PYDANTIC_AI,
                system_prompt=str(system_prompt) if system_prompt else None,
                tool_choice="auto" if tools_metadata else None,
                max_iterations=25,
                metadata={
                    "framework": "pydantic-ai",
                    "name": name,
                    "model": str(model),
                },
            ),
            name=f"pydantic-ai-{name}",
            registered_at=datetime.now(timezone.utc).isoformat(),
            pubsub=None,
            memory=MemoryMetadata(
                short_term=MemoryStoreMetadata(type="DaprWorkflow"),
            ),
            llm=llm_metadata,
            tools=tools_metadata,
            tool_choice="auto" if tools_metadata else None,
            registry=RegistryMetadata(
                resource_name=None,
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

        function_tools = get_pydantic_ai_tools(agent)

        for tool_name, tool_info in function_tools.items():
            tool_description = getattr(tool_info, "description", "") or ""

            # Extract parameter schema
            tool_args = ""
            func_schema = getattr(tool_info, "function_schema", None)
            if func_schema is not None:
                json_schema = getattr(func_schema, "json_schema", None)
                if json_schema is not None:
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
