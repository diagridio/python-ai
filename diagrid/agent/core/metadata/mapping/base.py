# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1


from abc import ABC, abstractmethod
from typing import Any

from dapr_agents import AgentMetadataSchema


class BaseAgentMapper(ABC):
    """Abstract base class for agent metadata mappers.

    Provides common functionality for extracting metadata from different
    agent frameworks (Strands, LangGraph, Dapr Agents).
    """

    @staticmethod
    def _extract_provider(module_name: str) -> str:
        """Extract provider name from module path.

        Args:
            module_name: Python module name (e.g., 'langchain_openai.chat_models')

        Returns:
            Provider identifier (e.g., 'openai', 'anthropic', 'azure_openai')
        """
        module_lower = module_name.lower()

        # Check more specific providers first
        if "vertexai" in module_lower:
            return "vertexai"
        elif "bedrock" in module_lower:
            return "bedrock"
        elif "azure" in module_lower:
            return "azure_openai"
        elif "openai" in module_lower:
            return "openai"
        elif "anthropic" in module_lower:
            return "anthropic"
        elif "ollama" in module_lower:
            return "ollama"
        elif "google" in module_lower or "gemini" in module_lower:
            return "google"
        elif "cohere" in module_lower:
            return "cohere"

        return "unknown"

    @abstractmethod
    def map_agent_metadata(
        self, agent: Any, schema_version: str
    ) -> AgentMetadataSchema:
        """Map agent to standardized metadata schema.

        Args:
            agent: Framework-specific agent instance
            schema_version: Schema version to use

        Returns:
            AgentMetadataSchema with extracted metadata
        """
        pass
