# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from .base import BaseAgentMapper

__all__ = [
    "BaseAgentMapper",
    "LangGraphMapper",
    "StrandsMapper",
    "CrewAIMapper",
    "ADKMapper",
    "OpenAIAgentsMapper",
    "PydanticAIMapper",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "LangGraphMapper":
        from .langgraph import LangGraphMapper

        return LangGraphMapper
    if name == "StrandsMapper":
        from .strands import StrandsMapper

        return StrandsMapper
    if name == "CrewAIMapper":
        from .crewai import CrewAIMapper

        return CrewAIMapper
    if name == "ADKMapper":
        from .adk import ADKMapper

        return ADKMapper
    if name == "OpenAIAgentsMapper":
        from .openai import OpenAIAgentsMapper

        return OpenAIAgentsMapper
    if name == "PydanticAIMapper":
        from .pydantic_ai import PydanticAIMapper

        return PydanticAIMapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
