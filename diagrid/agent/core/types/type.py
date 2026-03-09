# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from enum import StrEnum


class SupportedFrameworks(StrEnum):
    DAPR_AGENTS = "Dapr Agents"
    LANGGRAPH = "LangGraph"
    STRANDS = "Strands"
    CREWAI = "CrewAI"
    ADK = "ADK"
    OPENAI = "OpenAI"
    PYDANTIC_AI = "PydanticAI"
