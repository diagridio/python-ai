from enum import StrEnum


class SupportedFrameworks(StrEnum):
    DAPR_AGENTS = "Dapr Agents"
    LANGGRAPH = "LangGraph"
    STRANDS = "Strands"
    CREWAI = "CrewAI"
    ADK = "ADK"
    OPENAI = "OpenAI"
