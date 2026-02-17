from .metadata import (
    detect_framework,
    find_agent_in_stack,
)
from .types import (
    AgentMetadata,
    AgentMetadataSchema,
    LLMMetadata,
    MemoryMetadata,
    PubSubMetadata,
    RegistryMetadata,
    SupportedFrameworks,
    ToolMetadata,
)

__all__ = [
    "SupportedFrameworks",
    "AgentMetadataSchema",
    "AgentMetadata",
    "LLMMetadata",
    "PubSubMetadata",
    "ToolMetadata",
    "RegistryMetadata",
    "MemoryMetadata",
    "AgentRegistryAdapter",
    "find_agent_in_stack",
    "detect_framework",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "AgentRegistryAdapter":
        from .metadata import AgentRegistryAdapter

        return AgentRegistryAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
