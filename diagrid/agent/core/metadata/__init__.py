from .introspection import detect_framework, find_agent_in_stack


__all__ = [
    "AgentRegistryAdapter",
    "find_agent_in_stack",
    "detect_framework",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "AgentRegistryAdapter":
        from .metadata import AgentRegistryAdapter

        return AgentRegistryAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
