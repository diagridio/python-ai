from .base import BaseAgentMapper

__all__ = [
    "BaseAgentMapper",
    "LangGraphMapper",
    "StrandsMapper",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "LangGraphMapper":
        from .langgraph import LangGraphMapper

        return LangGraphMapper
    if name == "StrandsMapper":
        from .strands import StrandsMapper

        return StrandsMapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
