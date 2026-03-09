# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""DaprLLM — placeholder LLM for CrewAI agents that route through Dapr.

CrewAI eagerly validates the LLM at Agent construction time. When no LLM
is provided, it tries to use OpenAI and fails without an API key. This
module provides two mechanisms to avoid that crash:

1. **Auto-discovery (zero-config)** — ``_patch_crewai_llm_fallback()`` is
   called at ``diagrid.agent.crewai`` import time. It patches CrewAI's
   internal ``_llm_via_environment_or_fallback`` so that when the original
   fallback would raise, a ``DaprLLM()`` is returned instead. This means
   ``Agent(role=..., goal=..., backstory=...)`` just works with no ``llm=``.

2. **Explicit** — pass ``llm=DaprLLM()`` (or ``DaprLLM(component_name=...)``)
   to the CrewAI ``Agent`` constructor.

Usage (auto-discovery)::

    from diagrid.agent.crewai import DaprWorkflowAgentRunner  # patches fallback
    from crewai import Agent

    agent = Agent(role="Scout", goal="Find venues", backstory="Expert")

Usage (explicit)::

    from crewai import Agent
    from diagrid.agent.crewai import DaprLLM

    agent = Agent(role="Scout", goal="Find venues", backstory="Expert",
                  llm=DaprLLM(component_name="llm-provider"))
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DaprLLM:
    """Placeholder LLM that passes CrewAI's ``create_llm`` validation.

    ``create_llm`` checks ``isinstance(llm, (LLM, BaseLLM))`` first and
    returns matching instances immediately. For other objects it extracts
    ``model``, ``temperature``, etc. and tries to build a real ``LLM``,
    which fails without API keys.

    By subclassing CrewAI's ``LLM`` at runtime (so the import is lazy)
    and overriding ``__new__`` to skip provider validation, instances pass
    the ``isinstance`` check and are returned untouched.

    Args:
        component_name: Explicit Dapr conversation component name.
            When ``None``, the runner auto-discovers one from the sidecar.
    """

    _crewai_subclass: Optional[type] = None

    def __new__(cls, component_name: Optional[str] = None) -> "DaprLLM":
        # Lazily create a subclass of CrewAI's LLM so isinstance() passes.
        klass = cls._ensure_crewai_subclass()
        instance: DaprLLM = object.__new__(klass)
        return instance

    def __init__(self, component_name: Optional[str] = None) -> None:
        self.component_name = component_name
        # Minimal attributes CrewAI might inspect on an LLM instance.
        self.model = "dapr/conversation"
        self.stop: list[str] = []
        self.callbacks: list[Any] = []

    @classmethod
    def _ensure_crewai_subclass(cls) -> type:
        """Create (once) a subclass of crewai.LLM that skips __new__."""
        if cls._crewai_subclass is not None:
            return cls._crewai_subclass

        try:
            from crewai.llm import LLM as CrewAILLM

            class _DaprLLMBridge(CrewAILLM, DaprLLM):  # type: ignore[misc]
                """Runtime subclass so isinstance(inst, CrewAILLM) is True."""

                def __new__(inner_cls, *_args: Any, **_kwargs: Any) -> "_DaprLLMBridge":
                    return object.__new__(inner_cls)

                def __init__(self, component_name: Optional[str] = None) -> None:
                    # Skip CrewAILLM.__init__, only run DaprLLM.__init__.
                    DaprLLM.__init__(self, component_name=component_name)

                def call(self, *_args: Any, **_kwargs: Any) -> Any:
                    raise RuntimeError(
                        "DaprLLM is a placeholder and should not be called directly. "
                        "Use DaprWorkflowAgentRunner which routes LLM calls through "
                        "the Dapr Conversation API."
                    )

            cls._crewai_subclass = _DaprLLMBridge
        except ImportError:
            logger.debug(
                "crewai not installed; DaprLLM will not pass isinstance checks"
            )
            cls._crewai_subclass = cls

        return cls._crewai_subclass


_patched = False


def _patch_crewai_llm_fallback() -> None:
    """Patch CrewAI's ``_llm_via_environment_or_fallback`` to return ``DaprLLM()``.

    Only intercepts the ``llm=None`` path; explicit LLM configs are untouched.
    The patch activates only when the original fallback would have raised
    (e.g. no ``OPENAI_API_KEY``), so it is safe when a real API key is set.

    Called automatically when ``diagrid.agent.crewai`` is imported.
    """
    global _patched
    if _patched:
        return

    try:
        import crewai.utilities.llm_utils as _llm_mod

        _original = _llm_mod._llm_via_environment_or_fallback

        def _dapr_fallback() -> Any:
            try:
                return _original()
            except Exception:
                logger.info(
                    "CrewAI LLM fallback failed; substituting DaprLLM "
                    "(Dapr Conversation API will be auto-discovered at runtime)."
                )
                return DaprLLM()

        _llm_mod._llm_via_environment_or_fallback = _dapr_fallback  # type: ignore[attr-defined]
        _patched = True
        logger.debug("Patched CrewAI LLM fallback with DaprLLM")
    except (ImportError, AttributeError):
        logger.debug("Could not patch CrewAI LLM fallback; crewai may not be installed")
