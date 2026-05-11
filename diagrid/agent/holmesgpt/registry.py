# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Worker-local HolmesGPT primitives.

Activities reuse the same ``HolmesRegistry`` instance for the lifetime of
the worker process, so the slow paths (loading toolsets, validating LLM
config, running prerequisite checks) only run once at startup.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


_HOLMES_INSTALL_HINT = (
    "HolmesGPT is required for the diagrid.agent.holmesgpt integration. "
    "Install it from PyPI (`pip install holmesgpt`) or from source "
    "(https://github.com/HolmesGPT/holmesgpt) before importing this module."
)


def _import_holmes() -> Any:
    """Lazy-import HolmesGPT with a clean error message if missing."""
    try:
        import holmes  # noqa: F401  (sanity-check import)
        from holmes.config import Config
        from holmes.core.tools import (
            PrerequisiteCacheMode,
            ToolsetTag,
        )
    except ImportError as e:
        raise ImportError(_HOLMES_INSTALL_HINT) from e

    return Config, ToolsetTag, PrerequisiteCacheMode


@dataclass
class HolmesRegistry:
    """Cached HolmesGPT primitives for use inside Dapr Workflow activities.

    Holds both the assembled ``ToolCallingLLM`` (needed by HolmesGPT's
    ``build_chat_messages`` to render the system prompt + user content) and
    its component ``llm`` / ``tool_executor`` for direct use inside
    activities. The ``skills`` catalog is loaded once at build time so the
    runner can pass it into prompt construction without re-walking config.
    """

    config: Any
    ai: Any  # holmes.core.tool_calling_llm.ToolCallingLLM
    llm: Any
    tool_executor: Any
    openai_tools: List[dict]
    skills: Any = None  # Optional[holmes.plugins.skills.SkillCatalog]

    @classmethod
    def build(
        cls,
        *,
        config_path: Optional[str] = None,
        model: Optional[str] = None,
        toolset_tags: Optional[List[str]] = None,
        enable_all_toolsets_possible: bool = False,
    ) -> "HolmesRegistry":
        Config, ToolsetTag, PrerequisiteCacheMode = _import_holmes()

        if config_path is None:
            cfg = Config.load_from_env()
        else:
            cfg = Config.load_from_file(config_path)

        if model:
            cfg.model = model
            # When the caller is explicit about which model to use, don't try
            # to silently fall back to Robusta AI (which requires Robusta DAL
            # credentials we don't carry in this integration).
            cfg.should_try_robusta_ai = False

        tags = [ToolsetTag(t) for t in (toolset_tags or ["core", "cluster"])]

        ai = cfg.create_toolcalling_llm(
            dal=None,
            toolset_tag_filter=tags,
            enable_all_toolsets_possible=enable_all_toolsets_possible,
            prerequisite_cache=PrerequisiteCacheMode.DISABLED,
            reuse_executor=True,
            model=model,
        )

        openai_tools = ai.tool_executor.get_all_tools_openai_format(
            include_restricted=True,
            user_id=None,
        )

        skills = None
        try:
            skills = cfg.get_skill_catalog()
        except Exception as e:
            logger.debug("Skill catalog unavailable: %s", e)

        logger.info(
            "HolmesRegistry built: model=%s tools=%d skills=%s",
            getattr(ai.llm, "model", "<unknown>"),
            len(openai_tools),
            "yes" if skills else "no",
        )

        return cls(
            config=cfg,
            ai=ai,
            llm=ai.llm,
            tool_executor=ai.tool_executor,
            openai_tools=openai_tools,
            skills=skills,
        )


# Process-wide singleton, populated by the runner before activities run.
_REGISTRY: Optional[HolmesRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def set_registry(registry: HolmesRegistry) -> None:
    """Install the worker-local registry. Called by the runner at start()."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = registry


def get_registry() -> HolmesRegistry:
    """Return the worker-local registry, raising if not installed."""
    reg = _REGISTRY
    if reg is None:
        raise RuntimeError(
            "HolmesRegistry not initialized. Did you call DaprWorkflowHolmesRunner.start()?"
        )
    return reg
