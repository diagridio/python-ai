# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the worker-local ``HolmesRegistry`` singleton."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from diagrid.agent.holmesgpt import registry as registry_module
from diagrid.agent.holmesgpt.registry import HolmesRegistry


def test_set_and_get_registry_round_trip():
    fake = HolmesRegistry(
        config=None, ai=None, llm=None, tool_executor=None, openai_tools=[]
    )
    original = registry_module._REGISTRY
    try:
        registry_module.set_registry(fake)
        assert registry_module.get_registry() is fake
    finally:
        registry_module._REGISTRY = original


def test_get_registry_raises_when_not_initialized():
    original = registry_module._REGISTRY
    try:
        registry_module._REGISTRY = None
        with pytest.raises(RuntimeError, match="not initialized"):
            registry_module.get_registry()
    finally:
        registry_module._REGISTRY = original


def _fake_config(
    *,
    model=None,
    should_try_robusta_ai=True,
    tools_returned=None,
    skill_catalog=None,
    skill_catalog_raises=False,
):
    """Build a fake holmes.config.Config drop-in.

    Tests patch ``registry.Config`` to a class whose ``load_from_env`` returns
    this fake. The fake exposes the subset of ``Config`` the registry touches:
    ``model``, ``should_try_robusta_ai``, ``get_skill_catalog``, and
    ``create_toolcalling_llm``.
    """

    captured = {"create_toolcalling_llm_kwargs": None}

    class FakeConfig:
        def __init__(self):
            self.model = model
            self.should_try_robusta_ai = should_try_robusta_ai
            captured["instance"] = self

        @classmethod
        def load_from_env(cls):
            return cls()

        @classmethod
        def load_from_file(cls, _path):
            return cls()

        def get_skill_catalog(self):
            if skill_catalog_raises:
                raise RuntimeError("skill catalog not available")
            return skill_catalog

        def create_toolcalling_llm(self, **kwargs):
            captured["create_toolcalling_llm_kwargs"] = kwargs
            return SimpleNamespace(
                llm=SimpleNamespace(model=kwargs.get("model") or self.model or "x"),
                tool_executor=SimpleNamespace(
                    get_all_tools_openai_format=lambda **_: tools_returned or [],
                ),
            )

    return FakeConfig, captured


class _FakeTag(str):
    """``str`` subclass standing in for ``holmes.core.tools.ToolsetTag``."""

    def __new__(cls, v):
        return str.__new__(cls, v)


_FAKE_PREREQ_MODE = SimpleNamespace(DISABLED="disabled")


def _patch_holmes(FakeConfig):
    """Patch the registry module's holmes-imported names for one test."""
    return mock.patch.multiple(
        registry_module,
        Config=FakeConfig,
        ToolsetTag=_FakeTag,
        PrerequisiteCacheMode=_FAKE_PREREQ_MODE,
    )


def test_build_disables_robusta_ai_when_model_is_explicit():
    """If the user passed a model, the integration must disable Robusta AI."""
    FakeConfig, captured = _fake_config(
        should_try_robusta_ai=True, tools_returned=[{"name": "bash"}]
    )
    with _patch_holmes(FakeConfig):
        reg = HolmesRegistry.build(model="anthropic/foo")

    inst = captured["instance"]
    assert inst.model == "anthropic/foo"
    assert inst.should_try_robusta_ai is False
    assert reg.openai_tools == [{"name": "bash"}]
    assert reg.llm is reg.ai.llm


def test_build_loads_skill_catalog_when_available():
    FakeConfig, _ = _fake_config(
        model="anthropic/foo",
        should_try_robusta_ai=False,
        skill_catalog="my-skill-catalog",
    )
    with _patch_holmes(FakeConfig):
        reg = HolmesRegistry.build(model="anthropic/foo")
    assert reg.skills == "my-skill-catalog"


def test_build_swallows_skill_catalog_errors():
    """If skill catalog loading raises, the registry should still build."""
    FakeConfig, _ = _fake_config(
        model="anthropic/foo",
        should_try_robusta_ai=False,
        skill_catalog_raises=True,
    )
    with _patch_holmes(FakeConfig):
        reg = HolmesRegistry.build(model="anthropic/foo")
    assert reg.skills is None
