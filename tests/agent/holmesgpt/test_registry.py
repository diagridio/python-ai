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


def _build_fake_holmes(
    *,
    cfg_attrs: dict,
    tools_returned=None,
    skill_catalog=None,
    skill_catalog_raises=False,
):
    """Construct fake holmes primitives and a stub for ``_import_holmes``.

    Returns ``(FakeConfig, captured_state)`` where ``captured_state`` is a
    dict the test can inspect after ``build()`` runs (e.g. to confirm flags
    were flipped).
    """

    captured = {"create_toolcalling_llm_kwargs": None, "instance": None}

    class FakeConfig:
        model: str = cfg_attrs.get("model")  # type: ignore[assignment]
        should_try_robusta_ai: bool = cfg_attrs.get("should_try_robusta_ai", True)

        @classmethod
        def load_from_env(cls):
            inst = cls()
            for k, v in cfg_attrs.items():
                setattr(inst, k, v)
            captured["instance"] = inst
            return inst

        @classmethod
        def load_from_file(cls, _path):
            return cls.load_from_env()

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

    class FakeTag(str):
        def __new__(cls, v):
            return str.__new__(cls, v)

    FakePrereqMode = SimpleNamespace(DISABLED="disabled")
    return FakeConfig, FakeTag, FakePrereqMode, captured


def test_build_disables_robusta_ai_when_model_is_explicit():
    """If the user passed a model, the integration must disable Robusta AI."""
    FakeConfig, FakeTag, FakePrereq, captured = _build_fake_holmes(
        cfg_attrs={"model": None, "should_try_robusta_ai": True},
        tools_returned=[{"name": "bash"}],
    )
    with mock.patch.object(
        registry_module,
        "_import_holmes",
        return_value=(FakeConfig, FakeTag, FakePrereq),
    ):
        reg = HolmesRegistry.build(model="anthropic/foo")

    inst = captured["instance"]
    assert inst.model == "anthropic/foo"
    assert inst.should_try_robusta_ai is False
    assert reg.openai_tools == [{"name": "bash"}]
    assert reg.llm is reg.ai.llm


def test_build_loads_skill_catalog_when_available():
    FakeConfig, FakeTag, FakePrereq, _ = _build_fake_holmes(
        cfg_attrs={"model": "anthropic/foo", "should_try_robusta_ai": False},
        skill_catalog="my-skill-catalog",
    )
    with mock.patch.object(
        registry_module,
        "_import_holmes",
        return_value=(FakeConfig, FakeTag, FakePrereq),
    ):
        reg = HolmesRegistry.build(model="anthropic/foo")
    assert reg.skills == "my-skill-catalog"


def test_build_swallows_skill_catalog_errors():
    """If skill catalog loading raises, the registry should still build."""
    FakeConfig, FakeTag, FakePrereq, _ = _build_fake_holmes(
        cfg_attrs={"model": "anthropic/foo", "should_try_robusta_ai": False},
        skill_catalog_raises=True,
    )
    with mock.patch.object(
        registry_module,
        "_import_holmes",
        return_value=(FakeConfig, FakeTag, FakePrereq),
    ):
        reg = HolmesRegistry.build(model="anthropic/foo")
    assert reg.skills is None


def test_import_holmes_clean_error_when_missing():
    """If ``holmes`` is not importable, surface an actionable error."""
    import builtins

    real_import = builtins.__import__

    def _explode(name, *args, **kwargs):
        if name == "holmes" or name.startswith("holmes."):
            raise ImportError("No module named 'holmes'")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", side_effect=_explode):
        with pytest.raises(ImportError, match="HolmesGPT is required"):
            registry_module._import_holmes()
