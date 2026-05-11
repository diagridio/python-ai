# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for ``DaprWorkflowHolmesRunner`` construction, prompt building,
approval forwarding, and FastAPI surface (without booting Dapr)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from diagrid.agent.holmesgpt import event_log
from diagrid.agent.holmesgpt.runner import DaprWorkflowHolmesRunner


def _runner(**overrides):
    """Construct a runner without starting it (no Dapr connection required)."""
    return DaprWorkflowHolmesRunner(name=overrides.pop("name", "unit"), **overrides)


def test_workflow_name_follows_framework_convention():
    r = _runner(name="my-agent")
    assert r.workflow_name == "dapr.holmesgpt.MyAgent.workflow"


def test_runner_defaults_events_to_actor_state_store():
    r = _runner()
    # Default value, not the legacy ``holmes-events`` component.
    assert r._events_store_name == event_log.DEFAULT_STORE_NAME == "statestore"
    assert r._events_key_prefix == "holmes.stream"
    assert r._events_ttl_seconds == 86400


def test_runner_accepts_explicit_event_store_overrides():
    r = _runner(
        events_store_name="my-store",
        events_key_prefix="x.y.z",
        events_ttl_seconds=120,
    )
    assert r._events_store_name == "my-store"
    assert r._events_key_prefix == "x.y.z"
    assert r._events_ttl_seconds == 120


def test_ask_requires_started_runner():
    r = _runner()
    with pytest.raises(RuntimeError, match="Runner not started"):
        r._build_messages(
            question="hi",
            conversation_history=None,
            additional_system_prompt=None,
            skills=None,
            images=None,
            prompt_component_overrides=None,
            global_instructions=None,
        )


def test_build_messages_delegates_to_holmes_build_chat_messages():
    """``ask()`` and friends must forward to ``holmes.core.conversations``."""
    r = _runner()
    r._started = True
    r._registry = SimpleNamespace(
        ai="ai-sentinel",
        config="cfg-sentinel",
        skills="default-skills",
    )

    fake_messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Q"},
    ]
    captured: dict = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        return fake_messages

    with mock.patch(
        "holmes.core.conversations.build_chat_messages", side_effect=_fake_build
    ):
        result = r._build_messages(
            question="Q",
            conversation_history=[{"role": "user", "content": "prev"}],
            additional_system_prompt="extra",
            skills=None,  # → should fall back to registry.skills
            images=None,
            prompt_component_overrides={"detail": True},
            global_instructions="globals",
        )

    assert result is fake_messages
    # The runner uses the registry's cached skills when caller passes None.
    assert captured["skills"] == "default-skills"
    assert captured["ai"] == "ai-sentinel"
    assert captured["config"] == "cfg-sentinel"
    assert captured["ask"] == "Q"
    assert captured["additional_system_prompt"] == "extra"
    assert captured["global_instructions"] == "globals"
    assert captured["prompt_component_overrides"] == {"detail": True}
    assert captured["conversation_history"] == [{"role": "user", "content": "prev"}]


def test_build_messages_caller_supplied_skills_override_default():
    r = _runner()
    r._started = True
    r._registry = SimpleNamespace(ai=None, config=None, skills="default-skills")

    captured: dict = {}
    with mock.patch(
        "holmes.core.conversations.build_chat_messages",
        side_effect=lambda **kw: captured.update(kw) or [],
    ):
        r._build_messages(
            question="Q",
            conversation_history=None,
            additional_system_prompt=None,
            skills="caller-skills",
            images=None,
            prompt_component_overrides=None,
            global_instructions=None,
        )
    assert captured["skills"] == "caller-skills"


def test_approve_calls_raise_workflow_event_with_expected_payload():
    r = _runner()
    r._started = True
    r._workflow_client = mock.MagicMock()
    r.approve(
        "wf-1",
        "tc-1",
        approved=True,
        reason="LGTM",
        session_approved_prefixes=["ls"],
    )
    r._workflow_client.raise_workflow_event.assert_called_once_with(
        instance_id="wf-1",
        event_name="resume:tc-1",
        data={
            "approved": True,
            "reason": "LGTM",
            "session_approved_prefixes": ["ls"],
        },
    )


def test_approve_rejects_when_not_started():
    r = _runner()
    with pytest.raises(RuntimeError, match="Runner not started"):
        r.approve("wf-1", "tc-1")


def test_submit_frontend_result_forwards_payload():
    r = _runner()
    r._started = True
    r._workflow_client = mock.MagicMock()
    r.submit_frontend_result("wf-1", "tc-9", "user clicked Yes")
    r._workflow_client.raise_workflow_event.assert_called_once_with(
        instance_id="wf-1",
        event_name="resume:tc-9",
        data={"frontend_result": "user clicked Yes"},
    )


def test_read_events_after_delegates_to_event_log():
    r = _runner()
    with mock.patch(
        "diagrid.agent.holmesgpt.event_log.read_after",
        return_value=[{"seq": 1, "event": "x", "data": {}}],
    ) as m:
        events = r.read_events_after("wf-1", since_seq=3, limit=10)
    m.assert_called_once_with(
        "wf-1",
        3,
        10,
        store_name=r._events_store_name,
        key_prefix=r._events_key_prefix,
    )
    assert events == [{"seq": 1, "event": "x", "data": {}}]


def test_build_fastapi_app_exposes_expected_routes():
    r = _runner()
    app = r.build_fastapi_app()
    paths = {route.path for route in app.routes}
    assert "/investigations" in paths
    assert "/investigations/{workflow_id}" in paths
    assert "/investigations/{workflow_id}/stream" in paths
    assert "/investigations/{workflow_id}/approve" in paths
    assert "/investigations/{workflow_id}/frontend_result" in paths
