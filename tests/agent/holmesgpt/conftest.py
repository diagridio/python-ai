# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Shared fixtures and stubs for the durable HolmesGPT integration tests.

The activities under test reach into HolmesGPT (``LLM.completion``,
``Tool.invoke``, ``ToolExecutor.get_tool_by_name``); to keep the suite
hermetic we install a stubbed :class:`HolmesRegistry` for the duration of
each test.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, List, Optional

import pytest

from holmes.core.llm import LLM, ContextWindowUsage

from diagrid.agent.holmesgpt import registry as registry_module
from diagrid.agent.holmesgpt.registry import HolmesRegistry


class StubLLM(LLM):
    """Stand-in for ``holmes.core.llm.DefaultLLM``.

    Subclasses HolmesGPT's :class:`LLM` so that Pydantic validation on
    :class:`holmes.core.tools.ToolInvokeContext` accepts the instance.
    """

    def __init__(self, completions: Optional[List[Any]] = None):
        self.model = "stub-model"
        self._completions = list(completions or [])
        self.calls: List[dict] = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        if not self._completions:
            raise AssertionError("StubLLM.completion called with no scripted responses")
        return self._completions.pop(0)

    def get_max_token_count_for_single_tool(self) -> int:
        return 50_000

    def get_context_window_size(self) -> int:
        return 128_000

    def get_maximum_output_token(self) -> int:
        return 16_000

    def count_tokens(self, messages, tools=None):
        return ContextWindowUsage(
            total_tokens=0,
            tools_tokens=0,
            system_tokens=0,
            user_tokens=0,
            tools_to_call_tokens=0,
            assistant_tokens=0,
            other_tokens=0,
        )


class StubTool:
    """Stand-in for ``holmes.core.tools.Tool``.

    Each call to :meth:`invoke` returns the next scripted result.
    """

    def __init__(self, name: str, results: List[Any]):
        self.name = name
        self._results = list(results)
        self.invocations: List[dict] = []

    def invoke(self, *, params, context):
        self.invocations.append({"params": params, "context": context})
        if not self._results:
            raise AssertionError(
                f"StubTool({self.name!r}).invoke called with no scripted results"
            )
        return self._results.pop(0)


class StubToolExecutor:
    def __init__(self, tools: Optional[dict] = None):
        self.tools_by_name = dict(tools or {})

    def get_tool_by_name(self, name: str, user_id=None):
        return self.tools_by_name.get(name)


def make_llm_response(
    *,
    content: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
    finish_reason: str = "stop",
    response_id: str = "stub-response-id",
    usage: Optional[dict] = None,
) -> Any:
    """Build a minimal stand-in for ``litellm.types.utils.ModelResponse``.

    Only the fields read by ``call_llm_activity`` are populated.
    """
    tc_objs = []
    for tc in tool_calls or []:
        import json

        tc_objs.append(
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=json.dumps(tc.get("arguments", {})),
                ),
            )
        )

    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tc_objs or None,
        model_dump=lambda exclude_none=False: {
            k: v
            for k, v in {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (tc_objs or [])
                ]
                or None,
            }.items()
            if not (exclude_none and v is None)
        },
    )

    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage_obj = None
    if usage is not None:
        usage_obj = SimpleNamespace(model_dump=lambda: usage)
    return SimpleNamespace(choices=[choice], usage=usage_obj, id=response_id)


@pytest.fixture
def install_stub_registry() -> Callable[..., HolmesRegistry]:
    """Install a stub HolmesRegistry as the process-wide singleton for the test."""

    original = registry_module._REGISTRY

    def _install(
        *,
        completions: Optional[List[Any]] = None,
        tools: Optional[dict] = None,
        skills: Any = None,
    ) -> HolmesRegistry:
        llm = StubLLM(completions=completions)
        tool_executor = StubToolExecutor(tools=tools)
        reg = HolmesRegistry(
            config=SimpleNamespace(cluster_name=None),
            ai=SimpleNamespace(llm=llm, tool_executor=tool_executor),
            llm=llm,
            tool_executor=tool_executor,
            openai_tools=[],
            skills=skills,
        )
        registry_module.set_registry(reg)
        return reg

    yield _install

    registry_module._REGISTRY = original


# ---------------------------------------------------------------------------
# Workflow generator driver
# ---------------------------------------------------------------------------


class _Activity:
    """Sentinel for ``ctx.call_activity(fn, input=...)``."""

    def __init__(self, fn, input):
        self.fn = fn
        self.input = input

    def __repr__(self) -> str:
        return f"Activity({self.fn.__name__!s}, {self.input!r})"


class _WaitEvent:
    """Sentinel for ``ctx.wait_for_external_event(name)``."""

    def __init__(self, event_name):
        self.event_name = event_name

    def __repr__(self) -> str:
        return f"WaitEvent({self.event_name!r})"


class _WhenAll:
    """Sentinel for ``when_all(tasks)``."""

    def __init__(self, tasks):
        self.tasks = tasks

    def __repr__(self) -> str:
        return f"WhenAll({len(self.tasks)} tasks)"


class FakeWorkflowContext:
    """Minimal stand-in for ``DaprWorkflowContext`` for unit tests."""

    def __init__(self, instance_id: str = "wf-test"):
        self.instance_id = instance_id

    def call_activity(self, fn, *, input=None, retry_policy=None):
        return _Activity(fn, input)

    def wait_for_external_event(self, event_name: str):
        return _WaitEvent(event_name)


@pytest.fixture
def fake_when_all(monkeypatch):
    """Replace ``when_all`` in the workflow module with a sentinel builder."""

    def _wa(tasks):
        return _WhenAll(list(tasks))

    monkeypatch.setattr("diagrid.agent.holmesgpt.workflow.when_all", _wa)
    return _wa


@pytest.fixture
def drive_workflow(fake_when_all):
    """Return a driver that walks a workflow generator against a yield script.

    Each entry in ``script`` is a tuple ``(matcher, response)`` where matcher
    is one of:

    - ``("activity", <fn>)``      — must be ``ctx.call_activity(<fn>, ...)``
    - ``("wait", "<event_name>")`` — must be ``ctx.wait_for_external_event(...)``
    - ``("when_all", N)``          — must be ``when_all([... N tasks])``

    ``response`` is the value the workflow runtime would normally produce —
    a dict for activities, a list of dicts for ``when_all``, an event payload
    for waits.
    """

    def _drive(workflow_fn, payload, script, *, instance_id="wf-test"):
        ctx = FakeWorkflowContext(instance_id=instance_id)
        gen = workflow_fn(ctx, payload)
        sent = None
        for i, (matcher, response) in enumerate(script):
            try:
                yielded = gen.send(sent)
            except StopIteration as e:
                raise AssertionError(
                    f"step {i}: workflow returned early with value={e.value!r} "
                    f"but script expected {matcher!r}"
                )
            kind = matcher[0]
            if kind == "activity":
                expected_fn = matcher[1]
                assert isinstance(yielded, _Activity), (
                    f"step {i}: expected Activity({expected_fn.__name__}), got {yielded!r}"
                )
                assert yielded.fn is expected_fn, (
                    f"step {i}: expected {expected_fn.__name__}, got {yielded.fn.__name__}"
                )
            elif kind == "wait":
                expected_event = matcher[1]
                assert isinstance(yielded, _WaitEvent), (
                    f"step {i}: expected WaitEvent({expected_event!r}), got {yielded!r}"
                )
                assert yielded.event_name == expected_event, (
                    f"step {i}: expected wait on {expected_event!r}, "
                    f"got {yielded.event_name!r}"
                )
            elif kind == "when_all":
                expected_count = matcher[1]
                assert isinstance(yielded, _WhenAll), (
                    f"step {i}: expected WhenAll(N={expected_count}), got {yielded!r}"
                )
                assert len(yielded.tasks) == expected_count, (
                    f"step {i}: expected {expected_count} parallel tasks, "
                    f"got {len(yielded.tasks)}"
                )
            else:
                raise AssertionError(f"unknown matcher kind: {kind!r}")
            sent = response

        # Workflow should now return.
        try:
            extra = gen.send(sent)
            raise AssertionError(f"workflow yielded more than scripted: {extra!r}")
        except StopIteration as e:
            return e.value

    return _drive


@pytest.fixture(autouse=True)
def patch_event_log_record(monkeypatch):
    """Capture event-log writes from inside activities without touching Dapr.

    Only the alias used inside ``workflow.py`` is patched so that
    ``test_event_log.py`` can still exercise the real ``event_log.record``
    function against its own DaprClient mock.
    """
    captured: list[dict] = []

    def _record(*, instance_id, seq, event, data, **_):
        captured.append(
            {"instance_id": instance_id, "seq": seq, "event": event, "data": data}
        )

    monkeypatch.setattr(
        "diagrid.agent.holmesgpt.workflow.record_event_to_store", _record
    )
    return captured


@pytest.fixture(autouse=True)
def _stub_workflow_registration(monkeypatch):
    """Skip ``WorkflowRuntime.register_workflow/activity`` calls.

    The runtime de-dupes workflow names by tagging the function itself with
    ``_workflow_registered``; constructing multiple runners in the same
    process therefore raises ``ValueError``. Tests don't need real
    registration — they exercise the workflow function directly.
    """
    monkeypatch.setattr(
        "diagrid.agent.holmesgpt.runner.register_workflow_components",
        lambda runtime, *, workflow_name: None,
    )
