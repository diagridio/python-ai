# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runs streamed through the state-store event log, with a fake store and client.

The workflow runs inline on a thread (as in the golden harness) and records the
custom status it sets, which is how a reader on any replica finds the run's
units. ``activity_runner`` lets another runner's activities (another replica
of the app) run the nodes while the first one serves the stream.
"""

import json
import threading
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest import mock

from dapr.ext.workflow import WorkflowStatus

from diagrid.agent.core.workflow.stream_log import (
    StateLogStreamConfig,
    StateLogTransport,
)
from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.models import GraphWorkflowInput

from ..core.log_store_fake import FakeLogStore
from .streaming_helpers import _IdCtx

FAST = StateLogStreamConfig(flush_interval_s=0.01, flush_max_chunks=4, read_poll_s=0.01)


class StatusCtx(_IdCtx):
    """An inline workflow context that keeps the custom status the run sets."""

    def __init__(self, runner: Any, workflow_id: str, client: "LogClient") -> None:
        super().__init__(runner, workflow_id)
        self.instance_id = workflow_id
        self._client = client

    def set_custom_status(self, status: str) -> None:
        self._client.custom[self._workflow_id] = status

    def wait_for_external_event(self, name: str) -> Any:
        return self._client.wait_event(self._workflow_id, name)


class LogClient:
    """A workflow client: schedules ``agent_workflow`` on a thread, answers polls."""

    def __init__(
        self, runner: DaprWorkflowGraphRunner, activity_runner: Any = None
    ) -> None:
        self._runner = runner
        self._activity_runner = activity_runner or runner
        self.done: Dict[str, Any] = {}
        self.failed: Dict[str, str] = {}
        self.custom: Dict[str, str] = {}
        self.inputs: Dict[str, Any] = {}
        self._events: Dict[str, Any] = {}
        self._cond = threading.Condition()

    def schedule_new_workflow(self, *, workflow: Any, input: Any, instance_id: str):
        self.inputs[instance_id] = json.loads(json.dumps(input))
        threading.Thread(
            target=self._work, args=(input, instance_id), daemon=True
        ).start()

    def _work(self, input: Any, instance_id: str) -> None:
        from diagrid.agent.langgraph.workflow import agent_workflow

        wf_input = GraphWorkflowInput.from_dict(json.loads(json.dumps(input)))
        ctx = StatusCtx(self._activity_runner, instance_id, self)
        try:
            with mock.patch(
                "diagrid.agent.langgraph.workflow.when_all", side_effect=list
            ):
                workflow = agent_workflow(
                    ctx, json.loads(json.dumps(wf_input.to_dict()))
                )
                result = None
                try:
                    while True:
                        result = workflow.send(result)
                except StopIteration as stop:
                    self.done[instance_id] = stop.value
        except Exception as e:  # the workflow failed
            self.failed[instance_id] = str(e)

    def raise_workflow_event(self, instance_id: str, event_name: str, data: Any = None):
        with self._cond:
            self._events[f"{instance_id}/{event_name}"] = json.loads(json.dumps(data))
            self._cond.notify_all()

    def wait_event(self, instance_id: str, name: str) -> Any:
        key = f"{instance_id}/{name}"
        with self._cond:
            self._cond.wait_for(lambda: key in self._events, timeout=30)
            return self._events.pop(key, None)

    def get_workflow_state(self, *, instance_id: str, fetch_payloads: bool = True):
        if instance_id not in self.inputs:
            return None
        common = dict(
            serialized_custom_status=self.custom.get(instance_id),
            serialized_input=json.dumps(self.inputs[instance_id]),
            created_at=None,
            last_updated_at=None,
        )
        if instance_id in self.failed:
            return SimpleNamespace(
                runtime_status=WorkflowStatus.FAILED,
                serialized_output=None,
                failure_details=SimpleNamespace(message=self.failed[instance_id]),
                **common,
            )
        output = self.done.get(instance_id)
        if output is None:
            return SimpleNamespace(
                runtime_status=WorkflowStatus.RUNNING,
                serialized_output=None,
                failure_details=None,
                **common,
            )
        return SimpleNamespace(
            runtime_status=WorkflowStatus.COMPLETED,
            serialized_output=json.dumps(output),
            failure_details=None,
            **common,
        )


def log_runner(
    compiled: Any,
    name: str,
    *,
    store: Optional[FakeLogStore] = None,
    config: StateLogStreamConfig = FAST,
    activity_runner: Any = None,
) -> DaprWorkflowGraphRunner:
    """A started runner whose streams go through ``store``'s event log."""
    store = store if store is not None else FakeLogStore()
    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        name=name,
        stream_transport=StateLogTransport(config, store=store),
    )
    runner._started = True
    runner._workflow_client = LogClient(runner, activity_runner)  # type: ignore[assignment]
    return runner


def replica(compiled: Any, name: str, store: FakeLogStore) -> DaprWorkflowGraphRunner:
    """Another process of the same app: same graph, same store, its own activities."""
    return DaprWorkflowGraphRunner(
        graph=compiled,
        name=name,
        stream_transport=StateLogTransport(FAST, store=store),
    )
