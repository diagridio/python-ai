# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``GET /agent/run/{workflow_id}`` tells how a run ended, with its result.

A client that lost its stream polls this endpoint for the run's result, so a
finished run's body carries what a plain ``POST /agent/run`` returns for it.
The state lookup is a blocking gRPC call and runs off the event loop.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest import mock

from dapr.ext.workflow import WorkflowStatus

from diagrid.agent.langgraph.models import ChannelState, GraphWorkflowOutput

from .streaming_helpers import _chat_graph, serve_app
from .test_runner import _RunnerTestCase


def _wf_state(
    status: WorkflowStatus, output: Optional[GraphWorkflowOutput] = None, **kw: Any
) -> Any:
    return SimpleNamespace(
        runtime_status=status,
        serialized_output=json.dumps(output.to_dict()) if output else None,
        serialized_custom_status=None,
        created_at=None,
        last_updated_at=None,
        failure_details=kw.get("failure_details"),
    )


class TestRunStatusResult(_RunnerTestCase):
    def _get(self, state: Any, loops: Optional[List[bool]] = None) -> Any:
        from fastapi.testclient import TestClient

        runner = _chat_graph(name="status-result")
        runner._started = True

        def lookup(*, instance_id: str) -> Any:
            if loops is not None:
                try:
                    asyncio.get_running_loop()
                    loops.append(True)
                except RuntimeError:
                    loops.append(False)
            return state

        runner._workflow_client = mock.Mock()
        runner._workflow_client.get_workflow_state.side_effect = lookup
        return TestClient(serve_app(runner)).get("/agent/run/wf-1")

    def test_a_completed_run_carries_its_output(self):
        out = GraphWorkflowOutput(
            output={"x": 1},
            channel_state=ChannelState(values={"x": 1}),
            steps=2,
            status="completed",
            executed=[["a"], ["b"]],
        )
        resp = self._get(_wf_state(WorkflowStatus.COMPLETED, out))
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["workflow_id"], "wf-1")
        self.assertEqual(body["output"], {"x": 1})
        self.assertEqual(body["steps"], 2)
        self.assertEqual(body["run_status"], "completed")
        # what is for the stream only stays out of it
        self.assertNotIn("executed", body)
        self.assertNotIn("channel_state", body)

    def test_a_run_that_ended_in_error_says_why(self):
        out = GraphWorkflowOutput(
            output={}, steps=1, status="error", error="Node 'a' failed: boom"
        )
        body = self._get(_wf_state(WorkflowStatus.COMPLETED, out)).json()
        self.assertEqual(body["run_status"], "error")
        self.assertEqual(body["error"], "Node 'a' failed: boom")

    def test_a_failed_workflow_carries_its_failure(self):
        state = _wf_state(
            WorkflowStatus.FAILED,
            failure_details=SimpleNamespace(message="activity blew up"),
        )
        body = self._get(state).json()
        self.assertEqual(body["error"], "activity blew up")
        self.assertNotIn("output", body)

    def test_a_running_run_has_no_output_yet(self):
        body = self._get(_wf_state(WorkflowStatus.RUNNING)).json()
        self.assertNotIn("output", body)
        self.assertIn("RUNNING", body["status"])

    def test_an_unknown_run_is_404(self):
        self.assertEqual(self._get(None).status_code, 404)

    def test_the_lookup_runs_off_the_event_loop(self):
        loops: List[bool] = []
        self._get(_wf_state(WorkflowStatus.RUNNING), loops)
        self.assertEqual(loops, [False])


if __name__ == "__main__":
    unittest.main()
