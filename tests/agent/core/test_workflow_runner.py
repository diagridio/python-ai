# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for BaseWorkflowRunner."""

import asyncio
from typing import Any, AsyncIterator
from unittest import TestCase, mock

from diagrid.agent.core.workflow.runner import BaseWorkflowRunner


class ConcreteRunner(BaseWorkflowRunner):
    """Minimal concrete subclass for testing."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("framework", "test")
        super().__init__(kwargs.pop("name", "test_agent"), **kwargs)

    def _setup_telemetry(self) -> None:
        pass

    def _setup_serve_defaults(self) -> None:
        pass

    async def _serve_run(
        self,
        request: dict,
        session_id: str,  # type: ignore[type-arg]
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "workflow_completed", "workflow_id": "test"}

    def _register_workflow_components(self) -> None:
        pass


@mock.patch("diagrid.agent.core.workflow.runner.DaprWorkflowClient")
@mock.patch("diagrid.agent.core.workflow.runner.WorkflowRuntime")
class TestBaseWorkflowRunnerLifecycle(TestCase):
    """Tests for start/shutdown lifecycle."""

    def test_start(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.start()

        mock_runtime_cls.return_value.start.assert_called_once()
        mock_client_cls.assert_called_once_with(host=None, port=None)
        self.assertTrue(runner.is_running)

    def test_start_idempotent(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.start()
        runner.start()

        mock_runtime_cls.return_value.start.assert_called_once()

    def test_shutdown(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.start()
        runner.shutdown()

        mock_runtime_cls.return_value.shutdown.assert_called_once()
        self.assertFalse(runner.is_running)

    def test_shutdown_when_not_started(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.shutdown()  # Should not error

        mock_runtime_cls.return_value.shutdown.assert_not_called()

    def test_shutdown_closes_state_store(self, mock_runtime_cls, mock_client_cls):
        mock_store = mock.MagicMock()
        runner = ConcreteRunner(state_store=mock_store)
        runner.start()
        runner.shutdown()

        mock_store.close.assert_called_once()


@mock.patch("diagrid.agent.core.workflow.runner.DaprWorkflowClient")
@mock.patch("diagrid.agent.core.workflow.runner.WorkflowRuntime")
class TestWorkflowManagement(TestCase):
    """Tests for workflow status/terminate/purge."""

    def test_get_workflow_status(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.start()

        mock_state = mock.MagicMock()
        mock_state.runtime_status = "COMPLETED"
        mock_state.serialized_custom_status = None
        mock_state.created_at = None
        mock_state.last_updated_at = None
        mock_client_cls.return_value.get_workflow_state.return_value = mock_state

        status = runner.get_workflow_status("wf-1")
        self.assertIsNotNone(status)
        self.assertEqual(status["workflow_id"], "wf-1")
        self.assertEqual(status["status"], "COMPLETED")

    def test_get_workflow_status_not_found(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.start()

        mock_client_cls.return_value.get_workflow_state.return_value = None
        status = runner.get_workflow_status("wf-missing")
        self.assertIsNone(status)

    def test_get_workflow_status_not_started(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        with self.assertRaises(RuntimeError):
            runner.get_workflow_status("wf-1")

    def test_terminate_workflow(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.start()
        runner.terminate_workflow("wf-1")

        mock_client_cls.return_value.terminate_workflow.assert_called_once_with(
            instance_id="wf-1"
        )

    def test_terminate_not_started(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        with self.assertRaises(RuntimeError):
            runner.terminate_workflow("wf-1")

    def test_purge_workflow(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.start()
        runner.purge_workflow("wf-1")

        mock_client_cls.return_value.purge_workflow.assert_called_once_with(
            instance_id="wf-1"
        )

    def test_purge_not_started(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        with self.assertRaises(RuntimeError):
            runner.purge_workflow("wf-1")


@mock.patch("diagrid.agent.core.workflow.runner.DaprWorkflowClient")
@mock.patch("diagrid.agent.core.workflow.runner.WorkflowRuntime")
class TestPollWorkflow(TestCase):
    """Tests for the shared polling loop."""

    def _run_async(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_poll_completed(self, mock_runtime_cls, mock_client_cls):
        from dapr.ext.workflow import WorkflowStatus

        runner = ConcreteRunner()
        runner.start()

        mock_state = mock.MagicMock()
        mock_state.runtime_status = WorkflowStatus.COMPLETED
        mock_state.serialized_custom_status = None
        mock_state.serialized_output = '{"result": "done"}'
        mock_client_cls.return_value.get_workflow_state.return_value = mock_state

        async def _collect():
            events = []
            async for event in runner._poll_workflow(
                "wf-1", "sess-1", poll_interval=0.01
            ):
                events.append(event)
            return events

        events = self._run_async(_collect())
        types = [e["type"] for e in events]
        self.assertIn("workflow_status_changed", types)
        self.assertIn("workflow_completed", types)

    def test_poll_failed(self, mock_runtime_cls, mock_client_cls):
        from dapr.ext.workflow import WorkflowStatus

        runner = ConcreteRunner()
        runner.start()

        mock_fd = mock.MagicMock()
        mock_fd.message = "some error"
        mock_fd.error_type = "RuntimeError"
        mock_fd.stack_trace = None

        mock_state = mock.MagicMock()
        mock_state.runtime_status = WorkflowStatus.FAILED
        mock_state.serialized_custom_status = None
        mock_state.failure_details = mock_fd
        mock_client_cls.return_value.get_workflow_state.return_value = mock_state

        async def _collect():
            events = []
            async for event in runner._poll_workflow(
                "wf-1", "sess-1", poll_interval=0.01
            ):
                events.append(event)
            return events

        events = self._run_async(_collect())
        failed_events = [e for e in events if e["type"] == "workflow_failed"]
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0]["error"]["message"], "some error")

    def test_poll_terminated(self, mock_runtime_cls, mock_client_cls):
        from dapr.ext.workflow import WorkflowStatus

        runner = ConcreteRunner()
        runner.start()

        mock_state = mock.MagicMock()
        mock_state.runtime_status = WorkflowStatus.TERMINATED
        mock_state.serialized_custom_status = None
        mock_client_cls.return_value.get_workflow_state.return_value = mock_state

        async def _collect():
            events = []
            async for event in runner._poll_workflow(
                "wf-1", "sess-1", poll_interval=0.01
            ):
                events.append(event)
            return events

        events = self._run_async(_collect())
        types = [e["type"] for e in events]
        self.assertIn("workflow_terminated", types)

    def test_poll_state_not_found(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.start()

        mock_client_cls.return_value.get_workflow_state.return_value = None

        async def _collect():
            events = []
            async for event in runner._poll_workflow(
                "wf-1", "sess-1", poll_interval=0.01
            ):
                events.append(event)
            return events

        events = self._run_async(_collect())
        self.assertEqual(events[0]["type"], "workflow_error")


@mock.patch("diagrid.agent.core.workflow.runner.DaprWorkflowClient")
@mock.patch("diagrid.agent.core.workflow.runner.WorkflowRuntime")
class TestRunSync(TestCase):
    """Tests for _run_sync helper."""

    def test_run_sync(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()

        async def _coro():
            return 42

        result = runner._run_sync(_coro())
        self.assertEqual(result, 42)

    def test_run_sync_timeout(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()

        async def _slow():
            await asyncio.sleep(10)
            return 42

        with self.assertRaises(asyncio.TimeoutError):
            runner._run_sync(_slow(), timeout=0.01)


@mock.patch("diagrid.agent.core.workflow.runner.DaprWorkflowClient")
@mock.patch("diagrid.agent.core.workflow.runner.WorkflowRuntime")
class TestConstructor(TestCase):
    """Tests for constructor arguments."""

    def test_default_args(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        self.assertIsNone(runner._host)
        self.assertIsNone(runner._port)
        self.assertEqual(runner._max_iterations, 25)
        self.assertIsNone(runner._state_store)
        self.assertFalse(runner.is_running)

    def test_custom_args(self, mock_runtime_cls, mock_client_cls):
        mock_store = mock.MagicMock()
        runner = ConcreteRunner(
            host="localhost",
            port="50001",
            max_iterations=50,
            state_store=mock_store,
        )
        self.assertEqual(runner._host, "localhost")
        self.assertEqual(runner._port, "50001")
        self.assertEqual(runner._max_iterations, 50)
        self.assertIs(runner._state_store, mock_store)

    def test_workflow_runtime_created_with_host_port(
        self, mock_runtime_cls, mock_client_cls
    ):
        ConcreteRunner(host="myhost", port="12345")
        mock_runtime_cls.assert_called_once_with(host="myhost", port="12345")
