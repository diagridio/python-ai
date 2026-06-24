# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for graceful shutdown (SignalMixin parity) on BaseWorkflowRunner."""

import asyncio
from typing import Any, AsyncIterator
from unittest import TestCase, mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

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


def _run_async(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@mock.patch("diagrid.agent.core.workflow.runner.DaprWorkflowClient")
@mock.patch("diagrid.agent.core.workflow.runner.WorkflowRuntime")
class TestGracefulShutdown(TestCase):
    """``graceful_shutdown()`` routes to ``shutdown()`` and is idempotent."""

    def test_graceful_shutdown_stops_runtime(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.start()

        _run_async(runner.graceful_shutdown())

        mock_runtime_cls.return_value.shutdown.assert_called_once()
        self.assertFalse(runner.is_running)

    def test_graceful_shutdown_closes_state_store(
        self, mock_runtime_cls, mock_client_cls
    ):
        mock_store = mock.MagicMock()
        runner = ConcreteRunner(state_store=mock_store)
        runner.start()

        _run_async(runner.graceful_shutdown())

        mock_store.close.assert_called_once()

    def test_graceful_shutdown_idempotent(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        runner.start()

        _run_async(runner.graceful_shutdown())
        _run_async(runner.graceful_shutdown())

        mock_runtime_cls.return_value.shutdown.assert_called_once()

    def test_graceful_shutdown_when_not_started(
        self, mock_runtime_cls, mock_client_cls
    ):
        runner = ConcreteRunner()

        _run_async(runner.graceful_shutdown())  # Should not error

        mock_runtime_cls.return_value.shutdown.assert_not_called()


@mock.patch("diagrid.agent.core.workflow.runner.DaprWorkflowClient")
@mock.patch("diagrid.agent.core.workflow.runner.WorkflowRuntime")
class TestServeLifespan(TestCase):
    """The serve lifespan starts the runtime on enter and stops it on exit."""

    def test_lifespan_starts_and_stops(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        app = FastAPI()

        async def _drive() -> None:
            async with runner._serve_lifespan(app):
                # Runtime is up inside the lifespan.
                self.assertTrue(runner.is_running)
                mock_runtime_cls.return_value.start.assert_called_once()
            # Runtime is shut down on exit.
            self.assertFalse(runner.is_running)
            mock_runtime_cls.return_value.shutdown.assert_called_once()

        _run_async(_drive())

    def test_lifespan_shuts_down_on_error(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        app = FastAPI()

        async def _drive() -> None:
            with self.assertRaises(ValueError):
                async with runner._serve_lifespan(app):
                    raise ValueError("boom")
            mock_runtime_cls.return_value.shutdown.assert_called_once()

        _run_async(_drive())

    def test_serve_app_triggers_shutdown(self, mock_runtime_cls, mock_client_cls):
        """End-to-end: TestClient context manager runs startup + shutdown."""
        runner = ConcreteRunner()
        app = FastAPI(lifespan=runner._serve_lifespan)

        with TestClient(app):
            self.assertTrue(runner.is_running)
            mock_runtime_cls.return_value.start.assert_called_once()

        # Exiting the client context drives lifespan shutdown.
        self.assertFalse(runner.is_running)
        mock_runtime_cls.return_value.shutdown.assert_called_once()


@mock.patch("diagrid.agent.core.workflow.runner.DaprWorkflowClient")
@mock.patch("diagrid.agent.core.workflow.runner.WorkflowRuntime")
class TestSignalMixinParity(TestCase):
    """The reused SignalMixin API is wired onto the runner."""

    def test_shutdown_not_requested_initially(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()
        self.assertFalse(runner.is_shutdown_requested())

    def test_request_shutdown_sets_event(self, mock_runtime_cls, mock_client_cls):
        runner = ConcreteRunner()

        async def _drive() -> None:
            runner.install_signal_handlers()
            self.assertFalse(runner.is_shutdown_requested())
            runner.request_shutdown()
            # request_shutdown bounces through the loop (call_soon_threadsafe)
            # and schedules graceful_shutdown(); give it cycles to settle so
            # no coroutine is left pending when the loop closes.
            await asyncio.sleep(0.05)
            self.assertTrue(runner.is_shutdown_requested())
            runner.remove_signal_handlers()

        _run_async(_drive())
