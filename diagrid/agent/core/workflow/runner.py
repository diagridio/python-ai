# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Base workflow runner - shared logic for all framework runners."""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from dapr.ext.workflow import DaprWorkflowClient, WorkflowRuntime, WorkflowStatus

from diagrid.agent.core.chat import close_chat_clients
from diagrid.agent.core.discovery import discover_components
from diagrid.agent.core.metadata.mixins import AgentRegistryMixin
from diagrid.agent.core.observability import resolve_observability_config

logger = logging.getLogger(__name__)


class BaseWorkflowRunner(AgentRegistryMixin, ABC):
    """Base class for all framework workflow runners.

    Extracts the shared lifecycle, polling, status management, and serve
    logic that is duplicated across CrewAI, ADK, OpenAI Agents, Strands,
    and LangGraph runners.

    Subclasses must implement:
        - ``_setup_telemetry(app)`` — framework-specific OTEL setup
        - ``_build_serve_workflow_input(request)`` — map HTTP request to
          workflow input dict
        - ``_register_workflow_components()`` — register workflow and
          activities on ``self._workflow_runtime``
    """

    def __init__(
        self,
        name: str,
        *,
        framework: str,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_iterations: int = 25,
        component_name: Optional[str] = None,
        state_store: Any = None,
    ) -> None:
        self._name = name
        self._framework = framework
        self._host = host
        self._port = port
        self._max_iterations = max_iterations
        self._component_name = component_name
        self._state_store = state_store
        self._dapr_chat_client: Any = None
        self._workflow_runtime = WorkflowRuntime(host=host, port=port)
        self._workflow_client: Optional[DaprWorkflowClient] = None
        self._started = False

        # Run unified discovery and resolve observability config
        discovered = discover_components()
        self._observability_config = resolve_observability_config(
            runtime_conf=discovered.runtime_conf or None,
        )

        # Auto-configure state store from discovery if not explicitly provided
        if self._state_store is None and discovered.memory_store_name:
            from diagrid.agent.core.state import DaprStateStore

            self._state_store = DaprStateStore(store_name=discovered.memory_store_name)

    @property
    def workflow_name(self) -> str:
        """Return the canonical workflow name: ``dapr.<framework>.<name>.workflow``."""
        return f"dapr.{self._framework}.{self._name}.workflow"

    # ------------------------------------------------------------------
    # Shared lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the workflow runtime.

        Must be called before running any workflows.
        """
        if self._started:
            return

        self._workflow_runtime.start()
        self._workflow_client = DaprWorkflowClient(host=self._host, port=self._port)
        self._started = True
        logger.info("Dapr Workflow runtime started")

    def shutdown(self) -> None:
        """Shutdown the workflow runtime and clean up resources."""
        if not self._started:
            return

        self._workflow_runtime.shutdown()
        if self._dapr_chat_client:
            self._dapr_chat_client.close()
            self._dapr_chat_client = None
        close_chat_clients()
        if self._state_store is not None:
            self._state_store.close()
        self._started = False
        logger.info("Dapr Workflow runtime stopped")

    # ------------------------------------------------------------------
    # Shared workflow management
    # ------------------------------------------------------------------

    def get_workflow_status(self, workflow_id: str) -> Optional[dict[str, Any]]:
        """Get the status of a workflow.

        Args:
            workflow_id: The workflow instance ID.

        Returns:
            Dictionary with workflow status or None if not found.
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        state = self._workflow_client.get_workflow_state(instance_id=workflow_id)
        if state is None:
            return None

        return {
            "workflow_id": workflow_id,
            "status": str(state.runtime_status),
            "custom_status": state.serialized_custom_status,
            "created_at": (str(state.created_at) if state.created_at else None),
            "last_updated_at": (
                str(state.last_updated_at) if state.last_updated_at else None
            ),
        }

    def terminate_workflow(self, workflow_id: str) -> None:
        """Terminate a running workflow.

        Args:
            workflow_id: The workflow instance ID.
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        self._workflow_client.terminate_workflow(instance_id=workflow_id)
        logger.info("Terminated workflow: %s", workflow_id)

    def purge_workflow(self, workflow_id: str) -> None:
        """Purge a completed or terminated workflow.

        Args:
            workflow_id: The workflow instance ID.
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        self._workflow_client.purge_workflow(instance_id=workflow_id)
        logger.info("Purged workflow: %s", workflow_id)

    # ------------------------------------------------------------------
    # Shared polling loop
    # ------------------------------------------------------------------

    async def _poll_workflow(
        self,
        workflow_id: str,
        session_id: str,
        *,
        poll_interval: float = 0.5,
        parse_output: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Poll a running workflow and yield status/completion events.

        Args:
            workflow_id: The workflow instance ID.
            session_id: Session identifier for event metadata.
            poll_interval: Seconds between polls.
            parse_output: Optional callable to parse serialized output
                into a completion event dict. If None, raw output is
                returned.

        Yields:
            Event dictionaries with workflow progress updates.
        """
        assert self._workflow_client is not None

        previous_status = None

        while True:
            await asyncio.sleep(poll_interval)

            state = self._workflow_client.get_workflow_state(instance_id=workflow_id)

            if state is None:
                yield {
                    "type": "workflow_error",
                    "workflow_id": workflow_id,
                    "error": "Workflow state not found",
                }
                break

            if state.runtime_status != previous_status:
                yield {
                    "type": "workflow_status_changed",
                    "workflow_id": workflow_id,
                    "status": str(state.runtime_status),
                    "custom_status": state.serialized_custom_status,
                }
                previous_status = state.runtime_status

            if state.runtime_status == WorkflowStatus.COMPLETED:
                output_data = state.serialized_output
                if output_data:
                    try:
                        output_dict = (
                            json.loads(output_data)
                            if isinstance(output_data, str)
                            else output_data
                        )
                        if parse_output:
                            yield parse_output(workflow_id, output_dict)
                        else:
                            yield {
                                "type": "workflow_completed",
                                "workflow_id": workflow_id,
                                "output": output_dict,
                            }
                    except Exception as e:
                        yield {
                            "type": "workflow_completed",
                            "workflow_id": workflow_id,
                            "raw_output": output_data,
                            "parse_error": str(e),
                        }
                else:
                    yield {
                        "type": "workflow_completed",
                        "workflow_id": workflow_id,
                    }
                break

            elif state.runtime_status == WorkflowStatus.FAILED:
                error_info = None
                if state.failure_details:
                    fd = state.failure_details
                    error_info = {
                        "message": getattr(fd, "message", str(fd)),
                        "error_type": getattr(fd, "error_type", None),
                        "stack_trace": getattr(fd, "stack_trace", None),
                    }
                yield {
                    "type": "workflow_failed",
                    "workflow_id": workflow_id,
                    "error": error_info,
                }
                break

            elif state.runtime_status == WorkflowStatus.TERMINATED:
                yield {
                    "type": "workflow_terminated",
                    "workflow_id": workflow_id,
                }
                break

    # ------------------------------------------------------------------
    # Shared sync wrapper
    # ------------------------------------------------------------------

    def _run_sync(self, coro: Any, timeout: float = 300.0) -> Any:
        """Run an async coroutine synchronously with a timeout.

        Args:
            coro: The coroutine to execute.
            timeout: Maximum wait time in seconds.

        Returns:
            The coroutine result.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Shared serve pattern
    # ------------------------------------------------------------------

    def serve(
        self,
        *,
        port: int = 5001,
        host: str = "0.0.0.0",
        pubsub_name: Optional[str] = None,
        subscribe_topic: Optional[str] = None,
        publish_topic: Optional[str] = None,
    ) -> None:
        """Start an HTTP server exposing /agent/run endpoints.

        Creates a FastAPI app, calls ``_setup_telemetry()``, starts the
        runtime, and registers POST/GET ``/agent/run`` endpoints.

        When pub/sub parameters are provided, also registers Dapr
        programmatic subscription endpoints so the agent can receive
        tasks via pub/sub and publish results.

        Args:
            port: Port to listen on (default: 5001).
            host: Host to bind to (default: 0.0.0.0).
            pubsub_name: Dapr pub/sub component name for event-driven
                communication (e.g. ``"agent-pubsub"``).
            subscribe_topic: Topic to subscribe to for incoming tasks.
            publish_topic: Topic to publish results to after completion.
        """
        try:
            from fastapi import FastAPI, HTTPException, Request
            import uvicorn
        except ImportError:
            raise ImportError(
                "fastapi and uvicorn are required for serve(). "
                "Install them with: pip install fastapi uvicorn[standard]"
            )

        app = FastAPI()

        self._setup_telemetry()
        self._setup_serve_defaults()
        self.start()

        # Set up pub/sub publisher if configured
        pubsub_client = None
        if pubsub_name and publish_topic:
            from diagrid.agent.core.pubsub import DaprPubSub

            pubsub_client = DaprPubSub(pubsub_name=pubsub_name)

        runner_ref = self

        async def _run_and_collect(request: dict) -> dict:  # type: ignore[type-arg]
            """Run the agent and collect the result."""
            session_id = request.get(
                "session_id",
                request.get("thread_id", uuid.uuid4().hex[:8]),
            )
            result: dict[str, Any] = {}
            async for event in runner_ref._serve_run(
                request=request, session_id=session_id
            ):
                if event["type"] == "workflow_started":
                    result["instance_id"] = event["workflow_id"]
                elif event["type"] == "workflow_completed":
                    result.update(event)
                    break
                elif event["type"] == "workflow_failed":
                    result.update(event)
                    break

            # Publish result to topic if configured
            if pubsub_client and publish_topic:
                try:
                    pubsub_client.publish(publish_topic, result)
                    logger.info("Published result to topic=%s", publish_topic)
                except Exception as e:
                    logger.error("Failed to publish result: %s", str(e))

            return result

        @app.post("/agent/run")
        async def run_agent(request: dict) -> dict:  # type: ignore[type-arg]
            return await _run_and_collect(request)

        @app.get("/agent/run/{workflow_id}")
        async def get_status(workflow_id: str) -> dict:  # type: ignore[type-arg]
            status = self.get_workflow_status(workflow_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Workflow not found")
            return status

        # Register Dapr programmatic subscription if configured
        if pubsub_name and subscribe_topic:

            @app.get("/dapr/subscribe")
            async def dapr_subscribe() -> list:  # type: ignore[type-arg]
                return [
                    {
                        "pubsubname": pubsub_name,
                        "topic": subscribe_topic,
                        "route": f"/events/{subscribe_topic}",
                    }
                ]

            @app.post(f"/events/{subscribe_topic}")
            async def handle_topic_event(request: Request) -> dict:  # type: ignore[type-arg]
                body = await request.json()

                # Extract task from CloudEvent data
                data = body.get("data", body)
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        data = {"task": data}
                if isinstance(data, dict) and "task" not in data:
                    data = {"task": json.dumps(data) if data else ""}

                logger.info("Received event on topic=%s", subscribe_topic)
                result = await _run_and_collect(data)
                return result

        uvicorn.run(app, host=host, port=port)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the workflow runtime is running."""
        return self._started

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _setup_telemetry(self) -> None:
        """Set up framework-specific OpenTelemetry instrumentation."""
        ...

    @abstractmethod
    def _setup_serve_defaults(self) -> None:
        """Set up framework-specific serve defaults (e.g. workflow input factories)."""
        ...

    @abstractmethod
    async def _serve_run(
        self,
        request: dict,
        session_id: str,  # type: ignore[type-arg]
    ) -> AsyncIterator[dict[str, Any]]:
        """Handle a POST /agent/run request.

        Subclasses should map the request to a workflow input,
        start the workflow, and yield events.
        """
        ...
        # Make this an async generator
        yield  # type: ignore[misc]  # pragma: no cover

    @abstractmethod
    def _register_workflow_components(self) -> None:
        """Register workflow and activities on ``self._workflow_runtime``."""
        ...
