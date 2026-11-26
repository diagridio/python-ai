# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Base workflow runner - shared logic for all framework runners."""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from importlib import metadata
from typing import Any, AsyncIterator, Callable, Optional, Tuple

from dapr.ext.workflow import DaprWorkflowClient, WorkflowRuntime, WorkflowStatus
from dapr_agents.utils.signal.mixin import SignalMixin

from diagrid.agent.core.discovery import discover_components
from diagrid.agent.core.metadata.mixins import AgentRegistryMixin
from diagrid.agent.core.observability import resolve_observability_config
from diagrid.agent.core.workflow.naming import build_workflow_name
from diagrid.agent.core.workflow.stream import frame_sse

# ``diagrid-core`` ships the reporter. This distribution only pins
# ``diagrid-core>=0.1.0`` (through ``diagrid-cli``), so an application holding
# an older ``diagrid-core`` must keep working: fall back to a no-op.
try:
    from diagrid.core.analytics import report_usage
except ImportError:  # pragma: no cover - older diagrid-core

    def report_usage(package: str, **dimensions: object) -> None:
        return None


logger = logging.getLogger(__name__)

# The distribution that carries each framework, keyed by the lower-cased
# framework label. First installed match wins. Used only for the
# ``framework_version`` usage dimension, so a miss costs nothing.
_FRAMEWORK_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "langgraph": ("langgraph",),
    "strands": ("strands-agents",),
    "crewai": ("crewai",),
    "adk": ("google-adk",),
    "openai": ("openai-agents",),
    "pydanticai": ("pydantic-ai-slim", "pydantic-ai"),
    "claudeagents": ("claude-agent-sdk",),
    "deepagents": ("deepagents",),
    "langchain": ("langchain-core", "langchain"),
    "smolagents": ("smolagents",),
    "dapr agents": ("dapr-agents",),
    "holmesgpt": ("holmesgpt",),
}


#: A failed status poll is retried this many times in a row (only when the
#: caller asks, and only for a transient error: see ``_is_transient_poll_error``)
#: before the run's outcome is given up as unknown, waiting
#: ``POLL_RETRY_BASE_SECONDS`` (doubling, capped at ``POLL_RETRY_MAX_SECONDS``)
#: between tries: about a minute in all, enough for a sidecar restart or a
#: rolling update.
POLL_MAX_ERRORS = 10
POLL_RETRY_BASE_SECONDS = 0.5
POLL_RETRY_MAX_SECONDS = 10.0


def _transient_grpc_codes() -> frozenset:  # type: ignore[type-arg]
    import grpc

    return frozenset(
        {
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.ABORTED,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            grpc.StatusCode.CANCELLED,
        }
    )


def _is_transient_poll_error(error: BaseException) -> bool:
    """A failed poll worth retrying: the call may succeed if tried again.

    A gRPC error with a status that says the sidecar was briefly unreachable
    or busy (``UNAVAILABLE``, ``DEADLINE_EXCEEDED``, ``ABORTED``,
    ``RESOURCE_EXHAUSTED``, ``CANCELLED``), or a connection or timeout error.
    Anything else cannot get better by waiting and surfaces at once: a
    permission, authentication or argument status, a bad socket path
    (``FileNotFoundError``), a TLS failure, a programming error.
    """
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    try:
        import grpc
    except ImportError:  # pragma: no cover - dapr depends on grpcio
        return False
    if not isinstance(error, grpc.RpcError):
        return False
    code_of = getattr(error, "code", None)
    if not callable(code_of):
        return False
    try:
        return code_of() in _transient_grpc_codes()
    except Exception:
        return False


def _poll_retry_delay(
    error: BaseException, failed_polls: int, retry: bool, workflow_id: str
) -> Optional[float]:
    """Seconds to wait before polling again, or ``None`` to raise ``error``.

    ``failed_polls`` counts consecutive failures, this one included.
    """
    if (
        not retry
        or not _is_transient_poll_error(error)
        or failed_polls > POLL_MAX_ERRORS
    ):
        return None
    delay = min(
        POLL_RETRY_BASE_SECONDS * 2 ** (failed_polls - 1), POLL_RETRY_MAX_SECONDS
    )
    logger.warning(
        "Polling workflow %s failed (%d of %d), retrying in %.1fs: %s",
        workflow_id,
        failed_polls,
        POLL_MAX_ERRORS,
        delay,
        error,
    )
    return delay


def _completed_event(
    state: Any, workflow_id: str, parse_output: Optional[Callable[..., Any]]
) -> dict[str, Any]:
    """The ``workflow_completed`` event for a finished workflow's ``state``."""
    output_data = state.serialized_output
    if not output_data:
        return {"type": "workflow_completed", "workflow_id": workflow_id}
    try:
        output_dict = (
            json.loads(output_data) if isinstance(output_data, str) else output_data
        )
        if parse_output:
            return parse_output(workflow_id, output_dict)  # type: ignore[no-any-return]
        return {
            "type": "workflow_completed",
            "workflow_id": workflow_id,
            "output": output_dict,
        }
    except Exception as e:
        return {
            "type": "workflow_completed",
            "workflow_id": workflow_id,
            "raw_output": output_data,
            "parse_error": str(e),
        }


def _terminal_event(
    state: Any, workflow_id: str, parse_output: Optional[Callable[..., Any]]
) -> Optional[dict[str, Any]]:
    """The event that ends a poll for ``state``, or ``None`` while it is running."""
    status = state.runtime_status
    if status == WorkflowStatus.COMPLETED:
        return _completed_event(state, workflow_id, parse_output)
    if status == WorkflowStatus.FAILED:
        error_info = None
        if state.failure_details:
            fd = state.failure_details
            error_info = {
                "message": getattr(fd, "message", str(fd)),
                "error_type": getattr(fd, "error_type", None),
                "stack_trace": getattr(fd, "stack_trace", None),
            }
        return {
            "type": "workflow_failed",
            "workflow_id": workflow_id,
            "error": error_info,
        }
    if status == WorkflowStatus.TERMINATED:
        return {"type": "workflow_terminated", "workflow_id": workflow_id}
    return None


def fold_run_event(result: dict, event: dict) -> bool:  # type: ignore[type-arg]
    """Fold one event of a run into ``result``, the body of ``POST /agent/run``.

    Returns ``True`` when ``event`` is the run's last. The same fold builds the
    returned body of a plain call and the result published for a streamed one,
    so the two cannot drift apart.
    """
    kind = event["type"]
    if kind == "workflow_started":
        result["instance_id"] = event["workflow_id"]
        if event.get("thread_id"):
            result["thread_id"] = event["thread_id"]
    elif kind in ("workflow_completed", "workflow_failed"):
        result.update(event)
        return True
    return False


def _framework_version(framework: str) -> Optional[str]:
    """Return the installed version of the library behind ``framework``, or None."""
    for distribution in _FRAMEWORK_DISTRIBUTIONS.get(
        str(framework).strip().lower(), ()
    ):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return None


async def _next_state(
    client: Any, workflow_id: str, poll_interval: float, retry_failed_polls: bool
) -> Any:
    """The workflow's state after ``poll_interval``, retrying a failed poll.

    The poll failed, not the workflow (a sidecar restart, a dropped call):
    retry, and give up only on a poll that keeps failing, so the caller learns
    the run's fate is unknown.
    """
    failed_polls = 0
    while True:
        await asyncio.sleep(poll_interval)
        try:
            # A blocking gRPC call: off the loop, or every other run's
            # tokens (and all HTTP handling) wait for the sidecar.
            return await asyncio.to_thread(
                client.get_workflow_state, instance_id=workflow_id
            )
        except Exception as e:
            failed_polls += 1
            delay = _poll_retry_delay(e, failed_polls, retry_failed_polls, workflow_id)
            if delay is None:
                raise
            await asyncio.sleep(delay)


class BaseWorkflowRunner(SignalMixin, AgentRegistryMixin, ABC):
    """Base class for all framework workflow runners.

    Extracts the shared lifecycle, polling, status management, and serve
    logic that is duplicated across CrewAI, ADK, OpenAI Agents, Strands,
    and LangGraph runners.

    Mixes in ``SignalMixin`` so served processes shut the Dapr workflow
    runtime down gracefully on SIGINT/SIGTERM (e.g. Kubernetes pod
    termination), matching the dapr-agents behavior.

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
        state_store: Any = None,
    ) -> None:
        # Initialize SignalMixin state (shutdown event, captured loop, etc.)
        # so graceful shutdown on SIGINT/SIGTERM is available.
        super().__init__()
        # One anonymous usage event per process, never blocking. See
        # ``diagrid.core.analytics`` and the README "Usage analytics" section.
        report_usage(
            "diagrid",
            kind="agent",
            framework=framework,
            framework_version=_framework_version(framework),
        )
        self._name = name
        self._framework = framework
        self._host = host
        self._port = port
        self._max_iterations = max_iterations
        self._state_store = state_store
        self._workflow_runtime = WorkflowRuntime(host=host, port=port)
        self._workflow_client: Optional[DaprWorkflowClient] = None
        self._started = False

        # Run unified discovery and resolve observability config
        discovered = discover_components()
        self._observability_config = resolve_observability_config(
            runtime_conf=discovered.runtime_conf or None,
        )

        #: The state store workflows keep their state in, if discovery saw one.
        self._workflow_store_name: Optional[str] = getattr(
            discovered, "workflow_store_name", None
        )

        # Auto-configure state store from discovery if not explicitly provided
        if self._state_store is None and discovered.memory_store_name:
            from diagrid.agent.core.state import DaprStateStore

            self._state_store = DaprStateStore(store_name=discovered.memory_store_name)

    @property
    def workflow_name(self) -> str:
        """Return the canonical workflow name: ``dapr.<framework>.<name>.workflow``.

        The agent name is sanitized to TitleCase to match the convention
        used by dapr-agents (e.g. ``dapr.openai.CateringCoordinator.workflow``).
        """
        return build_workflow_name(self._framework, self._name)

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
        if self._state_store is not None:
            self._state_store.close()
        self._started = False
        logger.info("Dapr Workflow runtime stopped")

    async def graceful_shutdown(self) -> None:
        """Stop the Dapr workflow runtime and close resources.

        Overrides ``SignalMixin.graceful_shutdown`` so the same cleanup path
        runs on SIGINT/SIGTERM (or a programmatic ``request_shutdown()``).
        ``shutdown()`` is synchronous and idempotent.
        """
        self.shutdown()

    @asynccontextmanager
    async def _serve_lifespan(self, _app: Any) -> AsyncIterator[None]:
        """FastAPI lifespan that ties the Dapr runtime to the server lifecycle.

        Starts the runtime on server startup and runs ``graceful_shutdown()``
        on server shutdown. uvicorn's own SIGINT/SIGTERM handling drives the
        shutdown phase, so the Dapr runtime and state store are always closed
        cleanly when the process is terminated.
        """
        self.start()
        try:
            yield
        finally:
            await self.graceful_shutdown()

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
            **self._status_result(state),
        }

    def _status_result(self, state: Any) -> dict[str, Any]:
        """What :meth:`get_workflow_status` adds about how the run ended.

        A runner whose workflow output it can read overrides this; by default a
        failed workflow carries its failure message and nothing else is added.
        """
        if state.runtime_status != WorkflowStatus.FAILED:
            return {}
        details = state.failure_details
        if details is None:
            return {"error": "Workflow failed"}
        return {"error": str(getattr(details, "message", details))}

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
        retry_failed_polls: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Poll a running workflow and yield status/completion events.

        ``parse_output`` turns the serialized output into a completion event
        dict (the raw output is returned without one). ``retry_failed_polls``
        rides out a failed poll that looks transient (see ``POLL_MAX_ERRORS``)
        instead of raising at once; a runner opts in when it can tell its
        caller that the run's fate is unknown.
        """
        assert self._workflow_client is not None
        previous_status = None
        while True:
            state = await _next_state(
                self._workflow_client, workflow_id, poll_interval, retry_failed_polls
            )
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

            terminal = _terminal_event(state, workflow_id, parse_output)
            if terminal is not None:
                yield terminal
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

    @staticmethod
    def _result_publisher(
        pubsub_name: Optional[str], publish_topic: Optional[str]
    ) -> Optional[Callable[[dict], None]]:  # type: ignore[type-arg]
        """The function that publishes a run's result, or ``None`` without a topic.

        Blocking: callers run it off the event loop. A failure is logged, not
        raised, so publishing never fails the run.
        """
        if not (pubsub_name and publish_topic):
            return None
        from diagrid.agent.core.pubsub import DaprPubSub

        client = DaprPubSub(pubsub_name=pubsub_name)

        def publish(result: dict) -> None:  # type: ignore[type-arg]
            try:
                client.publish(publish_topic, result)
                logger.info("Published result to topic=%s", publish_topic)
            except Exception as e:
                logger.error("Failed to publish result: %s", str(e))

        return publish

    @staticmethod
    def _add_subscription_routes(
        app: Any,
        pubsub_name: str,
        subscribe_topic: str,
        run_and_collect: Callable[[dict], Any],  # type: ignore[type-arg]
    ) -> None:
        """Register the Dapr programmatic subscription and its event route."""
        from fastapi import Request

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
            return await run_and_collect(data)  # type: ignore[no-any-return]

    def _stream_response(
        self,
        request: dict,  # type: ignore[type-arg]
        on_result: Optional[Callable[[dict], None]],  # type: ignore[type-arg]
    ) -> Any:
        """The SSE response of a ``POST /agent/run`` with ``"stream": true``."""
        from fastapi import HTTPException
        from fastapi.responses import StreamingResponse

        # A bad request is refused with a 4xx now: once the response starts,
        # its 200 is committed and an error can only be in-band.
        try:
            self._validate_stream_request(request)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        session_id = request.get(
            "session_id", request.get("thread_id", uuid.uuid4().hex[:8])
        )
        return StreamingResponse(
            frame_sse(self._serve_stream(request, session_id, on_result=on_result)),
            media_type="text/event-stream",
            # Without these an ingress or reverse proxy may buffer the stream
            # and hand the client its tokens in one batch.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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

        A runner that supports it (the LangGraph runner does) answers
        ``POST /agent/run`` with ``"stream": true`` (the JSON boolean, nothing
        else) as server-sent events. The
        keys ``stream`` and ``stream_mode`` are reserved on such a request. The
        last event is ``workflow_completed`` or ``workflow_failed``; its data is
        also what gets published to ``publish_topic``. Streaming works while the
        node activities run in the serving process (single replica; see the
        runner's README). A client that disconnects ends the stream but not the
        workflow, which keeps running; with ``publish_topic`` set its result is
        still published when the run ends.

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
            from fastapi import FastAPI, HTTPException
            import uvicorn
        except ImportError:
            raise ImportError(
                "fastapi and uvicorn are required for serve(). "
                "Install them with: pip install fastapi uvicorn[standard]"
            )

        app = FastAPI(lifespan=self._serve_lifespan)

        self._setup_telemetry()
        self._setup_serve_defaults()

        publish = self._result_publisher(pubsub_name, publish_topic)

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
                if fold_run_event(result, event):
                    break

            if publish:
                await asyncio.to_thread(publish, result)
            return result

        # Only a run that is published needs to outlive its client.
        on_result = publish

        @app.post("/agent/run")
        async def run_agent(request: dict) -> Any:  # type: ignore[type-arg]
            # Only the JSON value true streams. Anything else ("false", an
            # object) is a plain request whose "stream" key may be the graph's own.
            if request.get("stream") is True and self._supports_serve_stream:
                return self._stream_response(request, on_result)
            return await _run_and_collect(request)

        @app.get("/agent/run/{workflow_id}")
        async def get_status(workflow_id: str) -> dict:  # type: ignore[type-arg]
            # Off the loop: the state lookup is a blocking gRPC call.
            status = await asyncio.to_thread(self.get_workflow_status, workflow_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Workflow not found")
            return status

        if pubsub_name and subscribe_topic:
            self._add_subscription_routes(
                app, pubsub_name, subscribe_topic, _run_and_collect
            )

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

    # Runners that can stream a run as Server-Sent Events set this and
    # implement ``_serve_stream`` (used by ``POST /agent/run`` with
    # ``"stream": true``).
    _supports_serve_stream = False

    def _validate_stream_request(self, request: dict) -> None:  # type: ignore[type-arg]
        """Raise ``ValueError`` if a streaming request is invalid.

        Called before the SSE response is started, so the client gets a 422
        rather than a 200 whose only event is a failure. Default: accept all.
        """
        return None

    def _serve_stream(
        self,
        request: dict,  # type: ignore[type-arg]
        session_id: str,
        on_result: Optional[Callable[[dict], None]] = None,  # type: ignore[type-arg]
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Yield ``(event_name, data)`` pairs for a run. Opt-in per runner.

        The last pair is ``workflow_completed`` or ``workflow_failed`` with the
        run's terminal event as data. Framing as SSE is done by ``serve()``.

        ``on_result`` is how a run gets published: the runner calls it once
        with the same result a plain ``POST /agent/run`` would return (see
        :func:`fold_run_event`) when the run ends.
        """
        raise NotImplementedError

    @abstractmethod
    def _register_workflow_components(self) -> None:
        """Register workflow and activities on ``self._workflow_runtime``."""
        ...
