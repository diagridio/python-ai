# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runner for executing HolmesGPT investigations as Dapr Workflows.

Each iteration of HolmesGPT's agent loop becomes a durable activity
invocation, enabling fault tolerance, durable approvals, and replayable
investigations without requiring upstream changes to HolmesGPT.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import uvicorn
from dapr.ext.workflow import WorkflowStatus
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from holmes.core.conversations import build_chat_messages

from diagrid.agent.core.telemetry import instrument_grpc, setup_telemetry
from diagrid.agent.core.workflow import BaseWorkflowRunner

from . import event_log
from .models import InvestigationInput, InvestigationOutput
from .registry import HolmesRegistry, set_registry
from .workflow import (
    _current_trace_carrier,
    investigation_workflow,
    register_workflow_components,
)

logger = logging.getLogger(__name__)

_FRAMEWORK_NAME = "HolmesGPT"


class DaprWorkflowHolmesRunner(BaseWorkflowRunner):
    """Runs HolmesGPT investigations as durable Dapr Workflows.

    Each LLM iteration and tool call becomes a Dapr activity. Approvals
    pause the workflow on ``wait_for_external_event``; clients resume by
    calling :meth:`approve`. A polling-based SSE bridge surfaces the same
    9 stream events the HolmesGPT HTTP server emits today.

    Example:
        ```python
        from diagrid.agent.holmesgpt import DaprWorkflowHolmesRunner

        runner = DaprWorkflowHolmesRunner(
            name="sre-agent",
            config_path="~/.holmes/config.yaml",
            model="anthropic/claude-sonnet-4-5-20250929",
        )
        runner.start()

        result = runner.invoke(
            messages=[{"role": "user", "content": "Why is checkout-api crashlooping?"}],
            max_steps=10,
        )
        print(result["final"]["content"])

        runner.shutdown()
        ```

    Args:
        name: Logical agent name (used in the workflow registration name).
        config_path: Optional path to a HolmesGPT config file. If ``None``,
            HolmesGPT loads from the environment / default config locations.
        model: Optional LLM model override, e.g.
            ``"anthropic/claude-sonnet-4-5-20250929"``.
        toolset_tags: Toolset tags to enable. Defaults to ``["core", "cluster"]``
            to match HolmesGPT's HTTP server defaults.
        max_steps: Default iteration cap (overridable per call).
        events_store_name: Dapr state store component used for the polling
            event tape. Defaults to ``HOLMES_EVENTS_STORE`` env or
            ``"holmes-events"``.
        events_key_prefix: Key prefix used in the event tape store.
        events_ttl_seconds: TTL for tape entries (0 = no TTL).
        host: Dapr sidecar gRPC host.
        port: Dapr sidecar gRPC port.
    """

    def __init__(
        self,
        *,
        name: str = "holmes-sre",
        config_path: Optional[str] = None,
        model: Optional[str] = None,
        toolset_tags: Optional[List[str]] = None,
        enable_all_toolsets_possible: bool = False,
        max_steps: int = 10,
        events_store_name: Optional[str] = None,
        events_key_prefix: Optional[str] = None,
        events_ttl_seconds: Optional[int] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
    ) -> None:
        super().__init__(
            name,
            framework=_FRAMEWORK_NAME,
            host=host,
            port=port,
            max_iterations=max_steps,
        )
        self._config_path = config_path
        self._model = model
        self._toolset_tags = toolset_tags
        self._enable_all_toolsets_possible = enable_all_toolsets_possible
        self._max_steps = max_steps

        self._events_store_name = events_store_name or event_log.DEFAULT_STORE_NAME
        self._events_key_prefix = events_key_prefix or event_log.DEFAULT_KEY_PREFIX
        self._events_ttl_seconds = (
            events_ttl_seconds
            if events_ttl_seconds is not None
            else event_log.DEFAULT_TTL_SECONDS
        )

        self._registry: Optional[HolmesRegistry] = None
        self._input_mapper: Optional[Callable[[dict], dict]] = None

        self._register_workflow_components()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the HolmesGPT registry and start the workflow runtime."""
        if self._started:
            return

        self._registry = HolmesRegistry.build(
            config_path=self._config_path,
            model=self._model,
            toolset_tags=self._toolset_tags,
            enable_all_toolsets_possible=self._enable_all_toolsets_possible,
        )
        set_registry(self._registry)
        super().start()
        logger.info(
            "DaprWorkflowHolmesRunner started: workflow=%s tools=%d",
            self.workflow_name,
            len(self._registry.openai_tools),
        )

    # Required abstract overrides (BaseWorkflowRunner)

    def _setup_telemetry(self) -> None:
        setup_telemetry(self.__class__.__name__, config=self._observability_config)
        instrument_grpc(config=self._observability_config)

    def _setup_serve_defaults(self) -> None:
        # No additional serve-time defaults required for HolmesGPT.
        return None

    def _register_workflow_components(self) -> None:
        register_workflow_components(
            self._workflow_runtime,
            workflow_name=self.workflow_name,
        )

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        additional_system_prompt: Optional[str] = None,
        skills: Optional[Any] = None,
        images: Optional[List[Any]] = None,
        prompt_component_overrides: Optional[Dict[str, bool]] = None,
        global_instructions: Optional[Any] = None,
        max_steps: Optional[int] = None,
        request_context: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run an investigation the way ``holmes ask`` would, durably.

        Renders HolmesGPT's full system prompt (toolset instructions, skills
        catalog, global instructions, behaviour overrides, etc.) via
        :func:`holmes.core.conversations.build_chat_messages`, then schedules
        the resulting message list as a Dapr workflow. Returns the final
        :class:`InvestigationOutput` dict.

        Args mirror HolmesGPT's HTTP server's ``/api/chat`` request:

        - ``conversation_history``: prior messages in OpenAI format; system
          message will be added/refreshed by HolmesGPT.
        - ``additional_system_prompt``: extra instructions appended to the
          rendered system prompt.
        - ``skills``: a :class:`SkillCatalog`. Defaults to whatever
          ``cfg.get_skill_catalog()`` produced at runner build time.
        - ``prompt_component_overrides``: HolmesGPT behaviour-control toggles
          (the ``behavior_controls`` field from ``/api/chat``).

        All other args are forwarded to :meth:`invoke`.
        """
        messages = self._build_messages(
            question=question,
            conversation_history=conversation_history,
            additional_system_prompt=additional_system_prompt,
            skills=skills,
            images=images,
            prompt_component_overrides=prompt_component_overrides,
            global_instructions=global_instructions,
        )
        return self.invoke(
            messages=messages,
            max_steps=max_steps,
            request_context=request_context,
            response_format=response_format,
            temperature=temperature,
            workflow_id=workflow_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    async def ask_async(
        self,
        question: str,
        *,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        additional_system_prompt: Optional[str] = None,
        skills: Optional[Any] = None,
        images: Optional[List[Any]] = None,
        prompt_component_overrides: Optional[Dict[str, bool]] = None,
        global_instructions: Optional[Any] = None,
        max_steps: Optional[int] = None,
        request_context: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.3,
        last_event_seq: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming counterpart to :meth:`ask`. Same prompt rendering, yields
        events from the polling tape."""
        messages = self._build_messages(
            question=question,
            conversation_history=conversation_history,
            additional_system_prompt=additional_system_prompt,
            skills=skills,
            images=images,
            prompt_component_overrides=prompt_component_overrides,
            global_instructions=global_instructions,
        )
        async for ev in self.run_async(
            messages=messages,
            max_steps=max_steps,
            request_context=request_context,
            response_format=response_format,
            temperature=temperature,
            workflow_id=workflow_id,
            poll_interval=poll_interval,
            last_event_seq=last_event_seq,
        ):
            yield ev

    def _build_messages(
        self,
        *,
        question: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        additional_system_prompt: Optional[str],
        skills: Optional[Any],
        images: Optional[List[Any]],
        prompt_component_overrides: Optional[Dict[str, bool]],
        global_instructions: Optional[Any],
    ) -> List[Dict[str, Any]]:
        """Delegate prompt construction to HolmesGPT's ``build_chat_messages``."""
        if not self._started or self._registry is None:
            raise RuntimeError("Runner not started. Call start() first.")

        return build_chat_messages(
            ask=question,
            conversation_history=conversation_history,
            ai=self._registry.ai,
            config=self._registry.config,
            global_instructions=global_instructions,
            additional_system_prompt=additional_system_prompt,
            skills=skills if skills is not None else self._registry.skills,
            images=images,
            prompt_component_overrides=prompt_component_overrides,
        )

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_steps: Optional[int] = None,
        request_context: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run an investigation synchronously and return the final result.

        Returns a dict shaped like :class:`InvestigationOutput`.
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None
        assert self._registry is not None

        workflow_id = workflow_id or f"holmes-{uuid.uuid4().hex[:12]}"
        wf_input = InvestigationInput(
            messages=messages,
            tools=self._registry.openai_tools,
            max_steps=max_steps or self._max_steps,
            response_format=response_format,
            temperature=temperature,
            request_context=request_context,
            trace_context=_current_trace_carrier(),
        )

        self._workflow_client.schedule_new_workflow(
            workflow=investigation_workflow,
            input=wf_input.model_dump(),
            instance_id=workflow_id,
        )
        logger.info("Scheduled investigation workflow: %s", workflow_id)

        deadline = (time.time() + timeout) if timeout else None
        while True:
            time.sleep(poll_interval)
            if deadline and time.time() > deadline:
                raise TimeoutError(f"Workflow {workflow_id} timed out after {timeout}s")

            state = self._workflow_client.get_workflow_state(instance_id=workflow_id)
            if state is None:
                raise RuntimeError(f"Workflow {workflow_id} state not found")

            if state.runtime_status == WorkflowStatus.COMPLETED:
                if state.serialized_output:
                    payload = state.serialized_output
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    return InvestigationOutput.model_validate(payload).model_dump()
                return {}
            if state.runtime_status == WorkflowStatus.FAILED:
                raise RuntimeError(
                    getattr(
                        state.failure_details, "message", str(state.failure_details)
                    )
                )
            if state.runtime_status == WorkflowStatus.TERMINATED:
                raise RuntimeError(f"Workflow {workflow_id} was terminated")

    async def run_async(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_steps: Optional[int] = None,
        request_context: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.3,
        last_event_seq: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Schedule an investigation and stream events from the polling tape."""
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None
        assert self._registry is not None

        workflow_id = workflow_id or f"holmes-{uuid.uuid4().hex[:12]}"
        wf_input = InvestigationInput(
            messages=messages,
            tools=self._registry.openai_tools,
            max_steps=max_steps or self._max_steps,
            response_format=response_format,
            temperature=temperature,
            request_context=request_context,
            trace_context=_current_trace_carrier(),
        )

        self._workflow_client.schedule_new_workflow(
            workflow=investigation_workflow,
            input=wf_input.model_dump(),
            instance_id=workflow_id,
        )
        yield {
            "type": "workflow_started",
            "workflow_id": workflow_id,
        }

        async for ev in self.stream_events(
            workflow_id,
            poll_interval=poll_interval,
            last_event_seq=last_event_seq,
        ):
            yield ev

    async def stream_events(
        self,
        workflow_id: str,
        *,
        poll_interval: float = 0.3,
        last_event_seq: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Poll the event tape for an existing workflow and yield events.

        Use this to resume a stream after an SSE reconnect: pass
        ``last_event_seq`` from the client's ``Last-Event-Id``.
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        seq = last_event_seq
        while True:
            events = await asyncio.to_thread(
                event_log.read_after,
                workflow_id,
                seq,
                64,
                store_name=self._events_store_name,
                key_prefix=self._events_key_prefix,
            )
            for ev in events:
                seq = ev["seq"]
                yield {"type": "event", "workflow_id": workflow_id, **ev}

            state = self._workflow_client.get_workflow_state(instance_id=workflow_id)
            if state is None:
                yield {
                    "type": "workflow_error",
                    "workflow_id": workflow_id,
                    "error": "Workflow state not found",
                }
                return

            if state.runtime_status == WorkflowStatus.COMPLETED:
                # Drain anything that landed between the last read and now.
                tail = await asyncio.to_thread(
                    event_log.read_after,
                    workflow_id,
                    seq,
                    64,
                    store_name=self._events_store_name,
                    key_prefix=self._events_key_prefix,
                )
                for ev in tail:
                    seq = ev["seq"]
                    yield {"type": "event", "workflow_id": workflow_id, **ev}

                output = state.serialized_output
                if isinstance(output, str):
                    output = json.loads(output) if output else None
                yield {
                    "type": "workflow_completed",
                    "workflow_id": workflow_id,
                    "output": output,
                }
                return

            if state.runtime_status == WorkflowStatus.FAILED:
                fd = state.failure_details
                yield {
                    "type": "workflow_failed",
                    "workflow_id": workflow_id,
                    "error": {
                        "message": getattr(fd, "message", str(fd)),
                        "error_type": getattr(fd, "error_type", None),
                    }
                    if fd
                    else None,
                }
                return

            if state.runtime_status == WorkflowStatus.TERMINATED:
                yield {
                    "type": "workflow_terminated",
                    "workflow_id": workflow_id,
                }
                return

            await asyncio.sleep(0 if events else poll_interval)

    def read_events_after(
        self,
        workflow_id: str,
        since_seq: int = 0,
        limit: int = 64,
    ) -> List[Dict[str, Any]]:
        """Synchronous one-shot read of the event tape."""
        return event_log.read_after(
            workflow_id,
            since_seq,
            limit,
            store_name=self._events_store_name,
            key_prefix=self._events_key_prefix,
        )

    # ------------------------------------------------------------------
    # Approval / pause resume
    # ------------------------------------------------------------------

    def approve(
        self,
        workflow_id: str,
        tool_call_id: str,
        *,
        approved: bool = True,
        reason: Optional[str] = None,
        session_approved_prefixes: Optional[List[str]] = None,
    ) -> None:
        """Resume a workflow paused on a HolmesGPT tool approval."""
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        self._workflow_client.raise_workflow_event(
            instance_id=workflow_id,
            event_name=f"resume:{tool_call_id}",
            data={
                "approved": approved,
                "reason": reason,
                "session_approved_prefixes": session_approved_prefixes or [],
            },
        )

    def submit_frontend_result(
        self,
        workflow_id: str,
        tool_call_id: str,
        frontend_result: str,
    ) -> None:
        """Resume a workflow paused on a frontend-executed tool."""
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")
        assert self._workflow_client is not None

        self._workflow_client.raise_workflow_event(
            instance_id=workflow_id,
            event_name=f"resume:{tool_call_id}",
            data={"frontend_result": frontend_result},
        )

    # ------------------------------------------------------------------
    # FastAPI server with SSE streaming
    # ------------------------------------------------------------------

    def build_fastapi_app(self) -> FastAPI:
        """Return a FastAPI app with /investigations and /investigations/{id}/stream."""
        app = FastAPI()
        runner = self

        @app.post("/investigations")
        async def create(req: dict) -> dict:  # type: ignore[type-arg]
            messages = req.get("messages")
            if not messages:
                question = req.get("question") or req.get("ask") or req.get("task")
                if not question:
                    raise HTTPException(
                        status_code=400,
                        detail="Provide either 'messages' or 'question' in the request body.",
                    )
                # Default path: render HolmesGPT's full system prompt via
                # build_chat_messages so this endpoint behaves like a durable
                # version of ``holmes ask``.
                messages = runner._build_messages(
                    question=question,
                    conversation_history=req.get("conversation_history"),
                    additional_system_prompt=req.get("additional_system_prompt"),
                    skills=req.get("skills"),
                    images=req.get("images"),
                    prompt_component_overrides=req.get("behavior_controls"),
                    global_instructions=req.get("global_instructions"),
                )

            workflow_id = req.get("workflow_id") or f"holmes-{uuid.uuid4().hex[:12]}"
            wf_input = InvestigationInput(
                messages=messages,
                tools=runner._registry.openai_tools if runner._registry else None,
                max_steps=req.get("max_steps") or runner._max_steps,
                response_format=req.get("response_format"),
                temperature=req.get("temperature"),
                request_context=req.get("request_context"),
                trace_context=_current_trace_carrier(),
            )
            assert runner._workflow_client is not None
            runner._workflow_client.schedule_new_workflow(
                workflow=investigation_workflow,
                input=wf_input.model_dump(),
                instance_id=workflow_id,
            )
            return {"workflow_id": workflow_id, "status": "scheduled"}

        @app.get("/investigations/{workflow_id}")
        async def get_status(workflow_id: str) -> dict:  # type: ignore[type-arg]
            status = runner.get_workflow_status(workflow_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Workflow not found")
            return status

        @app.get("/investigations/{workflow_id}/stream")
        async def stream(
            workflow_id: str,
            request: Request,
            last_event_id: Optional[str] = Header(default=None),
        ) -> StreamingResponse:
            try:
                last_seq = int(last_event_id) if last_event_id else 0
            except ValueError:
                last_seq = 0

            async def gen() -> AsyncIterator[bytes]:
                async for ev in runner.stream_events(
                    workflow_id, last_event_seq=last_seq
                ):
                    if await request.is_disconnected():
                        return
                    if ev.get("type") == "event":
                        seq = ev.get("seq", 0)
                        event_name = ev.get("event", "message")
                        data = ev.get("data", {})
                        yield (
                            f"id: {seq}\n"
                            f"event: {event_name}\n"
                            f"data: {json.dumps(data)}\n\n"
                        ).encode()
                    else:
                        yield (
                            f"event: {ev.get('type', 'workflow_status')}\n"
                            f"data: {json.dumps(ev)}\n\n"
                        ).encode()

            return StreamingResponse(gen(), media_type="text/event-stream")

        @app.post("/investigations/{workflow_id}/approve")
        async def approve_endpoint(workflow_id: str, body: dict) -> dict:  # type: ignore[type-arg]
            tool_call_id = body.get("tool_call_id")
            if not tool_call_id:
                raise HTTPException(status_code=400, detail="tool_call_id is required")
            runner.approve(
                workflow_id,
                tool_call_id,
                approved=bool(body.get("approved", True)),
                reason=body.get("reason"),
                session_approved_prefixes=body.get("session_approved_prefixes"),
            )
            return {"workflow_id": workflow_id, "status": "resumed"}

        @app.post("/investigations/{workflow_id}/frontend_result")
        async def frontend_result_endpoint(workflow_id: str, body: dict) -> dict:  # type: ignore[type-arg]
            tool_call_id = body.get("tool_call_id")
            frontend_result = body.get("frontend_result", "")
            if not tool_call_id:
                raise HTTPException(status_code=400, detail="tool_call_id is required")
            runner.submit_frontend_result(workflow_id, tool_call_id, frontend_result)
            return {"workflow_id": workflow_id, "status": "resumed"}

        @app.post("/investigations/{workflow_id}/stop")
        async def stop_endpoint(workflow_id: str) -> dict:  # type: ignore[type-arg]
            """Terminate a running workflow.

            Delegates to ``BaseWorkflowRunner.terminate_workflow`` which calls
            Dapr's ``terminate_workflow`` API. The workflow's state remains in
            the actor store; call ``/purge`` to delete it.
            """
            runner.terminate_workflow(workflow_id)
            return {"workflow_id": workflow_id, "status": "terminated"}

        @app.post("/investigations/{workflow_id}/purge")
        async def purge_endpoint(workflow_id: str) -> dict:  # type: ignore[type-arg]
            """Purge a completed / failed / terminated workflow.

            Removes the workflow's actor state from Dapr's store. Only valid
            for workflows in a terminal status.
            """
            runner.purge_workflow(workflow_id)
            return {"workflow_id": workflow_id, "status": "purged"}

        return app

    def serve(  # type: ignore[override]
        self,
        *,
        port: int = 5001,
        host: str = "0.0.0.0",
        **_: Any,
    ) -> None:
        """Run a uvicorn server exposing investigation endpoints."""
        self._setup_telemetry()
        app = self.build_fastapi_app()
        # Tie the Dapr runtime to the server lifecycle: start on startup and
        # shut down gracefully on SIGINT/SIGTERM (driven by uvicorn).
        app.router.lifespan_context = self._serve_lifespan
        uvicorn.run(app, host=host, port=port)

    # ------------------------------------------------------------------
    # Required base override; we don't use the base ``serve()`` flow.
    # ------------------------------------------------------------------

    async def _serve_run(
        self,
        request: dict,
        session_id: str,  # type: ignore[type-arg]
    ) -> AsyncIterator[Dict[str, Any]]:  # pragma: no cover
        messages = request.get("messages") or [
            {
                "role": "user",
                "content": request.get("question") or request.get("task") or "",
            }
        ]
        async for ev in self.run_async(
            messages=messages,
            max_steps=request.get("max_steps") or self._max_steps,
            request_context=request.get("request_context"),
        ):
            yield ev

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def registry(self) -> Optional[HolmesRegistry]:
        """The worker-local HolmesGPT registry (None until ``start()``)."""
        return self._registry
