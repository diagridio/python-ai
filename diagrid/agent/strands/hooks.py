# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr Workflow Hooks - Hook providers for Dapr workflow integration.

This module provides HookProvider implementations for integrating Strands
agents with Dapr workflows, including telemetry, distributed tracing,
and workflow-specific behaviors.
"""

import logging
from typing import Any, TYPE_CHECKING

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)

if TYPE_CHECKING:
    from strands.agent import Agent

logger = logging.getLogger(__name__)


class DaprWorkflowHookProvider(HookProvider):
    """Hook provider for Dapr workflow integration.

    This hook provider adds Dapr-specific functionality to Strands agents:

    - Distributed tracing correlation
    - Workflow telemetry emission
    - Tool call checkpointing signals
    - Activity execution tracking

    Example:
        ```python
        from diagrid.agent.strands import DaprWorkflowHookProvider
        from strands import Agent

        hooks = DaprWorkflowHookProvider(
            enable_tracing=True,
            emit_telemetry=True,
        )

        agent = Agent(
            model="us.amazon.nova-pro-v1:0",
            tools=[my_tool],
            hooks=[hooks],
        )
        ```

    Args:
        enable_tracing: Enable distributed tracing integration (default: True)
        emit_telemetry: Emit Dapr telemetry events (default: True)
        trace_tool_inputs: Include tool inputs in traces (default: False for security)
    """

    def __init__(
        self,
        enable_tracing: bool = True,
        emit_telemetry: bool = True,
        trace_tool_inputs: bool = False,
    ) -> None:
        """Initialize the Dapr workflow hook provider.

        Args:
            enable_tracing: Enable distributed tracing
            emit_telemetry: Emit telemetry events
            trace_tool_inputs: Include tool inputs in traces
        """
        self.enable_tracing = enable_tracing
        self.emit_telemetry = emit_telemetry
        self.trace_tool_inputs = trace_tool_inputs

        self._dapr_client: Any = None
        self._invocation_count = 0
        self._tool_call_count = 0
        self._model_call_count = 0

    def _get_client(self) -> Any:
        """Get or create the Dapr client for telemetry.

        Returns:
            The DaprClient instance or None if not available
        """
        if self._dapr_client is None:
            try:
                from dapr.clients import DaprClient

                self._dapr_client = DaprClient()
            except ImportError:
                logger.debug("Dapr client not available - telemetry disabled")
                return None

        return self._dapr_client

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks with the agent's hook registry.

        Args:
            registry: The agent's hook registry
            **kwargs: Additional arguments
        """
        # Invocation hooks
        registry.add_callback(BeforeInvocationEvent, self._on_before_invocation)
        registry.add_callback(
            AfterInvocationEvent,
            self._on_after_invocation,
        )

        # Model call hooks
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)
        registry.add_callback(AfterModelCallEvent, self._on_after_model_call)

        # Tool call hooks
        registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)
        registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)

    async def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Handle before invocation event.

        Args:
            event: The BeforeInvocationEvent
        """
        self._invocation_count += 1
        self._tool_call_count = 0
        self._model_call_count = 0

        if self.emit_telemetry:
            await self._emit_event(
                "strands.agent.invocation.start",
                {
                    "invocation_id": self._invocation_count,
                    "message_count": len(event.agent.messages),
                },
            )

        logger.debug(
            "invocation=%d | agent invocation starting",
            self._invocation_count,
        )

    async def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Handle after invocation event.

        Args:
            event: The AfterInvocationEvent
        """
        if self.emit_telemetry:
            await self._emit_event(
                "strands.agent.invocation.end",
                {
                    "invocation_id": self._invocation_count,
                    "tool_calls": self._tool_call_count,
                    "model_calls": self._model_call_count,
                    "stop_reason": str(getattr(event, "stop_reason", None))
                    if getattr(event, "stop_reason", None)
                    else None,
                },
            )

        logger.debug(
            "invocation=%d tools=%d models=%d | agent invocation completed",
            self._invocation_count,
            self._tool_call_count,
            self._model_call_count,
        )

    async def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Handle before model call event.

        Args:
            event: The BeforeModelCallEvent
        """
        self._model_call_count += 1

        if self.emit_telemetry:
            await self._emit_event(
                "strands.agent.model.call.start",
                {
                    "invocation_id": self._invocation_count,
                    "model_call_id": self._model_call_count,
                },
            )

        logger.debug(
            "invocation=%d model_call=%d | model call starting",
            self._invocation_count,
            self._model_call_count,
        )

    async def _on_after_model_call(self, event: AfterModelCallEvent) -> None:
        """Handle after model call event.

        Args:
            event: The AfterModelCallEvent
        """
        if self.emit_telemetry:
            await self._emit_event(
                "strands.agent.model.call.end",
                {
                    "invocation_id": self._invocation_count,
                    "model_call_id": self._model_call_count,
                    "retry": event.retry,
                },
            )

        logger.debug(
            "invocation=%d model_call=%d retry=%s | model call completed",
            self._invocation_count,
            self._model_call_count,
            event.retry,
        )

    async def _on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Handle before tool call event.

        Args:
            event: The BeforeToolCallEvent
        """
        self._tool_call_count += 1

        tool_name = event.tool_use.get("name", "unknown")
        tool_use_id = event.tool_use.get("toolUseId", "unknown")

        if self.emit_telemetry:
            telemetry_data: dict[str, Any] = {
                "invocation_id": self._invocation_count,
                "tool_call_id": self._tool_call_count,
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
            }

            if self.trace_tool_inputs:
                telemetry_data["tool_input"] = event.tool_use.get("input")

            await self._emit_event(
                "strands.agent.tool.call.start",
                telemetry_data,
            )

        logger.debug(
            "invocation=%d tool_call=%d tool=%s | tool call starting",
            self._invocation_count,
            self._tool_call_count,
            tool_name,
        )

    async def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Handle after tool call event.

        Args:
            event: The AfterToolCallEvent
        """
        tool_name = event.tool_use.get("name", "unknown")
        tool_use_id = event.tool_use.get("toolUseId", "unknown")
        status = event.result.get("status", "unknown")

        if self.emit_telemetry:
            await self._emit_event(
                "strands.agent.tool.call.end",
                {
                    "invocation_id": self._invocation_count,
                    "tool_call_id": self._tool_call_count,
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "status": status,
                    "retry": event.retry,
                    "had_exception": event.exception is not None,
                },
            )

        logger.debug(
            "invocation=%d tool_call=%d tool=%s status=%s | tool call completed",
            self._invocation_count,
            self._tool_call_count,
            tool_name,
            status,
        )

    async def _emit_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Emit a telemetry event to Dapr.

        Args:
            event_name: The event name/type
            data: The event data
        """
        client = self._get_client()
        if client is None:
            return

        try:
            # Publish to a pubsub topic for telemetry
            import json

            client.publish_event(
                pubsub_name="strands-telemetry",
                topic_name="agent-events",
                data=json.dumps(
                    {
                        "event_type": event_name,
                        "data": data,
                    }
                ),
            )
        except Exception as e:
            # Don't fail the agent on telemetry errors
            logger.debug("Failed to emit telemetry event: %s", str(e))


class DaprRetryHookProvider(HookProvider):
    """Hook provider for Dapr-aware retry behavior.

    This hook provider integrates with Dapr's retry policies and
    circuit breaker patterns for model and tool calls.

    Example:
        ```python
        from diagrid.agent.strands.hooks import DaprRetryHookProvider
        from strands import Agent

        retry_hooks = DaprRetryHookProvider(
            max_model_retries=3,
            max_tool_retries=2,
        )

        agent = Agent(
            model="us.amazon.nova-pro-v1:0",
            tools=[my_tool],
            hooks=[retry_hooks],
        )
        ```
    """

    def __init__(
        self,
        max_model_retries: int = 3,
        max_tool_retries: int = 2,
        retry_on_rate_limit: bool = True,
    ) -> None:
        """Initialize the retry hook provider.

        Args:
            max_model_retries: Maximum model call retries
            max_tool_retries: Maximum tool call retries
            retry_on_rate_limit: Retry on rate limit errors
        """
        self.max_model_retries = max_model_retries
        self.max_tool_retries = max_tool_retries
        self.retry_on_rate_limit = retry_on_rate_limit

        self._model_retry_counts: dict[int, int] = {}
        self._tool_retry_counts: dict[str, int] = {}
        self._current_invocation = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks.

        Args:
            registry: The hook registry
            **kwargs: Additional arguments
        """
        registry.add_callback(BeforeInvocationEvent, self._on_before_invocation)
        registry.add_callback(AfterModelCallEvent, self._on_after_model_call)
        registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)

    async def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Reset retry counts for new invocation."""
        self._current_invocation += 1
        self._model_retry_counts.clear()
        self._tool_retry_counts.clear()

    async def _on_after_model_call(self, event: AfterModelCallEvent) -> None:
        """Handle model call retries.

        Args:
            event: The AfterModelCallEvent
        """
        invocation_key = self._current_invocation
        current_retries = self._model_retry_counts.get(invocation_key, 0)

        # Check if we should request retry
        if event.exception is not None and current_retries < self.max_model_retries:
            exception_str = str(event.exception).lower()

            # Check for retryable errors
            if self.retry_on_rate_limit and (
                "rate" in exception_str
                or "throttl" in exception_str
                or "429" in exception_str
            ):
                self._model_retry_counts[invocation_key] = current_retries + 1
                event.retry = True

                logger.info(
                    "model_retry=%d/%d | retrying model call due to rate limit",
                    current_retries + 1,
                    self.max_model_retries,
                )

    async def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Handle tool call retries.

        Args:
            event: The AfterToolCallEvent
        """
        tool_use_id = event.tool_use.get("toolUseId", "")
        current_retries = self._tool_retry_counts.get(tool_use_id, 0)

        # Check if we should request retry
        if event.exception is not None and current_retries < self.max_tool_retries:
            self._tool_retry_counts[tool_use_id] = current_retries + 1
            event.retry = True

            logger.info(
                "tool=%s retry=%d/%d | retrying tool call due to exception",
                event.tool_use.get("name"),
                current_retries + 1,
                self.max_tool_retries,
            )
