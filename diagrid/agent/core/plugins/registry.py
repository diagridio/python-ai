# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""PluginRegistry — the chain dispatcher for the plugin system.

Runs the plugin chain on every lifecycle event and satisfies dapr-agents'
``LifecycleDispatcher`` Protocol (``attach``, ``detach``, ``dispatch``) so
it can be passed as ``DurableAgent(lifecycle_dispatcher=registry)``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

from diagrid.agent.core.plugins.context import LifecycleContext
from diagrid.agent.core.plugins.spi import (
    Deny,
    HookDecision,
    LifecycleEvent,
    Mutate,
    Plugin,
    Proceed,
    RequireApproval,
    Skip,
)

logger = logging.getLogger(__name__)

_FAILURE_MODE_CLOSED: Final = "closed"
_FAILURE_MODE_OPEN: Final = "open"

# Decision discriminators emitted in the returned decision dict.
_PROCEED: Final = "proceed"
_SKIP: Final = "skip"
_MUTATE: Final = "mutate"
_REQUIRE_APPROVAL: Final = "require_approval"
_DENY: Final = "deny"

# Optional fields copied verbatim onto the serialized decision when set.
_REQUIRE_APPROVAL_FIELDS: Final[Tuple[str, ...]] = (
    "timeout_seconds",
    "instructions",
    "reason",
    "required_approver_scopes",
    "allowed_approver_subjects",
    "approver_audience",
)
_DENY_FIELDS: Final[Tuple[str, ...]] = ("reason", "code", "details")

# Structured log event name for a plugin that raises from on_event.
_PLUGIN_EXCEPTION_EVENT: Final = "plugin.exception"


class PluginRegistry:
    """Holds an ordered chain of plugins and dispatches lifecycle events.

    Plugins run in ascending ``priority`` order, with registration order
    breaking ties.
    The first non-``Proceed`` ``Deny`` / ``Skip`` / ``RequireApproval``
    short-circuits the chain, while ``Mutate`` decisions accumulate so each
    subsequent plugin observes the merged payload.

    An exception raised from ``plugin.on_event`` is handled per the plugin's
    ``failure_mode``: ``"closed"`` short-circuits with
    ``Deny(code="plugin.<name>.exception")``, while ``"open"`` logs the error
    and treats the plugin as ``Proceed``.
    """

    def __init__(self, plugins: Optional[Sequence[Plugin]] = None) -> None:
        # Python's sort is stable, so a key of priority alone keeps
        # registration order for ties, which is the tie-break the chain relies
        # on.
        self._plugins: List[Plugin] = sorted(plugins or [], key=lambda p: p.priority)
        self._agent: Optional[Any] = None

    # ----- LifecycleDispatcher Protocol -----

    def attach(self, agent: Any) -> None:
        self._agent = agent
        for plugin in self._plugins:
            plugin.configure(agent)

    def detach(self) -> None:
        self._agent = None

    def dispatch(
        self,
        event_name: str,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run the plugin chain for a lifecycle event.

        Returns a serialized decision dict when a plugin alters the step, or
        ``None`` when the whole chain proceeds.
        """
        ctx = self._build_context(event_name, context)
        decision = self._run_chain_sync(ctx)
        return self._to_decision_dict(decision) if decision else None

    # ----- internal -----

    def _build_context(
        self, event_name: str, context: Dict[str, Any]
    ) -> LifecycleContext:
        known = {f.name for f in fields(LifecycleContext)} - {"event"}
        mapped = {k: v for k, v in context.items() if k in known}
        # LifecycleContext requires these; default them so a caller can
        # dispatch with a partial context without tripping construction.
        mapped.setdefault("workflow_instance_id", "")
        mapped.setdefault("agent_identity", {})
        return LifecycleContext(event=LifecycleEvent(event_name), **mapped)

    def _run_chain_sync(self, ctx: LifecycleContext) -> Optional[HookDecision]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._run_chain_async(ctx))
        # Already inside a running loop: drive the chain on a private loop in a
        # worker thread so we never touch the caller's live loop, which can't be
        # re-entered with run_until_complete.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(self._run_chain_async(ctx))).result()

    async def _run_chain_async(self, ctx: LifecycleContext) -> Optional[HookDecision]:
        accumulated_mutate_payload: Dict[str, Any] = {}

        for plugin in self._plugins:
            if not self._plugin_handles_event(plugin, ctx.event):
                continue

            try:
                decision = await self._invoke_plugin(plugin, ctx)
            except Exception as exc:
                decision = self._handle_plugin_exception(plugin, exc)

            if decision is None or isinstance(decision, Proceed):
                continue
            if isinstance(decision, Mutate):
                payload = decision.payload or {}
                accumulated_mutate_payload.update(payload)
                # Merge back onto the context so later plugins observe the
                # accumulated payload, not just the original input.
                ctx.payload.update(payload)
                continue
            return decision

        if accumulated_mutate_payload:
            return Mutate(payload=accumulated_mutate_payload)
        return None

    def _plugin_handles_event(self, plugin: Plugin, event: LifecycleEvent) -> bool:
        capabilities = getattr(plugin, "capabilities", None)
        if capabilities is None:
            return True
        return event in capabilities

    async def _invoke_plugin(
        self, plugin: Plugin, ctx: LifecycleContext
    ) -> Optional[HookDecision]:
        if inspect.iscoroutinefunction(plugin.on_event):
            return await plugin.on_event(ctx)
        # Sync hook: run off the event loop so a blocking plugin can't stall
        # the chain. Still await the result in case it returns an awaitable.
        result = await asyncio.to_thread(plugin.on_event, ctx)
        if inspect.isawaitable(result):
            return await result
        return result

    def _handle_plugin_exception(self, plugin: Plugin, exc: Exception) -> HookDecision:
        mode = getattr(plugin, "failure_mode", _FAILURE_MODE_CLOSED)
        if mode == _FAILURE_MODE_OPEN:
            logger.warning(
                _PLUGIN_EXCEPTION_EVENT,
                extra={"plugin": plugin.name, "error": str(exc)},
                exc_info=exc,
            )
            return Proceed()
        logger.error(
            _PLUGIN_EXCEPTION_EVENT,
            extra={"plugin": plugin.name, "error": str(exc)},
            exc_info=exc,
        )
        return Deny(
            code=f"plugin.{plugin.name}.exception",
            details={"plugin": plugin.name, "error": str(exc)},
        )

    def _to_decision_dict(self, decision: HookDecision) -> Dict[str, Any]:
        if isinstance(decision, Proceed):
            return {"type": _PROCEED}
        if isinstance(decision, Skip):
            return {"type": _SKIP, "result": decision.result}
        if isinstance(decision, Mutate):
            return {"type": _MUTATE, "payload": decision.payload or {}}
        if isinstance(decision, RequireApproval):
            approval: Dict[str, Any] = {"type": _REQUIRE_APPROVAL}
            for f in _REQUIRE_APPROVAL_FIELDS:
                v = getattr(decision, f, None)
                if v is not None:
                    approval[f] = list(v) if isinstance(v, (set, frozenset)) else v
            return approval
        if isinstance(decision, Deny):
            deny: Dict[str, Any] = {"type": _DENY}
            for f in _DENY_FIELDS:
                v = getattr(decision, f, None)
                if v is not None:
                    deny[f] = v
            return deny
        raise TypeError(f"unknown HookDecision: {type(decision)!r}")


def dispatch_plugin_event(
    registry: PluginRegistry,
    event: LifecycleEvent,
    context: Dict[str, Any],
    *,
    propagate_lineage: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run the plugin chain and propagate workflow lineage.

    Sub-workflow scheduling sites call this so the lineage propagation step
    lives inside the dispatch helper itself.
    Folding it in here means a per-framework runner cannot accidentally
    schedule a sub-workflow without joining the workflow history lineage
    chain.
    """
    if propagate_lineage:
        # Imported lazily so this module does not hard-depend on the workflow
        # runtime being initialized.
        from dapr.ext.workflow import wfctx  # type: ignore

        try:
            wfctx.PropagateLineage()
        except Exception as exc:
            logger.debug("PropagateLineage skipped (no workflow context): %s", exc)

    return registry.dispatch(event.value, context)
