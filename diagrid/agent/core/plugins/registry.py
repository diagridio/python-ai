# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
PluginRegistry runs the plugin chain on every lifecycle event.

Implements the ``LifecycleDispatcher`` Protocol so it can be passed to an
agent via the ``lifecycle_dispatcher=`` keyword.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import fields
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

from dapr_agents.hooks import (
    Deny,
    HookDecision,
    Mutate,
    Proceed,
    RequireApproval,
    Skip,
)
from dapr_agents.lifecycle import DecisionDict, LifecycleDispatcher

from .context import LifecycleContext
from .spi import FAILURE_MODE_CLOSED, FAILURE_MODE_OPEN, LifecycleEvent, Plugin

logger = logging.getLogger(__name__)

# Decision discriminators emitted in ``DecisionDict["type"]``.
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

# Structured log event name and the key collecting unmapped context entries.
_PLUGIN_EXCEPTION_EVENT: Final = "plugin.exception"
_EXTRA_KEY: Final = "extra"


class PluginRegistry(LifecycleDispatcher):
    """Chain dispatcher running plugins in priority order.

    Plugins are stable-sorted by ``(priority, registration_index)``.
    The first non-``Proceed`` ``Deny`` / ``Skip`` / ``RequireApproval``
    short-circuits the chain.
    ``Mutate`` decisions accumulate, so each subsequent plugin sees the
    merged payload.

    An exception raised from ``plugin.on_event`` is handled per the
    plugin's ``failure_mode``: ``"closed"`` short-circuits the chain with
    ``Deny(code="plugin.<name>.exception")``, while ``"open"`` logs the
    error and treats the plugin as ``Proceed``.
    """

    def __init__(self, plugins: Sequence[Plugin] = ()):
        self._plugins: List[Plugin] = list(plugins)
        registration_index = {id(p): i for i, p in enumerate(self._plugins)}
        self._plugins.sort(key=lambda p: (p.priority, registration_index[id(p)]))
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
    ) -> Optional[DecisionDict]:
        """Sync entry point called by the agent.

        Bridges into the async chain via ``asyncio.run`` when no loop is
        running, otherwise drives the existing loop to completion.
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
        unmapped = {k: v for k, v in context.items() if k not in known and k != "event"}
        if unmapped:
            mapped.setdefault(_EXTRA_KEY, {}).update(unmapped)
        return LifecycleContext(event=LifecycleEvent(event_name), **mapped)

    def _run_chain_sync(self, ctx: LifecycleContext) -> Optional[HookDecision]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._run_chain_async(ctx))
        return loop.run_until_complete(self._run_chain_async(ctx))

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
                accumulated_mutate_payload.update(decision.payload or {})
                continue
            return decision

        if accumulated_mutate_payload:
            return Mutate(payload=accumulated_mutate_payload)
        return None

    def _plugin_handles_event(self, plugin: Plugin, event: LifecycleEvent) -> bool:
        capabilities = getattr(plugin, "capabilities", None)
        if not capabilities:
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
        mode = getattr(plugin, "failure_mode", FAILURE_MODE_CLOSED)
        if mode == FAILURE_MODE_OPEN:
            logger.warning(
                _PLUGIN_EXCEPTION_EVENT,
                extra={"plugin": plugin.name, "error": str(exc)},
            )
            return Proceed()
        logger.error(
            _PLUGIN_EXCEPTION_EVENT,
            extra={"plugin": plugin.name, "error": str(exc)},
        )
        return Deny(
            code=f"plugin.{plugin.name}.exception",
            details={"plugin": plugin.name, "error": str(exc)},
        )

    def _to_decision_dict(self, decision: HookDecision) -> DecisionDict:
        if isinstance(decision, Proceed):
            return {"type": _PROCEED}
        if isinstance(decision, Skip):
            return {"type": _SKIP, "result": decision.result}
        if isinstance(decision, Mutate):
            return {"type": _MUTATE, "payload": decision.payload or {}}
        if isinstance(decision, RequireApproval):
            d: DecisionDict = {"type": _REQUIRE_APPROVAL}
            for f in _REQUIRE_APPROVAL_FIELDS:
                v = getattr(decision, f, None)
                if v is not None:
                    d[f] = list(v) if isinstance(v, (set, frozenset)) else v
            return d
        if isinstance(decision, Deny):
            d = {"type": _DENY}
            for f in _DENY_FIELDS:
                v = getattr(decision, f, None)
                if v is not None:
                    d[f] = v
            return d
        raise TypeError(f"unknown HookDecision: {type(decision)!r}")


def dispatch_plugin_event(
    registry: PluginRegistry,
    event: LifecycleEvent,
    context: Dict[str, Any],
    *,
    propagate_lineage: bool = True,
) -> Optional[DecisionDict]:
    """Run the plugin chain and propagate workflow lineage.

    Sub-workflow scheduling sites call this so that the lineage propagation
    step lives inside the dispatch helper itself.
    Folding it in here means a per-framework runner cannot accidentally
    schedule a sub-workflow without joining the workflow history lineage
    chain.
    """
    if propagate_lineage:
        # Imported lazily so this module does not hard-depend on the
        # workflow runtime being initialized.
        from dapr.ext.workflow import wfctx  # type: ignore

        try:
            wfctx.PropagateLineage()
        except Exception as exc:
            logger.debug("PropagateLineage skipped (no workflow context): %s", exc)

    return registry.dispatch(event.value, context)
