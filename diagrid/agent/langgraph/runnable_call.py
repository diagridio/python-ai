# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Calling a node's Runnable or coroutine from a (synchronous) activity.

Async nodes run on one process-wide event loop (see :class:`_BackgroundLoop`);
a Runnable with a real async entry point is called through ``ainvoke``.
"""

import asyncio
import concurrent.futures
import functools
import logging
import threading
from typing import Any, Dict, Optional, Tuple, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import run_in_executor

logger = logging.getLogger(__name__)


def _has_async_entry(runnable: Any) -> bool:
    """Whether ``runnable`` has a real async entry point, which a node should use.

    ``ToolNode`` has a sync ``func`` and an ``afunc``; its sync one cannot run
    an ``async def`` tool (MCP adapter tools are such), its async one runs both,
    A plain sync node also gets an ``afunc``, but
    only a ``run_in_executor`` wrapper around its function: that is not one.

    A composite (a compiled subgraph, a ``RunnableSequence``) has no function of
    its own and no ``afunc``, but a step inside it may be an ``async def`` with
    no sync entry, so it goes through ``ainvoke`` too: that runs a sync step in
    an executor.
    """
    afunc = getattr(runnable, "afunc", None)
    if afunc is None:
        return not hasattr(runnable, "func") and hasattr(runnable, "ainvoke")
    return not (isinstance(afunc, functools.partial) and afunc.func is run_in_executor)


class _BackgroundLoop:
    """One event loop per process, in a daemon thread, that async nodes run on.

    An async client a node keeps (a chat model's HTTP connection pool) is bound
    to the loop it was first used on. A loop made and closed for each activity
    would break that client for the next activity, so every coroutine runs on
    this one, which is never closed.

    ``asyncio`` re-raises ``SystemExit`` and ``KeyboardInterrupt`` out of
    ``run_forever``, which would end the loop's thread and leave every waiting
    and later async node hanging. A node's own coroutine is guarded so those
    fail that node alone, and a loop whose thread has died anyway is replaced.
    """

    #: Seconds between looks at whether the loop's thread is still alive.
    _CHECK_SECONDS = 1.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _running_loop(self) -> Tuple[asyncio.AbstractEventLoop, threading.Thread]:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            if self._loop is None or self._loop.is_closed() or not alive:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, name="langgraph-node-loop", daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            assert self._loop is not None and self._thread is not None
            return self._loop, self._thread

    @staticmethod
    async def _guarded(coro: Any) -> Any:
        """``coro``, with an exit request turned into an error of the node."""
        try:
            return await coro
        except (SystemExit, KeyboardInterrupt) as e:
            raise RuntimeError(
                f"The node raised {type(e).__name__}({', '.join(map(repr, e.args))}); "
                "it fails the node, not the process"
            ) from e

    def run(self, coro: Any) -> Any:
        """Run ``coro`` to completion on the loop and return its result.

        ``run_coroutine_threadsafe`` starts the task in a copy of the calling
        thread's context, so the ambient config ``set_config_context`` set for
        the node is what the coroutine sees.
        """
        loop, thread = self._running_loop()
        future = asyncio.run_coroutine_threadsafe(self._guarded(coro), loop)
        while True:
            concurrent.futures.wait([future], timeout=self._CHECK_SECONDS)
            if future.done():
                return future.result()
            if not thread.is_alive():
                future.cancel()
                raise RuntimeError(
                    "The event loop async nodes run on stopped while this node "
                    "was running (something ended it, as SystemExit does)"
                )


_background_loop = _BackgroundLoop()


def _await_if_coroutine(r: Any) -> Any:
    """Run ``r`` to completion on the process's node loop if it is a coroutine."""
    if asyncio.iscoroutine(r):
        return _background_loop.run(r)
    return r


def _invoke_runnable(runnable: Any, value: Any, config: Dict[str, Any]) -> Any:
    """Call a Runnable; one with an async entry returns a coroutine to await."""
    if not _has_async_entry(runnable):
        return runnable.invoke(value, config=config)
    if getattr(runnable, "func", None) is not None:
        return _ainvoke_or_sync(runnable, value, config)
    return runnable.ainvoke(value, config=config)


def _is_sync_only_middleware(error: BaseException) -> bool:
    """Whether ``error`` is a middleware that has only a sync wrap hook.

    LangChain's ``AgentMiddleware`` raises it from ``awrap_model_call`` /
    ``awrap_tool_call`` when only ``wrap_model_call`` / ``wrap_tool_call`` is
    defined, before the wrapped call runs.
    """
    text = str(error)
    return (
        isinstance(error, NotImplementedError)
        and "Asynchronous implementation of awrap_" in text
        and "is not available" in text
    )


async def _ainvoke_or_sync(runnable: Any, value: Any, config: Dict[str, Any]) -> Any:
    """``ainvoke``, or ``invoke`` for a node whose middleware is sync-only.

    Only a node with its own sync function falls back: the error comes before
    the model or tool call it wraps, so nothing ran twice. A composite (a
    subgraph) is not retried, as its earlier steps did run.
    """
    try:
        return await runnable.ainvoke(value, config=config)
    except NotImplementedError as e:
        if not _is_sync_only_middleware(e):
            raise
        logger.info(
            "A middleware of node %r has only a sync wrap hook; running the node "
            "through its sync entry",
            getattr(runnable, "name", None),
        )
        # in an executor, with this context: the ambient config goes with it
        return await run_in_executor(
            cast(RunnableConfig, config), runnable.invoke, value, config=config
        )
