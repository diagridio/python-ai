# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The loop async nodes share survives what a node does to it.

``asyncio`` re-raises ``SystemExit`` and ``KeyboardInterrupt`` out of the loop's
``run_forever``, which ends the loop's thread. Every async node and async
condition of the process runs on that one loop, so it must not stay dead: the
node that raised fails, and the next one runs.
"""

import asyncio
import concurrent.futures
import threading
import unittest
from typing import Any

from diagrid.agent.langgraph.workflow import _BackgroundLoop

_WAIT = 10


def _run_in_thread(loop: _BackgroundLoop, make_coro: Any) -> concurrent.futures.Future:
    """``loop.run(...)`` on a worker, so a hang fails the test instead of freezing it."""
    future: concurrent.futures.Future = concurrent.futures.Future()

    def work() -> None:
        try:
            future.set_result(loop.run(make_coro()))
        except BaseException as e:  # reported through the future
            future.set_exception(e)

    threading.Thread(target=work, daemon=True).start()
    return future


async def _ok() -> int:
    return 7


async def _exits() -> None:
    raise SystemExit(3)


async def _interrupts() -> None:
    raise KeyboardInterrupt()


class TestBackgroundLoop(unittest.TestCase):
    def test_a_coroutine_runs_and_returns(self):
        loop = _BackgroundLoop()
        self.assertEqual(_run_in_thread(loop, _ok).result(_WAIT), 7)

    def test_system_exit_fails_that_node_and_the_next_one_runs(self):
        for raiser in (_exits, _interrupts):
            with self.subTest(raiser=raiser.__name__):
                loop = _BackgroundLoop()
                with self.assertRaises(RuntimeError) as caught:
                    _run_in_thread(loop, raiser).result(_WAIT)
                self.assertIn(
                    "SystemExit" if raiser is _exits else "KeyboardInterrupt",
                    str(caught.exception),
                )
                self.assertEqual(_run_in_thread(loop, _ok).result(_WAIT), 7)

    def test_a_dead_loop_is_replaced_and_a_waiting_node_fails(self):
        loop = _BackgroundLoop()
        started = threading.Event()

        async def stuck() -> None:
            started.set()
            await asyncio.sleep(3600)

        waiting = _run_in_thread(loop, stuck)
        self.assertTrue(started.wait(_WAIT))
        # a task of the node's own ends the loop's thread, as a stray one would
        dead = loop._loop
        assert dead is not None

        dead.call_soon_threadsafe(lambda: dead.create_task(_exits()))
        with self.assertRaises(RuntimeError):
            waiting.result(_WAIT)
        self.assertEqual(_run_in_thread(loop, _ok).result(_WAIT), 7)
