# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

# Alias `dapr_agents.hooks` to the local fallback shim so the plugin and these
# tests share the same hook-decision classes until dapr-agents ships the module.

import asyncio
import inspect
import sys

import pytest

from diagrid.agent.core.plugins import _hooks

sys.modules.setdefault("dapr_agents.hooks", _hooks)


# pytest-asyncio is not a repo dependency, so run coroutine tests on a fresh
# event loop here.
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "asyncio: run an async test on a fresh event loop"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function):
    func = pyfuncitem.obj
    if inspect.iscoroutinefunction(func):
        kwargs = {
            name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames
        }
        asyncio.run(func(**kwargs))
        return True
    return None
