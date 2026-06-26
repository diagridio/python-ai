# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

# TODO(AI-596): dapr-agents has not yet released its `dapr_agents.hooks` module.
# Alias it to the local fallback shim so the OAuthPlugin and these tests import
# the same hook-decision classes. Remove this once AI-596 ships and the real
# module is installable.

import asyncio
import inspect
import sys

import pytest

from diagrid.agent.core.plugins import _hooks

sys.modules.setdefault("dapr_agents.hooks", _hooks)


# pytest-asyncio is not a dependency of this repo, so run coroutine test
# functions on a fresh event loop here. Keeps the @pytest.mark.asyncio test
# bodies unchanged without adding a test-only dependency.
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
