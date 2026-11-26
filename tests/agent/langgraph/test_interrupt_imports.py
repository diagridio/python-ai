# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``interrupt()`` support reads LangGraph internals: a langgraph without them is named."""

import builtins
import unittest
from typing import Any
from unittest import mock

from diagrid.agent.langgraph.interrupts import interrupt_config


class TestInternalImports(unittest.TestCase):
    def test_a_langgraph_without_the_internals_gets_a_clear_error(self):
        real = builtins.__import__

        def fake(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("langgraph._internal"):
                raise ImportError(f"No module named {name!r}")
            return real(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake):
            with self.assertRaises(ImportError) as caught:
                interrupt_config(
                    namespace="n", step=0, stop=None, resume=None, send=lambda w: None
                )
        self.assertIn("langgraph", str(caught.exception))
        self.assertIn("1.2", str(caught.exception))
