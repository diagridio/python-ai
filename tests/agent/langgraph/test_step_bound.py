# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""How many steps a run may take: ``recursion_limit`` and the runner's ``max_steps``.

A ``recursion_limit`` bounds the run as LangGraph counts it: that many node steps
run, and the run fails when it has taken them, whether or not more is pending.
The runner's ``max_steps`` is its own guard; a limit that only the graph's
``with_config`` supplies cannot raise it.
"""

import asyncio
from typing import Any, List

from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .golden_harness import _FakeDaprTest, _FakeWorkflowClient, _runner


class _S(TypedDict, total=False):
    n: int


def _forever() -> Any:
    def step(state: _S) -> dict:
        return {"n": state.get("n", 0) + 1}

    graph = StateGraph(_S)
    graph.add_node("step", step)
    graph.add_edge(START, "step")
    graph.add_conditional_edges("step", lambda s: "step", ["step"])
    return graph.compile()


def _talking_chain(length: int) -> Any:
    graph = StateGraph(_S)
    names = [f"n{i}" for i in range(length)]
    for name in names:

        def node(state: _S, name: str = name) -> dict:
            get_stream_writer()(name)
            return {"n": state.get("n", 0) + 1}

        graph.add_node(name, node)
    graph.add_edge(START, names[0])
    for a, b in zip(names, names[1:]):
        graph.add_edge(a, b)
    graph.add_edge(names[-1], END)
    return graph.compile()


def _talking_loop() -> Any:
    def step(state: _S) -> dict:
        get_stream_writer()(state.get("n", 0))
        return {"n": state.get("n", 0) + 1}

    graph = StateGraph(_S)
    graph.add_node("step", step)
    graph.add_edge(START, "step")
    graph.add_conditional_edges("step", lambda s: "step", ["step"])
    return graph.compile()


def _runner_with(compiled: Any, name: str, **kwargs: Any) -> DaprWorkflowGraphRunner:
    runner = DaprWorkflowGraphRunner(graph=compiled, name=name, **kwargs)
    runner._started = True
    runner._workflow_client = _FakeWorkflowClient(runner)  # type: ignore[assignment]
    return runner


async def _customs(runner: Any, config: Any, name: str) -> List[Any]:
    """The custom chunks of a run, up to the recursion error it ends with."""
    seen: List[Any] = []
    try:
        async for mode, chunk in runner.astream(
            {},
            config=config,
            workflow_id=f"wf-{name}",
            stream_mode=["custom"],
            poll_interval=0.01,
        ):
            if mode == "custom":
                seen.append(chunk)
    except GraphRecursionError:
        seen.append("GraphRecursionError")
    return seen


def _native_customs(compiled: Any, config: Any) -> List[Any]:
    seen: List[Any] = []
    try:
        for mode, chunk in compiled.stream({}, config, stream_mode=["custom"]):
            seen.append(chunk)
    except GraphRecursionError:
        seen.append("GraphRecursionError")
    return seen


class TestRecursionLimitCountsLikeLangGraph(_FakeDaprTest):
    async def test_a_chain_cut_by_the_limit_runs_every_step_the_limit_allows(self):
        compiled = _talking_chain(4)
        config = {"recursion_limit": 4}
        native = _native_customs(compiled, config)
        self.assertEqual(native, ["n0", "n1", "n2", "n3", "GraphRecursionError"])
        runner = _runner(compiled, "limit-chain")
        self.assertEqual(await _customs(runner, config, "limit-chain"), native)

    async def test_a_loop_runs_as_many_steps_as_the_limit(self):
        compiled = _talking_loop()
        config = {"recursion_limit": 10}
        native = _native_customs(compiled, config)
        self.assertEqual(len(native), 11)
        runner = _runner(compiled, "limit-loop")
        self.assertEqual(await _customs(runner, config, "limit-loop"), native)

    async def test_a_limit_one_above_the_steps_a_chain_needs_completes(self):
        compiled = _talking_chain(4)
        config = {"recursion_limit": 5}
        native = _native_customs(compiled, config)
        self.assertEqual(native, ["n0", "n1", "n2", "n3"])
        runner = _runner(compiled, "limit-enough")
        self.assertEqual(await _customs(runner, config, "limit-enough"), native)


class TestRunnerMaxStepsAgainstTheGraphsBoundLimit(_FakeDaprTest):
    def _invoke_error(self, runner: Any, config: Any = None) -> str:
        try:
            runner.invoke({}, config=config, poll_interval=0.01)
        except GraphRecursionError as e:
            return str(e)
        return ""

    async def test_an_explicit_max_steps_caps_a_limit_only_the_graph_binds(self):
        compiled = _forever().with_config(recursion_limit=60)
        runner = _runner_with(compiled, "bound-capped", max_steps=5)
        message = await asyncio.to_thread(self._invoke_error, runner)
        self.assertIn("(5)", message)

    async def test_the_cap_holds_on_the_streaming_path_too(self):
        compiled = _talking_loop().with_config(recursion_limit=60)
        runner = _runner_with(compiled, "bound-capped-stream", max_steps=5)
        seen = await _customs(runner, None, "bound-capped-stream")
        self.assertEqual(seen, [0, 1, 2, 3, 4, "GraphRecursionError"])

    async def test_without_an_explicit_max_steps_the_bound_limit_applies(self):
        compiled = _talking_loop().with_config(recursion_limit=150)
        runner = _runner_with(compiled, "bound-uncapped")
        seen = await _customs(runner, None, "bound-uncapped")
        self.assertEqual(len(seen), 151)

    async def test_a_limit_the_caller_passes_wins_over_max_steps(self):
        compiled = _talking_loop().with_config(recursion_limit=60)
        runner = _runner_with(compiled, "caller-limit", max_steps=5)
        seen = await _customs(runner, {"recursion_limit": 20}, "caller-limit")
        self.assertEqual(len(seen), 21)

    async def test_a_bound_limit_below_max_steps_is_the_graphs_own(self):
        compiled = _talking_loop().with_config(recursion_limit=3)
        runner = _runner_with(compiled, "bound-lower", max_steps=50)
        seen = await _customs(runner, None, "bound-lower")
        self.assertEqual(seen, [0, 1, 2, "GraphRecursionError"])
