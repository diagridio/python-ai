# -*- coding: utf-8 -*-

"""
Copyright 2025 Diagrid Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Tests for workflow functions."""

import unittest
from unittest import mock

from diagrid.agent.langgraph.models import (
    ChannelState,
    ExecuteNodeInput,
    ExecuteNodeOutput,
    EvaluateConditionInput,
    EvaluateConditionOutput,
    NodeWrite,
)
from diagrid.agent.langgraph.workflow import (
    register_node,
    register_condition,
    register_channel_reducer,
    get_registered_node,
    get_registered_condition,
    get_channel_reducer,
    clear_registries,
    _reconstruct_state,
    _result_to_writes,
    _serialize_value,
    _apply_write,
    execute_node_activity,
    evaluate_condition_activity,
)


class TestRegistries(unittest.TestCase):
    """Tests for registry functions."""

    def setUp(self):
        clear_registries()

    def tearDown(self):
        clear_registries()

    def test_register_node(self):
        def my_node(state):
            return {"value": 1}

        register_node("my_node", my_node)
        retrieved = get_registered_node("my_node")

        self.assertEqual(retrieved, my_node)

    def test_get_unregistered_node(self):
        result = get_registered_node("nonexistent")
        self.assertIsNone(result)

    def test_register_condition(self):
        def my_condition(state):
            return "next_node"

        register_condition("my_cond", my_condition)
        retrieved = get_registered_condition("my_cond")

        self.assertEqual(retrieved, my_condition)

    def test_register_channel_reducer(self):
        def my_reducer(a, b):
            return a + b

        register_channel_reducer("messages", my_reducer)
        retrieved = get_channel_reducer("messages")

        self.assertEqual(retrieved, my_reducer)

    def test_clear_registries(self):
        register_node("test", lambda x: x)
        register_condition("test", lambda x: "a")
        register_channel_reducer("test", lambda a, b: a)

        clear_registries()

        self.assertIsNone(get_registered_node("test"))
        self.assertIsNone(get_registered_condition("test"))
        self.assertIsNone(get_channel_reducer("test"))


class TestReconstructState(unittest.TestCase):
    """Tests for _reconstruct_state."""

    def test_simple_values(self):
        channel_state = ChannelState(
            values={"messages": ["hello"], "count": 5},
            versions={"messages": 1, "count": 1},
        )
        state = _reconstruct_state(channel_state)

        self.assertEqual(state["messages"], ["hello"])
        self.assertEqual(state["count"], 5)

    def test_empty_state(self):
        channel_state = ChannelState()
        state = _reconstruct_state(channel_state)

        self.assertEqual(state, {})


class TestResultToWrites(unittest.TestCase):
    """Tests for _result_to_writes."""

    def test_dict_result(self):
        result = {"messages": ["updated"], "count": 10}
        writes = _result_to_writes(result)

        self.assertEqual(len(writes), 2)
        channel_names = [w.channel for w in writes]
        self.assertIn("messages", channel_names)
        self.assertIn("count", channel_names)

    def test_none_result(self):
        writes = _result_to_writes(None)
        self.assertEqual(len(writes), 0)

    def test_non_dict_result(self):
        result = "just a string"
        writes = _result_to_writes(result)

        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].channel, "__result__")


class TestSerializeValue(unittest.TestCase):
    """Tests for _serialize_value."""

    def test_primitives(self):
        self.assertIsNone(_serialize_value(None))
        self.assertEqual(_serialize_value("hello"), "hello")
        self.assertEqual(_serialize_value(42), 42)
        self.assertEqual(_serialize_value(3.14), 3.14)
        self.assertEqual(_serialize_value(True), True)

    def test_list(self):
        result = _serialize_value([1, "two", 3.0])
        self.assertEqual(result, [1, "two", 3.0])

    def test_dict(self):
        result = _serialize_value({"a": 1, "b": "two"})
        self.assertEqual(result, {"a": 1, "b": "two"})

    def test_nested(self):
        result = _serialize_value({
            "list": [1, 2, {"nested": True}],
            "dict": {"inner": [1, 2]},
        })
        self.assertEqual(result["list"][2]["nested"], True)


class TestApplyWrite(unittest.TestCase):
    """Tests for _apply_write."""

    def setUp(self):
        clear_registries()

    def tearDown(self):
        clear_registries()

    def test_simple_write(self):
        channel_state = ChannelState(
            values={},
            versions={},
        )
        _apply_write(channel_state, "messages", ["hello"])

        self.assertEqual(channel_state.values["messages"], ["hello"])
        self.assertEqual(channel_state.versions["messages"], 1)

    def test_overwrite(self):
        channel_state = ChannelState(
            values={"count": 5},
            versions={"count": 1},
        )
        _apply_write(channel_state, "count", 10)

        self.assertEqual(channel_state.values["count"], 10)
        self.assertEqual(channel_state.versions["count"], 2)

    def test_with_reducer(self):
        def list_reducer(a, b):
            return a + b

        register_channel_reducer("messages", list_reducer)

        channel_state = ChannelState(
            values={"messages": ["hello"]},
            versions={"messages": 1},
        )
        _apply_write(channel_state, "messages", [" world"])

        self.assertEqual(channel_state.values["messages"], ["hello", " world"])


class TestExecuteNodeActivity(unittest.TestCase):
    """Tests for execute_node_activity."""

    def setUp(self):
        clear_registries()

    def tearDown(self):
        clear_registries()

    def test_node_not_found(self):
        ctx = mock.Mock()
        input_data = ExecuteNodeInput(
            node_name="nonexistent",
            channel_state=ChannelState(),
        ).to_dict()

        result = execute_node_activity(ctx, input_data)
        output = ExecuteNodeOutput.from_dict(result)

        self.assertEqual(output.node_name, "nonexistent")
        self.assertIsNotNone(output.error)
        self.assertIn("not found", output.error)

    def test_successful_execution(self):
        def my_node(state):
            return {"result": state.get("input", 0) + 1}

        register_node("my_node", my_node)

        ctx = mock.Mock()
        input_data = ExecuteNodeInput(
            node_name="my_node",
            channel_state=ChannelState(values={"input": 5}),
        ).to_dict()

        result = execute_node_activity(ctx, input_data)
        output = ExecuteNodeOutput.from_dict(result)

        self.assertEqual(output.node_name, "my_node")
        self.assertIsNone(output.error)
        self.assertEqual(len(output.writes), 1)
        self.assertEqual(output.writes[0].channel, "result")
        self.assertEqual(output.writes[0].value, 6)

    def test_node_error(self):
        def failing_node(state):
            raise ValueError("Test error")

        register_node("failing", failing_node)

        ctx = mock.Mock()
        input_data = ExecuteNodeInput(
            node_name="failing",
            channel_state=ChannelState(),
        ).to_dict()

        result = execute_node_activity(ctx, input_data)
        output = ExecuteNodeOutput.from_dict(result)

        self.assertEqual(output.node_name, "failing")
        self.assertIsNotNone(output.error)
        self.assertIn("Test error", output.error)


class TestEvaluateConditionActivity(unittest.TestCase):
    """Tests for evaluate_condition_activity."""

    def setUp(self):
        clear_registries()

    def tearDown(self):
        clear_registries()

    def test_condition_not_found(self):
        ctx = mock.Mock()
        input_data = EvaluateConditionInput(
            source_node="node_a",
            condition_name="nonexistent",
            channel_state=ChannelState(),
        ).to_dict()

        result = evaluate_condition_activity(ctx, input_data)
        output = EvaluateConditionOutput.from_dict(result)

        self.assertIsNotNone(output.error)
        self.assertIn("not found", output.error)

    def test_condition_returns_string(self):
        def my_condition(state):
            return "next_node"

        register_condition("my_cond", my_condition)

        ctx = mock.Mock()
        input_data = EvaluateConditionInput(
            source_node="node_a",
            condition_name="my_cond",
            channel_state=ChannelState(),
        ).to_dict()

        result = evaluate_condition_activity(ctx, input_data)
        output = EvaluateConditionOutput.from_dict(result)

        self.assertIsNone(output.error)
        self.assertEqual(output.next_nodes, ["next_node"])

    def test_condition_returns_list(self):
        def my_condition(state):
            return ["node_b", "node_c"]

        register_condition("multi_cond", my_condition)

        ctx = mock.Mock()
        input_data = EvaluateConditionInput(
            source_node="node_a",
            condition_name="multi_cond",
            channel_state=ChannelState(),
        ).to_dict()

        result = evaluate_condition_activity(ctx, input_data)
        output = EvaluateConditionOutput.from_dict(result)

        self.assertIsNone(output.error)
        self.assertEqual(output.next_nodes, ["node_b", "node_c"])

    def test_condition_based_on_state(self):
        def routing_condition(state):
            if state.get("count", 0) > 5:
                return "high_path"
            return "low_path"

        register_condition("router", routing_condition)

        ctx = mock.Mock()

        # Test low path
        input_data = EvaluateConditionInput(
            source_node="node_a",
            condition_name="router",
            channel_state=ChannelState(values={"count": 3}),
        ).to_dict()

        result = evaluate_condition_activity(ctx, input_data)
        output = EvaluateConditionOutput.from_dict(result)
        self.assertEqual(output.next_nodes, ["low_path"])

        # Test high path
        input_data = EvaluateConditionInput(
            source_node="node_a",
            condition_name="router",
            channel_state=ChannelState(values={"count": 10}),
        ).to_dict()

        result = evaluate_condition_activity(ctx, input_data)
        output = EvaluateConditionOutput.from_dict(result)
        self.assertEqual(output.next_nodes, ["high_path"])


if __name__ == "__main__":
    unittest.main()
