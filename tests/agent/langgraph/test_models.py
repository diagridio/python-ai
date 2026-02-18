# Copyright 2025 Diagrid Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for data models."""

import json
import unittest

from diagrid.agent.langgraph.models import (
    ChannelState,
    EdgeConfig,
    ExecuteNodeInput,
    ExecuteNodeOutput,
    EvaluateConditionInput,
    EvaluateConditionOutput,
    GraphConfig,
    GraphWorkflowInput,
    GraphWorkflowOutput,
    NodeConfig,
    NodeWrite,
    WorkflowStatus,
)


class TestNodeConfig(unittest.TestCase):
    """Tests for NodeConfig."""

    def test_to_dict(self):
        config = NodeConfig(
            name="test_node",
            triggers=["channel_a"],
            channels_read=["channel_a", "channel_b"],
            channels_write=["channel_c"],
        )
        result = config.to_dict()

        self.assertEqual(result["name"], "test_node")
        self.assertEqual(result["triggers"], ["channel_a"])
        self.assertEqual(result["channels_read"], ["channel_a", "channel_b"])
        self.assertEqual(result["channels_write"], ["channel_c"])

    def test_from_dict(self):
        data = {
            "name": "test_node",
            "triggers": ["channel_a"],
            "channels_read": ["channel_a"],
            "channels_write": [],
        }
        config = NodeConfig.from_dict(data)

        self.assertEqual(config.name, "test_node")
        self.assertEqual(config.triggers, ["channel_a"])

    def test_roundtrip(self):
        original = NodeConfig(
            name="test",
            triggers=["t1"],
            channels_read=["r1"],
            channels_write=["w1"],
        )
        roundtrip = NodeConfig.from_dict(original.to_dict())

        self.assertEqual(original.name, roundtrip.name)
        self.assertEqual(original.triggers, roundtrip.triggers)


class TestChannelState(unittest.TestCase):
    """Tests for ChannelState."""

    def test_to_dict(self):
        state = ChannelState(
            values={"messages": ["hello"], "count": 1},
            versions={"messages": 1, "count": 1},
            updated_channels=["messages"],
        )
        result = state.to_dict()

        self.assertEqual(result["values"]["messages"], ["hello"])
        self.assertEqual(result["versions"]["count"], 1)
        self.assertEqual(result["updated_channels"], ["messages"])

    def test_from_dict(self):
        data = {
            "values": {"key": "value"},
            "versions": {"key": 2},
            "updated_channels": ["key"],
        }
        state = ChannelState.from_dict(data)

        self.assertEqual(state.values["key"], "value")
        self.assertEqual(state.versions["key"], 2)

    def test_json_serializable(self):
        state = ChannelState(
            values={"messages": ["hello", "world"], "count": 42},
            versions={"messages": 3, "count": 1},
            updated_channels=["messages", "count"],
        )
        # Should not raise
        json_str = json.dumps(state.to_dict())
        parsed = json.loads(json_str)
        self.assertEqual(parsed["values"]["count"], 42)


class TestGraphConfig(unittest.TestCase):
    """Tests for GraphConfig."""

    def test_to_dict(self):
        config = GraphConfig(
            name="test_graph",
            nodes=[NodeConfig(name="a"), NodeConfig(name="b")],
            edges=[EdgeConfig(source="a", target="b")],
            entry_point="a",
            finish_points=["b"],
            input_channels=["messages"],
            output_channels=["messages"],
        )
        result = config.to_dict()

        self.assertEqual(result["name"], "test_graph")
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["entry_point"], "a")

    def test_from_dict(self):
        data = {
            "name": "test",
            "nodes": [{"name": "a"}, {"name": "b"}],
            "edges": [{"source": "a", "target": "b"}],
            "entry_point": "a",
            "finish_points": ["b"],
            "input_channels": ["in"],
            "output_channels": ["out"],
        }
        config = GraphConfig.from_dict(data)

        self.assertEqual(config.name, "test")
        self.assertEqual(len(config.nodes), 2)
        self.assertEqual(config.nodes[0].name, "a")


class TestExecuteNodeInput(unittest.TestCase):
    """Tests for ExecuteNodeInput."""

    def test_roundtrip(self):
        original = ExecuteNodeInput(
            node_name="test_node",
            channel_state=ChannelState(
                values={"messages": ["hello"]},
                versions={"messages": 1},
            ),
            config={"thread_id": "123"},
        )

        data = original.to_dict()
        restored = ExecuteNodeInput.from_dict(data)

        self.assertEqual(original.node_name, restored.node_name)
        self.assertEqual(
            original.channel_state.values,
            restored.channel_state.values,
        )


class TestExecuteNodeOutput(unittest.TestCase):
    """Tests for ExecuteNodeOutput."""

    def test_with_writes(self):
        output = ExecuteNodeOutput(
            node_name="test",
            writes=[
                NodeWrite(channel="messages", value=["updated"]),
                NodeWrite(channel="count", value=2),
            ],
        )
        data = output.to_dict()

        self.assertEqual(data["node_name"], "test")
        self.assertEqual(len(data["writes"]), 2)
        self.assertIsNone(data["error"])

    def test_with_error(self):
        output = ExecuteNodeOutput(
            node_name="test",
            error="Something went wrong",
        )
        data = output.to_dict()

        self.assertEqual(data["error"], "Something went wrong")
        self.assertEqual(len(data["writes"]), 0)


class TestGraphWorkflowInput(unittest.TestCase):
    """Tests for GraphWorkflowInput."""

    def test_full_roundtrip(self):
        original = GraphWorkflowInput(
            graph_config=GraphConfig(
                name="test",
                nodes=[NodeConfig(name="a")],
                edges=[],
                entry_point="a",
            ),
            channel_state=ChannelState(values={"x": 1}),
            step=5,
            max_steps=100,
            pending_nodes=["a", "b"],
            config={"key": "value"},
            thread_id="thread-123",
        )

        data = original.to_dict()
        # Verify JSON serializable
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = GraphWorkflowInput.from_dict(parsed)

        self.assertEqual(original.step, restored.step)
        self.assertEqual(original.max_steps, restored.max_steps)
        self.assertEqual(original.pending_nodes, restored.pending_nodes)
        self.assertEqual(original.thread_id, restored.thread_id)


class TestGraphWorkflowOutput(unittest.TestCase):
    """Tests for GraphWorkflowOutput."""

    def test_completed_output(self):
        output = GraphWorkflowOutput(
            output={"messages": ["result"], "count": 5},
            channel_state=ChannelState(values={"messages": ["result"]}),
            steps=10,
            status="completed",
        )
        data = output.to_dict()

        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["steps"], 10)
        self.assertEqual(data["output"]["count"], 5)
        self.assertIsNone(data["error"])

    def test_error_output(self):
        output = GraphWorkflowOutput(
            output={},
            steps=3,
            status="error",
            error="Node failed",
        )
        data = output.to_dict()

        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"], "Node failed")


class TestWorkflowStatus(unittest.TestCase):
    """Tests for WorkflowStatus enum."""

    def test_values(self):
        self.assertEqual(WorkflowStatus.PENDING.value, "pending")
        self.assertEqual(WorkflowStatus.RUNNING.value, "running")
        self.assertEqual(WorkflowStatus.COMPLETED.value, "completed")
        self.assertEqual(WorkflowStatus.FAILED.value, "failed")
        self.assertEqual(WorkflowStatus.TERMINATED.value, "terminated")


if __name__ == "__main__":
    unittest.main()
