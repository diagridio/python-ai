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

import unittest

from diagrid.agent.adk.workflow import (
    register_tool,
    get_registered_tool,
    clear_tool_registry,
)


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str):
        self.name = name

    async def run_async(self, args, tool_context):
        return f"Mock result for {self.name}"


class TestToolRegistry(unittest.TestCase):
    """Tests for tool registry functions."""

    def setUp(self):
        """Clear registry before each test."""
        clear_tool_registry()

    def tearDown(self):
        """Clear registry after each test."""
        clear_tool_registry()

    def test_register_and_get_tool(self):
        """Test registering and retrieving a tool."""
        tool = MockTool("test_tool")
        register_tool("test_tool", tool)

        retrieved = get_registered_tool("test_tool")
        self.assertEqual(retrieved, tool)
        self.assertEqual(retrieved.name, "test_tool")

    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        result = get_registered_tool("nonexistent")
        self.assertIsNone(result)

    def test_clear_registry(self):
        """Test clearing the registry."""
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")
        register_tool("tool1", tool1)
        register_tool("tool2", tool2)

        self.assertIsNotNone(get_registered_tool("tool1"))
        self.assertIsNotNone(get_registered_tool("tool2"))

        clear_tool_registry()

        self.assertIsNone(get_registered_tool("tool1"))
        self.assertIsNone(get_registered_tool("tool2"))

    def test_overwrite_tool(self):
        """Test that registering a tool with same name overwrites."""
        tool1 = MockTool("original")
        tool2 = MockTool("replacement")

        register_tool("my_tool", tool1)
        self.assertEqual(get_registered_tool("my_tool").name, "original")

        register_tool("my_tool", tool2)
        self.assertEqual(get_registered_tool("my_tool").name, "replacement")


if __name__ == "__main__":
    unittest.main()
