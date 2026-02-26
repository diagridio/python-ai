#!/usr/bin/env python3

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

"""
Example: Simple OpenAI Agents SDK Agent with Dapr Workflows

This example demonstrates how to run an OpenAI Agents SDK agent with durable tool execution
using Dapr Workflows. Each tool call is executed as a workflow activity,
providing fault tolerance and durability.

Prerequisites:
    1. Dapr installed and initialized: dapr init
    2. Required packages: pip install diagrid openai-agents openai

Run with Dapr:
    dapr run --app-id openai-agents-demo --resources-path ./components -- python3 simple_agent.py
"""

import asyncio
import os
from datetime import datetime

from agents import Agent, function_tool
from diagrid.agent.openai_agents import DaprWorkflowAgentRunner


@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a specified city."""
    weather_data = {
        "Tokyo": "Sunny, 22°C",
        "London": "Cloudy, 15°C",
        "New York": "Partly cloudy, 18°C",
        "Paris": "Rainy, 12°C",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@function_tool
def search_web(query: str) -> str:
    """Search the web for information on a given topic."""
    return f"Search results for '{query}': Found 10 relevant articles about {query}."


@function_tool
def get_datetime() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main():
    # Create an OpenAI Agents SDK agent with tools
    agent = Agent(
        name="research_assistant",
        instructions="""You are an expert research assistant with access to various
        information sources. You excel at finding and synthesizing information
        to provide comprehensive answers to user queries.
        Use the available tools when needed to complete the task.
        When you have the final answer, provide it clearly without using tools.""",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        tools=[get_weather, search_web, get_datetime],
    )

    # Create the Dapr Workflow runner
    runner = DaprWorkflowAgentRunner(
        agent=agent,
        max_iterations=10,
    )

    try:
        # Start the workflow runtime
        print("Starting Dapr Workflow runtime...")
        runner.start()
        print("Runtime started successfully!")

        # Run the agent with a user message
        session_id = "demo-session-001"
        user_message = (
            "Find out the current weather in Tokyo and search for "
            "recent news about AI developments. Provide a brief summary."
        )
        print(f"\nExecuting agent with session: {session_id}")
        print(f"User message: {user_message}")
        print("=" * 60)

        # Process events as they come
        async for event in runner.run_async(
            user_message=user_message,
            session_id=session_id,
        ):
            event_type = event["type"]

            if event_type == "workflow_started":
                print(f"\nWorkflow started: {event.get('workflow_id')}")

            elif event_type == "workflow_status_changed":
                print(f"Status: {event.get('status')}")

            elif event_type == "workflow_completed":
                print("\n" + "=" * 60)
                print("AGENT COMPLETED")
                print("=" * 60)
                print(f"Iterations: {event.get('iterations')}")
                print(f"Status: {event.get('status')}")
                print("\nFinal Response:")
                print("-" * 40)
                print(event.get("final_response", "No response"))

            elif event_type == "workflow_failed":
                print(f"\nWorkflow FAILED: {event.get('error')}")

            elif event_type == "workflow_error":
                print(f"\nWorkflow ERROR: {event.get('error')}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Shutdown the runtime
        print("\nShutting down Dapr Workflow runtime...")
        runner.shutdown()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
