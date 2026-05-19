#!/usr/bin/env python3

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Example: Simple Claude Agent SDK Agent with Dapr Workflows

This example demonstrates how to run a Claude Agent SDK agent with durable
execution using Dapr Workflows. Each LLM call and each tool call runs as a
separate Dapr workflow activity, providing fault tolerance and durability.

Prerequisites:
    1. Dapr installed and initialized: dapr init
    2. Required packages: pip install diagrid claude-agent-sdk anthropic
    3. ANTHROPIC_API_KEY environment variable set

Run with Dapr:
    dapr run --app-id claude-agents-demo --resources-path ./components -- python3 simple_agent.py
"""

import asyncio
import os
from datetime import datetime

from claude_agent_sdk import ClaudeAgentOptions, tool

from diagrid.agent.claude_agents import DaprWorkflowAgentRunner


@tool("get_weather", "Get the current weather for a specified city", {"city": str})
async def get_weather(args):
    weather_data = {
        "Tokyo": "Sunny, 22 C",
        "London": "Cloudy, 15 C",
        "New York": "Partly cloudy, 18 C",
        "Paris": "Rainy, 12 C",
    }
    text = weather_data.get(
        args["city"], f"Weather data not available for {args['city']}"
    )
    return {"content": [{"type": "text", "text": text}]}


@tool("search_web", "Search the web for information on a given topic", {"query": str})
async def search_web(args):
    text = (
        f"Search results for '{args['query']}': "
        f"Found 10 relevant articles about {args['query']}."
    )
    return {"content": [{"type": "text", "text": text}]}


@tool("get_datetime", "Get the current date and time", {})
async def get_datetime(args):
    text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"content": [{"type": "text", "text": text}]}


async def main():
    options = ClaudeAgentOptions(
        system_prompt=(
            "You are an expert research assistant with access to various "
            "information sources. You excel at finding and synthesizing "
            "information to provide comprehensive answers to user queries. "
            "Use the available tools when needed to complete the task. "
            "When you have the final answer, provide it clearly without using tools."
        ),
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
    )

    runner = DaprWorkflowAgentRunner(
        name="simple-agent",
        options=options,
        tools=[get_weather, search_web, get_datetime],
        max_iterations=10,
    )

    try:
        print("Starting Dapr Workflow runtime...")
        runner.start()
        print("Runtime started successfully!")

        session_id = "demo-session-001"
        user_message = (
            "Find out the current weather in Tokyo and search for "
            "recent news about AI developments. Provide a brief summary."
        )
        print(f"\nExecuting agent with session: {session_id}")
        print(f"User message: {user_message}")
        print("=" * 60)

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
        print("\nShutting down Dapr Workflow runtime...")
        runner.shutdown()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
