# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""DurableAgent - A Strands Agent with automatic Dapr workflow durability.

Each tool call becomes a separate Dapr workflow activity, providing:
- Checkpointing after each tool call
- Ability to resume from the last successful tool
- Per-tool observability and retry
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Generator

from strands import Agent
from strands.types.tools import ToolUse

logger = logging.getLogger(__name__)


class DurableAgent:
    """A wrapper that adds Dapr workflow durability to any Strands Agent.

    Each tool call becomes a separate Dapr workflow activity, enabling:
    - Checkpointing after each tool execution
    - Resume from last successful tool on failure
    - Per-tool retry policies
    - Distributed tool execution

    Architecture:
        Workflow
          +-- Activity: call_model (get next action from LLM)
          +-- Activity: tool_calculate (if model requests)
          +-- Activity: tool_search (if model requests)
          +-- Activity: call_model (with tool results)
          +-- ... repeat until model says done

    Example:
        ```python
        from strands import Agent, tool
        from strands.models.openai import OpenAIModel
        from diagrid.agent.strands import DurableAgent

        @tool
        def search(query: str) -> str:
            return f"Results for {query}"

        agent = Agent(
            model=OpenAIModel(model_id="gpt-4o"),
            tools=[search],
        )

        durable = DurableAgent(agent)
        result = durable("Search for AI papers")
        ```
    """

    def __init__(
        self,
        agent: Agent,
        *,
        name: str,
    ):
        """Create a durable wrapper around a Strands Agent.

        Args:
            agent: The Strands Agent instance to make durable
            name: Required name (produces ``dapr.strands.<name>.workflow``)
        """
        self._agent = agent
        self._name = name
        self._workflow_name = f"dapr.strands.{name}.workflow"

        # Dapr components (initialized on first call)
        self._workflow_runtime: Any = None
        self._workflow_client: Any = None
        self._workflow_func: Any = None  # Reference to the registered workflow function
        self._initialized = False

    def _initialize_dapr(self) -> None:
        """Initialize Dapr workflow runtime and register activities."""
        if self._initialized:
            return

        from dapr.ext.workflow import WorkflowRuntime, DaprWorkflowClient

        agent = self._agent

        self._workflow_runtime = WorkflowRuntime()

        # ============================================================
        # Activity: Call the model (LLM)
        # ============================================================
        def call_model_activity(ctx: Any, input_data: dict) -> dict:  # type: ignore[type-arg]
            """Activity that calls the LLM model."""
            from strands.event_loop.streaming import process_stream

            messages = input_data.get("messages", [])
            system_prompt = input_data.get("system_prompt", "")

            print(
                f"[ACTIVITY:call_model] Calling model with {len(messages)} messages",
                flush=True,
            )

            try:
                # Get tool specs
                tool_specs = [
                    tool.tool_spec for tool in agent.tool_registry.registry.values()
                ]

                # Call the model and process stream using Strands' built-in parser
                async def _call() -> tuple[list[Any], str]:
                    stop_reason = "end_turn"
                    message_content: list[Any] = []

                    # Get raw stream from model
                    raw_stream = agent.model.stream(
                        messages=messages,
                        tool_specs=tool_specs if tool_specs else None,
                        system_prompt=system_prompt,
                    )

                    # Process using Strands' stream processor
                    async for event in process_stream(raw_stream):
                        if isinstance(event, dict) and "stop" in event:
                            stop_data = event["stop"]
                            if isinstance(stop_data, tuple) and len(stop_data) >= 2:
                                stop_reason = stop_data[0]
                                message = stop_data[1]
                                if isinstance(message, dict) and "content" in message:
                                    message_content = message["content"]
                                    print(
                                        f"[ACTIVITY:call_model] Got message with "
                                        f"{len(message_content)} content blocks",
                                        flush=True,
                                    )

                    return message_content, stop_reason

                content, stop_reason = asyncio.run(_call())

                # Parse tool uses from the content
                tool_uses: list[Any] = []
                text_content: list[str] = []

                print(
                    f"[ACTIVITY:call_model] Parsing {len(content)} content blocks",
                    flush=True,
                )
                for i, block in enumerate(content):
                    print(
                        f"[ACTIVITY:call_model] Block {i}: "
                        f"{list(block.keys()) if isinstance(block, dict) else type(block)}",
                        flush=True,
                    )
                    if isinstance(block, dict):
                        if "toolUse" in block:
                            tool_uses.append(block["toolUse"])
                        elif "text" in block:
                            text_content.append(block["text"])

                print(
                    f"[ACTIVITY:call_model] Model returned: "
                    f"{len(tool_uses)} tool calls, stop_reason={stop_reason}",
                    flush=True,
                )

                return {
                    "tool_uses": tool_uses,
                    "text": "".join(text_content),
                    "stop_reason": stop_reason,
                    "content": content,
                }

            except Exception as e:
                print(f"[ACTIVITY:call_model] Error: {e}", flush=True)
                import traceback

                traceback.print_exc()
                return {
                    "error": str(e),
                    "tool_uses": [],
                    "text": "",
                    "stop_reason": "error",
                }

        # ============================================================
        # Activity: Execute a tool
        # ============================================================
        def execute_tool_activity(ctx: Any, input_data: dict) -> dict:  # type: ignore[type-arg]
            """Activity that executes a single tool."""
            tool_name = input_data.get("tool_name")
            tool_use_id = input_data.get("tool_use_id")
            tool_input = input_data.get("tool_input", {})

            print(
                f"[ACTIVITY:tool:{tool_name}] Executing with input: {tool_input}",
                flush=True,
            )

            try:
                tool = agent.tool_registry.registry.get(tool_name or "")
                if not tool:
                    return {
                        "tool_use_id": tool_use_id,
                        "status": "error",
                        "content": [{"text": f"Unknown tool: {tool_name}"}],
                    }

                tool_use: ToolUse = {
                    "name": tool_name or "",
                    "toolUseId": tool_use_id or "",
                    "input": tool_input,
                }

                # Execute the tool
                async def _execute() -> Any:
                    result = None
                    async for event in tool.stream(tool_use, {}):
                        result = event
                    return result

                result = asyncio.run(_execute())

                # Handle different result formats
                if isinstance(result, dict) and "toolResult" in result:
                    tool_result = result["toolResult"]
                elif isinstance(result, dict) and "status" in result:
                    tool_result = result
                else:
                    tool_result = {
                        "toolUseId": tool_use_id,
                        "status": "success",
                        "content": [{"text": str(result)}],
                    }

                print(
                    f"[ACTIVITY:tool:{tool_name}] Completed: {tool_result.get('status')}",
                    flush=True,
                )
                return tool_result

            except Exception as e:
                print(f"[ACTIVITY:tool:{tool_name}] Error: {e}", flush=True)
                import traceback

                traceback.print_exc()
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": f"Error: {str(e)}"}],
                }

        # ============================================================
        # Workflow: Orchestrate model + tool calls
        # ============================================================
        def agent_workflow(ctx: Any, input_data: dict) -> Generator[Any, Any, dict]:  # type: ignore[type-arg]
            """Workflow that orchestrates the agent loop.

            Each iteration:
            1. Call model activity
            2. If model wants tools: call tool activities (can be parallel)
            3. Repeat until model says done
            """
            task = input_data.get("task", "")
            system_prompt = agent.system_prompt or ""

            print(
                f"[WORKFLOW] Starting with task: {task[:100]}...",
                flush=True,
            )

            # Build initial messages
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": [{"text": task}]}
            ]
            final_text = ""
            all_tool_calls: list[dict[str, Any]] = []
            max_iterations = 10

            for iteration in range(max_iterations):
                print(f"[WORKFLOW] Iteration {iteration + 1}", flush=True)

                # Step 1: Call the model (as an activity)
                model_result = yield ctx.call_activity(
                    call_model_activity,
                    input={
                        "messages": messages,
                        "system_prompt": system_prompt,
                    },
                )

                if model_result.get("error"):
                    final_text = f"Error: {model_result['error']}"
                    break

                tool_uses = model_result.get("tool_uses", [])
                text = model_result.get("text", "")
                stop_reason = model_result.get("stop_reason")
                content = model_result.get("content", [])

                # Add assistant message to history
                messages.append({"role": "assistant", "content": content})

                # If no tool calls, we're done
                if not tool_uses or stop_reason == "end_turn":
                    final_text = text
                    print(
                        f"[WORKFLOW] Model finished: {stop_reason}",
                        flush=True,
                    )
                    break

                # Step 2: Execute each tool (as separate activities)
                print(
                    f"[WORKFLOW] Executing {len(tool_uses)} tools",
                    flush=True,
                )

                tool_results = []
                for tool_use in tool_uses:
                    tool_name = tool_use.get("name")
                    tool_use_id = tool_use.get("toolUseId")
                    tool_input = tool_use.get("input", {})

                    all_tool_calls.append(
                        {
                            "tool_name": tool_name,
                            "tool_use_id": tool_use_id,
                            "input": tool_input,
                        }
                    )

                    # Each tool call is a separate activity!
                    result = yield ctx.call_activity(
                        execute_tool_activity,
                        input={
                            "tool_name": tool_name,
                            "tool_use_id": tool_use_id,
                            "tool_input": tool_input,
                        },
                    )
                    tool_results.append({"toolResult": result})

                # Add tool results to messages
                messages.append({"role": "user", "content": tool_results})

            print(
                f"[WORKFLOW] Completed with {len(all_tool_calls)} total tool calls",
                flush=True,
            )

            return {  # type: ignore[return-value]
                "result": final_text,
                "tool_calls": all_tool_calls,
            }

        # Register everything
        self._workflow_runtime.register_workflow(
            agent_workflow, name=self._workflow_name
        )
        self._workflow_runtime.register_activity(
            call_model_activity, name=f"dapr.strands.{self._name}.call_model"
        )
        self._workflow_runtime.register_activity(
            execute_tool_activity, name=f"dapr.strands.{self._name}.execute_tool"
        )

        # Store reference to workflow function for scheduling
        self._workflow_func = agent_workflow

        # Start runtime
        self._workflow_runtime.start()
        print("[DurableAgent] Initialized with per-tool activities", flush=True)

        # Create workflow client (new API, not deprecated)
        self._workflow_client = DaprWorkflowClient()
        time.sleep(1)

        self._initialized = True

    def __call__(self, task: str, timeout_seconds: int = 120) -> str:
        """Run the agent with the given task."""
        self._initialize_dapr()

        instance_id = f"{self._workflow_name}_{uuid.uuid4().hex[:8]}"

        # Use the new DaprWorkflowClient API with function reference
        self._workflow_client.schedule_new_workflow(
            workflow=self._workflow_func,
            input={"task": task},
            instance_id=instance_id,
        )

        # Wait for completion
        start_time = time.time()
        while True:
            state = self._workflow_client.get_workflow_state(
                instance_id=instance_id,
            )

            status = str(state.runtime_status).upper()
            if "COMPLETED" in status or "FAILED" in status or "TERMINATED" in status:
                break

            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Agent timed out after {timeout_seconds}s")

            time.sleep(0.5)

        if "FAILED" in status:
            raise RuntimeError("Agent workflow failed")

        # Get result
        output = None
        for attr in ["serialized_output", "workflow_output", "output", "result"]:
            if hasattr(state, attr):
                output = getattr(state, attr)
                break

        if output and isinstance(output, str):
            output = json.loads(output)

        if isinstance(output, dict):
            return output.get("result", str(output))
        return str(output) if output else ""

    def shutdown(self) -> None:
        """Shutdown the Dapr workflow runtime."""
        if self._workflow_runtime:
            self._workflow_runtime.shutdown()
            print("[DurableAgent] Shutdown complete", flush=True)

    def __enter__(self) -> "DurableAgent":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    @property
    def agent(self) -> Agent:
        """Access the underlying Strands Agent."""
        return self._agent

    @property
    def messages(self) -> Any:
        return self._agent.messages

    @property
    def tools(self) -> Any:
        return self._agent.tool_registry

    @property
    def model(self) -> Any:
        return self._agent.model
