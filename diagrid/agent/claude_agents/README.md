# Diagrid Agent Claude Agent SDK

Durable execution of [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
agents using Dapr Workflows.

## What it does

Wraps each agent invocation in a Dapr workflow, with every LLM turn and every
tool call running as its own durable activity:

- **Workflow** = one agent invocation (one user message + the agent loop).
- **Activity `claude_call_llm`** = one Anthropic `messages.create` call (one assistant turn).
- **Activity `claude_execute_tool`** = one tool invocation. When the model
  emits multiple `tool_use` blocks in a turn, they are fanned out in parallel.

A crash in the middle of a tool resumes from the last completed activity on
restart — the conversation state, prior LLM responses, and completed tool
results are all checkpointed.

## Installation

```bash
pip install diagrid claude-agent-sdk anthropic
```

## Quick start

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions, tool
from diagrid.agent.claude_agents import DaprWorkflowAgentRunner


@tool("get_weather", "Get the weather for a city", {"city": str})
async def get_weather(args):
    return {"content": [{"type": "text", "text": f"Sunny in {args['city']}"}]}


async def main():
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful weather assistant.",
        model="claude-sonnet-4-6",
    )

    runner = DaprWorkflowAgentRunner(
        name="weather-agent",
        options=options,
        tools=[get_weather],
    )
    runner.start()
    try:
        async for event in runner.run_async(
            user_message="What's the weather in Tokyo?",
            session_id="demo-1",
        ):
            print(event)
    finally:
        runner.shutdown()


asyncio.run(main())
```

Run with the Dapr sidecar:

```bash
dapr run --app-id claude-agents-demo --resources-path ./resources -- python3 simple_agent.py
```

## Configuration

`DaprWorkflowAgentRunner` accepts:

| arg              | purpose                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| `name`           | Required. Used to derive the workflow name.                             |
| `system_prompt`  | Claude system prompt.                                                   |
| `model`          | Claude model ID (e.g. `claude-sonnet-4-6`).                             |
| `tools`          | List of tools (Claude Agent SDK `@tool` instances or plain callables).  |
| `max_tokens`     | Max output tokens per LLM call. Default: 4096.                          |
| `max_iterations` | Maximum LLM iterations per workflow. Default: 25.                       |
| `options`        | Optional `ClaudeAgentOptions` — `system_prompt` / `model` are read from it when the explicit args are absent. |
| `state_store`    | Optional `DaprStateStore` for memory persistence.                       |

## Tools

Any of the following can be passed to `tools=[...]`:

- A function decorated with `@tool(...)` from `claude_agent_sdk`. Its
  `input_schema` is converted to JSON Schema for the Anthropic API call.
- A plain Python `def` / `async def` function — its parameters become the
  tool's input schema only if you also pass an `input_schema` attribute on it;
  otherwise the schema is the empty object.
- Any object with `name`, `description`, `input_schema`, and a `handler`
  coroutine.

Tools execute in the same process as the workflow runtime — they have access
to your application's state.

## Examples

See [`examples/`](./examples/) for a runnable demo with three tools.
