#!/usr/bin/env python3

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Durable Sub-Agent Workflows with AsyncSubAgent + DaprWorkflowDeepAgentRunner

Demonstrates that each sub-agent runs as its own Dapr workflow and the
parent supervisor orchestration is also durable.

Architecture (three Dapr app instances):

    Supervisor  (DaprWorkflowDeepAgentRunner — Dapr Workflow)
    │
    ├── start_async_task("researcher") ──► HTTP ──► Researcher  (DaprWorkflowDeepAgentRunner)
    │     check_async_task(task_id)     ◄── HTTP ◄──              independent Dapr Workflow
    │
    └── start_async_task("analyst")    ──► HTTP ──► Analyst     (DaprWorkflowDeepAgentRunner)
          check_async_task(task_id)     ◄── HTTP ◄──              independent Dapr Workflow

Each agent is wrapped in DaprWorkflowDeepAgentRunner.  Every LLM call
and tool invocation becomes a durable Dapr workflow activity.  The
supervisor delegates to sub-agents via the deepagents AsyncSubAgent
middleware, which communicates over the Agent Protocol (HTTP).

A thin Agent Protocol adapter (FastAPI) bridges DaprWorkflowDeepAgentRunner
with the HTTP endpoints that the AsyncSubAgent middleware expects.

Usage:
    # Start each agent in a separate terminal with its own Dapr sidecar:

    # Terminal 1 — Researcher sub-agent (port 8001)
    dapr run --app-id researcher --app-port 8001 --resources-path ./components -- \\
        python3 subagent_workflows.py researcher

    # Terminal 2 — Analyst sub-agent (port 8002)
    dapr run --app-id analyst --app-port 8002 --resources-path ./components -- \\
        python3 subagent_workflows.py analyst

    # Terminal 3 — Supervisor (delegates to sub-agents via HTTP)
    dapr run --app-id supervisor --resources-path ./components -- \\
        python3 subagent_workflows.py supervisor

Prerequisites:
    - Dapr initialized (dapr init) with Redis running
    - OPENAI_API_KEY environment variable set
"""

import asyncio
import sys
import time
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request

from deepagents import AsyncSubAgent, create_deep_agent
from langchain.agents import create_agent
from langchain_core.tools import tool

from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner


def log(msg: str = ""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Agent Protocol adapter
# ---------------------------------------------------------------------------
# The AsyncSubAgentMiddleware tools use the LangGraph SDK client to
# communicate with sub-agents over the Agent Protocol (HTTP).  This
# adapter wraps a DaprWorkflowDeepAgentRunner with the minimal set of
# endpoints:
#   POST   /threads                        — create thread
#   GET    /threads/{id}                   — get thread (with values)
#   POST   /threads/{id}/runs             — create run (starts workflow)
#   GET    /threads/{id}/runs/{id}        — check run status
#   POST   /threads/{id}/runs/{id}/cancel — cancel run
# ---------------------------------------------------------------------------


class AgentProtocolAdapter:
    """Expose a DaprWorkflowDeepAgentRunner as an Agent Protocol HTTP server."""

    def __init__(self, runner: DaprWorkflowDeepAgentRunner) -> None:
        self.runner = runner
        self.threads: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}

    def create_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/threads")
        async def create_thread(request: Request):
            body: dict = {}
            try:
                body = await request.json()
            except Exception:
                pass
            thread_id = str(uuid4())
            self.threads[thread_id] = {
                "thread_id": thread_id,
                "values": {},
                "metadata": body.get("metadata", {}),
                "created_at": "",
                "updated_at": "",
            }
            return self.threads[thread_id]

        @app.get("/threads/{thread_id}")
        async def get_thread(thread_id: str):
            return self.threads.get(
                thread_id,
                {"thread_id": thread_id, "values": {}, "metadata": {}},
            )

        @app.post("/threads/{thread_id}/runs")
        async def create_run(thread_id: str, request: Request):
            body = await request.json()
            run_id = str(uuid4())
            run_record: dict = {
                "run_id": run_id,
                "thread_id": thread_id,
                "assistant_id": body.get("assistant_id", ""),
                "status": "pending",
            }
            self.runs[run_id] = run_record
            input_data = body.get("input", {})
            asyncio.create_task(
                self._execute(thread_id, run_id, input_data)
            )
            return run_record

        @app.get("/threads/{thread_id}/runs/{run_id}")
        async def get_run(thread_id: str, run_id: str):
            return self.runs.get(
                run_id,
                {"run_id": run_id, "thread_id": thread_id, "status": "error"},
            )

        @app.post("/threads/{thread_id}/runs/{run_id}/cancel")
        async def cancel_run(thread_id: str, run_id: str):
            if run_id in self.runs:
                self.runs[run_id]["status"] = "cancelled"
            return {}

        return app

    async def _execute(
        self, thread_id: str, run_id: str, input_data: dict
    ) -> None:
        """Run the Dapr workflow and track status."""
        self.runs[run_id]["status"] = "running"
        try:
            async for event in self.runner.run_async(
                input=input_data, thread_id=thread_id
            ):
                etype = event["type"]
                if etype == "workflow_completed":
                    output = event.get("output", {})
                    # Convert messages to plain dicts for JSON transport
                    messages = output.get("messages", [])
                    safe_messages = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            safe_messages.append(msg)
                        elif hasattr(msg, "model_dump"):
                            safe_messages.append(msg.model_dump())
                        else:
                            safe_messages.append({"content": str(msg)})
                    self.threads[thread_id]["values"] = {
                        "messages": safe_messages
                    }
                    self.runs[run_id]["status"] = "success"
                    log(f"  [{thread_id[:8]}] Workflow completed")
                    break
                elif etype == "workflow_failed":
                    self.runs[run_id]["status"] = "error"
                    self.runs[run_id]["error"] = event.get(
                        "error", "unknown"
                    )
                    log(f"  [{thread_id[:8]}] Workflow failed")
                    break
        except Exception as exc:
            self.runs[run_id]["status"] = "error"
            self.runs[run_id]["error"] = str(exc)
            log(f"  [{thread_id[:8]}] Error: {exc}")


# ---------------------------------------------------------------------------
# Sub-agent tools
# ---------------------------------------------------------------------------


@tool
def search_web(query: str) -> str:
    """Search the web for information on a given topic.
    Args: query — the search query."""
    log(f"  [Researcher] Searching: {query}")
    time.sleep(0.5)
    return (
        f"Found 3 sources on '{query}':\n"
        f"1. Recent advances in {query} (Nature, 2025)\n"
        f"2. A comprehensive survey of {query} (arXiv, 2025)\n"
        f"3. Industry applications of {query} (IEEE, 2025)"
    )


@tool
def analyze_data(topic: str, findings: str) -> str:
    """Analyze research findings and produce a structured report.
    Args: topic — the research topic, findings — the research summary."""
    log(f"  [Analyst] Analyzing: {topic}")
    time.sleep(0.5)
    return (
        f"Analysis of '{topic}':\n"
        f"- Key insight: {findings[:150]}\n"
        f"- Trend: Growing adoption in industry applications\n"
        f"- Recommendation: Focus on practical deployment patterns"
    )


# ---------------------------------------------------------------------------
# Agent graph factories
# ---------------------------------------------------------------------------


def make_researcher():
    """Create the researcher sub-agent graph.

    Uses ``create_agent`` (not ``create_deep_agent``) because sub-agents
    only need their own tools — no sandbox middleware required.
    """
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=[search_web],
        system_prompt=(
            "You are a research agent. When given a topic, use search_web "
            "to find information, then summarize your findings concisely."
        ),
        name="researcher",
    )


def make_analyst():
    """Create the analyst sub-agent graph.

    Uses ``create_agent`` (not ``create_deep_agent``) because sub-agents
    only need their own tools — no sandbox middleware required.
    """
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=[analyze_data],
        system_prompt=(
            "You are an analyst agent. When given research findings, use "
            "analyze_data to produce a structured analysis, then provide "
            "your conclusions."
        ),
        name="analyst",
    )


RESEARCHER_URL = "http://localhost:8001"
ANALYST_URL = "http://localhost:8002"


def make_supervisor():
    """Create the supervisor deep-agent with async sub-agent specs."""
    return create_deep_agent(
        model="openai:gpt-4o-mini",
        subagents=[
            AsyncSubAgent(
                name="researcher",
                description=(
                    "Research agent that searches the web for information "
                    "on a topic and returns a summary of findings."
                ),
                graph_id="researcher",
                url=RESEARCHER_URL,
            ),
            AsyncSubAgent(
                name="analyst",
                description=(
                    "Analyst agent that takes research findings and produces "
                    "a structured analysis report with insights."
                ),
                graph_id="analyst",
                url=ANALYST_URL,
            ),
        ],
        system_prompt=(
            "You are a supervisor that orchestrates research and analysis.\n\n"
            "When given a topic:\n"
            "1. Start the researcher sub-agent with the topic\n"
            "2. Check the researcher until it completes, then read its result\n"
            "3. Start the analyst sub-agent with the research findings\n"
            "4. Check the analyst until it completes, then read its result\n"
            "5. Provide a final synthesis combining both results\n\n"
            "Always wait for each sub-agent to complete before starting "
            "the next one."
        ),
        name="supervisor",
    )


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------

TOPIC = "Research and analyze: advances in durable AI agent orchestration"


async def run_subagent(name: str, port: int, graph) -> None:
    """Run a sub-agent with DaprWorkflowDeepAgentRunner + Agent Protocol."""
    runner = DaprWorkflowDeepAgentRunner(
        agent=graph, name=name, max_steps=50
    )
    runner.start()
    log(f"  {name} — Dapr workflow runtime started")
    await asyncio.sleep(1)

    adapter = AgentProtocolAdapter(runner)
    app = adapter.create_app()

    log(f"  {name} — Agent Protocol server listening on port {port}")
    config = uvicorn.Config(
        app, host="0.0.0.0", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except KeyboardInterrupt:
        log(f"  {name} — shutting down")
    finally:
        runner.shutdown()
        log(f"  {name} — stopped")


async def run_supervisor() -> None:
    """Run the supervisor with DaprWorkflowDeepAgentRunner."""
    graph = make_supervisor()
    runner = DaprWorkflowDeepAgentRunner(
        agent=graph, name="supervisor", max_steps=100
    )

    try:
        runner.start()
        log("  supervisor — Dapr workflow runtime started")
        await asyncio.sleep(2)

        log(f"\n{'=' * 64}")
        log(f"  SUPERVISOR — {TOPIC}")
        log(f"{'=' * 64}\n")

        async for event in runner.run_async(
            input={"messages": [{"role": "user", "content": TOPIC}]},
            thread_id="supervisor-demo",
        ):
            etype = event["type"]
            if etype == "workflow_started":
                log(f"  Workflow started: {event.get('workflow_id')}")
            elif etype == "workflow_completed":
                output = event.get("output", {})
                messages = output.get("messages", [])
                if messages:
                    last = messages[-1]
                    content = (
                        last.get("content", "")
                        if isinstance(last, dict)
                        else getattr(last, "content", str(last))
                    )
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "")
                            if isinstance(b, dict)
                            else str(b)
                            for b in content
                        )
                    log(f"\n{'=' * 64}")
                    log("  SUPERVISOR FINAL RESPONSE")
                    log(f"{'=' * 64}")
                    log(f"  {content[:800]}")
                log(f"{'=' * 64}")
                break
            elif etype == "workflow_failed":
                log(f"  Workflow FAILED: {event.get('error')}")
                break

    except KeyboardInterrupt:
        log("  Interrupted")
    finally:
        runner.shutdown()
        log("  Runtime stopped.")


def main() -> None:
    if len(sys.argv) < 2:
        log(
            "Usage: python3 subagent_workflows.py "
            "<researcher|analyst|supervisor>"
        )
        log()
        log("Start each agent in a separate terminal with its own Dapr "
            "sidecar:")
        log()
        log("  # Terminal 1")
        log("  dapr run --app-id researcher --app-port 8001 "
            "--resources-path ./components -- \\")
        log("      python3 subagent_workflows.py researcher")
        log()
        log("  # Terminal 2")
        log("  dapr run --app-id analyst --app-port 8002 "
            "--resources-path ./components -- \\")
        log("      python3 subagent_workflows.py analyst")
        log()
        log("  # Terminal 3")
        log("  dapr run --app-id supervisor "
            "--resources-path ./components -- \\")
        log("      python3 subagent_workflows.py supervisor")
        sys.exit(1)

    role = sys.argv[1].lower()

    if role == "researcher":
        log("\n  Starting RESEARCHER sub-agent...")
        asyncio.run(run_subagent("researcher", 8001, make_researcher()))

    elif role == "analyst":
        log("\n  Starting ANALYST sub-agent...")
        asyncio.run(run_subagent("analyst", 8002, make_analyst()))

    elif role == "supervisor":
        log("\n  Starting SUPERVISOR...")
        asyncio.run(run_supervisor())

    else:
        log(f"Unknown role: {role}")
        sys.exit(1)


if __name__ == "__main__":
    main()
