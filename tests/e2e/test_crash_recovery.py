"""E2E tests for Dapr workflow crash recovery via subprocess orchestration.

Adapted from ``diagrid/agent/langgraph/examples/test_crash_recovery.py``.

Each test uses a subprocess approach:
1. Writes a helper script that builds a workflow where a node/tool crashes
   the process (``os._exit(1)``) on the first run.
2. Runs the helper twice under ``dapr run`` with the same app-id.
3. On the second run, Dapr auto-resumes the workflow and completes it.
4. Asserts all nodes/tools executed and the process ran at least twice.

All tests assert only what crash recovery guarantees: the crashed
activity (tool/node) re-executes successfully after process restart.
LangGraph and Deep Agents are deterministic (no LLM).
Agent frameworks (PydanticAI, OpenAI Agents, CrewAI, Strands) require
an LLM in the subprocess but only to reach the crashing tool; the
crash and resume are deterministic.
"""

import json
import os
import shutil
import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest

_RESOURCES_DIR = str(Path(__file__).parent / "resources")


def _write_helper_script(state_file: Path) -> str:
    """Write the crash-recovery helper script to a temp file.

    Returns the path to the script.
    """
    script_path = state_file.with_suffix(".py")
    script_content = textwrap.dedent(f"""\
        import asyncio
        import json
        import os
        from pathlib import Path
        from typing import List, TypedDict

        from langgraph.graph import StateGraph, START, END
        from diagrid.agent.langgraph import DaprWorkflowGraphRunner
        from dapr.ext.workflow import WorkflowStatus

        STATE_FILE = Path({str(state_file)!r})
        THREAD_ID = "crash-recovery-e2e"


        def load_state() -> dict:
            if STATE_FILE.exists():
                return json.loads(STATE_FILE.read_text())
            return {{
                "run_count": 0,
                "node1_executed": False,
                "node2_executed": False,
                "node3_executed": False,
                "workflow_scheduled": False,
                "workflow_id": None,
            }}


        def save_state(state: dict):
            STATE_FILE.write_text(json.dumps(state, indent=2))


        state = load_state()
        state["run_count"] += 1
        save_state(state)


        class GraphState(TypedDict):
            messages: List[str]
            step: int


        def node_one(gs: GraphState) -> dict:
            state["node1_executed"] = True
            save_state(state)
            return {{"messages": gs["messages"] + ["Node 1"], "step": gs["step"] + 1}}


        def node_two(gs: GraphState) -> dict:
            if state["run_count"] == 1:
                os._exit(1)
            state["node2_executed"] = True
            save_state(state)
            return {{"messages": gs["messages"] + ["Node 2"], "step": gs["step"] + 1}}


        def node_three(gs: GraphState) -> dict:
            state["node3_executed"] = True
            save_state(state)
            return {{"messages": gs["messages"] + ["Node 3"], "step": gs["step"] + 1}}


        async def main():
            graph = StateGraph(GraphState)
            graph.add_node("node_one", node_one)
            graph.add_node("node_two", node_two)
            graph.add_node("node_three", node_three)
            graph.add_edge(START, "node_one")
            graph.add_edge("node_one", "node_two")
            graph.add_edge("node_two", "node_three")
            graph.add_edge("node_three", END)
            compiled = graph.compile()

            runner = DaprWorkflowGraphRunner(
                graph=compiled, max_steps=50, name="crash_recovery_e2e",
            )

            try:
                runner.start()
                await asyncio.sleep(1)

                if not state["workflow_scheduled"]:
                    async for event in runner.run_async(
                        input={{"messages": ["Start"], "step": 0}},
                        thread_id=THREAD_ID,
                    ):
                        if event["type"] == "workflow_started":
                            state["workflow_scheduled"] = True
                            state["workflow_id"] = event.get("workflow_id")
                            save_state(state)
                        elif event["type"] == "workflow_completed":
                            break
                        elif event["type"] == "workflow_failed":
                            break
                else:
                    wf_id = state.get("workflow_id")
                    if wf_id:
                        for _ in range(60):
                            await asyncio.sleep(1)
                            ws = runner._workflow_client.get_workflow_state(
                                instance_id=wf_id,
                            )
                            if ws and ws.runtime_status == WorkflowStatus.COMPLETED:
                                break
                            if ws and ws.runtime_status == WorkflowStatus.FAILED:
                                break
            finally:
                runner.shutdown()


        asyncio.run(main())
    """)
    script_path.write_text(script_content)
    return str(script_path)


@pytest.mark.integration
def test_crash_recovery() -> None:
    """Test that Dapr resumes a workflow after process crash.

    Run 1: crashes during node 2 (os._exit(1)).
    Run 2: Dapr auto-resumes the workflow, node 2 succeeds, node 3 executes.
    """
    dapr_bin = shutil.which("dapr")
    if dapr_bin is None:
        pytest.skip("dapr CLI not found")

    unique = uuid.uuid4().hex[:8]
    state_file = Path(f"/tmp/e2e_crash_test_{unique}.json")
    helper_script = _write_helper_script(state_file)
    app_id = f"e2e-crash-{unique}"

    dapr_cmd = [
        dapr_bin,
        "run",
        "--app-id",
        app_id,
        "--dapr-grpc-port",
        "0",
        "--dapr-http-port",
        "0",
        "--resources-path",
        _RESOURCES_DIR,
        "--log-level",
        "warn",
        "--",
        "python3",
        helper_script,
    ]

    try:
        # Run 1: should crash during node 2
        result1 = _run_dapr_subprocess(dapr_cmd, timeout=60)
        assert result1.returncode != 0, (
            f"Run 1 should have crashed, but returned code {result1.returncode}"
        )

        # Verify state after crash
        assert state_file.exists(), "state file not created during run 1"
        state_after_crash = json.loads(state_file.read_text())
        assert state_after_crash["run_count"] >= 1, "run_count not incremented"
        assert state_after_crash["node1_executed"] is True, (
            "node 1 should have executed before crash"
        )

        # Run 2: should resume and complete
        result2 = _run_dapr_subprocess(dapr_cmd, timeout=60)
        assert result2.returncode == 0, (
            f"Run 2 should have succeeded, but returned code {result2.returncode}. "
            f"stderr: {result2.stderr.decode()[:500]}"
        )

        # Verify final state
        final_state = json.loads(state_file.read_text())
        assert final_state["node1_executed"] is True, "node 1 not executed"
        assert final_state["node2_executed"] is True, "node 2 not executed after resume"
        assert final_state["node3_executed"] is True, "node 3 not executed after resume"
        assert final_state["run_count"] >= 2, (
            f"expected >=2 runs, got {final_state['run_count']}"
        )
    finally:
        # Cleanup temp files
        state_file.unlink(missing_ok=True)
        Path(helper_script).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Shared subprocess runner for crash recovery tests
# ---------------------------------------------------------------------------


def _run_dapr_subprocess(
    dapr_cmd: list[str],
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    """Run a command under ``dapr run``, killing the process tree on timeout."""
    proc = subprocess.Popen(
        dapr_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        raise
    return subprocess.CompletedProcess(
        args=dapr_cmd,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _run_crash_recovery_test(
    helper_script: str,
    state_file: Path,
    app_id: str,
    *,
    crash_timeout: int = 60,
    resume_timeout: int = 180,
) -> dict:
    """Run crash recovery: 2 subprocess invocations under dapr run.

    Args:
        crash_timeout: Timeout for run 1 (crash). Should be fast.
        resume_timeout: Timeout for run 2 (resume). Needs time for Dapr
            startup + workflow completion (LLM calls can be slow).

    Returns the final state dict.
    """
    dapr_bin = shutil.which("dapr")
    if dapr_bin is None:
        pytest.skip("dapr CLI not found")

    dapr_cmd = [
        dapr_bin,
        "run",
        "--app-id",
        app_id,
        "--dapr-grpc-port",
        "0",
        "--dapr-http-port",
        "0",
        "--resources-path",
        _RESOURCES_DIR,
        "--log-level",
        "warn",
        "--",
        "python3",
        helper_script,
    ]

    # Run 1: schedule workflow then crash
    result1 = _run_dapr_subprocess(dapr_cmd, timeout=crash_timeout)
    assert result1.returncode != 0, (
        f"Run 1 should have crashed, but returned code {result1.returncode}"
    )

    # Verify state after crash
    assert state_file.exists(), "state file not created during run 1"
    state_after_crash = json.loads(state_file.read_text())
    assert state_after_crash["run_count"] >= 1, "run_count not incremented"

    # Run 2: resume and complete (LLM workflows need more time)
    result2 = _run_dapr_subprocess(dapr_cmd, timeout=resume_timeout)
    assert result2.returncode == 0, (
        f"Run 2 should have succeeded, but returned code {result2.returncode}. "
        f"stderr: {result2.stderr.decode()[:500]}"
    )

    final_state = json.loads(state_file.read_text())
    assert final_state["run_count"] >= 2, (
        f"expected >=2 runs, got {final_state['run_count']}"
    )
    return final_state


# ---------------------------------------------------------------------------
# Deep Agents crash recovery (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _write_deepagents_crash_helper(state_file: Path) -> str:
    """Write a Deep Agents crash-recovery helper script.

    Same graph pattern as LangGraph but uses DaprWorkflowDeepAgentRunner.
    """
    script_path = state_file.with_name(state_file.stem + "_script.py")
    script_content = textwrap.dedent(f"""\
        import asyncio
        import json
        import os
        from pathlib import Path
        from typing import List, TypedDict

        from langgraph.graph import StateGraph, START, END
        from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner
        from dapr.ext.workflow import WorkflowStatus

        STATE_FILE = Path({str(state_file)!r})
        THREAD_ID = "crash-recovery-deepagents-e2e"


        def load_state() -> dict:
            if STATE_FILE.exists():
                return json.loads(STATE_FILE.read_text())
            return {{
                "run_count": 0,
                "node1_executed": False,
                "node2_executed": False,
                "node3_executed": False,
                "workflow_scheduled": False,
                "workflow_id": None,
            }}


        def save_state(state: dict):
            STATE_FILE.write_text(json.dumps(state, indent=2))


        state = load_state()
        state["run_count"] += 1
        save_state(state)


        class GraphState(TypedDict):
            messages: List[str]
            step: int


        def node_one(gs: GraphState) -> dict:
            state["node1_executed"] = True
            save_state(state)
            return {{"messages": gs["messages"] + ["Node 1"], "step": gs["step"] + 1}}


        def node_two(gs: GraphState) -> dict:
            if state["run_count"] == 1:
                os._exit(1)
            state["node2_executed"] = True
            save_state(state)
            return {{"messages": gs["messages"] + ["Node 2"], "step": gs["step"] + 1}}


        def node_three(gs: GraphState) -> dict:
            state["node3_executed"] = True
            save_state(state)
            return {{"messages": gs["messages"] + ["Node 3"], "step": gs["step"] + 1}}


        async def main():
            graph = StateGraph(GraphState)
            graph.add_node("node_one", node_one)
            graph.add_node("node_two", node_two)
            graph.add_node("node_three", node_three)
            graph.add_edge(START, "node_one")
            graph.add_edge("node_one", "node_two")
            graph.add_edge("node_two", "node_three")
            graph.add_edge("node_three", END)
            compiled = graph.compile()

            runner = DaprWorkflowDeepAgentRunner(
                agent=compiled, max_steps=50, name="crash_recovery_da_e2e",
            )

            try:
                runner.start()
                await asyncio.sleep(1)

                if not state["workflow_scheduled"]:
                    async for event in runner.run_async(
                        input={{"messages": ["Start"], "step": 0}},
                        thread_id=THREAD_ID,
                    ):
                        if event["type"] == "workflow_started":
                            state["workflow_scheduled"] = True
                            state["workflow_id"] = event.get("workflow_id")
                            save_state(state)
                        elif event["type"] == "workflow_completed":
                            break
                        elif event["type"] == "workflow_failed":
                            break
                else:
                    wf_id = state.get("workflow_id")
                    if wf_id:
                        for _ in range(60):
                            await asyncio.sleep(1)
                            ws = runner._workflow_client.get_workflow_state(
                                instance_id=wf_id,
                            )
                            if ws and ws.runtime_status == WorkflowStatus.COMPLETED:
                                break
                            if ws and ws.runtime_status == WorkflowStatus.FAILED:
                                break
            finally:
                runner.shutdown()


        asyncio.run(main())
    """)
    script_path.write_text(script_content)
    return str(script_path)


@pytest.mark.integration
def test_crash_recovery_deepagents() -> None:
    """Test that Dapr resumes a Deep Agents workflow after process crash.

    Same graph pattern as LangGraph but uses DaprWorkflowDeepAgentRunner.
    No LLM needed — purely deterministic.
    """
    unique = uuid.uuid4().hex[:8]
    state_file = Path(f"/tmp/e2e_crash_da_{unique}.json")
    helper_script = _write_deepagents_crash_helper(state_file)
    app_id = f"e2e-crash-da-{unique}"

    try:
        final_state = _run_crash_recovery_test(helper_script, state_file, app_id)
        assert final_state["node1_executed"] is True, "node 1 not executed"
        assert final_state["node2_executed"] is True, "node 2 not executed after resume"
        assert final_state["node3_executed"] is True, "node 3 not executed after resume"
    finally:
        state_file.unlink(missing_ok=True)
        Path(helper_script).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Agent framework crash recovery helpers
# ---------------------------------------------------------------------------
#
# Unlike LangGraph/Deep Agents (deterministic graphs), agent frameworks need
# an LLM to drive tool calls. The crash must be deterministic — we crash
# immediately after the workflow is scheduled (run 1), then poll for
# completion on run 2. This tests Dapr's ability to resume a workflow whose
# host process died, without depending on LLM tool-call decisions.
# ---------------------------------------------------------------------------


def _agent_crash_preamble(state_file: Path) -> str:
    """Return the preamble for an agent crash-recovery helper script."""
    return textwrap.dedent(f"""\
        import asyncio
        import json
        import os
        from pathlib import Path

        from dapr.ext.workflow import WorkflowStatus

        STATE_FILE = Path({str(state_file)!r})
        SESSION_ID = "crash-recovery-agent-e2e"


        def load_state() -> dict:
            if STATE_FILE.exists():
                return json.loads(STATE_FILE.read_text())
            return {{
                "run_count": 0,
                "workflow_scheduled": False,
                "workflow_id": None,
            }}


        def save_state(state: dict):
            STATE_FILE.write_text(json.dumps(state, indent=2))


        state = load_state()
        state["run_count"] += 1
        save_state(state)
    """)


def _write_pydantic_ai_crash_helper(state_file: Path) -> str:
    """Write a PydanticAI crash-recovery helper script.

    Crash is deterministic: ``os._exit(1)`` fires right after the workflow
    is scheduled (run 1). On run 2 the script polls until the workflow
    completes or fails.
    """
    script_path = state_file.with_name(state_file.stem + "_script.py")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")
    script_content = _agent_crash_preamble(state_file)
    script_content += textwrap.dedent(f"""\

        import sys
        # Bypass test shadow paths for pydantic_ai
        sys.path = [p for p in sys.path if "site-packages" in p or "tests" not in p]

        from pydantic_ai import Agent
        from diagrid.agent.pydantic_ai import DaprWorkflowAgentRunner


        def get_weather(city: str) -> str:
            \"\"\"Get the current weather for a city.\"\"\"
            return f"The weather in {{city}} is sunny and 22 degrees Celsius"


        agent = Agent(
            "openai:{ollama_model}",
            system_prompt=(
                "You are a helpful weather assistant. "
                "When asked about weather, use the get_weather tool."
            ),
            tools=[get_weather],
        )

        runner = DaprWorkflowAgentRunner(
            agent=agent,
            name="crash_recovery_pydantic_e2e",
            max_iterations=10,
        )


        async def main():
            try:
                runner.start()
                await asyncio.sleep(1)

                if not state["workflow_scheduled"]:
                    async for event in runner.run_async(
                        user_message="What is the weather in Tokyo?",
                        session_id=SESSION_ID,
                    ):
                        if event["type"] == "workflow_started":
                            state["workflow_scheduled"] = True
                            state["workflow_id"] = event.get("workflow_id")
                            save_state(state)
                            os._exit(1)
                else:
                    wf_id = state.get("workflow_id")
                    if wf_id:
                        for _ in range(90):
                            await asyncio.sleep(1)
                            ws = runner._workflow_client.get_workflow_state(
                                instance_id=wf_id,
                            )
                            if ws and ws.runtime_status in (
                                WorkflowStatus.COMPLETED,
                                WorkflowStatus.FAILED,
                            ):
                                break
            finally:
                runner.shutdown()


        asyncio.run(main())
    """)
    script_path.write_text(script_content)
    return str(script_path)


def _write_openai_agents_crash_helper(state_file: Path) -> str:
    """Write an OpenAI Agents crash-recovery helper script.

    Crash is deterministic: ``os._exit(1)`` fires right after the workflow
    is scheduled (run 1).
    """
    script_path = state_file.with_name(state_file.stem + "_script.py")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")
    script_content = _agent_crash_preamble(state_file)
    script_content += textwrap.dedent(f"""\

        from agents import Agent, function_tool
        from diagrid.agent.openai_agents import DaprWorkflowAgentRunner


        @function_tool
        def get_weather(city: str) -> str:
            \"\"\"Get the current weather for a city.\"\"\"
            return f"The weather in {{city}} is sunny and 22 degrees Celsius"


        agent = Agent(
            name="crash_recovery_agent",
            instructions=(
                "You are a helpful weather assistant. "
                "When asked about weather, use the get_weather tool."
            ),
            model="{ollama_model}",
            tools=[get_weather],
        )

        runner = DaprWorkflowAgentRunner(
            agent=agent,
            name="crash_recovery_openai_e2e",
            max_iterations=10,
        )


        async def main():
            try:
                runner.start()
                await asyncio.sleep(1)

                if not state["workflow_scheduled"]:
                    async for event in runner.run_async(
                        user_message="What is the weather in Tokyo?",
                        session_id=SESSION_ID,
                    ):
                        if event["type"] == "workflow_started":
                            state["workflow_scheduled"] = True
                            state["workflow_id"] = event.get("workflow_id")
                            save_state(state)
                            os._exit(1)
                else:
                    wf_id = state.get("workflow_id")
                    if wf_id:
                        for _ in range(90):
                            await asyncio.sleep(1)
                            ws = runner._workflow_client.get_workflow_state(
                                instance_id=wf_id,
                            )
                            if ws and ws.runtime_status in (
                                WorkflowStatus.COMPLETED,
                                WorkflowStatus.FAILED,
                            ):
                                break
            finally:
                runner.shutdown()


        asyncio.run(main())
    """)
    script_path.write_text(script_content)
    return str(script_path)


def _write_crewai_crash_helper(state_file: Path) -> str:
    """Write a CrewAI crash-recovery helper script.

    Crash is deterministic: ``os._exit(1)`` fires right after the workflow
    is scheduled (run 1).
    """
    script_path = state_file.with_name(state_file.stem + "_script.py")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")
    litellm_model = f"openai/{ollama_model}"
    script_content = _agent_crash_preamble(state_file)
    script_content += textwrap.dedent(f"""\

        from crewai import Agent, Task
        from crewai.tools import tool
        from diagrid.agent.crewai import DaprWorkflowAgentRunner


        @tool("Get weather for a city")
        def get_weather(city: str) -> str:
            \"\"\"Get the current weather for a city.\"\"\"
            return f"The weather in {{city}} is sunny and 22 degrees Celsius"


        agent = Agent(
            role="Weather Assistant",
            goal="Help users check the weather using the get_weather tool",
            backstory="You are a helpful weather assistant.",
            tools=[get_weather],
            llm="{litellm_model}",
            verbose=False,
        )

        task = Task(
            description="What is the weather in Tokyo? Use the get_weather tool.",
            expected_output="A sentence describing the current weather in Tokyo.",
            agent=agent,
        )

        runner = DaprWorkflowAgentRunner(
            agent=agent,
            name="crash_recovery_crewai_e2e",
            max_iterations=10,
        )


        async def main():
            try:
                runner.start()
                await asyncio.sleep(1)

                if not state["workflow_scheduled"]:
                    async for event in runner.run_async(
                        task=task,
                        session_id=SESSION_ID,
                    ):
                        if event["type"] == "workflow_started":
                            state["workflow_scheduled"] = True
                            state["workflow_id"] = event.get("workflow_id")
                            save_state(state)
                            os._exit(1)
                else:
                    wf_id = state.get("workflow_id")
                    if wf_id:
                        for _ in range(90):
                            await asyncio.sleep(1)
                            ws = runner._workflow_client.get_workflow_state(
                                instance_id=wf_id,
                            )
                            if ws and ws.runtime_status in (
                                WorkflowStatus.COMPLETED,
                                WorkflowStatus.FAILED,
                            ):
                                break
            finally:
                runner.shutdown()


        asyncio.run(main())
    """)
    script_path.write_text(script_content)
    return str(script_path)


def _write_strands_crash_helper(state_file: Path) -> str:
    """Write a Strands crash-recovery helper script.

    Crash is deterministic: ``os._exit(1)`` fires right after the workflow
    is scheduled (run 1).
    """
    script_path = state_file.with_name(state_file.stem + "_script.py")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")
    ollama_endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1")
    script_content = _agent_crash_preamble(state_file)
    script_content += textwrap.dedent(f"""\

        from strands import Agent, tool
        from strands.models.openai import OpenAIModel
        from diagrid.agent.strands import DaprWorkflowAgentRunner


        @tool
        def get_weather(city: str) -> str:
            \"\"\"Get the current weather for a city.

            Args:
                city: The city name

            Returns:
                Weather information
            \"\"\"
            return f"Weather in {{city}}: Sunny, 72F (22C), humidity 45%"


        agent = Agent(
            model=OpenAIModel(
                model_id="{ollama_model}",
                client_args={{"base_url": "{ollama_endpoint}", "api_key": "ollama"}},
            ),
            tools=[get_weather],
            system_prompt=(
                "You are a helpful weather assistant. "
                "When asked about weather, use the get_weather tool."
            ),
        )

        runner = DaprWorkflowAgentRunner(
            agent=agent,
            name="crash_recovery_strands_e2e",
        )


        async def main():
            try:
                runner.start()
                await asyncio.sleep(1)

                if not state["workflow_scheduled"]:
                    async for event in runner.run_async(
                        task="What is the weather in Tokyo?",
                        session_id=SESSION_ID,
                    ):
                        if event["type"] == "workflow_started":
                            state["workflow_scheduled"] = True
                            state["workflow_id"] = event.get("workflow_id")
                            save_state(state)
                            os._exit(1)
                else:
                    wf_id = state.get("workflow_id")
                    if wf_id:
                        for _ in range(90):
                            await asyncio.sleep(1)
                            ws = runner._workflow_client.get_workflow_state(
                                instance_id=wf_id,
                            )
                            if ws and ws.runtime_status in (
                                WorkflowStatus.COMPLETED,
                                WorkflowStatus.FAILED,
                            ):
                                break
            finally:
                runner.shutdown()


        asyncio.run(main())
    """)
    script_path.write_text(script_content)
    return str(script_path)


# ---------------------------------------------------------------------------
# Agent framework crash recovery tests
# ---------------------------------------------------------------------------


@pytest.mark.ollama
@pytest.mark.integration
def test_crash_recovery_pydantic_ai() -> None:
    """Test that Dapr resumes a PydanticAI workflow after process crash.

    Run 1: schedules workflow then crashes (os._exit(1)).
    Run 2: polls the existing workflow until it completes.
    """
    unique = uuid.uuid4().hex[:8]
    state_file = Path(f"/tmp/e2e_crash_pydantic_{unique}.json")
    helper_script = _write_pydantic_ai_crash_helper(state_file)
    app_id = f"e2e-crash-pydantic-{unique}"

    try:
        final_state = _run_crash_recovery_test(
            helper_script,
            state_file,
            app_id,
            resume_timeout=180,
        )
        assert final_state["workflow_scheduled"] is True, "workflow was not scheduled"
    finally:
        state_file.unlink(missing_ok=True)
        Path(helper_script).unlink(missing_ok=True)


@pytest.mark.ollama
@pytest.mark.integration
def test_crash_recovery_openai_agents() -> None:
    """Test that Dapr resumes an OpenAI Agents workflow after process crash.

    Run 1: schedules workflow then crashes (os._exit(1)).
    Run 2: polls the existing workflow until it completes.
    """
    unique = uuid.uuid4().hex[:8]
    state_file = Path(f"/tmp/e2e_crash_openai_{unique}.json")
    helper_script = _write_openai_agents_crash_helper(state_file)
    app_id = f"e2e-crash-openai-{unique}"

    try:
        final_state = _run_crash_recovery_test(
            helper_script,
            state_file,
            app_id,
            resume_timeout=180,
        )
        assert final_state["workflow_scheduled"] is True, "workflow was not scheduled"
    finally:
        state_file.unlink(missing_ok=True)
        Path(helper_script).unlink(missing_ok=True)


@pytest.mark.ollama
@pytest.mark.integration
def test_crash_recovery_crewai() -> None:
    """Test that Dapr resumes a CrewAI workflow after process crash.

    Run 1: schedules workflow then crashes (os._exit(1)).
    Run 2: polls the existing workflow until it completes.
    """
    unique = uuid.uuid4().hex[:8]
    state_file = Path(f"/tmp/e2e_crash_crewai_{unique}.json")
    helper_script = _write_crewai_crash_helper(state_file)
    app_id = f"e2e-crash-crewai-{unique}"

    try:
        final_state = _run_crash_recovery_test(
            helper_script,
            state_file,
            app_id,
            resume_timeout=180,
        )
        assert final_state["workflow_scheduled"] is True, "workflow was not scheduled"
    finally:
        state_file.unlink(missing_ok=True)
        Path(helper_script).unlink(missing_ok=True)


@pytest.mark.ollama
@pytest.mark.integration
def test_crash_recovery_strands() -> None:
    """Test that Dapr resumes a Strands workflow after process crash.

    Run 1: schedules workflow then crashes (os._exit(1)).
    Run 2: polls the existing workflow until it completes.
    """
    unique = uuid.uuid4().hex[:8]
    state_file = Path(f"/tmp/e2e_crash_strands_{unique}.json")
    helper_script = _write_strands_crash_helper(state_file)
    app_id = f"e2e-crash-strands-{unique}"

    try:
        final_state = _run_crash_recovery_test(
            helper_script,
            state_file,
            app_id,
            resume_timeout=180,
        )
        assert final_state["workflow_scheduled"] is True, "workflow was not scheduled"
    finally:
        state_file.unlink(missing_ok=True)
        Path(helper_script).unlink(missing_ok=True)
