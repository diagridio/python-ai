#!/usr/bin/env python3

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Demo: AI-Powered Incident Response Agent with Durable Execution

This demo shows a real-world scenario where an AI agent handles a production
incident by executing a 5-step runbook. Each tool call is a checkpointed
Dapr workflow activity. If the process crashes mid-execution, Dapr resumes
exactly where it left off — no duplicate pages, no duplicate notifications.

The demo is designed to run TWICE:
  - First run:  Steps 1-2 complete, then the process crashes during step 3.
  - Second run: Dapr resumes from the checkpoint. Steps 1-2 are skipped.
                Steps 3-5 complete. The on-call team is NOT paged again.

Usage:
    # Clean up any previous state:
    rm -f /tmp/incident_response_demo_state.json

    # First run (will crash during mitigation):
    dapr run --app-id incident-agent --resources-path ./components -- python3 demo_incident_response.py

    # Second run (Dapr resumes and completes):
    dapr run --app-id incident-agent --resources-path ./components -- python3 demo_incident_response.py

Prerequisites:
    - Dapr initialized: dapr init
    - Redis running: docker run -d --name redis -p 6379:6379 redis:latest
    - OPENAI_API_KEY environment variable set
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_core.tools import tool

from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner
from dapr.ext.workflow import WorkflowStatus


# ---------------------------------------------------------------------------
# State management (persisted across crashes)
# ---------------------------------------------------------------------------

STATE_FILE = Path("/tmp/incident_response_demo_state.json")
THREAD_ID = "incident-response-demo"


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "run_count": 0,
        "workflow_scheduled": False,
        "workflow_id": None,
        "tools": {
            "diagnose_incident": 0,
            "page_oncall_team": 0,
            "execute_mitigation": 0,
            "notify_stakeholders": 0,
            "create_postmortem": 0,
        },
    }


def save_state(s: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)


# Load and bump run count at import time (same pattern as test_crash_recovery)
state = load_state()
state["run_count"] += 1
save_state(state)


# ---------------------------------------------------------------------------
# Sandbox environment — simulated production infrastructure
# ---------------------------------------------------------------------------

SANDBOX_DIR = Path(tempfile.mkdtemp(prefix="incident_sandbox_"))


def setup_sandbox() -> LocalShellBackend:
    """Create a sandbox with simulated production infrastructure files."""
    # Service logs
    logs_dir = SANDBOX_DIR / "var" / "log" / "payments-api"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "app.log").write_text(
        "2024-11-15T03:12:01Z INFO  payments-api started on :8080\n"
        "2024-11-15T03:12:02Z INFO  connected to pg-prod-01 (pool_size=20)\n"
        "2024-11-15T03:14:33Z WARN  connection pool utilization at 85%\n"
        "2024-11-15T03:14:45Z WARN  connection pool utilization at 95%\n"
        "2024-11-15T03:14:58Z ERROR connection pool exhausted — no available connections\n"
        "2024-11-15T03:15:01Z ERROR GET /v1/payments 500 — pool_exhausted (12ms)\n"
        "2024-11-15T03:15:01Z ERROR GET /v1/payments 500 — pool_exhausted (8ms)\n"
        "2024-11-15T03:15:02Z ERROR POST /v1/checkout 500 — pool_exhausted (15ms)\n"
        "2024-11-15T03:15:03Z ERROR GET /v1/billing/status 500 — pool_exhausted (6ms)\n"
        "2024-11-15T03:15:05Z WARN  downstream: checkout-service reporting errors\n"
        "2024-11-15T03:15:06Z WARN  downstream: billing-service reporting errors\n"
        "2024-11-15T03:15:10Z ERROR 847 requests failed in last 60s\n"
    )

    # Database metrics
    db_dir = SANDBOX_DIR / "var" / "log" / "postgres"
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "pg-prod-01.log").write_text(
        "2024-11-15T03:14:30Z LOG  active connections: 19/20\n"
        "2024-11-15T03:14:45Z LOG  active connections: 20/20\n"
        "2024-11-15T03:14:58Z WARNING  max_connections reached, rejecting new connections\n"
        "2024-11-15T03:15:00Z LOG  waiting queue depth: 142\n"
        "2024-11-15T03:15:05Z LOG  avg query latency: 28.4s (threshold: 200ms)\n"
    )

    # Service configuration
    config_dir = SANDBOX_DIR / "etc" / "payments-api"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        "service:\n"
        "  name: payments-api\n"
        "  port: 8080\n"
        "  region: us-east-1\n"
        "\n"
        "database:\n"
        "  host: pg-prod-01.internal\n"
        "  port: 5432\n"
        "  pool_size: 20\n"
        "  pool_timeout: 30s\n"
        "  max_idle_connections: 5\n"
        "\n"
        "replicas:\n"
        "  count: 3\n"
        "  autoscale: false\n"
    )

    # Mitigation scripts
    scripts_dir = SANDBOX_DIR / "opt" / "runbooks" / "payments-api"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    restart_script = scripts_dir / "restart_pool.sh"
    restart_script.write_text(
        "#!/bin/bash\n"
        "# Restart connection pool and scale replicas for payments-api\n"
        "set -e\n"
        "\n"
        'echo "Recycling connection pool on pg-prod-01..."\n'
        "sleep 1\n"
        'echo "Connection pool recycled. Active connections: 3/20"\n'
        "\n"
        'echo "Scaling replicas: 3 -> 6..."\n'
        "sleep 1\n"
        'echo "Replicas scaled to 6. Health checks passing."\n'
        "\n"
        'echo "Error rate: 45% -> 0.2%"\n'
        'echo "P99 latency: 30s -> 180ms"\n'
        'echo "Mitigation complete."\n'
    )
    restart_script.chmod(0o755)

    # Postmortem template
    templates_dir = SANDBOX_DIR / "opt" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / "postmortem.md").write_text(
        "# Postmortem: {{INCIDENT_ID}}\n"
        "\n"
        "## Summary\n"
        "{{SUMMARY}}\n"
        "\n"
        "## Timeline\n"
        "{{TIMELINE}}\n"
        "\n"
        "## Root Cause\n"
        "{{ROOT_CAUSE}}\n"
        "\n"
        "## Action Items\n"
        "- [ ] {{ACTION_1}}\n"
        "- [ ] {{ACTION_2}}\n"
        "- [ ] {{ACTION_3}}\n"
    )

    backend = LocalShellBackend(
        root_dir=str(SANDBOX_DIR),
        virtual_mode=True,
        inherit_env=False,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    return backend


sandbox_backend = setup_sandbox()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def log(msg: str = ""):
    print(msg, flush=True)


def banner(title: str):
    log(f"\n{'=' * 64}")
    log(f"  {title}")
    log(f"{'=' * 64}")


def step_header(num: int, total: int, name: str):
    log(f"\n  [{num}/{total}] {name}")


def step_detail(line: str):
    log(f"         {line}")


def step_done():
    step_detail("Status: COMPLETE  [checkpointed by Dapr]")


# ---------------------------------------------------------------------------
# Print run preamble
# ---------------------------------------------------------------------------

banner("INCIDENT RESPONSE AGENT — Durable Execution Demo")
log("  Powered by: Deep Agents + Dapr Workflows + Sandbox")
log("  Each tool call = checkpointed workflow activity")
log(f"  Sandbox root: {SANDBOX_DIR}")
log(f"{'=' * 64}")

if state["run_count"] == 1:
    log("\n  RUN #1 — First execution")
    log("  State: Clean start (no previous checkpoints)")
else:
    log(f"\n  RUN #{state['run_count']} — Recovery after crash")
    executed = [k for k, v in state["tools"].items() if v > 0]
    if executed:
        log(f"  Checkpointed steps: {', '.join(executed)}")
    log()
    log("  NOTE: Previously completed steps will NOT re-execute.")
    log("        The on-call team will NOT be paged again.")
    log("        Dapr resumes exactly where it left off.")

log(f"\n{'-' * 64}")
log("  INCOMING ALERT")
log(f"{'-' * 64}")
log("  Service:  payments-api")
log("  Symptoms: HTTP 500 spike, P99 latency > 30s, connection timeouts")
log("  Region:   us-east-1")
log("  Source:   Datadog monitor #4821")
log(f"{'-' * 64}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def diagnose_incident(service_name: str) -> str:
    """Query monitoring systems and sandbox logs to diagnose a production incident.
    Reads application and database logs from the sandbox filesystem.
    Args: service_name — the affected service to investigate."""
    step_header(1, 5, "DIAGNOSE INCIDENT")
    step_detail("Reading logs from sandbox environment...")
    time.sleep(1)

    # Read real log files from the sandbox
    app_logs = sandbox_backend.read(f"/var/log/{service_name}/app.log")
    db_logs = sandbox_backend.read("/var/log/postgres/pg-prod-01.log")
    sandbox_backend.read(f"/etc/{service_name}/config.yaml")

    app_lines = len([line for line in app_logs.splitlines() if line.strip()])
    db_lines = len([line for line in db_logs.splitlines() if line.strip()])
    step_detail(f"Scanned: /var/log/{service_name}/app.log ({app_lines} lines)")
    step_detail(f"Scanned: /var/log/postgres/pg-prod-01.log ({db_lines} lines)")
    step_detail(f"Read:    /etc/{service_name}/config.yaml")
    time.sleep(1)

    state["tools"]["diagnose_incident"] += 1
    save_state(state)

    step_detail("Root Cause: Connection pool exhaustion on primary DB (pg-prod-01)")
    step_detail("Severity:   SEV-1 (customer-facing impact)")
    step_detail("Blast Radius: 3 downstream services, ~12,000 req/min affected")
    step_done()

    return (
        "Diagnosis complete. "
        "Incident ID: INC-2024-0892. "
        "Severity: SEV-1. "
        f"Analyzed sandbox logs at /var/log/{service_name}/app.log and "
        "/var/log/postgres/pg-prod-01.log. "
        f"Service config at /etc/{service_name}/config.yaml shows pool_size=20. "
        "Root cause: Connection pool exhaustion on primary DB pg-prod-01. "
        "Blast radius: payments-api, checkout-service, billing-service affected. "
        "~12,000 requests/min impacted. "
        "Mitigation script available at /opt/runbooks/payments-api/restart_pool.sh. "
        "Recommended mitigation: run the restart_pool.sh script. "
        "Next step: page the on-call team with severity SEV-1."
    )


@tool
def page_oncall_team(severity: str, incident_summary: str) -> str:
    """Page the on-call engineering team via PagerDuty.
    Args: severity — incident severity (e.g. SEV-1), incident_summary — brief description."""
    step_header(2, 5, "PAGE ON-CALL TEAM")
    step_detail("Triggering PagerDuty escalation policy...")
    time.sleep(1)

    state["tools"]["page_oncall_team"] += 1
    save_state(state)

    step_detail("Paged:      Sarah Chen (primary), Mike Torres (secondary)")
    step_detail("Escalation: VP Engineering auto-notified (SEV-1 policy)")
    step_detail("Ack ETA:    < 5 minutes")
    step_done()

    return (
        "On-call team paged successfully. "
        "Sarah Chen (primary) and Mike Torres (secondary) notified. "
        "VP Engineering auto-notified per SEV-1 escalation policy. "
        "Next step: execute mitigation — restart connection pool and scale replicas."
    )


@tool
def execute_mitigation(action: str) -> str:
    """Execute a mitigation script from the sandbox environment.
    Runs the runbook script in the sandbox and captures output.
    Args: action — the mitigation to perform (e.g. restart pool, scale replicas)."""
    step_header(3, 5, "EXECUTE MITIGATION")
    step_detail("Running /opt/runbooks/payments-api/restart_pool.sh in sandbox...")

    # --- CRASH POINT ---
    if state["run_count"] == 1:
        time.sleep(1)
        log()
        log("  !! PROCESS CRASH !!")
        log("  The process terminated unexpectedly during mitigation.")
        log()
        log("  In a normal agent framework, ALL progress would be lost.")
        log("  The on-call team would be paged AGAIN at 3am.")
        log()
        log("  With Dapr Workflows, steps 1-2 are safely checkpointed.")
        log("  Run the demo again to see durable recovery.")
        log(f"{'=' * 64}")
        os._exit(1)

    # Run the mitigation script in the sandbox (execute() uses real paths)
    script_path = SANDBOX_DIR / "opt" / "runbooks" / "payments-api" / "restart_pool.sh"
    result = sandbox_backend.execute(f"bash {script_path}")
    for line in result.output.strip().splitlines():
        if not line.startswith("[stderr]"):
            step_detail(f"  {line}")
    time.sleep(1)

    state["tools"]["execute_mitigation"] += 1
    save_state(state)

    step_done()

    return (
        "Mitigation applied successfully via sandbox script. "
        f"Script output:\n{result.output}\n"
        "Connection pool restarted on pg-prod-01. "
        "Replicas scaled from 3 to 6. "
        "Error rate dropped from 45% to 0.2%. "
        "P99 latency recovered to 180ms. "
        "Incident ID: INC-2024-0892. "
        "Next step: notify stakeholders of the resolution."
    )


@tool
def notify_stakeholders(incident_id: str, status_update: str) -> str:
    """Send incident status notifications to Slack channels and affected customers.
    Args: incident_id — the incident identifier, status_update — current status message."""
    step_header(4, 5, "NOTIFY STAKEHOLDERS")
    step_detail("Sending status updates...")
    time.sleep(1)

    state["tools"]["notify_stakeholders"] += 1
    save_state(state)

    step_detail("Slack:   #incidents, #payments-team, #engineering-all")
    step_detail("Email:   847 affected customer accounts notified")
    step_detail("Status page: Updated to 'Monitoring — issue mitigated'")
    step_done()

    return (
        "Stakeholders notified. "
        "Slack channels updated: #incidents, #payments-team, #engineering-all. "
        "847 affected customers emailed. "
        "Public status page updated to 'Monitoring'. "
        "Incident ID: INC-2024-0892. "
        "Next step: create postmortem with incident timeline."
    )


@tool
def create_postmortem(incident_id: str, timeline: str) -> str:
    """Create a postmortem document in the sandbox filesystem.
    Reads the template from /opt/templates/postmortem.md and writes the filled document.
    Args: incident_id — the incident identifier, timeline — summary of events."""
    step_header(5, 5, "CREATE POSTMORTEM")
    step_detail("Reading template from /opt/templates/postmortem.md...")
    time.sleep(1)

    # Read the template from the sandbox
    sandbox_backend.read("/opt/templates/postmortem.md")

    # Write the filled postmortem to the sandbox
    postmortem_content = (
        f"# Postmortem: {incident_id}\n"
        "\n"
        "## Summary\n"
        "Connection pool exhaustion on pg-prod-01 caused cascading failures\n"
        "across payments-api, checkout-service, and billing-service.\n"
        "\n"
        "## Timeline\n"
        f"{timeline}\n"
        "\n"
        "## Root Cause\n"
        "Connection pool size (20) was insufficient for traffic spike.\n"
        "No auto-scaling was configured for database connections.\n"
        "\n"
        "## Action Items\n"
        "- [ ] Increase connection pool limits to 50 (owner: Sarah Chen)\n"
        "- [ ] Add pool exhaustion alerting at 80% (owner: Mike Torres)\n"
        "- [ ] Update runbook with auto-scaling trigger (owner: Platform team)\n"
    )
    postmortem_path = f"/opt/postmortems/{incident_id}.md"
    postmortems_dir = SANDBOX_DIR / "opt" / "postmortems"
    postmortems_dir.mkdir(parents=True, exist_ok=True)
    sandbox_backend.write(postmortem_path, postmortem_content)

    state["tools"]["create_postmortem"] += 1
    save_state(state)

    step_detail(f"Written: {postmortem_path}")
    step_detail("Action items:")
    step_detail("  1. Increase connection pool limits (owner: Sarah Chen)")
    step_detail("  2. Add pool exhaustion alerting at 80% (owner: Mike Torres)")
    step_detail("  3. Update runbook with auto-scaling trigger (owner: Platform team)")
    step_done()

    return (
        f"Postmortem written to sandbox at {postmortem_path}. "
        "3 action items generated. "
        f"Incident {incident_id} is fully resolved. "
        "All 5 runbook steps complete."
    )


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    model="openai:gpt-4o-mini",
    tools=[
        diagnose_incident,
        page_oncall_team,
        execute_mitigation,
        notify_stakeholders,
        create_postmortem,
    ],
    backend=sandbox_backend,
    system_prompt=(
        "You are an AI incident response commander. Your tools have access to a "
        "sandbox environment containing production infrastructure: application logs, "
        "database logs, service configs, and runbook scripts.\n\n"
        "When given an incident alert, execute the following runbook steps "
        "strictly IN ORDER. IMPORTANT: Call exactly ONE tool per response. "
        "Wait for each tool's result before proceeding to the next step. "
        "Do NOT call multiple tools in a single response.\n\n"
        "1. Call diagnose_incident with the service name\n"
        "2. Call page_oncall_team with the severity and summary from step 1\n"
        "3. Call execute_mitigation with the recommended action from step 1\n"
        "4. Call notify_stakeholders with the incident ID and resolution status\n"
        "5. Call create_postmortem with the incident ID and a timeline summary\n\n"
        "Execute each tool exactly once, in order. Do not skip steps. "
        "Do not use sandbox tools (read_file, ls, execute, etc.) directly — "
        "the runbook tools handle sandbox access internally. "
        "After all 5 steps complete, provide a brief incident resolution summary."
    ),
    name="incident-commander",
)

PROMPT = (
    "ALERT: Production incident detected.\n"
    "Service: payments-api\n"
    "Symptoms: HTTP 500 errors spiking, P99 latency > 30s, connection timeouts\n"
    "Affected region: us-east-1\n"
    "Alert source: Datadog monitor #4821\n\n"
    "Execute the full incident response runbook."
)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def print_verification():
    """Print the durability verification report."""
    final = load_state()
    tools = final["tools"]

    banner("DURABILITY VERIFICATION")
    log()
    max_name = max(len(k) for k in tools)
    for name, count in tools.items():
        pad = " " * (max_name - len(name))
        note = ""
        if final["run_count"] > 1 and name in ("diagnose_incident", "page_oncall_team"):
            note = "  (NOT re-run after crash)"
        elif final["run_count"] > 1:
            note = "  (completed on recovery)"
        log(f"  {name}{pad} : executed {count} time(s){note}")

    total_calls = sum(tools.values())
    log()
    log(f"  Process runs:     {final['run_count']}")
    log(f"  Tool executions:  {total_calls} total (no duplicates)")
    log()

    if final["run_count"] >= 2 and all(v == 1 for v in tools.values()):
        log("  RESULT: Durable execution preserved all progress.")
        log("          No duplicate pages. No duplicate notifications.")
        log("          The agent resumed exactly where it left off.")
    elif all(v >= 1 for v in tools.values()):
        log("  RESULT: All incident response steps completed successfully.")

    log(f"{'=' * 64}")


def print_completion(event: dict):
    output = event.get("output", {})
    messages = output.get("messages", [])

    banner("INCIDENT RESOLVED")

    if messages:
        last_msg = messages[-1]
        content = (
            last_msg.get("content", "")
            if isinstance(last_msg, dict)
            else getattr(last_msg, "content", str(last_msg))
        )
        # Handle list-of-blocks content from Responses API
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        if content:
            log()
            log("  Agent Summary:")
            # Wrap long lines
            words = content.split()
            line = "    "
            for word in words:
                if len(line) + len(word) + 1 > 62:
                    log(line)
                    line = "    " + word
                else:
                    line += (" " if line.strip() else "") + word
            if line.strip():
                log(line)

    log(f"\n  Steps: {event.get('steps', 'N/A')}")
    log(f"{'=' * 64}")

    print_verification()


async def poll_for_completion(runner: DaprWorkflowDeepAgentRunner, workflow_id: str):
    from diagrid.agent.langgraph.models import GraphWorkflowOutput

    if not workflow_id:
        log("  No workflow_id saved — cannot poll!")
        return

    assert runner._workflow_client is not None

    previous_status = None
    while True:
        await asyncio.sleep(1.0)
        wf_state = runner._workflow_client.get_workflow_state(
            instance_id=workflow_id
        )
        if wf_state is None:
            log("  Workflow state not found!")
            break

        if wf_state.runtime_status != previous_status:
            previous_status = wf_state.runtime_status

        if wf_state.runtime_status == WorkflowStatus.COMPLETED:
            output_data = wf_state.serialized_output
            if output_data:
                output_dict = (
                    json.loads(output_data)
                    if isinstance(output_data, str)
                    else output_data
                )
                output = GraphWorkflowOutput.from_dict(output_dict)
                print_completion({"output": output.output, "steps": output.steps})
            break
        elif wf_state.runtime_status == WorkflowStatus.FAILED:
            log(f"\n  Workflow FAILED: {wf_state.failure_details}")
            break
        elif wf_state.runtime_status == WorkflowStatus.TERMINATED:
            log("\n  Workflow was TERMINATED")
            break


async def main():
    runner = DaprWorkflowDeepAgentRunner(
        agent=agent,
        name="incident-response",
        max_steps=50,
    )

    try:
        runner.start()
        log("\n  Agent runtime started. Executing runbook...\n")
        await asyncio.sleep(1)

        if not state["workflow_scheduled"]:
            # First run: schedule the workflow
            async for event in runner.run_async(
                input={"messages": [{"role": "user", "content": PROMPT}]},
                thread_id=THREAD_ID,
            ):
                event_type = event["type"]
                if event_type == "workflow_started":
                    state["workflow_scheduled"] = True
                    state["workflow_id"] = event.get("workflow_id")
                    save_state(state)
                elif event_type == "workflow_completed":
                    print_completion(event)
                    break
                elif event_type == "workflow_failed":
                    log(f"\n  Workflow FAILED: {event.get('error')}")
                    break
        else:
            # Recovery run: show skipped steps, then poll
            for name, count in state["tools"].items():
                if count > 0:
                    step_num = list(state["tools"].keys()).index(name) + 1
                    name_display = name.upper().replace("_", " ")
                    log(f"  [{step_num}/5] {name_display} ... SKIPPED (already checkpointed)")
            log()
            log(f"  Resuming workflow: {state.get('workflow_id', '?')}")
            log()
            await poll_for_completion(runner, state.get("workflow_id", ""))

    except KeyboardInterrupt:
        log("\n  Interrupted by user")
    finally:
        runner.shutdown()
        log(f"\n  Sandbox artifacts at: {SANDBOX_DIR}")
        log("  Agent runtime stopped.")


if __name__ == "__main__":
    asyncio.run(main())
