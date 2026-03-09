# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""diagridpy deploy command — build and deploy an agent to the kind cluster."""

from __future__ import annotations

import atexit
import base64
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import click
import httpx

from diagrid.cli.infra.docker import (
    build_image,
    push_to_registry,
    push_to_registry_parallel,
)
from diagrid.cli.infra.kind import (
    cluster_exists,
    ensure_registry_config,
    kind_available,
)
from diagrid.cli.infra.kubectl import apply_stdin, rollout_restart
from diagrid.cli.utils import console
from diagrid.cli.utils.deps import preflight_check
from diagrid.cli.utils.process import CommandError, run, run_capture
from diagrid.core.auth.device_code import DeviceCodeAuth
from diagrid.core.catalyst.appids import create_appid, get_appid
from diagrid.core.catalyst.client import CatalystAPIError, CatalystClient
from diagrid.core.catalyst.projects import get_project
from diagrid.core.config.constants import (
    DEFAULT_KIND_CLUSTER,
    DEFAULT_NAMESPACE,
    KIND_REGISTRY_PORT,
    OTEL_COLLECTOR_ENDPOINT,
    OTEL_COLLECTOR_NODEPORT_GRPC,
    ORCHESTRATOR_AGENTS,
    OrchestratorAgent,
)

# Dapr-agents AppIDs that use the dapr-agents OTEL env var convention
_DAPR_AGENTS_APP_IDS = {"dapr-agent", "event-orchestrator"}

DEPLOYMENT_TEMPLATE = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {namespace}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
      - name: {name}
        image: {image}
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: {port}
        env:
        - name: APP_PORT
          value: "{port}"
        - name: DAPR_APP_ID
          value: "{app_id}"
        - name: DAPR_API_TOKEN
          value: "{api_token}"
        - name: DAPR_HTTP_ENDPOINT
          value: "{http_endpoint}"
        - name: DAPR_GRPC_ENDPOINT
          value: "{grpc_endpoint}"
{otel_env_block}{secret_env_block}\
"""

_OTEL_STANDARD_BLOCK = """\
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "{otel_endpoint}"
        - name: OTEL_SERVICE_NAME
          value: "{app_id}"
"""

_OTEL_DAPR_AGENTS_BLOCK = """\
        - name: OTEL_ENABLED
          value: "true"
        - name: OTEL_ENDPOINT
          value: "{otel_endpoint}"
        - name: OTEL_TRACING_ENABLED
          value: "true"
        - name: OTEL_LOGGING_ENABLED
          value: "true"
        - name: OTEL_TRACING_EXPORTER
          value: "otlp_grpc"
        - name: OTEL_LOGGING_EXPORTER
          value: "otlp_grpc"
"""


def _otel_env_block(app_id: str, otel_endpoint: str) -> str:
    """Return the OTEL env var block for a given agent."""
    if app_id in _DAPR_AGENTS_APP_IDS:
        return _OTEL_DAPR_AGENTS_BLOCK.format(
            otel_endpoint=otel_endpoint, app_id=app_id
        )
    return _OTEL_STANDARD_BLOCK.format(otel_endpoint=otel_endpoint, app_id=app_id)


_SECRET_ENV_ENTRY = """\
        - name: {env_name}
          valueFrom:
            secretKeyRef:
              name: llm-secret
              key: {secret_key}
              optional: true
"""


def _secret_env_block(agent: OrchestratorAgent) -> str:
    """Return secretKeyRef env entries for an agent's required LLM keys."""
    if not agent.secret_env:
        return ""
    return "".join(
        _SECRET_ENV_ENTRY.format(env_name=name, secret_key=key)
        for name, key in agent.secret_env
    )


# ---------------------------------------------------------------------------
# Registry health check
# ---------------------------------------------------------------------------


def _ensure_registry_healthy() -> None:
    """Ensure the local registry is reachable; auto-repair Kind node config."""
    if kind_available() and cluster_exists(DEFAULT_KIND_CLUSTER):
        console.info("Ensuring registry config on Kind nodes...")
        ensure_registry_config(DEFAULT_KIND_CLUSTER)

    # Verify the registry is actually responding
    url = f"http://localhost:{KIND_REGISTRY_PORT}/v2/"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
    except Exception:
        raise click.ClickException(
            f"Local registry not reachable at localhost:{KIND_REGISTRY_PORT}. "
            "Ensure 'kind-registry' container is running:\n"
            "  docker ps | grep kind-registry\n"
            "Or re-run: diagridpy init"
        )


# ---------------------------------------------------------------------------
# LLM key resolution
# ---------------------------------------------------------------------------


def _read_secret_key(namespace: str, secret_name: str, key: str) -> str:
    """Read a single key from a Kubernetes secret. Returns '' on failure."""
    try:
        raw = run_capture(
            "kubectl",
            "get",
            "secret",
            secret_name,
            "-n",
            namespace,
            "-o",
            f"jsonpath={{.data.{key}}}",
        )
        if raw.strip():
            return base64.b64decode(raw.strip()).decode()
    except (CommandError, Exception):
        pass
    return ""


def _resolve_llm_keys(
    openai_flag: str | None,
    google_flag: str | None,
    namespace: str,
) -> dict[str, str]:
    """Resolve LLM API keys. Priority: CLI flag -> env var -> K8s secret -> prompt."""
    # OpenAI
    openai_key = (
        openai_flag
        or os.environ.get("OPENAI_API_KEY")
        or _read_secret_key(namespace, "llm-secret", "apiKey")
    )
    if not openai_key:
        openai_key = click.prompt(
            "Enter your OPENAI_API_KEY", hide_input=True, default=""
        )

    # Google
    google_key = (
        google_flag
        or os.environ.get("GOOGLE_API_KEY")
        or _read_secret_key(namespace, "llm-secret", "googleApiKey")
    )
    if not google_key:
        google_key = click.prompt(
            "Enter your GOOGLE_API_KEY (required for ADK agent)",
            hide_input=True,
            default="",
        )

    return {"OPENAI_API_KEY": openai_key, "GOOGLE_API_KEY": google_key}


def _patch_llm_secret(namespace: str, resolved_keys: dict[str, str]) -> None:
    """Create or update llm-secret with resolved keys so they persist for future deploys."""
    # Map of resolved key name -> secret data key
    key_map = {
        "OPENAI_API_KEY": "apiKey",
        "GOOGLE_API_KEY": "googleApiKey",
    }
    literal_args: list[str] = []
    for env_name, secret_key in key_map.items():
        value = resolved_keys.get(env_name, "")
        if value:
            literal_args.extend(["--from-literal", f"{secret_key}={value}"])

    if not literal_args:
        return

    # Ensure the namespace exists (no-op if it does)
    try:
        run("kubectl", "create", "namespace", namespace)
    except CommandError:
        pass  # already exists

    # Create-or-replace: dry-run + apply is idempotent and works whether
    # the secret already exists or not.
    try:
        yaml_output = run_capture(
            "kubectl",
            "create",
            "secret",
            "generic",
            "llm-secret",
            "-n",
            namespace,
            *literal_args,
            "--dry-run=client",
            "-o",
            "yaml",
        )
        apply_stdin(yaml_output, namespace=namespace)
    except CommandError:
        console.warning("Could not create/update llm-secret — keys may not persist")


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@click.command()
@click.option("--api-key", default=None, help="Diagrid API key")
@click.option("--no-browser", is_flag=True, help="Don't open browser for auth")
@click.option("--image", default="agent", help="Docker image name")
@click.option("--tag", default="latest", help="Docker image tag")
@click.option("--namespace", default=DEFAULT_NAMESPACE, help="Kubernetes namespace")
@click.option("--context", "kube_context", default=None, help="Kubectl context")
@click.option("--project", default="agents-quickstart", help="Catalyst project name")
@click.option(
    "--app-id", default=None, help="Catalyst AppID name (defaults to <project>-agent)"
)
@click.option(
    "--port", default=5001, type=int, help="Agent container port (default: 5001)"
)
@click.option(
    "--trigger",
    default=None,
    help='Trigger the agent with a prompt after deploy (e.g. --trigger "Plan a trip to Paris")',
)
@click.option(
    "--openai-api-key",
    default=None,
    help="OpenAI API key (or set OPENAI_API_KEY env)",
)
@click.option(
    "--google-api-key",
    default=None,
    help="Google API key for ADK (or set GOOGLE_API_KEY env)",
)
@click.pass_context
def deploy(
    ctx: click.Context,
    api_key: str | None,
    no_browser: bool,
    image: str,
    tag: str,
    namespace: str,
    kube_context: str | None,
    project: str,
    app_id: str | None,
    port: int,
    trigger: str | None,
    openai_api_key: str | None,
    google_api_key: str | None,
) -> None:
    """Build and deploy an agent to the kind cluster."""
    preflight_check()

    api_url: str | None = ctx.obj.get("api_url") if ctx.obj else None

    if kube_context:
        run("kubectl", "config", "use-context", kube_context)

    try:
        # Ensure local registry is healthy before building images
        _ensure_registry_healthy()

        if _is_orchestrator_project():
            # Resolve LLM keys for orchestrator deploys
            resolved_keys = _resolve_llm_keys(openai_api_key, google_api_key, namespace)
            _patch_llm_secret(namespace, resolved_keys)

            _deploy_orchestrator(
                api_url=api_url,
                api_key=api_key,
                no_browser=no_browser,
                tag=tag,
                namespace=namespace,
                project=project,
                trigger=trigger,
            )
        else:
            _deploy_single_agent(
                api_url=api_url,
                api_key=api_key,
                no_browser=no_browser,
                image=image,
                tag=tag,
                namespace=namespace,
                project=project,
                app_id=app_id,
                port=port,
                trigger=trigger,
            )
    except CommandError as exc:
        console.error(str(exc))
        raise SystemExit(1) from exc
    except click.ClickException:
        raise
    except Exception as exc:
        console.error(f"Deployment failed: {exc}")
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Orchestrator auto-detection
# ---------------------------------------------------------------------------


def _is_orchestrator_project() -> bool:
    """Detect if the current directory is an orchestrator project.

    Heuristic: ``shared-resources/`` dir exists and >= 3 sibling
    directories contain a Dockerfile.
    """
    cwd = Path.cwd()
    if not (cwd / "shared-resources").is_dir():
        return False
    dockerfile_count = sum(
        1
        for child in cwd.iterdir()
        if child.is_dir()
        and child.name != "shared-resources"
        and (child / "Dockerfile").exists()
    )
    # Also count nested dirs (e.g. dapr-agents/durable-agent)
    for child in cwd.iterdir():
        if child.is_dir() and child.name != "shared-resources":
            for grandchild in child.iterdir():
                if grandchild.is_dir() and (grandchild / "Dockerfile").exists():
                    dockerfile_count += 1
    return dockerfile_count >= 3


# ---------------------------------------------------------------------------
# Single-agent deploy (existing flow)
# ---------------------------------------------------------------------------


def _deploy_single_agent(
    *,
    api_url: str | None,
    api_key: str | None,
    no_browser: bool,
    image: str,
    tag: str,
    namespace: str,
    project: str,
    app_id: str | None,
    port: int,
    trigger: str | None,
) -> None:
    """Build and deploy a single agent (the original deploy flow)."""
    appid_name = app_id or f"{project}-agent"
    total_steps = 6 if trigger else 5

    # Step 1: Authenticate
    with console.spinner(1, total_steps, "Authenticating..."):
        auth = DeviceCodeAuth(
            api_url=api_url, api_key_flag=api_key, no_browser=no_browser
        )
        auth_ctx = auth.authenticate()
    console.success("Authenticated")

    # Step 2: Fetch Catalyst connection details
    with console.spinner(2, total_steps, "Fetching Catalyst connection details..."):
        client = CatalystClient(auth_ctx)
        conn = _get_connection_details(client, project, appid_name)
    console.success(f"AppID '{appid_name}' ready")

    # Step 3: Build Docker image
    with console.spinner(3, total_steps, f"Building Docker image {image}:{tag}..."):
        full_tag = build_image(image, tag)
    console.success(f"Image built: {full_tag}")

    # Step 4: Push to local registry
    with console.spinner(4, total_steps, "Pushing image to local registry..."):
        registry_tag = push_to_registry(full_tag)
    console.success("Image pushed to local registry")

    # Step 5: Apply Kubernetes manifest
    with console.spinner(5, total_steps, "Deploying to Kubernetes..."):
        manifest = DEPLOYMENT_TEMPLATE.format(
            name=image,
            namespace=namespace,
            image=registry_tag,
            port=port,
            app_id=conn["app_id"],
            api_token=conn["api_token"],
            http_endpoint=conn["http_endpoint"],
            grpc_endpoint=conn["grpc_endpoint"],
            otel_env_block="",
            secret_env_block="",
        )
        apply_stdin(manifest, namespace=namespace)
        rollout_restart(image, namespace=namespace)
    console.success("Agent deployed")

    # Step 6 (optional): Trigger the agent
    if trigger:
        with console.spinner(6, total_steps, "Triggering agent..."):
            result = _trigger_agent(image, namespace, trigger, container_port=port)
        console.success("Agent triggered")
        console.info(f"Instance ID: {result.get('instance_id', 'N/A')}")

    summary_lines = [
        f"Image: {full_tag}",
        f"Namespace: {namespace}",
        f"Deployment: {image}",
        f"AppID: {appid_name}",
    ]
    if trigger:
        summary_lines.append("")
        summary_lines.append(f'Prompt: "{trigger}"')

    console.print_summary("Deployment Complete", summary_lines)


# ---------------------------------------------------------------------------
# Orchestrator deploy
# ---------------------------------------------------------------------------


def _generate_compose_yaml(
    agents: tuple[OrchestratorAgent, ...],
    base_dir: str = ".",
    base_image: str | None = None,
) -> str:
    """Generate a docker-compose.yaml (build-only, no runtime config).

    When *base_image* is set, framework agents (non-dapr-agents) get the
    ``BASE_IMAGE`` build arg so they reuse the pre-built base layer.
    """
    lines = ["services:"]
    for agent in agents:
        is_framework = not agent.directory.startswith("dapr-agents")
        if base_image and is_framework:
            lines.extend(
                [
                    f"  {agent.app_id}:",
                    "    build:",
                    f"      context: {base_dir}/{agent.directory}",
                    "      args:",
                    f"        BASE_IMAGE: {base_image}",
                    f"    image: {agent.app_id}:latest",
                ]
            )
        else:
            lines.extend(
                [
                    f"  {agent.app_id}:",
                    f"    build: {base_dir}/{agent.directory}",
                    f"    image: {agent.app_id}:latest",
                ]
            )
    return "\n".join(lines) + "\n"


def _deploy_orchestrator(
    *,
    api_url: str | None,
    api_key: str | None,
    no_browser: bool,
    tag: str,
    namespace: str,
    project: str,
    trigger: str | None = None,
) -> None:
    """Build and deploy all orchestrator agents."""
    total_steps = 7 if trigger else 6
    agents = ORCHESTRATOR_AGENTS
    base_image_name = "diagrid-agent-base:latest"

    # Step 1: Authenticate
    with console.spinner(1, total_steps, "Authenticating..."):
        auth = DeviceCodeAuth(
            api_url=api_url, api_key_flag=api_key, no_browser=no_browser
        )
        auth_ctx = auth.authenticate()
    console.success("Authenticated")

    # Step 2: Fetch connection details for all agents
    with console.spinner(2, total_steps, "Fetching Catalyst connection details..."):
        client = CatalystClient(auth_ctx)
        connections: dict[str, dict[str, str]] = {}
        for agent in agents:
            conn = _get_connection_details(client, project, agent.app_id)
            connections[agent.app_id] = conn
    console.success(f"Connection details fetched for {len(agents)} agents")

    # Step 3: Build shared base image
    with console.spinner(3, total_steps, "Building shared base image..."):
        run("docker", "build", "-t", base_image_name, "base")
    console.success("Shared base image built")

    # Step 4: Build all agent images via docker compose
    with console.spinner(4, total_steps, f"Building {len(agents)} Docker images..."):
        compose_path = Path.cwd() / "docker-compose.yaml"
        if compose_path.exists():
            # Use existing compose file, pass BASE_IMAGE as build arg
            run(
                "docker",
                "compose",
                "build",
                "--build-arg",
                f"BASE_IMAGE={base_image_name}",
            )
        else:
            # Generate a temporary compose file
            compose_yaml = _generate_compose_yaml(agents, base_image=base_image_name)
            compose_path.write_text(compose_yaml)
            try:
                run("docker", "compose", "build")
            finally:
                compose_path.unlink(missing_ok=True)
    console.success("All images built")

    # Step 5: Push images to local registry (parallel)
    with console.spinner(5, total_steps, "Pushing images to local registry..."):
        image_tags = [f"{agent.app_id}:{tag}" for agent in agents]
        registry_tags = push_to_registry_parallel(image_tags)
        # Build a map from original tag to registry tag
        registry_map = dict(zip(image_tags, registry_tags))
    console.success("All images pushed to local registry")

    # Step 6: Deploy K8s manifests
    with console.spinner(6, total_steps, "Deploying to Kubernetes..."):
        for agent in agents:
            conn = connections[agent.app_id]
            otel_block = _otel_env_block(agent.app_id, OTEL_COLLECTOR_ENDPOINT)
            secret_block = _secret_env_block(agent)
            agent_image = registry_map[f"{agent.app_id}:{tag}"]
            manifest = DEPLOYMENT_TEMPLATE.format(
                name=agent.app_id,
                namespace=namespace,
                image=agent_image,
                port=agent.port,
                app_id=conn["app_id"],
                api_token=conn["api_token"],
                http_endpoint=conn["http_endpoint"],
                grpc_endpoint=conn["grpc_endpoint"],
                otel_env_block=otel_block,
                secret_env_block=secret_block,
            )
            apply_stdin(manifest, namespace=namespace)
            rollout_restart(agent.app_id, namespace=namespace)
    console.success(f"Deployed {len(agents)} agents")

    # TODO: Re-enable once Catalyst cloud supports these endpoints.
    # # Step N: Set up piko tunnel for remote traces
    # piko_proc = _setup_piko_tunnel(client, project)
    # # Step N+1: Configure Dapr tracing on Catalyst
    # _configure_dapr_tracing(client, project)
    piko_proc = None

    # Step 7 (optional): Trigger the orchestrator agent
    if trigger:
        with console.spinner(total_steps, total_steps, "Triggering agent..."):
            result = _trigger_agent(
                "event-orchestrator", namespace, trigger, container_port=8007
            )
        console.success("Agent triggered")
        console.info(f"Instance ID: {result.get('instance_id', 'N/A')}")

    summary_lines = [
        f"Project: {project}",
        f"Namespace: {namespace}",
        f"Agents deployed: {len(agents)}",
        "",
    ]
    for agent in agents:
        summary_lines.append(f"  - {agent.app_id} (port {agent.port})")
    if piko_proc:
        summary_lines.append("")
        summary_lines.append("Piko tunnel: running (remote traces → local Grafana)")
    if trigger:
        summary_lines.append("")
        summary_lines.append(f'Prompt: "{trigger}"')

    console.print_summary("Orchestrator Deployment Complete", summary_lines)


# ---------------------------------------------------------------------------
# Piko tunnel for remote trace collection
# ---------------------------------------------------------------------------


def _setup_piko_tunnel(client: CatalystClient, project: str) -> subprocess.Popen | None:  # type: ignore[type-arg]
    """Create an otel-collector AppID and start a piko tunnel.

    Returns the piko subprocess, or None if tunnel setup fails.
    """
    otel_appid = "otel-collector"

    # Ensure the otel-collector AppID exists
    try:
        create_appid(client, project, otel_appid)
    except CatalystAPIError as exc:
        if exc.status_code != 409:
            console.warning(f"Could not create otel-collector AppID: {exc}")
            return None

    # Request tunnel credentials (body required — Go CLI sends {"appHealthCheck": null})
    try:
        resp = client.post(
            f"/projects/{project}/apptunnels/{otel_appid}/connect",
            json_data={},
            params={"appTunnelProvider": "piko"},
            timeout=60.0,
        )
        tunnel_data = resp.json()
    except (CatalystAPIError, Exception) as exc:
        console.warning(f"Could not get tunnel credentials: {exc}")
        return None

    # Extract tunnel info
    endpoint = tunnel_data.get("endpoint", "")
    credentials = tunnel_data.get("credentials", {})
    cert_bundle = tunnel_data.get("certBundle", {})
    tunnel_id = credentials.get("id", "")
    token = credentials.get("token", "")

    if not endpoint or not tunnel_id or not token:
        console.warning("Incomplete tunnel credentials — skipping tunnel")
        return None

    # Write mTLS cert files to temp dir
    cert_dir = tempfile.mkdtemp(prefix="diagrid-piko-")
    cert_path = _write_b64_file(cert_dir, "cert.pem", cert_bundle.get("cert", ""))
    key_path = _write_b64_file(cert_dir, "key.pem", cert_bundle.get("key", ""))
    ca_path = _write_b64_file(cert_dir, "ca.pem", cert_bundle.get("ca", ""))

    # Build piko command
    cmd = [
        "piko",
        "agent",
        "listen",
        "--endpoint",
        endpoint,
        "--token",
        token,
        "--endpoint-id",
        tunnel_id,
        "--upstream-addr",
        f"localhost:{OTEL_COLLECTOR_NODEPORT_GRPC}",
    ]
    if cert_path and key_path and ca_path:
        cmd.extend(
            ["--tls-cert", cert_path, "--tls-key", key_path, "--tls-ca", ca_path]
        )

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        console.warning("piko binary not found — skipping tunnel")
        return None

    # Register cleanup on exit
    def _cleanup() -> None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        # Clean up temp cert files
        shutil.rmtree(cert_dir, ignore_errors=True)

    atexit.register(_cleanup)

    # Start token refresh thread
    _start_token_refresh(client, project, otel_appid, proc, cmd, cert_dir)

    return proc


def _write_b64_file(directory: str, filename: str, b64_data: str) -> str:
    """Decode base64 data and write to a file. Returns the file path."""
    if not b64_data:
        return ""
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return path


def _start_token_refresh(
    client: CatalystClient,
    project: str,
    otel_appid: str,
    proc: subprocess.Popen,  # type: ignore[type-arg]
    cmd: list[str],
    cert_dir: str,
) -> None:
    """Start a daemon thread that refreshes the tunnel token every 10 minutes."""

    def _refresh_loop() -> None:
        while True:
            time.sleep(600)  # 10 minutes
            if proc.poll() is not None:
                return  # piko has exited

            try:
                resp = client.post(
                    f"/projects/{project}/apptunnels/{otel_appid}/connect",
                    json_data={},
                    params={"appTunnelProvider": "piko"},
                    timeout=60.0,
                )
                tunnel_data = resp.json()
                new_token = tunnel_data.get("credentials", {}).get("token", "")
                new_endpoint = tunnel_data.get("endpoint", "")
                new_id = tunnel_data.get("credentials", {}).get("id", "")

                if not new_token:
                    continue

                # Update cert files if provided
                new_bundle = tunnel_data.get("certBundle", {})
                if new_bundle.get("cert"):
                    _write_b64_file(cert_dir, "cert.pem", new_bundle["cert"])
                if new_bundle.get("key"):
                    _write_b64_file(cert_dir, "key.pem", new_bundle["key"])
                if new_bundle.get("ca"):
                    _write_b64_file(cert_dir, "ca.pem", new_bundle["ca"])

                # Rebuild command with new token
                new_cmd = list(cmd)
                for i, arg in enumerate(new_cmd):
                    if arg == "--token" and i + 1 < len(new_cmd):
                        new_cmd[i + 1] = new_token
                    if arg == "--endpoint" and i + 1 < len(new_cmd) and new_endpoint:
                        new_cmd[i + 1] = new_endpoint
                    if arg == "--endpoint-id" and i + 1 < len(new_cmd) and new_id:
                        new_cmd[i + 1] = new_id

                # Restart piko with new credentials
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

                subprocess.Popen(
                    new_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Swap reference (the atexit handler still holds the old ref,
                # but that's acceptable — worst case double-terminate)
                # Update the outer proc reference via nonlocal isn't possible
                # in a clean way, so we just let atexit handle the old one.

            except Exception:
                # Refresh failed — keep the existing tunnel alive
                pass

    thread = threading.Thread(target=_refresh_loop, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Dapr tracing configuration
# ---------------------------------------------------------------------------


def _configure_dapr_tracing(client: CatalystClient, project: str) -> bool:
    """Create a Dapr tracing Configuration on the Catalyst project.

    Returns ``True`` on success (or if already exists), ``False`` on failure.
    """
    # Get the otel-collector AppID endpoint
    try:
        get_appid(client, project, "otel-collector")
    except CatalystAPIError:
        console.warning(
            "Could not fetch otel-collector AppID — skipping tracing config"
        )
        return False

    # Build the OTEL tunnel endpoint from project endpoints
    proj = get_project(client, project)
    otel_endpoint = ""
    if proj.status and proj.status.endpoints:
        if proj.status.endpoints.grpc and proj.status.endpoints.grpc.url:
            otel_endpoint = proj.status.endpoints.grpc.url

    if not otel_endpoint:
        console.warning("No gRPC endpoint available — skipping tracing config")
        return False

    tracing_config = {
        "apiVersion": "dapr.diagrid.io/v1beta1",
        "kind": "Configuration",
        "metadata": {"name": "tracing"},
        "spec": {
            "tracing": {
                "samplingRate": "1",
                "otel": {
                    "endpointAddress": otel_endpoint,
                    "protocol": "grpc",
                    "isSecure": False,
                },
            }
        },
    }
    try:
        client.post(
            f"/projects/{project}/configurations",
            json_data=tracing_config,
            api_group=CatalystClient.DAPR_API_GROUP,
        )
    except CatalystAPIError as exc:
        if exc.status_code == 409:
            console.info("Tracing configuration already exists")
            return True
        console.warning(f"Could not create tracing config: {exc}")
        return False
    return True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_connection_details(
    client: CatalystClient, project_name: str, appid_name: str
) -> dict[str, str]:
    """Fetch the Dapr connection env vars from Catalyst."""
    try:
        proj = get_project(client, project_name)
    except CatalystAPIError as exc:
        if exc.status_code == 404:
            raise click.ClickException(
                f"Project '{project_name}' not found. Create it with: diagridpy init"
            ) from exc
        raise

    try:
        appid = get_appid(client, project_name, appid_name)
    except CatalystAPIError as exc:
        if exc.status_code == 404:
            raise click.ClickException(
                f"AppID '{appid_name}' not found in project '{project_name}'. "
                f"Use --app-id to specify an existing AppID, "
                f"or create it with: diagridpy init"
            ) from exc
        raise

    http_endpoint = ""
    grpc_endpoint = ""
    if proj.status and proj.status.endpoints:
        if proj.status.endpoints.http and proj.status.endpoints.http.url:
            http_endpoint = proj.status.endpoints.http.url
        if proj.status.endpoints.grpc and proj.status.endpoints.grpc.url:
            grpc_endpoint = proj.status.endpoints.grpc.url

    api_token = ""
    if appid.status and appid.status.api_token:
        api_token = appid.status.api_token

    if not api_token:
        raise click.ClickException(
            f"AppID '{appid_name}' has no API token yet. "
            "It may still be provisioning — try again in a few seconds."
        )
    if not http_endpoint:
        raise click.ClickException(
            f"Project '{project_name}' has no HTTP endpoint yet. "
            "It may still be provisioning — try again in a few seconds."
        )

    return {
        "app_id": appid_name,
        "api_token": api_token,
        "http_endpoint": http_endpoint,
        "grpc_endpoint": grpc_endpoint,
    }


def _wait_for_rollout(name: str, namespace: str, timeout: int = 120) -> None:
    """Wait for a deployment rollout to complete."""
    run(
        "kubectl",
        "rollout",
        "status",
        f"deployment/{name}",
        "-n",
        namespace,
        f"--timeout={timeout}s",
    )


def _find_free_port() -> int:
    """Find a free local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _trigger_agent(
    name: str,
    namespace: str,
    prompt: str,
    *,
    container_port: int = 5001,
    timeout: int = 120,
) -> dict:  # type: ignore[type-arg]
    """Wait for the agent pod, port-forward, and POST to /run.

    Returns the JSON response from the agent.
    """
    # Wait for deployment to be ready
    _wait_for_rollout(name, namespace)

    local_port = _find_free_port()

    # Start port-forward in background
    pf = subprocess.Popen(
        [
            "kubectl",
            "port-forward",
            f"deployment/{name}",
            f"{local_port}:{container_port}",
            "-n",
            namespace,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for port-forward to be ready
        _wait_for_port(local_port, timeout=15)

        # Wait for the agent's HTTP server to be ready, then send the task
        url = f"http://localhost:{local_port}/agent/run"
        return _post_with_retry(url, {"task": prompt}, timeout=timeout)
    finally:
        pf.terminate()
        pf.wait(timeout=5)


def _post_with_retry(
    url: str,
    json_data: dict,  # type: ignore[type-arg]
    *,
    timeout: int = 120,
    startup_timeout: int = 60,
) -> dict:  # type: ignore[type-arg]
    """POST to the agent, retrying until the server is ready."""
    deadline = time.monotonic() + startup_timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=float(timeout)) as client:
                resp = client.post(url, json=json_data)
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]
        except (
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            httpx.HTTPStatusError,
        ) as exc:
            last_error = exc
            time.sleep(3)

    raise last_error or TimeoutError(f"Agent not ready after {startup_timeout}s")


def _wait_for_port(port: int, *, timeout: int = 15) -> None:
    """Wait until a local port is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Port {port} not ready after {timeout}s")
