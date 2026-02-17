"""diagridpy deploy command — build and deploy an agent to the kind cluster."""

from __future__ import annotations

import json
import socket
import subprocess
import time

import click
import httpx

from diagrid.cli.infra.docker import build_image, load_into_kind
from diagrid.cli.infra.kubectl import apply_stdin
from diagrid.cli.utils import console
from diagrid.cli.utils.process import CommandError, run, run_capture
from diagrid.core.auth.device_code import DeviceCodeAuth
from diagrid.core.catalyst.appids import get_appid
from diagrid.core.catalyst.client import CatalystClient
from diagrid.core.catalyst.projects import get_project
from diagrid.core.config.constants import DEFAULT_KIND_CLUSTER, DEFAULT_NAMESPACE


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
"""


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
def deploy(
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
) -> None:
    """Build and deploy an agent to the kind cluster."""
    appid_name = app_id or f"{project}-agent"
    total_steps = 6 if trigger else 5

    try:
        # Step 1: Authenticate
        console.step(1, total_steps, "Authenticating...")
        auth = DeviceCodeAuth(api_key_flag=api_key, no_browser=no_browser)
        auth_ctx = auth.authenticate()
        console.success("Authenticated")

        # Step 2: Fetch Catalyst connection details
        console.step(2, total_steps, "Fetching Catalyst connection details...")
        client = CatalystClient(auth_ctx)
        conn = _get_connection_details(client, project, appid_name)
        console.success(f"AppID '{appid_name}' ready")

        # Step 3: Build Docker image
        console.step(3, total_steps, f"Building Docker image {image}:{tag}...")
        full_tag = build_image(image, tag)
        console.success(f"Image built: {full_tag}")

        # Step 4: Load into kind
        console.step(4, total_steps, "Loading image into kind cluster...")
        load_into_kind(full_tag, DEFAULT_KIND_CLUSTER)
        console.success("Image loaded into kind")

        # Step 5: Apply Kubernetes manifest
        console.step(5, total_steps, "Deploying to Kubernetes...")
        if kube_context:
            run("kubectl", "config", "use-context", kube_context)

        manifest = DEPLOYMENT_TEMPLATE.format(
            name=image,
            namespace=namespace,
            image=full_tag,
            port=port,
            app_id=conn["app_id"],
            api_token=conn["api_token"],
            http_endpoint=conn["http_endpoint"],
            grpc_endpoint=conn["grpc_endpoint"],
        )
        apply_stdin(manifest, namespace=namespace)
        console.success("Agent deployed")

        # Step 6 (optional): Trigger the agent
        if trigger:
            console.step(6, total_steps, "Triggering agent...")
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

    except CommandError as exc:
        console.error(str(exc))
        raise SystemExit(1) from exc
    except Exception as exc:
        console.error(f"Deployment failed: {exc}")
        raise SystemExit(1) from exc


def _get_connection_details(
    client: CatalystClient, project_name: str, appid_name: str
) -> dict[str, str]:
    """Fetch the Dapr connection env vars from Catalyst."""
    proj = get_project(client, project_name)
    appid = get_appid(client, project_name, appid_name)

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

        # Wait for the agent's HTTP server to be ready, then send the prompt
        url = f"http://localhost:{local_port}/agent/run"
        return _post_with_retry(url, {"prompt": prompt}, timeout=timeout)
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
    from diagrid.cli.utils import console

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
