"""diagrid init command — initialize a local agent development environment."""

from __future__ import annotations

import os
import shutil
import tempfile

import click

from diagrid.cli.infra.helm import install_dapr_agents
from diagrid.cli.infra.kind import cluster_exists, create_cluster, kind_available
from diagrid.cli.utils import console
from diagrid.cli.utils.process import CommandError, run, run_capture
from diagrid.core.auth.device_code import DeviceCodeAuth
from diagrid.core.catalyst.appids import create_appid
from diagrid.core.catalyst.client import CatalystAPIError, CatalystClient
from diagrid.core.catalyst.projects import create_project
from diagrid.core.config.constants import (
    DEFAULT_KIND_CLUSTER,
    ENV_OPENAI_API_KEY,
    QUICKSTART_REPO_URL,
    QUICKSTART_SUBDIR,
)


@click.command()
@click.argument("project_name", default="agents-quickstart")
@click.option("--api-key", default=None, help="Diagrid API key")
@click.option("--no-browser", is_flag=True, help="Don't open browser for auth")
@click.option(
    "--openai-api-key",
    default=None,
    help="OpenAI API key (or set OPENAI_API_KEY env)",
)
def init(
    project_name: str,
    api_key: str | None,
    no_browser: bool,
    openai_api_key: str | None,
) -> None:
    """Initialize a local agent development environment."""
    total_steps = 7

    try:
        # Step 1: Authenticate
        console.step(1, total_steps, "Authenticating...")
        auth = DeviceCodeAuth(api_key_flag=api_key, no_browser=no_browser)
        auth_ctx = auth.authenticate()
        console.success("Authenticated successfully")

        # Step 2: Get LLM API key
        console.step(2, total_steps, "Checking LLM API key...")
        llm_key = openai_api_key or os.environ.get(ENV_OPENAI_API_KEY)
        if llm_key:
            console.success("LLM API key found")
        else:
            llm_key = click.prompt("Enter your OPENAI_API_KEY", hide_input=True)
            if not llm_key:
                raise click.ClickException("OPENAI_API_KEY is required")
            console.success("LLM API key configured")

        # Step 3: Create project
        console.step(3, total_steps, f"Creating project '{project_name}'...")
        client = CatalystClient(auth_ctx)
        try:
            create_project(client, project_name, agent_infrastructure_enabled=True)
            console.success(f"Project '{project_name}' created")
        except CatalystAPIError as exc:
            if exc.status_code == 409:
                console.info(f"Project '{project_name}' already exists, continuing")
            else:
                raise

        # Step 4: Clone quickstart
        console.step(4, total_steps, "Cloning quickstart template...")
        _clone_quickstart(project_name)
        console.success(f"Quickstart cloned to ./{project_name}/")

        # Step 5: Provision cluster
        console.step(5, total_steps, "Provisioning Kubernetes cluster...")
        _provision_cluster()

        # Step 6: Deploy helm chart
        console.step(6, total_steps, "Installing dapr-agents helm chart...")
        install_dapr_agents(llm_key)
        console.success("Helm chart installed")

        # Step 7: Create AppID
        console.step(7, total_steps, "Creating AppID...")
        try:
            create_appid(client, project_name, f"{project_name}-agent")
            console.success("AppID created")
        except CatalystAPIError as exc:
            if exc.status_code == 409:
                console.info("AppID already exists, continuing")
            else:
                raise

        # Print summary
        console.print_summary(
            "Initialization Complete",
            [
                f"Project: {project_name}",
                f"Directory: ./{project_name}/",
                "",
                "Next steps:",
                f"  cd {project_name}",
                "  diagridpy deploy",
            ],
        )

    except CommandError as exc:
        console.error(str(exc))
        raise SystemExit(1) from exc
    except click.ClickException:
        raise
    except Exception as exc:
        console.error(f"Initialization failed: {exc}")
        raise SystemExit(1) from exc


def _clone_quickstart(project_name: str) -> None:
    """Clone the quickstart repo and copy the Python agent template."""
    with tempfile.TemporaryDirectory() as tmp:
        run(
            "git",
            "clone",
            "--depth",
            "1",
            QUICKSTART_REPO_URL,
            tmp,
        )

        source = os.path.join(tmp, QUICKSTART_SUBDIR)
        if not os.path.isdir(source):
            raise click.ClickException(
                f"Quickstart template not found at {QUICKSTART_SUBDIR}"
            )

        dest = project_name
        if os.path.exists(dest):
            raise click.ClickException(f"Directory '{dest}' already exists")

        shutil.copytree(source, dest)

        # Patch agent code to read port from APP_PORT env var
        _patch_agent_port(dest)

        # Initialize a fresh git repo
        try:
            run("git", "init", cwd=dest)
        except CommandError:
            console.warning("Could not initialize git repo")


def _patch_agent_port(dest: str) -> None:
    """Patch the agent's main.py to read port from APP_PORT env var."""
    main_py = os.path.join(dest, "main.py")
    if not os.path.isfile(main_py):
        return

    with open(main_py) as f:
        content = f.read()

    # Replace hardcoded port with env var lookup
    # e.g. runner.serve(travel_assistant, port=5001)
    # becomes runner.serve(travel_assistant, port=int(os.environ.get("APP_PORT", "5001")))
    import re

    content = re.sub(
        r"(runner\.serve\([^)]*port=)\d+(\))",
        r'\1int(os.environ.get("APP_PORT", "5001"))\2',
        content,
    )

    # Ensure os is imported
    if "import os" not in content:
        content = "import os\n" + content

    with open(main_py, "w") as f:
        f.write(content)


def _provision_cluster() -> None:
    """Provision a kind cluster or verify existing kubectl context."""
    if kind_available():
        if not cluster_exists(DEFAULT_KIND_CLUSTER):
            console.info(f"Creating kind cluster '{DEFAULT_KIND_CLUSTER}'...")
            create_cluster(DEFAULT_KIND_CLUSTER)
        else:
            console.info(f"Kind cluster '{DEFAULT_KIND_CLUSTER}' already exists")
        run(
            "kubectl",
            "config",
            "use-context",
            f"kind-{DEFAULT_KIND_CLUSTER}",
        )
        console.success("Cluster ready")
    else:
        console.warning("kind not found, checking current kubectl context...")
        try:
            ctx = run_capture("kubectl", "config", "current-context")
            console.info(f"Using kubectl context: {ctx}")
        except CommandError:
            raise click.ClickException(
                "No kubectl context found and kind is not installed. "
                "Please install kind or configure kubectl."
            )
