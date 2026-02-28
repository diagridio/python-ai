"""diagrid init command — initialize a local agent development environment."""

from __future__ import annotations

import os
import re
import shutil
import tempfile

import click

from diagrid.cli.infra.helm import install_dapr_agents
from diagrid.cli.utils.deps import preflight_check
from diagrid.cli.infra.kind import (
    cluster_exists,
    create_cluster,
    ensure_registry_config,
    kind_available,
)
from diagrid.cli.utils import console
from diagrid.cli.utils.process import CommandError, run, run_capture
from diagrid.core.auth.device_code import DeviceCodeAuth
from diagrid.core.catalyst.appids import create_appid
from diagrid.core.catalyst.client import CatalystAPIError, CatalystClient
from diagrid.core.catalyst.projects import create_project, list_projects
from diagrid.core.config.constants import (
    DEFAULT_KIND_CLUSTER,
    ENV_OPENAI_API_KEY,
    ORCHESTRATOR_AGENTS,
    QUICKSTART_REPO_URL,
    QUICKSTART_SUBDIRS,
)


SUPPORTED_FRAMEWORKS = list(QUICKSTART_SUBDIRS.keys())


@click.command()
@click.argument("project_name", default="agents-quickstart")
@click.option("--api-key", default=None, help="Diagrid API key")
@click.option("--no-browser", is_flag=True, help="Don't open browser for auth")
@click.option(
    "--openai-api-key",
    default=None,
    help="OpenAI API key (or set OPENAI_API_KEY env)",
)
@click.option(
    "--google-api-key",
    default=None,
    help="Google API key for ADK framework (or set GOOGLE_API_KEY env)",
)
@click.option(
    "--framework",
    type=click.Choice(SUPPORTED_FRAMEWORKS, case_sensitive=False),
    default="dapr-agents",
    help="Agent framework to use (default: dapr-agents)",
)
@click.option(
    "--existing-path",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Path to an existing agent project (skips clone step)",
)
@click.option(
    "--existing-project",
    is_flag=True,
    default=False,
    help="Select an existing Catalyst project instead of creating a new one",
)
@click.pass_context
def init(
    ctx: click.Context,
    project_name: str,
    api_key: str | None,
    no_browser: bool,
    openai_api_key: str | None,
    google_api_key: str | None,
    framework: str,
    existing_path: str | None,
    existing_project: bool,
) -> None:
    """Initialize a local agent development environment."""
    if existing_path and framework != "dapr-agents":
        raise click.ClickException(
            "--existing-path and --framework cannot be used together"
        )

    preflight_check()

    api_url: str | None = ctx.obj.get("api_url") if ctx.obj else None
    total_steps = 7

    try:
        # Step 1: Authenticate
        console.step(1, total_steps, "Authenticating...")
        auth = DeviceCodeAuth(
            api_url=api_url, api_key_flag=api_key, no_browser=no_browser
        )
        auth_ctx = auth.authenticate()
        console.success("Authenticated successfully")

        # Step 2: Get LLM API key
        console.step(2, total_steps, "Checking LLM API key...")
        google_key = ""
        if framework == "orchestrator":
            # Orchestrator runs all frameworks; OpenAI key is required, Google is optional
            llm_key = openai_api_key or os.environ.get(ENV_OPENAI_API_KEY)
            if not llm_key:
                llm_key = click.prompt("Enter your OPENAI_API_KEY", hide_input=True)
            google_key = google_api_key or os.environ.get("GOOGLE_API_KEY") or ""
            if not google_key:
                console.warning(
                    "GOOGLE_API_KEY not set — ADK agent will use OpenAI via Dapr conversation API"
                )
            console.success("LLM API key(s) configured")
        elif framework == "adk":
            key_name = "GOOGLE_API_KEY"
            llm_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
            if llm_key:
                google_key = llm_key
                console.success("LLM API key found")
            else:
                llm_key = click.prompt(f"Enter your {key_name}", hide_input=True)
                if not llm_key:
                    raise click.ClickException(f"{key_name} is required")
                google_key = llm_key
                console.success("LLM API key configured")
        else:
            key_name = "OPENAI_API_KEY"
            llm_key = openai_api_key or os.environ.get(ENV_OPENAI_API_KEY)
            if llm_key:
                console.success("LLM API key found")
            else:
                llm_key = click.prompt(f"Enter your {key_name}", hide_input=True)
                if not llm_key:
                    raise click.ClickException(f"{key_name} is required")
                console.success("LLM API key configured")

        # Step 3: Create or select project
        client = CatalystClient(auth_ctx)
        if existing_project:
            console.step(3, total_steps, "Listing existing projects...")
            projects = list_projects(client)
            if not projects:
                raise click.ClickException(
                    "No existing projects found. Run without --existing-project "
                    "to create a new one."
                )
            project_names = [
                p.metadata.name for p in projects if p.metadata and p.metadata.name
            ]
            if not project_names:
                raise click.ClickException("No projects found with valid names.")
            project_name = click.prompt(
                "Select a project",
                type=click.Choice(project_names, case_sensitive=False),
            )
            console.success(f"Using existing project '{project_name}'")
        else:
            console.step(3, total_steps, f"Creating project '{project_name}'...")
            try:
                create_project(client, project_name, agent_infrastructure_enabled=True)
                console.success(f"Project '{project_name}' created")
            except CatalystAPIError as exc:
                if exc.status_code == 409:
                    console.info(f"Project '{project_name}' already exists, continuing")
                else:
                    raise

        # Step 4: Clone quickstart (or use existing path)
        if existing_path:
            console.step(4, total_steps, "Using existing project...")
            project_dir = existing_path
            console.success(f"Using existing project at {project_dir}")
        else:
            console.step(4, total_steps, "Cloning quickstart template...")
            cloned = _clone_quickstart(project_name, framework)
            project_dir = project_name
            if cloned:
                console.success(f"Quickstart cloned to ./{project_name}/")
            else:
                console.info(
                    f"Directory './{project_name}/' already exists, skipping clone"
                )

        # Step 5: Provision cluster
        console.step(5, total_steps, "Provisioning Kubernetes cluster...")
        _provision_cluster()

        # Step 6: Deploy helm chart
        console.step(6, total_steps, "Installing catalyst-agents helm chart...")
        install_dapr_agents(llm_key, google_api_key=google_key)
        console.success("Helm chart installed")

        # Step 7: Create AppID(s)
        if framework == "orchestrator":
            console.step(7, total_steps, "Creating AppIDs...")
            for agent in ORCHESTRATOR_AGENTS:
                try:
                    create_appid(client, project_name, agent.app_id)
                except CatalystAPIError as exc:
                    if exc.status_code != 409:
                        raise
            console.success(f"Created {len(ORCHESTRATOR_AGENTS)} AppIDs")
        else:
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
        if framework == "orchestrator":
            console.print_summary(
                "Initialization Complete",
                [
                    f"Project: {project_name}",
                    f"Directory: ./{project_dir}/",
                    "",
                    "Next steps:",
                    f"  cd {project_dir}",
                    "  diagrid dev run -f dev-python-orchestration.yaml",
                ],
            )
        else:
            console.print_summary(
                "Initialization Complete",
                [
                    f"Project: {project_name}",
                    f"Directory: ./{project_dir}/",
                    "",
                    "Next steps:",
                    f"  cd {project_dir}",
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


def _clone_quickstart(project_name: str, framework: str) -> bool:
    """Clone the quickstart repo and copy the framework template.

    Returns ``True`` if the quickstart was cloned, ``False`` if the
    target directory already exists (skip).
    """
    subdir = QUICKSTART_SUBDIRS.get(framework)
    if not subdir:
        raise click.ClickException(f"Unknown framework: {framework}")

    dest = project_name
    if os.path.exists(dest):
        return False

    with tempfile.TemporaryDirectory() as tmp:
        run(
            "git",
            "clone",
            "--depth",
            "1",
            QUICKSTART_REPO_URL,
            tmp,
        )

        source = os.path.join(tmp, subdir)
        if not os.path.isdir(source):
            raise click.ClickException(f"Quickstart template not found at {subdir}")

        shutil.copytree(source, dest)

        # Patch agent code to read port from APP_PORT env var (dapr-agents only)
        if framework == "dapr-agents":
            _patch_agent_port(dest)

        # Initialize a fresh git repo
        try:
            run("git", "init", cwd=dest)
        except CommandError:
            console.warning("Could not initialize git repo")

    return True


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
            ensure_registry_config(DEFAULT_KIND_CLUSTER)
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
