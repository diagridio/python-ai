"""Kind cluster provisioning with port mappings and local registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

from diagrid.cli.infra.kubectl import apply_stdin
from diagrid.cli.utils.process import CommandError, has_command, run, run_capture
from diagrid.core.config.constants import (
    DEFAULT_KIND_CLUSTER,
    KIND_CONFIG_YAML,
    KIND_NODE_IMAGE,
    KIND_REGISTRY_NAME,
    KIND_REGISTRY_PORT,
    REGISTRY_MIRRORS,
)


def kind_available() -> bool:
    """Check if kind is available."""
    return has_command("kind")


def cluster_exists(name: str = DEFAULT_KIND_CLUSTER) -> bool:
    """Check if a kind cluster exists."""
    if not kind_available():
        return False
    output = run_capture("kind", "get", "clusters")
    return name in output.splitlines()


def _ensure_container(name: str, run_args: list[str]) -> None:
    """Ensure a Docker container is running, handling 3 states.

    - **Running** — no-op.
    - **Stopped** — ``docker start <name>``.
    - **Missing** (docker inspect fails) — ``docker run`` with *run_args*.
    """
    try:
        result = run_capture(
            "docker",
            "inspect",
            "-f",
            "{{.State.Running}}",
            name,
        )
        if result.strip() == "true":
            return
        # Container exists but is stopped — restart it.
        run("docker", "start", name)
    except CommandError:
        # Container does not exist — create it.
        run("docker", "run", *run_args)


def _ensure_registry() -> None:
    """Start the local Docker registry if not running."""
    _ensure_container(
        KIND_REGISTRY_NAME,
        [
            "-d",
            "--restart=always",
            "-p",
            f"127.0.0.1:{KIND_REGISTRY_PORT}:5000",
            "--network",
            "bridge",
            "--name",
            KIND_REGISTRY_NAME,
            "registry:2",
        ],
    )


def _ensure_mirrors() -> None:
    """Start transparent mirrors for public registries."""
    for host, url in REGISTRY_MIRRORS:
        name = host.replace(".", "-")
        mirror_name = f"kind-mirror-{name}"

        _ensure_container(
            mirror_name,
            [
                "-d",
                "--restart=always",
                "--network",
                "bridge",
                "--name",
                mirror_name,
                "-e",
                f"REGISTRY_PROXY_REMOTEURL={url}",
                "registry:2",
            ],
        )


def _configure_registry_on_nodes(cluster_name: str) -> None:
    """Configure containerd registry access on all cluster nodes."""
    nodes_output = run_capture("kind", "get", "nodes", "--name", cluster_name)
    nodes = [n.strip() for n in nodes_output.splitlines() if n.strip()]

    for node in nodes:
        # Local registry
        reg_dir = f"/etc/containerd/certs.d/localhost:{KIND_REGISTRY_PORT}"
        run("docker", "exec", node, "mkdir", "-p", reg_dir)
        hosts_toml = f'[host."http://{KIND_REGISTRY_NAME}:5000"]'
        run(
            "docker",
            "exec",
            "-i",
            node,
            "sh",
            "-c",
            f"echo '{hosts_toml}' > {reg_dir}/hosts.toml",
        )

        # Public registry mirrors
        for host, _ in REGISTRY_MIRRORS:
            name = host.replace(".", "-")
            mirror_name = f"kind-mirror-{name}"
            mirror_dir = f"/etc/containerd/certs.d/{host}"
            run("docker", "exec", node, "mkdir", "-p", mirror_dir)
            mirror_toml = f'[host."http://{mirror_name}:5000"]'
            run(
                "docker",
                "exec",
                "-i",
                node,
                "sh",
                "-c",
                f"echo '{mirror_toml}' > {mirror_dir}/hosts.toml",
            )


def _connect_registries_to_network(cluster_name: str) -> None:
    """Connect registries to the kind Docker network."""
    # Connect local registry
    network_info = run_capture(
        "docker",
        "inspect",
        "-f",
        "{{json .NetworkSettings.Networks.kind}}",
        KIND_REGISTRY_NAME,
    )
    if network_info.strip() == "null":
        run("docker", "network", "connect", "kind", KIND_REGISTRY_NAME)

    # Connect mirrors
    for host, _ in REGISTRY_MIRRORS:
        name = host.replace(".", "-")
        mirror_name = f"kind-mirror-{name}"
        network_info = run_capture(
            "docker",
            "inspect",
            "-f",
            "{{json .NetworkSettings.Networks.kind}}",
            mirror_name,
        )
        if network_info.strip() == "null":
            run("docker", "network", "connect", "kind", mirror_name)


def _apply_registry_configmap() -> None:
    """Document the local registry via a ConfigMap."""
    configmap = (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: local-registry-hosting\n"
        "  namespace: kube-public\n"
        "data:\n"
        "  localRegistryHosting.v1: |\n"
        f'    host: "localhost:{KIND_REGISTRY_PORT}"\n'
        '    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"\n'
    )
    apply_stdin(configmap)


def ensure_registry_config(name: str = DEFAULT_KIND_CLUSTER) -> None:
    """Ensure local registry is running, connected, and configured on all nodes.

    Idempotent — safe to call on existing clusters.
    """
    _ensure_registry()
    _ensure_mirrors()
    _configure_registry_on_nodes(name)
    _connect_registries_to_network(name)
    _apply_registry_configmap()


def create_cluster(name: str = DEFAULT_KIND_CLUSTER) -> None:
    """Create a kind cluster with port mappings, local registry, and mirrors."""
    if cluster_exists(name):
        return

    # Start registry and mirrors
    _ensure_registry()
    _ensure_mirrors()

    # Write kind config to temp file
    config_content = KIND_CONFIG_YAML.format(
        cluster_name=name,
        node_image=KIND_NODE_IMAGE,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        run("kind", "create", "cluster", "--name", name, "--config", config_path)
    finally:
        Path(config_path).unlink(missing_ok=True)

    # Configure registry access on nodes
    _configure_registry_on_nodes(name)

    # Connect registries to kind network
    _connect_registries_to_network(name)

    # Document local registry
    _apply_registry_configmap()


def ensure_cluster(name: str = DEFAULT_KIND_CLUSTER) -> str:
    """Ensure a kind cluster exists and set kubectl context.

    Returns the kubectl context name.
    """
    if kind_available():
        create_cluster(name)
        context = f"kind-{name}"
        run("kubectl", "config", "use-context", context)
        return context
    else:
        # Fall back to current kubectl context
        return run_capture("kubectl", "config", "current-context")
