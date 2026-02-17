"""Docker image build and kind load operations."""

from __future__ import annotations

from diagrid.cli.utils.process import run
from diagrid.core.config.constants import DEFAULT_KIND_CLUSTER


def build_image(
    image: str,
    tag: str,
    context: str = ".",
) -> str:
    """Build a Docker image. Returns the full image:tag string."""
    full_tag = f"{image}:{tag}"
    run("docker", "build", "-t", full_tag, context)
    return full_tag


def load_into_kind(
    image_tag: str,
    cluster: str = DEFAULT_KIND_CLUSTER,
) -> None:
    """Load a Docker image into a kind cluster."""
    run("kind", "load", "docker-image", image_tag, "--name", cluster)
