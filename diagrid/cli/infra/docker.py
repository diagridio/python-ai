"""Docker image build and kind load operations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from diagrid.cli.utils.process import run
from diagrid.core.config.constants import (
    DEFAULT_KIND_CLUSTER,
    KIND_REGISTRY_PORT,
)

_LOCAL_REGISTRY = f"localhost:{KIND_REGISTRY_PORT}"


def build_image(
    image: str,
    tag: str,
    context: str = ".",
) -> str:
    """Build a Docker image. Returns the full image:tag string."""
    full_tag = f"{image}:{tag}"
    run("docker", "build", "-t", full_tag, context)
    return full_tag


def push_to_registry(image_tag: str) -> str:
    """Tag and push an image to the local kind registry.

    Returns the registry-qualified image reference.
    """
    registry_tag = f"{_LOCAL_REGISTRY}/{image_tag}"
    run("docker", "tag", image_tag, registry_tag)
    run("docker", "push", registry_tag)
    return registry_tag


def push_to_registry_parallel(image_tags: list[str]) -> list[str]:
    """Push multiple images to the local kind registry in parallel.

    Returns the registry-qualified image references.
    """
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(push_to_registry, tag): tag for tag in image_tags}
        results = []
        for future in as_completed(futures):
            results.append(future.result())
        return results


def load_into_kind(
    image_tag: str,
    cluster: str = DEFAULT_KIND_CLUSTER,
) -> None:
    """Load a Docker image into a kind cluster."""
    run("kind", "load", "docker-image", image_tag, "--name", cluster)


def load_into_kind_parallel(
    image_tags: list[str],
    cluster: str = DEFAULT_KIND_CLUSTER,
) -> None:
    """Load multiple Docker images into a kind cluster in parallel."""
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(load_into_kind, tag, cluster): tag for tag in image_tags
        }
        for future in as_completed(futures):
            future.result()  # raises if any load failed
