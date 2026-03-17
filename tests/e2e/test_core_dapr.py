"""E2E tests for core Dapr infrastructure used by all frameworks.

Tests state store CRUD, component discovery, and workflow naming utilities.
These tests require a Dapr sidecar but not an LLM backend.
"""

import logging
import uuid

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test A: State store CRUD
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_state_store_crud() -> None:
    """Test DaprStateStore save, get, and delete operations.

    Uses the default ``statestore`` component created by ``dapr init``.
    """
    from diagrid.agent.core.state import DaprStateStore

    store = DaprStateStore(store_name="statestore")
    test_key = f"e2e-test-{uuid.uuid4().hex[:8]}"
    test_value = {"data": "hello", "nested": {"count": 42}}

    try:
        # Save
        store.save(test_key, test_value)

        # Get
        result = store.get(test_key)
        assert result is not None, "get returned None after save"
        assert result == test_value, f"expected {test_value}, got {result}"

        # Overwrite
        updated_value = {"data": "updated", "nested": {"count": 99}}
        store.save(test_key, updated_value)
        result = store.get(test_key)
        assert result == updated_value, f"expected {updated_value}, got {result}"

        # Delete
        store.delete(test_key)
        result = store.get(test_key)
        assert result is None, f"expected None after delete, got {result}"
    finally:
        # Clean up in case of assertion failure
        try:
            store.delete(test_key)
        except Exception as exc:
            logger.debug("Cleanup delete failed for key %s: %s", test_key, exc)
        store.close()


@pytest.mark.integration
def test_state_store_get_nonexistent_key() -> None:
    """Test that getting a nonexistent key returns None."""
    from diagrid.agent.core.state import DaprStateStore

    store = DaprStateStore(store_name="statestore")
    try:
        result = store.get(f"nonexistent-{uuid.uuid4().hex}")
        assert result is None, f"expected None for nonexistent key, got {result}"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test B: Component discovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_component_discovery() -> None:
    """Test that Dapr component discovery completes without error.

    After ``dapr init``, the default components are named ``statestore``
    and ``pubsub`` (not ``agent-memory`` / ``agent-pubsub``), so the
    discovered fields will be None. The test validates that discovery
    runs successfully and returns a valid object.
    """
    from diagrid.agent.core.discovery import (
        DiscoveredComponents,
        _reset_discovery_cache,
        discover_components,
    )

    # Reset cache so we get a fresh discovery
    _reset_discovery_cache()

    discovered = discover_components()
    assert discovered is not None, "discover_components returned None"
    assert isinstance(discovered, DiscoveredComponents)

    # Reset cache after test to avoid polluting other tests
    _reset_discovery_cache()


# ---------------------------------------------------------------------------
# Test C: Workflow naming (pure logic, no Dapr needed)
# ---------------------------------------------------------------------------


def test_sanitize_agent_name_basic() -> None:
    """Test agent name sanitization with common patterns."""
    from diagrid.agent.core.workflow.naming import sanitize_agent_name

    assert sanitize_agent_name("my-agent") == "MyAgent"
    assert sanitize_agent_name("hello_world") == "HelloWorld"
    assert sanitize_agent_name("catering-coordinator") == "CateringCoordinator"
    assert sanitize_agent_name("") == "unnamed_agent"


def test_sanitize_agent_name_special_chars() -> None:
    """Test agent name sanitization strips invalid characters."""
    from diagrid.agent.core.workflow.naming import sanitize_agent_name

    assert sanitize_agent_name("agent<name>") == "Agentname"
    assert sanitize_agent_name("path/to\\agent") == "Pathtoagent"
    assert sanitize_agent_name("pipe|test") == "Pipetest"


def test_normalize_to_title_case() -> None:
    """Test TitleCase normalization for different naming styles."""
    from diagrid.agent.core.workflow.naming import _normalize_to_title_case

    assert _normalize_to_title_case("get_user") == "GetUser"
    assert _normalize_to_title_case("get-user") == "GetUser"
    assert _normalize_to_title_case("get user") == "GetUser"
    assert _normalize_to_title_case("GetUser") == "GetUser"
    assert _normalize_to_title_case("UPPERCASE") == "Uppercase"
    assert _normalize_to_title_case("SamwiseGamgee") == "SamwiseGamgee"
    assert _normalize_to_title_case("") == ""
