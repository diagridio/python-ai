"""Tests for unified Dapr component discovery."""

import json
from collections import namedtuple
from unittest import TestCase, mock

from diagrid.agent.core.discovery import (
    DiscoveredComponents,
    discover_components,
    _reset_discovery_cache,
)

RegisteredComponents = namedtuple(
    "RegisteredComponents", ["name", "type", "version", "capabilities"]
)


def _make_metadata_response(components, app_id="test-app"):
    resp = mock.MagicMock()
    resp.application_id = app_id
    resp.registered_components = [
        RegisteredComponents(
            name=c["name"],
            type=c["type"],
            version=c.get("version", "v1"),
            capabilities=c.get("capabilities", []),
        )
        for c in components
    ]
    return resp


class TestDiscoverComponents(TestCase):
    def setUp(self):
        _reset_discovery_cache()

    def tearDown(self):
        _reset_discovery_cache()

    @mock.patch("diagrid.agent.core.discovery.DaprClient")
    def test_discovers_memory_store(self, mock_dapr_cls):
        mock_client = mock.MagicMock()
        mock_client.get_metadata.return_value = _make_metadata_response(
            [
                {"name": "agent-memory", "type": "state.redis"},
            ]
        )
        mock_dapr_cls.return_value.__enter__ = mock.MagicMock(return_value=mock_client)
        mock_dapr_cls.return_value.__exit__ = mock.MagicMock(return_value=False)

        result = discover_components()

        self.assertEqual(result.memory_store_name, "agent-memory")

    @mock.patch("diagrid.agent.core.discovery.DaprClient")
    def test_discovers_pubsub(self, mock_dapr_cls):
        mock_client = mock.MagicMock()
        mock_client.get_metadata.return_value = _make_metadata_response(
            [
                {"name": "agent-pubsub", "type": "pubsub.redis"},
            ]
        )
        mock_dapr_cls.return_value.__enter__ = mock.MagicMock(return_value=mock_client)
        mock_dapr_cls.return_value.__exit__ = mock.MagicMock(return_value=False)

        result = discover_components()

        self.assertEqual(result.pubsub_name, "agent-pubsub")

    @mock.patch("diagrid.agent.core.discovery.DaprClient")
    def test_discovers_registry(self, mock_dapr_cls):
        mock_client = mock.MagicMock()
        mock_client.get_metadata.return_value = _make_metadata_response(
            [
                {"name": "agent-registry", "type": "state.redis"},
            ]
        )
        mock_dapr_cls.return_value.__enter__ = mock.MagicMock(return_value=mock_client)
        mock_dapr_cls.return_value.__exit__ = mock.MagicMock(return_value=False)

        result = discover_components()

        self.assertEqual(result.registry_store_name, "agent-registry")

    @mock.patch("diagrid.agent.core.discovery.DaprClient")
    def test_discovers_runtime_config(self, mock_dapr_cls):
        mock_client = mock.MagicMock()
        mock_client.get_metadata.return_value = _make_metadata_response(
            [
                {"name": "agent-runtime", "type": "state.redis"},
            ]
        )
        state_resp = mock.MagicMock()
        state_resp.data = json.dumps({"OTEL_SERVICE_NAME": "my-svc"}).encode()
        mock_client.get_state.return_value = state_resp
        mock_dapr_cls.return_value.__enter__ = mock.MagicMock(return_value=mock_client)
        mock_dapr_cls.return_value.__exit__ = mock.MagicMock(return_value=False)

        result = discover_components()

        self.assertEqual(result.runtime_conf, {"OTEL_SERVICE_NAME": "my-svc"})

    @mock.patch("diagrid.agent.core.discovery.DaprClient")
    def test_graceful_fallback_on_error(self, mock_dapr_cls):
        mock_dapr_cls.side_effect = Exception("no sidecar")

        result = discover_components()

        self.assertIsNone(result.memory_store_name)
        self.assertEqual(result.runtime_conf, {})

    @mock.patch("diagrid.agent.core.discovery.DaprClient")
    def test_caches_result(self, mock_dapr_cls):
        mock_client = mock.MagicMock()
        mock_client.get_metadata.return_value = _make_metadata_response(
            [
                {"name": "agent-memory", "type": "state.redis"},
            ]
        )
        mock_dapr_cls.return_value.__enter__ = mock.MagicMock(return_value=mock_client)
        mock_dapr_cls.return_value.__exit__ = mock.MagicMock(return_value=False)

        result1 = discover_components()
        result2 = discover_components()

        self.assertIs(result1, result2)
        # DaprClient constructor called only once
        self.assertEqual(mock_dapr_cls.call_count, 1)

    def test_discovered_components_defaults(self):
        dc = DiscoveredComponents()
        self.assertIsNone(dc.configuration_name)
        self.assertIsNone(dc.memory_store_name)
        self.assertIsNone(dc.pubsub_name)
        self.assertIsNone(dc.registry_store_name)
        self.assertEqual(dc.runtime_conf, {})
