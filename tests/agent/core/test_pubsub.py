# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for DaprPubSub."""

from unittest import TestCase, mock

from diagrid.agent.core.pubsub.pubsub import DaprPubSub


class TestDaprPubSub(TestCase):
    """Tests for the DaprPubSub class."""

    def test_init_defaults(self):
        pubsub = DaprPubSub()
        self.assertEqual(pubsub.pubsub_name, "agent-pubsub")
        self.assertIsNone(pubsub._client)

    def test_init_custom_name(self):
        pubsub = DaprPubSub(pubsub_name="my-pubsub")
        self.assertEqual(pubsub.pubsub_name, "my-pubsub")

    @mock.patch("dapr.clients.DaprClient")
    def test_lazy_client_init(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client_cls.return_value = mock_client

        pubsub = DaprPubSub()
        self.assertIsNone(pubsub._client)

        pubsub.publish("test-topic", {"key": "value"})
        mock_client_cls.assert_called_once()
        self.assertIsNotNone(pubsub._client)

    @mock.patch("dapr.clients.DaprClient")
    def test_publish(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client_cls.return_value = mock_client

        pubsub = DaprPubSub(pubsub_name="test-pubsub")
        pubsub.publish("my-topic", {"message": "hello"})

        mock_client.publish_event.assert_called_once_with(
            pubsub_name="test-pubsub",
            topic_name="my-topic",
            data='{"message": "hello"}',
            data_content_type="application/json",
            publish_metadata={},
        )

    @mock.patch("dapr.clients.DaprClient")
    def test_publish_with_metadata(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client_cls.return_value = mock_client

        pubsub = DaprPubSub(pubsub_name="test-pubsub")
        pubsub.publish("my-topic", {"x": 1}, metadata={"rawPayload": "true"})

        mock_client.publish_event.assert_called_once_with(
            pubsub_name="test-pubsub",
            topic_name="my-topic",
            data='{"x": 1}',
            data_content_type="application/json",
            publish_metadata={"rawPayload": "true"},
        )

    @mock.patch("dapr.clients.DaprClient")
    def test_close(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client_cls.return_value = mock_client

        pubsub = DaprPubSub()
        pubsub.publish("topic", {})  # Force client creation
        pubsub.close()

        mock_client.close.assert_called_once()
        self.assertIsNone(pubsub._client)

    def test_close_no_client(self):
        pubsub = DaprPubSub()
        pubsub.close()  # Should not raise

    @mock.patch("dapr.clients.DaprClient")
    def test_client_reused(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client_cls.return_value = mock_client

        pubsub = DaprPubSub()
        pubsub.publish("topic-1", {"a": 1})
        pubsub.publish("topic-2", {"b": 2})

        mock_client_cls.assert_called_once()
        self.assertEqual(mock_client.publish_event.call_count, 2)
