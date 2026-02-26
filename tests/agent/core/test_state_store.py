"""Tests for DaprStateStore."""

import json
from unittest import TestCase, mock

from diagrid.agent.core.state import DaprStateStore


class TestDaprStateStore(TestCase):
    """Tests for the DaprStateStore class."""

    @mock.patch("dapr.clients.DaprClient")
    def test_save_serializes_json(self, mock_dapr_cls):
        store = DaprStateStore(store_name="mystore")
        store.save("key1", {"msg": "hello"})

        mock_dapr_cls.return_value.save_state.assert_called_once_with(
            store_name="mystore",
            key="key1",
            value=json.dumps({"msg": "hello"}),
            state_metadata=None,
        )

    @mock.patch("dapr.clients.DaprClient")
    def test_save_with_metadata(self, mock_dapr_cls):
        store = DaprStateStore(store_name="mystore")
        store.save("key1", "value", metadata={"partitionKey": "pk"})

        mock_dapr_cls.return_value.save_state.assert_called_once_with(
            store_name="mystore",
            key="key1",
            value=json.dumps("value"),
            state_metadata={"partitionKey": "pk"},
        )

    @mock.patch("dapr.clients.DaprClient")
    def test_get_returns_deserialized_data(self, mock_dapr_cls):
        mock_response = mock.MagicMock()
        mock_response.data = json.dumps({"msg": "hello"}).encode()
        mock_dapr_cls.return_value.get_state.return_value = mock_response

        store = DaprStateStore()
        result = store.get("key1")

        self.assertEqual(result, {"msg": "hello"})
        mock_dapr_cls.return_value.get_state.assert_called_once_with(
            store_name="statestore",
            key="key1",
        )

    @mock.patch("dapr.clients.DaprClient")
    def test_get_returns_none_for_missing_key(self, mock_dapr_cls):
        mock_response = mock.MagicMock()
        mock_response.data = b""
        mock_dapr_cls.return_value.get_state.return_value = mock_response

        store = DaprStateStore()
        result = store.get("missing")

        self.assertIsNone(result)

    @mock.patch("dapr.clients.DaprClient")
    def test_delete(self, mock_dapr_cls):
        store = DaprStateStore()
        store.delete("key1")

        mock_dapr_cls.return_value.delete_state.assert_called_once_with(
            store_name="statestore",
            key="key1",
        )

    @mock.patch("dapr.clients.DaprClient")
    def test_save_bulk(self, mock_dapr_cls):
        store = DaprStateStore(store_name="mystore")
        store.save_bulk([("k1", "v1"), ("k2", {"nested": True})])

        mock_dapr_cls.return_value.save_bulk_state.assert_called_once()
        call_args = mock_dapr_cls.return_value.save_bulk_state.call_args
        self.assertEqual(call_args.kwargs["store_name"], "mystore")
        states = call_args.kwargs["states"]
        self.assertEqual(len(states), 2)

    @mock.patch("dapr.clients.DaprClient")
    def test_close(self, mock_dapr_cls):
        store = DaprStateStore()
        # Access client to initialize it
        store._get_client()
        store.close()

        mock_dapr_cls.return_value.close.assert_called_once()
        self.assertIsNone(store._client)

    @mock.patch("dapr.clients.DaprClient")
    def test_close_without_init(self, mock_dapr_cls):
        store = DaprStateStore()
        store.close()  # Should not error
        mock_dapr_cls.return_value.close.assert_not_called()

    @mock.patch("dapr.clients.DaprClient")
    def test_lazy_client_init(self, mock_dapr_cls):
        mock_response = mock.MagicMock()
        mock_response.data = b""
        mock_dapr_cls.return_value.get_state.return_value = mock_response

        store = DaprStateStore()
        # Client not created until first use
        mock_dapr_cls.assert_not_called()

        store.get("key1")
        mock_dapr_cls.assert_called_once()

    def test_store_name_property(self):
        store = DaprStateStore(store_name="custom-store")
        self.assertEqual(store.store_name, "custom-store")

    def test_default_store_name(self):
        store = DaprStateStore()
        self.assertEqual(store.store_name, "statestore")
