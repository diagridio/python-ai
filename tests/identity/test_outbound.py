from diagrid.identity.outbound import (
    clear_current_token,
    current_user_token,
    outbound_identity_headers,
    set_current_token,
)


def test_set_and_get_token():
    set_current_token("abc123")
    assert current_user_token() == "abc123"
    clear_current_token()


def test_clear_token():
    set_current_token("abc123")
    clear_current_token()
    assert current_user_token() is None


def test_outbound_headers_with_token():
    set_current_token("tok")
    headers = outbound_identity_headers()
    assert headers == {"X-Diagrid-User-Token": "Bearer tok"}
    clear_current_token()


def test_outbound_headers_without_token():
    clear_current_token()
    assert outbound_identity_headers() == {}


def test_default_is_none():
    clear_current_token()
    assert current_user_token() is None
