from diagrid.identity.outbound import (
    clear_current_token,
    current_user_token,
    outbound_identity_headers,
    reset_current_token,
    set_current_token,
)


def test_set_and_get_token():
    tok = set_current_token("abc123")
    assert current_user_token() == "abc123"
    reset_current_token(tok)


def test_reset_restores_previous():
    tok1 = set_current_token("first")
    tok2 = set_current_token("second")
    assert current_user_token() == "second"
    reset_current_token(tok2)
    assert current_user_token() == "first"
    reset_current_token(tok1)


def test_clear_token():
    set_current_token("abc123")
    clear_current_token()
    assert current_user_token() is None


def test_outbound_headers_with_token():
    tok = set_current_token("tok")
    headers = outbound_identity_headers()
    assert headers == {"X-Diagrid-User-Token": "Bearer tok"}
    reset_current_token(tok)


def test_outbound_headers_without_token():
    clear_current_token()
    assert outbound_identity_headers() == {}


def test_default_is_none():
    clear_current_token()
    assert current_user_token() is None
