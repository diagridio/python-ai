from diagrid.identity import OAuthConfig, VerifiedUser

_UNSORTED_SCOPES = frozenset(
    {
        "agent.invoke",
        "admin.write",
        "billing.read",
        "catalog.list",
        "dataset.read",
        "events.publish",
        "files.write",
        "graph.read",
        "health.read",
        "index.write",
    }
)
_SORTED_SCOPES = sorted(_UNSORTED_SCOPES)


class TestOAuthConfig:
    def test_default_construction_is_fail_closed(self):
        config = OAuthConfig()
        assert config.require_auth is True
        assert config.scopes == frozenset()

    def test_allow_insecure_jwks_defaults_to_false(self):
        assert OAuthConfig().allow_insecure_jwks is False

    def test_scopes_iterate_in_sorted_order(self):
        assert list(OAuthConfig(scopes=_UNSORTED_SCOPES).scopes) == _SORTED_SCOPES

    def test_scopes_keep_set_semantics(self):
        config = OAuthConfig(scopes={"agent.invoke", "admin.write"})
        assert config.scopes == frozenset({"admin.write", "agent.invoke"})
        assert "agent.invoke" in config.scopes
        assert config.scopes - {"agent.invoke"} == frozenset({"admin.write"})


class TestVerifiedUser:
    def test_has_scope(self):
        user = VerifiedUser(subject="alice", scopes=frozenset({"agent.invoke"}))
        assert user.has_scope("agent.invoke") is True
        assert user.has_scope("admin.write") is False

    def test_has_scope_on_a_caller_with_no_scopes(self):
        assert VerifiedUser(subject="alice").has_scope("agent.invoke") is False

    def test_scopes_iterate_in_sorted_order(self):
        user = VerifiedUser(subject="alice", scopes=_UNSORTED_SCOPES)
        assert list(user.scopes) == _SORTED_SCOPES
