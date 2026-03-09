# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""OAuth2 Device Code Flow for Auth0, ported from Go CLI."""

from __future__ import annotations

import re
import time
import webbrowser
from datetime import datetime, timezone

import httpx
import jwt

from diagrid.core.auth.api_key import extract_org_id_from_api_key, get_api_key
from diagrid.core.auth.credentials import Credential, FileCredentialStore
from diagrid.core.auth.token import AuthContext, DeviceCodeResponse, TokenResponse
from diagrid.core.config.constants import (
    AUTH_SCOPE,
    DEFAULT_API_URL,
    ENV_API_URL,
    TOKEN_REFRESH_BUFFER_SECONDS,
)
from diagrid.core.config.envs import EnvConfig, get_env_config_sync
from diagrid.core.config.user_config import FileUserConfigStore

ORG_CLAIM_REGEX = re.compile(r"^https://diagrid\.io/org_([a-zA-Z0-9-]+)/roles$")
DEFAULT_ORG_CLAIM = "https://diagrid.io/defaultOrg"


class AuthenticationError(Exception):
    """Raised on authentication failures."""


class DeviceCodeAuth:
    """Manages the full authentication flow: API key, cached creds, refresh, device code."""

    def __init__(
        self,
        api_url: str | None = None,
        no_browser: bool = False,
        api_key_flag: str | None = None,
    ) -> None:
        import os

        self.api_url = api_url or os.environ.get(ENV_API_URL) or DEFAULT_API_URL
        self.no_browser = no_browser
        self.api_key_flag = api_key_flag
        self.cred_store = FileCredentialStore()
        self.config_store = FileUserConfigStore()

    def authenticate(self) -> AuthContext:
        """Run the full auth flow and return an AuthContext.

        1. Check for API key → extract orgID, return immediately
        2. Check cached credential (>5 min until expiry) → reuse
        3. Near-expiry credential → refresh
        4. Otherwise → full device code flow
        """
        # 1. API key path
        api_key = get_api_key(flag_key=self.api_key_flag)
        if api_key:
            org_id = extract_org_id_from_api_key(api_key)
            config = self.config_store.get()
            return AuthContext(
                api_url=self.api_url,
                org_id=org_id,
                project_id=config.current_project_id,
                api_key=api_key,
            )

        # 2. Check cached credential
        cred = self.cred_store.get()
        if cred.token_response and cred.expires_at:
            now = datetime.now(timezone.utc)
            expires_at = cred.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            remaining = (expires_at - now).total_seconds()

            if remaining > TOKEN_REFRESH_BUFFER_SECONDS:
                # Valid credential, reuse it
                return self._build_auth_context(cred)

            # 3. Near-expiry → refresh
            if cred.token_response.refresh_token and cred.env:
                try:
                    cred = self._refresh_token(cred)
                    return self._build_auth_context(cred)
                except Exception:
                    pass  # Fall through to full device code flow

        # 4. Full device code flow
        cred = self._device_code_flow()
        return self._build_auth_context(cred)

    def _build_auth_context(self, cred: Credential) -> AuthContext:
        config = self.config_store.get()
        org_id = cred.default_org or config.current_org_id
        # Use the API URL from the stored credential's env if available,
        # since the actual endpoint (e.g. api.r1.diagrid.io) may differ
        # from the default (api.diagrid.io).
        api_url = self.api_url
        if cred.env and cred.env.api_url:
            api_url = cred.env.api_url
        return AuthContext(
            api_url=api_url,
            org_id=org_id,
            project_id=config.current_project_id,
            access_token=cred.bearer_token,
        )

    def _refresh_token(self, cred: Credential) -> Credential:
        """Refresh the access token using the refresh token."""
        assert cred.env is not None
        assert cred.token_response is not None

        client_id = cred.env.auth_client_id
        refresh_token = cred.token_response.refresh_token

        url = cred.env.token_endpoint
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }

        with httpx.Client() as client:
            resp = client.post(
                url,
                data=data,
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                raise AuthenticationError(f"Failed to refresh token: {resp.text}")
            tkn = TokenResponse.model_validate(resp.json())

        # Preserve the refresh token (not returned in refresh response)
        tkn.refresh_token = refresh_token

        new_cred = self._credential_from_token(cred.env, tkn)
        self.cred_store.set(new_cred)
        return new_cred

    def _device_code_flow(self) -> Credential:
        """Run the full OAuth2 device code flow."""
        env = get_env_config_sync(self.api_url)

        # Request device code
        dc_resp = self._request_device_code(env)

        # Show user the code and open browser
        if self.no_browser:
            print(
                f"Visit: {dc_resp.verification_uri_complete}\n"
                f"Confirm the code matches: {dc_resp.user_code}"
            )
        else:
            print("Attempting to open login page in your default browser.")
            print(f"Confirm device code matches: {dc_resp.user_code}")
            print(f"If the browser does not open, visit: {dc_resp.verification_uri}")
            try:
                webbrowser.open(dc_resp.verification_uri_complete)
            except Exception:
                pass

        # Poll for token
        tkn = self._poll_for_token(env, dc_resp)

        # Build credential from token
        cred = self._credential_from_token(env, tkn)

        # Persist
        self.cred_store.set(cred)
        self._update_user_config(cred)

        return cred

    def _request_device_code(self, env: EnvConfig) -> DeviceCodeResponse:
        url = env.device_authorization_endpoint
        data = {
            "client_id": env.auth_client_id,
            "scope": AUTH_SCOPE,
            "audience": env.auth_audience,
        }
        with httpx.Client() as client:
            resp = client.post(
                url,
                data=data,
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                raise AuthenticationError(f"Failed to get device code: {resp.text}")
            return DeviceCodeResponse.model_validate(resp.json())

    def _poll_for_token(self, env: EnvConfig, dc: DeviceCodeResponse) -> TokenResponse:
        """Poll for token with backoff, matching Go CLI behavior."""
        url = env.token_endpoint
        interval = dc.interval

        with httpx.Client() as client:
            while True:
                resp = client.post(
                    url,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "device_code": dc.device_code,
                        "client_id": env.auth_client_id,
                    },
                    headers={"content-type": "application/x-www-form-urlencoded"},
                )

                if resp.status_code == 200:
                    return TokenResponse.model_validate(resp.json())

                error_data = resp.json()
                error = error_data.get("error", "")

                if error == "access_denied":
                    raise AuthenticationError(
                        f"Access denied: {error_data.get('error_description', '')}"
                    )
                if error == "expired_token":
                    raise AuthenticationError("Device code expired")
                if error == "slow_down":
                    interval += 1
                if error in ("authorization_pending", "slow_down"):
                    time.sleep(interval)
                    continue

                raise AuthenticationError(f"Token error: {error}")

    def _credential_from_token(self, env: EnvConfig, tkn: TokenResponse) -> Credential:
        """Parse JWT claims to build a Credential, matching Go CLI's getCredentialFromToken."""
        claims = jwt.decode(tkn.access_token, options={"verify_signature": False})

        subject = claims.get("sub", "")
        orgs: dict[str, list[str]] = {}

        for key, value in claims.items():
            match = ORG_CLAIM_REGEX.match(key)
            if match:
                org_id = match.group(1)
                roles = [str(v) for v in value] if isinstance(value, list) else []
                orgs[org_id] = roles

        # Get default org from ID token
        default_org = self._get_default_org_from_id_token(env, tkn)

        # Validate default org exists in orgs
        if default_org and default_org not in orgs:
            default_org = ""
        if not default_org and orgs:
            default_org = next(iter(orgs))

        # Get expiry
        exp = claims.get("exp", 0)
        expires_at = datetime.fromtimestamp(float(exp), tz=timezone.utc)

        return Credential(
            subject=subject,
            env=env,
            token_response=tkn,
            client_id=env.auth_client_id,
            client_secret="",
            invalid=False,
            default_org=default_org,
            orgs=orgs,
            expires_at=expires_at,
            timestamp=datetime.now(timezone.utc),
        )

    def _get_default_org_from_id_token(self, env: EnvConfig, tkn: TokenResponse) -> str:
        """Extract default org from the ID token claims."""
        if not tkn.id_token:
            return ""
        try:
            claims = jwt.decode(tkn.id_token, options={"verify_signature": False})
            return str(claims.get(DEFAULT_ORG_CLAIM, ""))
        except Exception:
            return ""

    def _update_user_config(self, cred: Credential) -> None:
        """Update user config with org/product info after login."""
        config = self.config_store.get()
        if cred.default_org:
            config.current_org_id = cred.default_org
            config.current_product = "catalyst"
        self.config_store.set(config)
