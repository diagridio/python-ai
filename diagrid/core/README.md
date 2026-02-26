# Diagrid Core

`diagrid-core` is the foundational library for Diagrid Catalyst Python SDKs. It provides shared utilities for authentication, configuration management, and API client interactions.

**Note:** This package is primarily intended for internal use by `diagrid` and `diagrid-cli`, or for advanced users building custom integrations with the Diagrid Catalyst API.

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

## Installation

```bash
pip install diagrid-core
```

## Features

- **Authentication:** Handles Catalyst API authentication, including API Key management and OAuth2 device code flows.
- **API Client:** A robust HTTP client for interacting with Diagrid Catalyst services, built on `httpx`.
- **Configuration:** Manages global configuration settings, environment variables, and context persistence.
- **Type Safety:** Fully typed with modern Python type hints.

## Requirements

- Python 3.11+
- `httpx`
- `pydantic`
- `pyjwt`
