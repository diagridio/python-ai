# Security policy

## Reporting a vulnerability

**Do not open a public issue, pull request or discussion for a security
problem.**

Report it privately through GitHub's private vulnerability reporting:

1. Open the repository's [Security tab](https://github.com/diagridio/python-ai/security).
2. Choose **Report a vulnerability**, or go straight to
   <https://github.com/diagridio/python-ai/security/advisories/new>.
3. Describe the issue, the affected version (`pip show diagrid`), and the steps
   to reproduce it.

Only the maintainers can see the report. We aim to acknowledge it within three
business days, keep you updated while we fix it, and credit you in the published
advisory unless you ask us not to.

## Supported versions

The `diagrid`, `diagrid-core` and `diagrid-cli` packages are released together
under one version. Security fixes go into the latest release only. Upgrade to
the newest version before you report a problem, to check it still happens there.

## Scope

In scope: the code in this repository, which covers the three Python packages
and the `catalyst-agents` Helm chart under `charts/`.

Report problems in Dapr itself to the [Dapr project](https://github.com/dapr/dapr/security),
and problems in a third-party agent framework to that framework's maintainers.
