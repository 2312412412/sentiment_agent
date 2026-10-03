# Security Policy

## Secrets

Do not commit `.env`, API keys, certificates, generated memory data, model files, or private reports.

If a secret has been committed, revoke it at the provider immediately and generate a new one. Removing it from the latest commit is not enough if it has already entered Git history.

## Reporting Issues

Please report security issues privately to the project maintainer when possible. Do not include real API keys, personal data, or private company reports in public issues.
