# CreditGuard-AI

CreditGuard-AI is a credit-risk modeling project with training, evaluation, explainability, and Streamlit UI components.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
streamlit run streamlit_app.py
```

## Production readiness checklist

Before going live, complete at least the following:

1. **CI gates must pass** on supported Python versions (3.11–3.13).
2. **Static checks** (`ruff`, `mypy`) and **test suite** (`pytest`) must be mandatory in pull requests.
3. **Pinned dependency strategy** should be used for deploy environments (lock file or frozen image digest).
4. **Security hardening**: run behind TLS/reverse proxy, define authN/authZ, store secrets outside the repo.
5. **Model governance**: track model version, training dataset fingerprint, and rollback artifact.
6. **Monitoring**: capture service health, latency, and model-drift metrics.
7. **Operational procedures**: documented backup/restore and incident response steps.

## Useful commands

```bash
make install-dev
make lint
make typecheck
make test
make docker
make docker-run
```
