# CreditGuard AI — common developer commands.

.PHONY: help install install-dev lint format typecheck test smoke train app docker docker-run clean

help:
	@echo "CreditGuard AI — make targets:"
	@echo "  install      - pip install -e ."
	@echo "  install-dev  - pip install -e .[dev]"
	@echo "  lint         - ruff check src tests"
	@echo "  format       - ruff format src tests"
	@echo "  typecheck    - mypy src"
	@echo "  test         - pytest -q"
	@echo "  smoke        - quick training run with no Optuna and no OpenML"
	@echo "  train        - full training run with Optuna tuning"
	@echo "  app          - launch the Streamlit UI"
	@echo "  docker       - build the Docker image"
	@echo "  docker-run   - docker compose up"
	@echo "  clean        - remove generated artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

test:
	pytest -q

smoke:
	creditguard train --no-optuna --no-openml --synthetic-rows 800

train:
	creditguard train --trials 30

app:
	streamlit run streamlit_app.py

docker:
	docker build -t creditguard-ai:latest .

docker-run:
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f models/*.joblib reports/figures/*.png reports/*.pdf
