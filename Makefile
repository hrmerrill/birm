.PHONY: format check test docs docs-serve build publish bump-patch bump-minor bump-major

format:
	uv run ruff format .
	uv run ruff check --fix .

check:
	uv run ruff check .
	uv run ty check .
	uv run interrogate --fail-under 100 src/

test:
	uv run pytest --cov=src --cov-report=term-missing --cov-report=xml

docs:
	uv run mkdocs build

docs-serve:
	uv run mkdocs serve

build:
	uv build

publish: test build
	uv publish --trusted-publishing always

bump-patch:  ## Bump patch version (e.g. 0.1.0 → 0.1.1)
	uv run bump-my-version bump patch

bump-minor:  ## Bump minor version (e.g. 0.1.0 → 0.2.0)
	uv run bump-my-version bump minor

bump-major:  ## Bump major version (e.g. 0.1.0 → 1.0.0)
	uv run bump-my-version bump major