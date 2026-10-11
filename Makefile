.PHONY: format check test docs docs-serve build publish patch minor major

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

patch:
	uv version --bump patch

minor:
	uv version --bump minor

major:
	uv version --bump major