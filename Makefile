.PHONY: install test lint demo clean
install:
	pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check .
demo:
	ben train --config configs/alpha_1_0.yaml
clean:
	rm -rf artifacts .pytest_cache .ruff_cache
