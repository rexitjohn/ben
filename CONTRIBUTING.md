# Contributing

## Development setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
```

## Pull requests
Include a problem statement, proposed change, tests, documentation updates, and confirmation that no private veterinary or owner data is included.

## Research claims
Do not present metrics from synthetic data, screenshots, or exploratory notebooks as clinical results. Record dataset provenance, split strategy, sample counts, model version, threshold, uncertainty, and limitations.
