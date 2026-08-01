# Contributing

Contributions are welcome. Keep changes focused on the reusable LFM2.5 embedding workflow and
avoid committing datasets, model weights, credentials, customer information, or environment
receipts.

```bash
uv sync --extra train --extra dev
uv run ruff format src tests
uv run ruff check src tests
uv run ty check
uv run pytest -q
```

Please include tests for behavior changes and explain any model-loading, loss, split, or metric
assumptions in the pull request.
