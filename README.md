# Homilivo_MCP

## CI

`.github/workflows/ci.yml` runs on every pull request and every push to
`main`: it installs dependencies, runs the full `pytest tests/` suite
(which includes the calibrated per-category golden-eval gates in
`tests/test_golden_eval.py`), then runs `tests/run_golden_eval.py`
(overall golden-set accuracy gate). Nothing in the job depends on a live
Ollama instance or the real Homilivo API - both are mocked.

**Branch protection:** add the `test` job from this workflow as a
required status check for `main` (GitHub repo Settings → Branches →
branch protection rule → "Require status checks to pass") so a PR that
fails tests or drops below a golden-eval gate cannot be merged.