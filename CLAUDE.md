# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The virtualenv lives in `venv/` (Python 3.11; code requires 3.10+ for `X | None` syntax). Use its binaries directly:

```bash
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn app.main:app --reload          # serves http://127.0.0.1:8000 (test page at /, API docs at /docs)
venv/bin/python -m pytest                       # all tests
venv/bin/python -m pytest tests/test_api.py::test_few_points_fallback   # single test
```

No linter/formatter or build step is configured.

## Architecture

FastAPI service that estimates velocity Vx(t) from noisy position samples x(t).

- `app/velocity.py` — pure numerics, no web dependencies. `predict_velocity(t, x, t_eval, smoothing)` sorts input and averages x over duplicate t (`_prepare`), then fits `scipy.interpolate.make_smoothing_spline` and returns `(t_eval, spline(t_eval), spline.derivative()(t_eval))`. `smoothing=None` → lambda chosen by GCV; `0` → exact interpolation. Below `MIN_POINTS_FOR_SPLINE` (5) distinct points it falls back to `np.gradient` + linear interpolation. Raises `ValueError` for < 2 distinct t.
- `app/main.py` — thin HTTP layer. `POST /predict` validates via Pydantic (`t`/`x` length match, `n_points` 2–10000, `smoothing` ≥ 0), builds a uniform `t_eval` grid when `n_points` is set (otherwise evaluates at the sorted unique input times), and maps `ValueError` from the numerics to HTTP 422. Also `GET /health` and `GET /` (serves the static page).
- `app/static/index.html` — self-contained test frontend (no build, no external libs): parses `t, x` text, generates sample data, calls `/predict`, and draws hand-rolled SVG charts. Colors are CSS tokens on `:root` with light/dark variants. `tests/test_api.py::test_index_page` asserts the page contains the string "Velocity predictor".

Tests go through `fastapi.testclient.TestClient` against the real app; numeric assertions ignore the edges of the range, where spline derivative estimates are least accurate.

## Repo

Remote: https://github.com/rapIsKal/velocity-predictor (public), branch `main`.
