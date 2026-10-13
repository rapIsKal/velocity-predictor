# Velocity predictor

A small FastAPI service that estimates velocity **Vx(t)** from noisy scatter
measurements of position **x(t)**. It fits a cubic smoothing spline to the data
and returns the spline and its derivative.

- Smoothing strength is picked automatically by generalized cross-validation,
  or you can set it yourself.
- Input points can be in any order; duplicate `t` values are averaged.
- With fewer than 5 distinct points it falls back to finite differences.

## Quick start

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:

- http://127.0.0.1:8000 — test page: paste `t, x` pairs (or generate sample
  data) and see the fit and predicted velocity plotted
- http://127.0.0.1:8000/docs — interactive API docs

## API

### `POST /predict`

| Field       | Type          | Required | Description                                                                 |
|-------------|---------------|----------|-----------------------------------------------------------------------------|
| `t`         | `float[]`     | yes      | Sample times                                                                |
| `x`         | `float[]`     | yes      | Positions at those times (same length as `t`)                               |
| `n_points`  | `int` (2–10000) | no     | Return results on a uniform grid of this many points instead of at the input times |
| `smoothing` | `float` ≥ 0   | no       | Spline smoothing λ; omit for automatic (GCV), `0` interpolates exactly      |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"t": [0, 1, 2, 3, 4, 5], "x": [0, 1, 4, 9, 16, 25], "n_points": 3}'
```

```json
{"t": [0.0, 2.5, 5.0], "x_fit": [0.0, 6.26, 25.0], "vx": [0.58, 5.0, 9.42]}
```

Returns `422` if the lengths of `t` and `x` differ or there are fewer than 2
distinct `t` values.

Note: velocity estimates near the ends of the data range are less accurate than
in the interior, as is usual for smoothing splines.

### `GET /health`

Returns `{"status": "ok"}`.

## Tests

```bash
pytest
```

## Project layout

```
app/
  main.py            FastAPI app and endpoints
  velocity.py        spline fitting / velocity estimation
  static/index.html  test frontend
tests/
  test_api.py
```
