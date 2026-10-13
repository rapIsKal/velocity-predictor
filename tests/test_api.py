import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_noisy_sine_velocity():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 2 * np.pi, 200))
    x = np.sin(t) + rng.normal(0, 0.05, t.size)

    r = client.post("/predict", json={"t": t.tolist(), "x": x.tolist(), "n_points": 100})
    assert r.status_code == 200
    body = r.json()
    t_out, vx = np.array(body["t"]), np.array(body["vx"])
    assert len(vx) == 100
    # ignore edges, where any derivative estimate is least reliable
    inner = slice(10, -10)
    assert np.max(np.abs(vx[inner] - np.cos(t_out[inner]))) < 0.2


def test_unsorted_with_duplicates_linear():
    t = [3, 1, 2, 2, 0, 4, 5]
    x = [6, 2, 4, 4, 0, 8, 10]  # x = 2t -> vx = 2
    r = client.post("/predict", json={"t": t, "x": x})
    assert r.status_code == 200
    assert np.allclose(r.json()["vx"], 2, atol=1e-6)


def test_few_points_fallback():
    r = client.post("/predict", json={"t": [0, 1, 2], "x": [0, 1, 4]})
    assert r.status_code == 200
    assert r.json()["vx"] == [1.0, 2.0, 3.0]


def test_length_mismatch():
    r = client.post("/predict", json={"t": [0, 1], "x": [0]})
    assert r.status_code == 422


def test_single_point():
    r = client.post("/predict", json={"t": [1, 1], "x": [0, 2]})
    assert r.status_code == 422


def test_index_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "Velocity predictor" in r.text
