from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from app.velocity import predict_velocity

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Velocity predictor", description="Predict Vx(t) from scatter x(t)")


class PredictRequest(BaseModel):
    t: list[float] = Field(..., description="Sample times")
    x: list[float] = Field(..., description="Positions at those times")
    n_points: int | None = Field(
        None, ge=2, le=10_000,
        description="If set, return Vx on a uniform grid of this many points; "
                    "otherwise at the (sorted, unique) input times",
    )
    smoothing: float | None = Field(
        None, ge=0, description="Spline smoothing lambda; omit for automatic (GCV)"
    )

    @model_validator(mode="after")
    def check_lengths(self):
        if len(self.t) != len(self.x):
            raise ValueError("t and x must have the same length")
        return self


class PredictResponse(BaseModel):
    t: list[float]
    x_fit: list[float]
    vx: list[float]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t_eval = None
    if req.n_points is not None and req.t:
        t_eval = np.linspace(min(req.t), max(req.t), req.n_points)
    try:
        t_out, x_fit, vx = predict_velocity(req.t, req.x, t_eval, req.smoothing)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictResponse(t=t_out.tolist(), x_fit=x_fit.tolist(), vx=vx.tolist())
