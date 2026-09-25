#!/usr/bin/env python3
"""Serve the forecast over HTTP."""

import datetime
import os

import fastapi
import pydantic
import uvicorn

import pmlcast
from pmlcast import config
from pmlcast import dashboard
from pmlcast import serving

TITLE = "PMLcast"
DESCRIPTION = (
    "Day-ahead hourly electricity price forecasting for Mexican grid nodes. "
    "Informational only; not financial advice."
)

DEFAULT_NODES = os.path.join(config.SNAPSHOT_DIR, "nodes_stage2.csv")

app = fastapi.FastAPI(
    title=TITLE, description=DESCRIPTION, version=pmlcast.__version__
)
_forecaster = None


class ForecastRequest(pydantic.BaseModel):
    """A forecast request: a node, and optionally the day to forecast."""

    node: str
    target_date: datetime.date = None


def get_forecaster():
    """Load the model once and reuse it across requests."""
    global _forecaster
    if _forecaster is None:
        _forecaster = serving.Forecaster(
            model_path=os.environ.get("PMLCAST_MODEL", serving.DEFAULT_MODEL),
            meta_path=os.environ.get("PMLCAST_META", serving.DEFAULT_META),
            backfill=os.environ.get("PMLCAST_BACKFILL") == "1",
        )

    return _forecaster


@app.get("/health")
def health():
    """Report whether the service can answer forecasts."""
    try:
        forecaster = get_forecaster()
    except serving.ForecastError as error:
        raise fastapi.HTTPException(status_code=503, detail=str(error))

    return {
        "status": "ok",
        "version": pmlcast.__version__,
        "model_version": forecaster.model_version,
        "market": forecaster.market,
        "input_hours": forecaster.meta["input_hours"],
        "today": config.today_local().isoformat(),
    }


@app.get("/", response_class=fastapi.responses.HTMLResponse)
def index():
    """Serve the dashboard page."""
    return dashboard.render_page()


@app.get("/dashboard/data")
def dashboard_data():
    """Return the node list, the model card and the data freshness."""
    return dashboard.dashboard_data(
        os.environ.get("PMLCAST_NODES", DEFAULT_NODES),
        os.environ.get("PMLCAST_META", serving.DEFAULT_META),
        config.GOLD_DIR,
        config.SILVER_DIR,
    )


@app.post("/forecast")
def forecast(request: ForecastRequest):
    """Return the 24 hourly prices of the target day for one node."""
    try:
        return get_forecaster().forecast(request.node, request.target_date)
    except serving.ForecastError as error:
        raise fastapi.HTTPException(status_code=422, detail=str(error))


def main():
    """Run the API with uvicorn."""
    uvicorn.run(
        app,
        host=os.environ.get("PMLCAST_HOST", "0.0.0.0"),
        # Render and most PaaS hand the port in $PORT; PMLCAST_PORT is
        # the local override and 8000 the development default.
        port=int(
            os.environ.get("PORT")
            or os.environ.get("PMLCAST_PORT")
            or "8000"
        ),
    )


if __name__ == "__main__":
    main()
