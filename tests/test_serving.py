#!/usr/bin/env python3
"""Test the contract the forecast service promises."""

import datetime

import numpy as np
import pandas as pd
import pytest

from pmlcast import cenace
from pmlcast import serving

D = datetime.date


def test_window_of_finds_the_contiguous_block():
    prices = np.array([100.0] * 24)
    prices[18:22] = 900.0

    expensive = serving.window_of(prices, 4, expensive=True)
    cheap = serving.window_of(prices, 4, expensive=False)

    assert expensive == {"start_hour": 19, "end_hour": 22}
    # the cheapest run is the first flat stretch, before the peak
    assert cheap["end_hour"] - cheap["start_hour"] == 3
    assert cheap["end_hour"] < 19


def test_window_of_picks_the_best_block_not_the_best_hours():
    # one huge hour alone should not beat four good ones together
    prices = np.array([100.0] * 24)
    prices[5] = 5000.0
    prices[10:14] = 1500.0

    window = serving.window_of(prices, 4, expensive=True)

    assert window == {"start_hour": 11, "end_hour": 14}


class FakeForecaster(serving.Forecaster):
    """A Forecaster that skips loading a model, to test its guards."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.market = "MDA"
        self.input_days = 14


def test_out_of_scope_node_is_rejected():
    catalog = pd.DataFrame(
        {
            "node": ["07PJZ-230", "01TUL-400"],
            "sistema": ["BCA", "SIN"],
            "region": ["BAJA CALIFORNIA", "CENTRAL"],
            "zone": ["TIJUANA", "CENTRO"],
            "kv": [230.0, 400.0],
        }
    )
    forecaster = FakeForecaster(catalog)

    with pytest.raises(serving.ForecastError) as error:
        forecaster.check_scope("07PJZ-230")
    assert "BCA" in str(error.value)

    # a SIN node passes; a key in no catalogue is named as such rather
    # than failing later for missing history
    forecaster.check_scope("01TUL-400")
    with pytest.raises(serving.ForecastError) as unknown:
        forecaster.check_scope("99XXX-115")
    assert "not in the CENACE catalogue" in str(unknown.value)


def test_uncatalogued_node_passes_when_no_catalogue_is_loaded():
    """With no catalogue at all, history is the only requirement."""
    forecaster = FakeForecaster(pd.DataFrame(columns=["node", "sistema"]))

    forecaster.check_scope("99XXX-115")


def test_evaluated_set_is_read_from_the_nodes_file(tmp_path):
    """A node outside the evaluation set is served, and marked as such."""
    path = tmp_path / "nodes.csv"
    path.write_text("node,sistema\n01TUL-400,SIN\n", encoding="utf-8")
    evaluated = set(cenace.read_nodes_file(str(path)))

    assert "01TUL-400" in evaluated
    assert "99XXX-115" not in evaluated


def test_more_than_one_day_ahead_is_rejected():
    forecaster = FakeForecaster(pd.DataFrame(columns=["node", "sistema"]))
    far = datetime.date.today() + datetime.timedelta(days=5)

    with pytest.raises(serving.ForecastError) as error:
        forecaster.forecast("01TUL-400", far)
    assert "single-horizon" in str(error.value)


def test_missing_model_is_reported_clearly(tmp_path):
    with pytest.raises(serving.ForecastError) as error:
        serving.Forecaster(model_path=str(tmp_path / "nope.keras"))
    assert "model not found" in str(error.value)


def _one_day(day, node="08MDP-230"):
    """Return 24 published silver-like rows of one node and day."""
    return pd.DataFrame(
        {
            "node": node,
            "market": "MDA",
            "sistema": "SIN",
            "ts_local": pd.date_range(pd.Timestamp(day), periods=24, freq="h"),
            "fecha": day,
            "hora": range(1, 25),
            "pml": 1000.0,
            "pml_ene": 900.0,
            "pml_per": 80.0,
            "pml_cng": 20.0,
            "quality": "ok",
        }
    )


def test_unpublished_target_day_gets_a_placeholder():
    """Tomorrow can be forecast before CENACE publishes it."""
    forecaster = FakeForecaster(pd.DataFrame(columns=["node", "sistema"]))
    origin = datetime.date(2026, 10, 1)
    out = forecaster._with_target_day(_one_day(origin), "08MDP-230", origin)

    target = out[out["fecha"] == origin + datetime.timedelta(days=1)]
    assert len(out) == 48 and len(target) == 24
    assert (target["quality"] == serving.PENDING_QUALITY).all()
    assert target["pml"].isna().all()
    assert list(target["ts_local"].dt.hour) == list(range(24))


def test_placeholder_is_not_added_over_real_data_or_a_gap():
    """Published targets are kept, and a gap before the origin is left."""
    forecaster = FakeForecaster(pd.DataFrame(columns=["node", "sistema"]))
    origin = datetime.date(2026, 10, 1)
    published = pd.concat(
        [_one_day(origin), _one_day(origin + datetime.timedelta(days=1))],
        ignore_index=True,
    )
    same = forecaster._with_target_day(published, "08MDP-230", origin)
    assert len(same) == 48 and not same["pml"].isna().any()

    stale = _one_day(origin - datetime.timedelta(days=2))
    kept = forecaster._with_target_day(stale, "08MDP-230", origin)
    assert len(kept) == 24
