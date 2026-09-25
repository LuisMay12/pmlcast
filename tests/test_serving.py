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

    # a SIN node and an uncatalogued one both pass the scope check
    forecaster.check_scope("01TUL-400")
    forecaster.check_scope("99XXX-115")


def test_evaluated_set_is_read_from_the_nodes_file(tmp_path):
    """A node outside the evaluation set is served, and marked as such."""
    path = tmp_path / "nodes.csv"
    path.write_text(
        "node,sistema\n01TUL-400,SIN\n", encoding="utf-8"
    )
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
