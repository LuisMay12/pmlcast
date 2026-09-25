#!/usr/bin/env python3
"""Test the baseline forecasters and their MLflow logging."""

import datetime
import types

import numpy as np
import pandas as pd
import pytest

from pmlcast import baselines


def test_naive_baselines_use_the_right_days(dataset_small, silver_small):
    ds = dataset_small
    train = ds.positions("train")
    positions = ds.positions("test")[:5]
    n24 = baselines.Naive24().fit(ds, train).predict(ds, positions)
    n168 = baselines.Naive168().fit(ds, train).predict(ds, positions)
    assert n24.shape == (5, 24) and n168.shape == (5, 24)
    for i, position in enumerate(positions):
        row = ds.index.iloc[position]
        node = silver_small[silver_small["node"] == row["node"]]
        node = node.set_index("ts_local")
        day_d = node.loc[str(row["origin_date"]), "pml"].to_numpy()
        assert np.allclose(n24[i], day_d, atol=0.05)
        week_ago = row["origin_date"] - datetime.timedelta(days=6)
        day_dm6 = node.loc[str(week_ago), "pml"].to_numpy()
        assert np.allclose(n168[i], day_dm6, atol=0.05)


def _synthetic_silver(node, start, days, amplitude=200.0):
    """Two-year hourly series with a flat profile and a yearly sinusoid."""
    index = pd.date_range(start, periods=days * 24, freq="h")
    doy = index.dayofyear.to_numpy()
    level = 1000.0 + amplitude * np.sin(2 * np.pi * doy / 365.0)
    return pd.DataFrame(
        {
            "node": node,
            "fecha": index.date,
            "hora": index.hour + 1,
            "ts_local": index,
            "pml": level,
            "quality": "ok",
        }
    )


def test_seasonal_ma_recovers_a_known_level():
    start = datetime.date(2022, 1, 1)
    silver = _synthetic_silver("N-115", start, 730)
    fit_targets = [datetime.date(2023, 12, 31)]
    predict_targets = [
        datetime.date(2024, 3, 21),
        datetime.date(2024, 6, 21),
        datetime.date(2024, 9, 22),
        datetime.date(2024, 12, 21),
    ]
    index = pd.DataFrame(
        {
            "node": "N-115",
            "target_date": fit_targets + predict_targets,
            "split": ["train"] + ["test"] * 4,
        }
    )
    ds = types.SimpleNamespace(index=index)
    model = baselines.SeasonalMA(half_window=15, history_years=2)
    model.fit(ds, [0], silver_df=silver)
    pred = model.predict(ds, [1, 2, 3, 4])
    assert pred.shape == (4, 24)
    for row, target in zip(pred, predict_targets):
        expected = 1000.0 + 200.0 * np.sin(
            2 * np.pi * pd.Timestamp(target).dayofyear / 365.0
        )
        assert np.allclose(row, expected, rtol=0.03)
        assert np.allclose(row, row[0])  # flat profile stays flat
    assert model.params() == {"half_window": 15, "history_years": 2}
    with pytest.raises(ValueError):
        baselines.SeasonalMA().fit(ds, [0], silver_df=None)


def test_linear_lags(dataset_small):
    ds = dataset_small
    train, val, test = (ds.positions(s) for s in ("train", "val", "test"))
    design = baselines.LinearLags.design(ds, test)
    assert design.shape == (len(test), 92)
    model = baselines.LinearLags().fit(ds, train, val_positions=val)
    assert model.alpha in model.alphas
    pred = model.predict(ds, test)
    assert pred.shape == (len(test), 24) and np.isfinite(pred).all()
    again = baselines.LinearLags().fit(ds, train, val_positions=val)
    assert np.allclose(again.predict(ds, test), pred)
    plain = baselines.LinearLags().fit(ds, train)
    assert plain.alpha == 1.0
