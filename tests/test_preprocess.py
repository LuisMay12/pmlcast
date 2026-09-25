#!/usr/bin/env python3
"""Test DST alignment, the hourly grid, scaling and feature encoders."""

import datetime

import numpy as np
import pandas as pd
import pytest

from pmlcast import preprocess
from tests.conftest import make_records
from tests.conftest import parsed_records

D = datetime.date
KNOWN_DST = {
    D(2016, 4, 3): "spring",
    D(2016, 10, 30): "fall",
    D(2017, 4, 2): "spring",
    D(2017, 10, 29): "fall",
    D(2018, 4, 1): "spring",
    D(2018, 10, 28): "fall",
    D(2019, 4, 7): "spring",
    D(2019, 10, 27): "fall",
    D(2020, 4, 5): "spring",
    D(2020, 10, 25): "fall",
    D(2021, 4, 4): "spring",
    D(2021, 10, 31): "fall",
    D(2022, 4, 3): "spring",
    D(2022, 10, 30): "fall",
}


def test_dst_days_pins_known_dates():
    assert preprocess.dst_days(range(2016, 2023)) == KNOWN_DST
    assert preprocess.dst_days(range(2023, 2027)) == {}


def _day_frame(node, day, horas):
    return pd.DataFrame(
        {
            "node": node,
            "fecha": day,
            "hora": horas,
            "pml": [100.0 * h for h in horas],
            "pml_ene": [90.0 * h for h in horas],
            "pml_per": [5.0 * h for h in horas],
            "pml_cng": [5.0 * h for h in horas],
        }
    )


def test_align_day_spring():
    out = preprocess.align_day(
        _day_frame("N", D(2022, 4, 3), range(1, 24)), "spring"
    )
    assert list(out["hora"]) == list(range(1, 25))
    filled = out[out["quality"] == "dst_fill"]
    assert len(filled) == 1 and filled["hora"].iloc[0] == 3
    assert filled["pml"].iloc[0] == pytest.approx(
        250.0
    )  # mean of hora 2 and 3
    assert np.isnan(filled["source_hora"].iloc[0])
    shifted = out[out["hora"] == 4]
    assert (
        shifted["source_hora"].iloc[0] == 3 and shifted["pml"].iloc[0] == 300.0
    )
    assert out["ts_local"].iloc[0] == pd.Timestamp("2022-04-03 00:00")
    assert out["ts_local"].iloc[-1] == pd.Timestamp("2022-04-03 23:00")


def test_align_day_fall():
    out = preprocess.align_day(
        _day_frame("N", D(2022, 10, 30), range(1, 26)), "fall"
    )
    assert list(out["hora"]) == list(range(1, 25))
    merged = out[out["quality"] == "dst_merge"]
    assert len(merged) == 1 and merged["hora"].iloc[0] == 2
    assert merged["pml"].iloc[0] == pytest.approx(250.0)
    assert merged["source_hora"].iloc[0] == 2
    assert out[out["hora"] == 3]["source_hora"].iloc[0] == 4
    assert out[out["hora"] == 24]["source_hora"].iloc[0] == 25


def test_records_to_hourly_handles_dst_and_strings():
    values = make_records(D(2022, 4, 1), 5) + make_records(D(2022, 10, 28), 5)
    records = parsed_records("07PNC-115", values)
    out = preprocess.records_to_hourly(records)
    assert list(out.columns) == preprocess.SILVER_COLUMNS
    assert out.groupby("fecha").size().eq(24).all()
    spring = out[out["fecha"] == D(2022, 4, 3)]
    assert (spring["quality"] == "dst_fill").sum() == 1
    fall = out[out["fecha"] == D(2022, 10, 30)]
    assert (fall["quality"] == "dst_merge").sum() == 1
    assert out["hora"].dtype.kind == "i"
    assert out["ts_local"].is_monotonic_increasing
    assert preprocess.records_to_hourly([]).empty


def test_build_hourly_grid_interp_and_missing(silver_small):
    node = silver_small[silver_small["node"] == "07PNC-115"]
    assert node["ts_local"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    counts = node["quality"].value_counts()
    assert counts["interp"] == 2
    assert counts["missing"] == 30
    interp = node[node["quality"] == "interp"]
    assert interp["pml"].notna().all()
    identity = interp["pml_ene"] + interp["pml_per"] + interp["pml_cng"]
    assert np.allclose(identity, interp["pml"], atol=0.05)
    missing = node[node["quality"] == "missing"]
    assert missing["pml"].isna().all()
    assert counts["dst_fill"] == 1 and counts["dst_merge"] == 1

    gaps = preprocess.gap_table(silver_small)
    assert len(gaps) == 2
    assert set(gaps["kind"]) == {"interp", "missing"}
    assert gaps.loc[gaps["kind"] == "missing", "n_hours"].iloc[0] == 30


def test_rolling_stats_matches_brute_force(silver_small):
    node = silver_small[silver_small["node"] == "07MRD-230"]
    stats = preprocess.rolling_stats(node)
    origin = D(2022, 5, 20)
    row = stats.loc[origin]
    start = pd.Timestamp(origin) - pd.Timedelta(days=27)
    end = pd.Timestamp(origin) + pd.Timedelta(hours=23)
    window = node.set_index("ts_local").loc[start:end, "pml"]
    assert len(window) == 672
    assert row["mu"] == pytest.approx(window.mean())
    assert row["std_raw"] == pytest.approx(window.std(ddof=0))
    assert row["n_obs"] == 672 and row["valid"]
    assert row["sigma"] == pytest.approx(
        max(window.std(ddof=0), 0.05 * abs(window.mean()), 10.0)
    )
    week = node.set_index("ts_local").loc[
        pd.Timestamp(origin) - pd.Timedelta(days=6) : end, "pml"
    ]
    assert row["mu_7d"] == pytest.approx(week.mean())
    # the first 27 days cannot have a valid window
    assert not stats.loc[D(2022, 3, 20), "valid"]
    assert np.isnan(stats.loc[D(2022, 3, 20), "sigma"])


def test_rolling_stats_floors():
    index = pd.date_range("2024-01-01", periods=30 * 24, freq="h")
    flat = pd.DataFrame(
        {
            "ts_local": index,
            "pml": 5.0,
            "pml_ene": 5.0,
            "pml_per": 0.0,
            "pml_cng": 0.0,
        }
    )
    stats = preprocess.rolling_stats(flat)
    assert stats.loc[D(2024, 1, 29), "sigma"] == 10.0
    noisy = flat.copy()
    rng = np.random.default_rng(0)
    noisy["pml"] = 1000.0 + 20.0 * rng.standard_normal(len(noisy))
    stats = preprocess.rolling_stats(noisy)
    assert stats.loc[D(2024, 1, 29), "sigma"] == pytest.approx(50.0, rel=0.05)


def test_level_features():
    stats = pd.DataFrame(
        {"mu": [1000.0, -2.0], "sigma": [200.0, 10.0], "mu_7d": [1100.0, -2.0]}
    )
    feats = preprocess.level_features(stats)
    assert feats.shape == (2, 3)
    assert feats[0].tolist() == [1.0, 0.2, 0.5]
    assert np.isfinite(feats).all()  # no explosion when mu is near zero
    logged = preprocess.level_features(stats, volatility="log1p_sigma")
    assert logged[0, 1] == pytest.approx(np.log1p(200.0))
    with pytest.raises(ValueError):
        preprocess.level_features(stats, volatility="sigma_over_mu")
