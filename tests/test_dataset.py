#!/usr/bin/env python3
"""Test window construction, splits, hold-out folds and persistence."""

import datetime

import numpy as np
import pandas as pd

from pmlcast import dataset
from pmlcast import preprocess

D = datetime.date


def test_enumerate_origins_rules():
    n = 12
    missing = np.zeros(n, bool)
    interp = np.zeros(n, int)
    valid_stats = np.ones(n, bool)
    valid = dataset.enumerate_origins(missing, interp, valid_stats, 12)
    assert valid.tolist() == [False] * 6 + [True] * 5 + [False]
    missing[9] = True  # kills origins whose input or target touch day 9
    valid = dataset.enumerate_origins(missing, interp, valid_stats, 12)
    assert valid[6] and valid[7]  # their inputs end before day 9
    assert not valid[8:11].any()  # origins 8..10 touch day 9
    missing[9] = False
    interp[10] = 1  # target-day interpolation kills origin 9 only
    valid = dataset.enumerate_origins(missing, interp, valid_stats, 12)
    assert not valid[9] and valid[8] and valid[10]
    interp[:] = 0
    interp[7] = 13  # too many interpolated hours in the input
    valid = dataset.enumerate_origins(missing, interp, valid_stats, 12)
    assert not valid[7:11].any()
    assert (
        dataset.enumerate_origins(
            missing[:5], interp[:5], valid_stats[:5], 12
        ).sum()
        == 0
    )


def test_enumerate_origins_follows_the_window_length():
    n = 12
    clean = np.zeros(n, bool)
    interp = np.zeros(n, int)
    valid_stats = np.ones(n, bool)

    week = dataset.enumerate_origins(clean, interp, valid_stats, 12, 7)
    three = dataset.enumerate_origins(clean, interp, valid_stats, 12, 3)

    # a shorter window needs less history, so more days become origins
    assert week.sum() == 5 and three.sum() == 9
    assert np.flatnonzero(week)[0] == 6
    assert np.flatnonzero(three)[0] == 2
    # the target day is still excluded from being an origin
    assert not week[-1] and not three[-1]
    # a window longer than the series yields nothing
    assert (
        dataset.enumerate_origins(clean, interp, valid_stats, 12, 20).sum()
        == 0
    )


def test_shapes_and_inversion(dataset_small, silver_small):
    ds = dataset_small
    n = len(ds)
    assert ds.arrays["X_seq"].shape == (n, 168, 9)
    assert ds.arrays["X_cal"].shape == (n, 10)
    assert ds.arrays["X_level"].shape == (n, 3)
    assert ds.arrays["X_meta"].shape == (n, 9)
    assert ds.arrays["y"].shape == (n, 24)
    assert ds.arrays["X_seq"].dtype == np.float32
    assert list(ds.index.columns) == dataset.INDEX_COLUMNS
    assert (
        np.isfinite(ds.arrays["X_seq"]).all()
        and np.isfinite(ds.arrays["y"]).all()
    )

    # inverted inputs equal the silver prices of the window; target equals D+1
    pos = 10
    row = ds.index.iloc[pos]
    node = silver_small[silver_small["node"] == row["node"]].set_index(
        "ts_local"
    )
    start = pd.Timestamp(row["origin_date"]) - pd.Timedelta(days=6)
    end = pd.Timestamp(row["origin_date"]) + pd.Timedelta(hours=23)
    window = node.loc[start:end, "pml"].to_numpy()
    back = preprocess.invert(
        ds.arrays["X_seq"][pos, :, 0], row["mu"], row["sigma"]
    )
    assert np.allclose(back, window, atol=0.05)
    target = node.loc[str(row["target_date"]), "pml"].to_numpy()
    assert np.allclose(ds.y_prices([pos])[0], target, atol=0.05)
    assert row["target_date"] == row["origin_date"] + datetime.timedelta(
        days=1
    )
    # components share sigma and keep the identity in z-space
    comp_sum = ds.arrays["X_seq"][pos, :, 1:4].sum(axis=1) * row["sigma"]
    comp_sum += ds.arrays["comp_mu"][pos].sum()
    assert np.allclose(comp_sum, window, atol=0.1)


def test_gap_windows_are_excluded(dataset_small):
    ds = dataset_small
    gappy = ds.index[ds.index["node"] == "07PNC-115"]
    targets = pd.to_datetime(gappy["target_date"])
    origins = pd.to_datetime(gappy["origin_date"])
    # the 30-hour gap spans 2022-06-01 05:00 .. 2022-06-02 10:00
    blocked = (
        origins - pd.Timedelta(days=6) <= pd.Timestamp("2022-06-02")
    ) & (targets >= pd.Timestamp("2022-06-01"))
    assert not blocked.any()
    # the 2-hour gap (2022-05-10) may sit in the input, never in the target
    assert (targets == pd.Timestamp("2022-05-10")).sum() == 0
    inputs_with_gap = gappy[
        (origins >= pd.Timestamp("2022-05-10"))
        & (origins <= pd.Timestamp("2022-05-16"))
    ]
    assert (inputs_with_gap["n_interp_input"] == 2).all() and len(
        inputs_with_gap
    ) > 0
    clean = ds.index[ds.index["node"] == "07TIZ-400"]
    assert (clean["n_interp_input"] == 0).all()
    # stats need 605 of 672 hours, so the first valid origin is day 25
    assert len(clean) == 240 - 26


def test_splits_and_walk_forward(dataset_small):
    ds = dataset_small
    index = ds.index
    targets = pd.to_datetime(index["target_date"])
    for split in dataset.SPLITS:
        assert (index["split"] == split).any()
    assert (
        targets[index["split"] == "train"].max()
        < targets[index["split"] == "val"].min()
    )
    assert (
        targets[index["split"] == "val"].max()
        < targets[index["split"] == "test"].min()
    )
    bounds = ds.meta["splits"]
    assert bounds["test_end"] == str(targets.max().date())
    assert (targets[index["split"] == "val"].nunique()) == 14

    dates = [d for d, _ in dataset.walk_forward(index, "test")]
    assert dates == sorted(dates) and len(dates) == len(set(dates))
    total = sum(len(p) for _, p in dataset.walk_forward(index, "test"))
    assert total == (index["split"] == "test").sum()
    assert len(ds.positions(split="test", nodes=["07TIZ-400"])) == len(dates)


def test_hold_out_folds(dataset_small):
    ds = dataset_small
    folds = list(dataset.leave_one_node_out(ds.index))
    assert [f["held_out"] for f in folds] == [
        "07MRD-230",
        "07PNC-115",
    ]  # MERIDA has 2 nodes
    for fold in folds:
        nodes_train = set(ds.index.iloc[fold["train"]]["node"])
        assert fold["held_out"] not in nodes_train
        assert (ds.index.iloc[fold["test"]]["node"] == fold["held_out"]).all()
        assert (ds.index.iloc[fold["test"]]["split"] == "test").all()
        assert (ds.index.iloc[fold["val"]]["split"] == "val").all()

    zone_folds = list(dataset.leave_one_zone_out(ds.index, max_folds=3))
    assert [f["held_out"] for f in zone_folds] == ["MERIDA", "CANCUN"]
    merida = zone_folds[0]
    assert merida["nodes"] == ["07MRD-230", "07PNC-115"]
    assert merida["zone_override"] == 0
    assert "MERIDA" not in set(ds.index.iloc[merida["train"]]["zone"])
    inputs = dataset.inputs_dict(
        ds, merida["test"], use_meta=True, use_zone=True, zone_override=0
    )
    assert set(inputs) == {"seq", "cal", "level", "meta", "zone"}
    assert (inputs["zone"] == 0).all()
    plain = dataset.inputs_dict(ds, merida["test"], use_zone=True)
    assert (plain["zone"] == ds.meta["zone_vocab"]["MERIDA"]).all()
