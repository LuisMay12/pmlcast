#!/usr/bin/env python3
"""Test the LSTM builders, the tf.data pipeline and a short training run."""

import os

import numpy as np
import pytest

from pmlcast import models
from pmlcast import train


def _tiny_inputs(n=6, with_meta=False, with_zone=False, with_node=False):
    rng = np.random.default_rng(0)
    inputs = {
        "seq": rng.normal(size=(n, 168, 9)).astype(np.float32),
        "cal": rng.normal(size=(n, 10)).astype(np.float32),
        "level": rng.normal(size=(n, 3)).astype(np.float32),
    }
    if with_meta:
        inputs["meta"] = rng.normal(size=(n, 9)).astype(np.float32)
    if with_zone:
        inputs["zone"] = rng.integers(0, 5, size=n).astype(np.int32)
    if with_node:
        inputs["node"] = rng.integers(0, 3, size=n).astype(np.int32)
    return inputs


@pytest.mark.parametrize(
    "kwargs, extras",
    [
        ({}, {}),
        ({"n_meta": 9, "n_zones": 5}, {"with_meta": True, "with_zone": True}),
        (
            {"n_meta": 9, "n_zones": 5, "n_nodes": 3},
            {"with_meta": True, "with_zone": True, "with_node": True},
        ),
    ],
)
def test_build_model_shapes(kwargs, extras):
    model = models.build_model((168, 9), 10, 3, units=8, **kwargs)
    inputs = _tiny_inputs(**extras)
    out = models.predict_z(model, inputs)
    assert out.shape == (6, 24)
    assert set(model.input.keys()) == set(inputs.keys())
    with pytest.raises(ValueError):
        models.build_model((168, 9), 10, 3, loss="mse")


def test_train_history_only_short_run(dataset_small, tmp_path):
    ds = dataset_small
    checkpoint = str(tmp_path / "model.keras")
    result = train.train_and_score(
        ds,
        kind="history",
        epochs=2,
        batch_size=64,
        units=8,
        patience=1,
        checkpoint_path=checkpoint,
        verbose=0,
    )
    n_test = len(ds.positions("test"))
    assert result["y_pred"].shape == (n_test, 24)
    assert np.isfinite(result["y_pred"]).all()
    assert "val_loss" in result["history"].history
    assert result["summary"]["all"]["n_samples"] == n_test
    assert set(result["references"]) == {"naive24", "naive168"}
    assert os.path.exists(checkpoint)

    docs = tmp_path / "docs" / "baselines.md"
    train.append_summary(
        str(docs),
        "lstm_history",
        "small",
        result["summary"],
        {"naive24": 1.0, "naive168": 2.0},
    )
    assert "## lstm_history on dataset small" in docs.read_text()
