#!/usr/bin/env python3
"""Test error, skill and product metrics and the report writer."""

import numpy as np
import pytest

from pmlcast import evaluate


def _curve(peak_hours, base=100.0, peak=200.0):
    y = np.full(24, base)
    for h in peak_hours:
        y[h] = peak
    return y


def test_error_metrics_by_hand():
    y_true = np.array([[100.0, 200.0, 300.0] + [0.0] * 21])
    y_pred = np.array([[110.0, 180.0, 300.0] + [0.0] * 21])
    assert evaluate.mae(y_true, y_pred) == pytest.approx(30.0 / 24)
    assert evaluate.rmse(y_true, y_pred) == pytest.approx(np.sqrt(500.0 / 24))
    # sMAPE: 0/0 terms count as 0
    expected = 100 * (2 * 10 / 210 + 2 * 20 / 380) / 24
    assert evaluate.smape(y_true, y_pred) == pytest.approx(expected)
    assert evaluate.per_hour_mae(y_true, y_pred)[1] == 20.0
    with pytest.raises(ValueError):
        evaluate.mae(y_true, y_pred[:, :5])


def test_top4_overlap_is_set_overlap_not_exact_match():
    y_true = np.array([_curve([18, 19, 20, 21])])
    y_pred = np.array([_curve([17, 19, 20, 21])])
    assert evaluate.top4_overlap(y_true, y_pred) == 0.75
    assert evaluate.top4_overlap(y_true, y_true) == 1.0
    disjoint = np.array([_curve([1, 2, 3, 4])])
    assert evaluate.top4_overlap(y_true, disjoint) == 0.0


def test_captured_value():
    y_true = np.array([_curve([18, 19, 20, 21], base=100.0, peak=200.0)])
    y_pred = np.array([_curve([17, 19, 20, 21])])
    # captured 100 + 3 * 200 out of 4 * 200
    assert evaluate.captured_value(y_true, y_pred) == pytest.approx(
        700.0 / 800.0
    )
    assert evaluate.captured_value(y_true, y_true) == 1.0
    zeros = np.zeros((1, 24))
    assert np.isnan(evaluate.captured_value(zeros, y_pred))
    both = np.vstack([y_true, zeros])
    values = evaluate.captured_value_per_sample(
        both, np.vstack([y_pred, y_pred])
    )
    assert np.isnan(values[1]) and values[0] == pytest.approx(0.875)
    assert evaluate.captured_value(
        both, np.vstack([y_pred, y_pred])
    ) == pytest.approx(0.875)


def test_spike_days():
    y_true = np.array(
        [
            _curve([18], base=100.0, peak=200.0),
            _curve([18], base=100.0, peak=900.0),
        ]
    )
    flags = evaluate.spike_days(
        y_true, mu=np.array([100.0, 100.0]), sigma=np.array([50.0, 50.0])
    )
    assert flags.tolist() == [False, True]
