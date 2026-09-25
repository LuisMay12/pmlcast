#!/usr/bin/env python3
"""Score forecasts with error, skill and product metrics and write reports."""

import os

import matplotlib
import numpy as np
import pandas as pd

from pmlcast import config

matplotlib.use("Agg")

TOP_K = 4
SUBSETS = ("all", "no_spike", "spike")
METRIC_COLUMNS = (
    "mae",
    "rmse",
    "smape",
    "skill",
    "top4_overlap",
    "captured_value",
    "n_samples",
)


# ---------------------------------------------------------------------------
# Error metrics (inputs are (n, 24) arrays in MXN/MWh)
# ---------------------------------------------------------------------------


def _check(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.ndim != 2:
        raise ValueError("expected arrays of shape (n_samples, 24)")
    return y_true, y_pred


def mae(y_true, y_pred):
    """Return the mean absolute error."""
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred):
    """Return the root mean squared error."""
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def smape(y_true, y_pred):
    """Return the symmetric MAPE in percent, treating 0/0 terms as 0."""
    y_true, y_pred = _check(y_true, y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    safe = np.where(denominator < 1e-6, 1.0, denominator)
    terms = np.where(
        denominator < 1e-6, 0.0, 2.0 * np.abs(y_pred - y_true) / safe
    )
    return float(100.0 * terms.mean())


def skill(mae_model, mae_reference):
    """Return ``1 - MAE_model / MAE_reference`` (NaN if reference is 0)."""
    if mae_reference is None or mae_reference <= 0:
        return float("nan")
    return float(1.0 - mae_model / mae_reference)


def per_hour_mae(y_true, y_pred):
    """Return the MAE of each of the 24 hours, shape (24,)."""
    y_true, y_pred = _check(y_true, y_pred)
    return np.abs(y_pred - y_true).mean(axis=0)


# ---------------------------------------------------------------------------
# Product metrics
# ---------------------------------------------------------------------------


def top_k_sets(y, k=TOP_K):
    """Return the indices of the k most expensive hours per sample."""
    return np.argsort(-np.asarray(y, dtype=float), axis=1, kind="stable")[
        :, :k
    ]


def top4_overlap_per_sample(y_true, y_pred, k=TOP_K):
    """Return ``|PredTopK ∩ ActualTopK| / k`` for every sample."""
    y_true, y_pred = _check(y_true, y_pred)
    true_sets = top_k_sets(y_true, k)
    pred_sets = top_k_sets(y_pred, k)
    matches = (pred_sets[:, :, None] == true_sets[:, None, :]).any(axis=2)
    return matches.sum(axis=1) / float(k)


def top4_overlap(y_true, y_pred, k=TOP_K):
    """Return the mean set overlap of the k most expensive hours."""
    return float(top4_overlap_per_sample(y_true, y_pred, k).mean())


def captured_value_per_sample(y_true, y_pred, k=TOP_K):
    """Return the share of the best k-hour revenue captured per sample.

    Samples whose actual top-k revenue is not positive get NaN.
    """
    y_true, y_pred = _check(y_true, y_pred)
    true_sets = top_k_sets(y_true, k)
    pred_sets = top_k_sets(y_pred, k)
    best = np.take_along_axis(y_true, true_sets, axis=1).sum(axis=1)
    got = np.take_along_axis(y_true, pred_sets, axis=1).sum(axis=1)
    return np.where(best > 0, got / np.where(best > 0, best, 1.0), np.nan)


def captured_value(y_true, y_pred, k=TOP_K):
    """Return the mean captured value, ignoring undefined samples."""
    values = captured_value_per_sample(y_true, y_pred, k)
    if np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def spike_days(y_true, mu, sigma, k=config.SPIKE_K_SIGMA):
    """Flag days whose highest price exceeds ``mu + k * sigma``.

    Args:
        y_true: Actual prices, shape (n, 24), in MXN/MWh.
        mu: Frozen per-sample means, shape (n,).
        sigma: Frozen per-sample (floored) standard deviations, shape (n,).
        k: Threshold in standard deviations.

    Returns:
        Boolean array of shape (n,).
    """
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return y_true.max(axis=1) > mu + k * sigma


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def metrics(y_true, y_pred, reference_pred=None):
    """Compute every metric on one set of samples.

    Args:
        y_true: Actual prices (n, 24).
        y_pred: Predicted prices (n, 24).
        reference_pred: Optional reference forecast for the skill score.

    Returns:
        Dict with ``METRIC_COLUMNS``; ``skill`` is NaN without reference
        and every value is NaN when there are no samples.
    """
    y_true, y_pred = _check(y_true, y_pred)
    if len(y_true) == 0:
        out = {key: float("nan") for key in METRIC_COLUMNS}
        out["n_samples"] = 0
        return out
    model_mae = mae(y_true, y_pred)
    reference_mae = None
    if reference_pred is not None:
        reference_mae = mae(y_true, reference_pred)
    return {
        "mae": model_mae,
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "skill": skill(model_mae, reference_mae),
        "top4_overlap": top4_overlap(y_true, y_pred),
        "captured_value": captured_value(y_true, y_pred),
        "n_samples": int(len(y_true)),
    }


def summarize(y_true, y_pred, is_spike=None, reference_pred=None):
    """Compute metrics for all samples and for the spike / no-spike subsets.

    Returns:
        Dict with keys ``all``, ``no_spike`` and ``spike``.
    """
    y_true, y_pred = _check(y_true, y_pred)
    if is_spike is None:
        is_spike = np.zeros(len(y_true), dtype=bool)
    is_spike = np.asarray(is_spike, dtype=bool)
    masks = {
        "all": np.ones(len(y_true), dtype=bool),
        "no_spike": ~is_spike,
        "spike": is_spike,
    }
    out = {}
    for name, mask in masks.items():
        ref = (
            None
            if reference_pred is None
            else np.asarray(reference_pred)[mask]
        )
        out[name] = metrics(y_true[mask], y_pred[mask], ref)
    return out


def summarize_by(y_true, y_pred, groups, reference_pred=None):
    """Compute metrics per group (e.g. region or node).

    Args:
        y_true: Actual prices (n, 24).
        y_pred: Predicted prices (n, 24).
        groups: Array-like of length n with a group label per sample.
        reference_pred: Optional reference forecast for the skill score.

    Returns:
        DataFrame with one row per group and ``METRIC_COLUMNS``.
    """
    y_true, y_pred = _check(y_true, y_pred)
    groups = np.asarray(groups)
    rows = []
    for label in sorted(set(groups.tolist())):
        mask = groups == label
        ref = (
            None
            if reference_pred is None
            else np.asarray(reference_pred)[mask]
        )
        row = metrics(y_true[mask], y_pred[mask], ref)
        row["group"] = label
        rows.append(row)
    return pd.DataFrame(rows, columns=["group"] + list(METRIC_COLUMNS))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _format_table(frame, floats=2):
    return frame.to_markdown(index=False, floatfmt=".{}f".format(floats))


def _plot_mae_by_hour(y_true, predictions, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, y_pred in predictions.items():
        ax.plot(range(1, 25), per_hour_mae(y_true, y_pred), label=name)
    ax.set_xlabel("hour of day (hora)")
    ax.set_ylabel("MAE (MXN/MWh)")
    ax.set_title("MAE by hour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_bars(table, column, path, title, ylabel):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table["model"], table[column])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_example(y_true, predictions, index, position, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    hours = range(1, 25)
    ax.plot(
        hours, y_true[position], color="black", linewidth=2, label="actual"
    )
    for name, y_pred in predictions.items():
        ax.plot(hours, y_pred[position], label=name)
    row = index.iloc[position]
    ax.set_title("{} target {}".format(row["node"], row["target_date"]))
    ax.set_xlabel("hour of day (hora)")
    ax.set_ylabel("PML (MXN/MWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(
    y_true,
    predictions,
    index,
    out_md,
    fig_dir,
    reference="naive168",
    title="Forecast evaluation",
):
    """Write a markdown report with tables and figures for several models.

    Args:
        y_true: Actual prices (n, 24) in MXN/MWh.
        predictions: Dict model name -> predicted prices (n, 24).
        index: DataFrame with one row per sample; needs ``node``,
            ``region``, ``target_date`` and ``is_spike_day``.
        out_md: Path of the markdown file to write.
        fig_dir: Directory for the PNG figures.
        reference: Name of the model used as skill reference.
        title: Report title.

    Returns:
        The markdown text.
    """
    y_true = np.asarray(y_true, dtype=float)
    if reference not in predictions:
        raise ValueError(
            "reference model {} not in predictions".format(reference)
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    is_spike = index["is_spike_day"].to_numpy(dtype=bool)
    reference_pred = np.asarray(predictions[reference], dtype=float)

    lines = ["# {}".format(title), ""]
    lines.append(
        "{} samples, {} nodes, targets {} to {}; {} spike days "
        "(max hour above mu + {:g} sigma). Skill is relative to `{}`.".format(
            len(index),
            index["node"].nunique(),
            index["target_date"].min(),
            index["target_date"].max(),
            int(is_spike.sum()),
            config.SPIKE_K_SIGMA,
            reference,
        )
    )
    lines.append("")

    summaries = {
        name: summarize(y_true, y_pred, is_spike, reference_pred)
        for name, y_pred in predictions.items()
    }
    for subset in SUBSETS:
        rows = []
        for name in predictions:
            row = {"model": name}
            row.update(summaries[name][subset])
            rows.append(row)
        table = pd.DataFrame(rows, columns=["model"] + list(METRIC_COLUMNS))
        lines.append("## Metrics: {}".format(subset.replace("_", " ")))
        lines.append("")
        lines.append(_format_table(table, 3))
        lines.append("")
        if subset == "all":
            overall = table

    lines.append("## MAE and skill by region")
    lines.append("")
    regions = index["region"].to_numpy()
    by_region = None
    for name, y_pred in predictions.items():
        table = summarize_by(y_true, y_pred, regions, reference_pred)
        table = table[["group", "mae", "skill", "n_samples"]].rename(
            columns={
                "group": "region",
                "mae": "mae_" + name,
                "skill": "skill_" + name,
            }
        )
        if by_region is None:
            by_region = table
        else:
            by_region = by_region.merge(
                table.drop(columns="n_samples"), on="region"
            )
    lines.append(_format_table(by_region, 3))
    lines.append("")

    figures = {
        "mae_by_hour.png": lambda p: _plot_mae_by_hour(y_true, predictions, p),
        "skill_by_model.png": lambda p: _plot_bars(
            overall, "skill", p, "Skill vs {}".format(reference), "skill"
        ),
        "top4_overlap.png": lambda p: _plot_bars(
            overall, "top4_overlap", p, "Top-4 overlap", "share"
        ),
    }
    if len(index):
        position = int(np.argmax(y_true.max(axis=1) - y_true.min(axis=1)))
        figures["example_forecast.png"] = lambda p: _plot_example(
            y_true, predictions, index, position, p
        )
    lines.append("## Figures")
    lines.append("")
    for name, plot in figures.items():
        path = os.path.join(fig_dir, name)
        plot(path)
        rel = os.path.relpath(path, os.path.dirname(os.path.abspath(out_md)))
        lines.append("![{}]({})".format(name, rel))
        lines.append("")

    text = "\n".join(lines)
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text
