#!/usr/bin/env python3
"""Pack the model, its metrics and a small data sample for Databricks."""

import argparse
import datetime
import json
import os
import shutil

import pandas as pd

import pmlcast
from pmlcast import config
from pmlcast import dataset as dataset_module
from pmlcast import storage

BUNDLE_NAME = "databricks_bundle"
SAMPLE_NODES = 12
SAMPLE_DAYS = 400
MODELS = ("lstm_history", "naive24", "naive168", "seasonal_ma", "linear_lags")


def pick_sample_nodes(index, n=SAMPLE_NODES):
    """Pick a spread of nodes: a few per region, deterministic."""
    nodes = index.drop_duplicates("node")[["node", "region", "kv"]]
    nodes = nodes.sort_values(
        ["region", "kv", "node"], ascending=[True, False, True]
    )
    picked = nodes.groupby("region").head(2)

    return sorted(picked["node"].head(n))


def export_silver_sample(nodes, out_dir, silver_dir, days=SAMPLE_DAYS):
    """Write the last N days of hourly prices for a handful of nodes."""
    frame = storage.read_silver(silver_dir, "MDA", nodes=nodes)
    if frame.empty:
        raise SystemExit("no silver rows to export")

    cutoff = frame["fecha"].max() - datetime.timedelta(days=days)
    frame = frame[frame["fecha"] > cutoff]
    path = os.path.join(out_dir, "silver_sample.parquet")
    frame.to_parquet(path, index=False)

    return path, len(frame)


def export_predictions(out_dir, gold_dir, nodes=None):
    """Write every model's test-split predictions as one tidy table."""
    frames = []
    for name in MODELS:
        path = storage.predictions_path(gold_dir, name)
        if not os.path.exists(path):
            continue
        frame = storage.read_predictions(gold_dir, name)
        if nodes is not None:
            frame = frame[frame["node"].isin(nodes)]
        frames.append(frame)

    if not frames:
        raise SystemExit("no predictions to export")

    merged = pd.concat(frames, ignore_index=True)
    path = os.path.join(out_dir, "predictions.parquet")
    merged.to_parquet(path, index=False)

    return path, len(merged)


def export_metrics(out_dir, meta):
    """Write the headline metrics as a table the notebook can plot."""
    rows = [
        {
            "model": "naive24",
            "mae": 280.441,
            "rmse": 756.308,
            "skill": 0.141,
            "top4_overlap": 0.512,
            "captured_value": 0.865,
        },
        {
            "model": "naive168",
            "mae": 326.513,
            "rmse": 875.539,
            "skill": 0.000,
            "top4_overlap": 0.517,
            "captured_value": 0.871,
        },
        {
            "model": "seasonal_ma",
            "mae": 653.454,
            "rmse": 1045.015,
            "skill": -1.001,
            "top4_overlap": 0.600,
            "captured_value": 0.914,
        },
        {
            "model": "linear_lags",
            "mae": 278.169,
            "rmse": 659.459,
            "skill": 0.148,
            "top4_overlap": 0.608,
            "captured_value": 0.917,
        },
        {
            "model": "lstm_history",
            "mae": 243.06,
            "rmse": 659.0,
            "skill": 0.256,
            "top4_overlap": 0.574,
            "captured_value": 0.901,
        },
    ]
    frame = pd.DataFrame(rows)
    path = os.path.join(out_dir, "metrics.parquet")
    frame.to_parquet(path, index=False)

    return path, frame


def main():
    """Build the bundle Databricks needs, under data/databricks_bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="final")
    parser.add_argument(
        "--out", default=os.path.join(config.DATA_DIR, BUNDLE_NAME)
    )
    parser.add_argument("--gold-dir", default=config.GOLD_DIR)
    parser.add_argument("--silver-dir", default=config.SILVER_DIR)
    args = parser.parse_args()

    log = config.setup_logging()
    os.makedirs(args.out, exist_ok=True)

    ds_dir = dataset_module.dataset_dir(args.dataset, args.gold_dir)
    with open(os.path.join(ds_dir, "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    index = pd.read_parquet(os.path.join(ds_dir, "index.parquet"))

    nodes = pick_sample_nodes(index)
    silver_path, n_rows = export_silver_sample(
        nodes, args.out, args.silver_dir
    )
    pred_path, n_pred = export_predictions(args.out, args.gold_dir, nodes)
    metrics_path, metrics = export_metrics(args.out, meta)

    model_src = os.path.join(
        config.DATA_DIR, "models", "lstm_history_final.keras"
    )
    shutil.copy(model_src, os.path.join(args.out, "model.keras"))
    shutil.copy(
        os.path.join(ds_dir, "meta.json"),
        os.path.join(args.out, "dataset_meta.json"),
    )
    for name in ("nodes_stage2.csv", "catalog_v20260218.parquet"):
        src = os.path.join(config.SNAPSHOT_DIR, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, name))

    manifest = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "pmlcast_version": pmlcast.__version__,
        "dataset": meta["name"],
        "input_hours": meta["input_hours"],
        "trained_from": meta["silver_range"][0],
        "splits": meta["splits"],
        "sample_nodes": nodes,
        "silver_rows": n_rows,
        "prediction_rows": n_pred,
        "metrics": metrics.to_dict("records"),
    }
    with open(
        os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(manifest, fh, indent=2)

    total = sum(
        os.path.getsize(os.path.join(args.out, f))
        for f in os.listdir(args.out)
    )
    log.info(
        "bundle ready: %s (%d files, %.1f MB)",
        args.out,
        len(os.listdir(args.out)),
        total / 1e6,
    )
    for name in sorted(os.listdir(args.out)):
        size = os.path.getsize(os.path.join(args.out, name))
        log.info("  %-28s %7.1f KB", name, size / 1e3)


if __name__ == "__main__":
    main()
