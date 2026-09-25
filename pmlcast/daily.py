#!/usr/bin/env python3
"""Run the daily forecast and score the forecasts CENACE has published."""

import argparse
import datetime
import os

import pandas as pd

import pmlcast
from pmlcast import cenace
from pmlcast import config
from pmlcast import evaluate
from pmlcast import serving
from pmlcast import storage

FORECAST_TABLE = "daily_forecast"
SCORE_TABLE = "daily_scores"
FORECAST_COLUMNS = [
    "node",
    "market",
    "origin_date",
    "target_date",
    "hora",
    "pml_pred",
    "model_version",
    "issued_at",
    "run_at",
]
SCORE_COLUMNS = [
    "node",
    "target_date",
    "mae",
    "rmse",
    "top4_overlap",
    "captured_value",
    "naive24_mae",
    "naive168_mae",
    "skill_vs_naive24",
    "scored_at",
]


def forecast_all(forecaster, nodes, target_date, log=None):
    """Forecast one day for every node, skipping the ones that cannot be.

    A node without enough published history is logged and left out
    rather than failing the whole run: the job has to survive one bad
    node on a morning when offers are about to close.
    """
    log = log or config.setup_logging()
    rows = []
    skipped = []

    for node in nodes:
        try:
            result = forecaster.forecast(node, target_date)
        except serving.ForecastError as error:
            skipped.append((node, str(error)))
            continue

        for hour in result["hours"]:
            rows.append(
                {
                    "node": node,
                    "market": result["market"],
                    "origin_date": target_date - datetime.timedelta(days=1),
                    "target_date": target_date,
                    "hora": hour["hora"],
                    "pml_pred": hour["pml"],
                    "model_version": result["model_version"],
                    "issued_at": result["issued_at"],
                    "run_at": pd.Timestamp.now(tz="UTC"),
                }
            )

    for node, reason in skipped:
        log.warning("skipped %s: %s", node, reason)
    log.info(
        "forecast %s: %d nodes, %d skipped",
        target_date,
        len(rows) // 24,
        len(skipped),
    )

    return pd.DataFrame(rows, columns=FORECAST_COLUMNS)


def merge_forecasts(gold_dir, frame):
    """Append today's forecasts, replacing any earlier run of the same day."""
    if frame.empty:
        return frame

    path = storage.predictions_path(gold_dir, FORECAST_TABLE)
    if os.path.exists(path):
        old = storage.read_predictions(gold_dir, FORECAST_TABLE)
        targets = set(frame["target_date"])
        old = old[~old["target_date"].isin(targets)]
        frame = pd.concat([old, frame], ignore_index=True)

    frame = frame.sort_values(["target_date", "node", "hora"])
    storage.write_predictions(gold_dir, FORECAST_TABLE, frame)

    return frame


def score_published(gold_dir, silver_dir, market="MDA", log=None):
    """Score every stored forecast whose real prices are now published."""
    log = log or config.setup_logging()
    path = storage.predictions_path(gold_dir, FORECAST_TABLE)
    if not os.path.exists(path):
        log.info("no forecasts to score yet")
        return pd.DataFrame(columns=SCORE_COLUMNS)

    forecasts = storage.read_predictions(gold_dir, FORECAST_TABLE)
    done = set()
    score_path = storage.predictions_path(gold_dir, SCORE_TABLE)
    if os.path.exists(score_path):
        scored = storage.read_predictions(gold_dir, SCORE_TABLE)
        done = set(zip(scored["node"], scored["target_date"]))

    rows = []
    for (node, target), group in forecasts.groupby(["node", "target_date"]):
        if (node, target) in done:
            continue

        actual = storage.read_silver(
            silver_dir, market, nodes=[node], start=target, end=target
        )
        if len(actual) < config.TARGET_HOURS or actual["pml"].isna().any():
            continue  # CENACE has not published this day yet

        group = group.sort_values("hora")
        y_true = actual.sort_values("hora")["pml"].to_numpy()[None, :]
        y_pred = group["pml_pred"].to_numpy()[None, :]

        history = storage.read_silver(
            silver_dir,
            market,
            nodes=[node],
            start=target - datetime.timedelta(days=7),
            end=target - datetime.timedelta(days=1),
        )
        naive24 = _day_prices(history, target - datetime.timedelta(days=1))
        naive168 = _day_prices(history, target - datetime.timedelta(days=7))

        row = {
            "node": node,
            "target_date": target,
            "mae": evaluate.mae(y_true, y_pred),
            "rmse": evaluate.rmse(y_true, y_pred),
            "top4_overlap": evaluate.top4_overlap(y_true, y_pred),
            "captured_value": evaluate.captured_value(y_true, y_pred),
            "naive24_mae": _safe_mae(y_true, naive24),
            "naive168_mae": _safe_mae(y_true, naive168),
            "scored_at": pd.Timestamp.now(tz="UTC"),
        }
        row["skill_vs_naive24"] = evaluate.skill(
            row["mae"], row["naive24_mae"]
        )
        rows.append(row)

    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    if not scores.empty:
        if os.path.exists(score_path):
            scores = pd.concat(
                [storage.read_predictions(gold_dir, SCORE_TABLE), scores],
                ignore_index=True,
            )
        storage.write_predictions(gold_dir, SCORE_TABLE, scores)
    log.info("scored %d new (node, day) forecasts", len(rows))

    return scores


def _day_prices(history, day):
    """Return the 24 prices of one day, or None when they are not there."""
    rows = history[history["fecha"] == day].sort_values("hora")
    if len(rows) != config.TARGET_HOURS or rows["pml"].isna().any():
        return None

    return rows["pml"].to_numpy()[None, :]


def _safe_mae(y_true, y_pred):
    """Return the MAE, or NaN when the reference day is missing."""
    if y_pred is None:
        return float("nan")

    return evaluate.mae(y_true, y_pred)


def recent_skill(gold_dir, days=30):
    """Summarize how the model has been doing over the last N scored days."""
    path = storage.predictions_path(gold_dir, SCORE_TABLE)
    if not os.path.exists(path):
        return {}

    scores = storage.read_predictions(gold_dir, SCORE_TABLE)
    if scores.empty:
        return {}

    cutoff = max(scores["target_date"]) - datetime.timedelta(days=days)
    recent = scores[scores["target_date"] > cutoff]

    return {
        "days": int(recent["target_date"].nunique()),
        "nodes": int(recent["node"].nunique()),
        "mae": float(recent["mae"].mean()),
        "top4_overlap": float(recent["top4_overlap"].mean()),
        "captured_value": float(recent["captured_value"].mean()),
        "skill_vs_naive24": float(recent["skill_vs_naive24"].mean()),
    }


def main():
    """Forecast tomorrow, then score whatever CENACE has published."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes-file",
        default=os.path.join(config.SNAPSHOT_DIR, "nodes_stage2.csv"),
    )
    parser.add_argument("--target-date", type=datetime.date.fromisoformat)
    parser.add_argument("--model", default=serving.DEFAULT_MODEL)
    parser.add_argument("--meta", default=serving.DEFAULT_META)
    parser.add_argument("--gold-dir", default=config.GOLD_DIR)
    parser.add_argument("--silver-dir", default=config.SILVER_DIR)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    log = config.setup_logging(os.path.join(config.DATA_DIR, "daily.log"))
    log.info("pmlcast %s daily job", pmlcast.__version__)

    if not args.score_only:
        forecaster = serving.Forecaster(
            model_path=args.model,
            meta_path=args.meta,
            silver_dir=args.silver_dir,
        )
        nodes = cenace.read_nodes_file(args.nodes_file)
        target = args.target_date or config.today_local() + datetime.timedelta(
            days=1
        )
        frame = forecast_all(forecaster, nodes, target, log)
        merge_forecasts(args.gold_dir, frame)

    score_published(args.gold_dir, args.silver_dir, log=log)
    summary = recent_skill(args.gold_dir)
    if summary:
        log.info("last 30 days: %s", summary)


if __name__ == "__main__":
    main()
