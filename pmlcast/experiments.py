#!/usr/bin/env python3
"""Run the hyperparameter search and the generalization experiments."""

import argparse
import datetime
import itertools
import os

import pandas as pd

from pmlcast import baselines
from pmlcast import catalog as catalog_module
from pmlcast import config
from pmlcast import dataset as dataset_module
from pmlcast import evaluate
from pmlcast import models
from pmlcast import storage
from pmlcast import train

SEARCH_EXPERIMENT = "pmlcast-search"
HOLDOUT_EXPERIMENT = "pmlcast-holdout"
SEARCH_EPOCHS = 20
SEARCH_PATIENCE = 3
FOLD_EPOCHS = 15
FOLD_PATIENCE = 3
BATCH_SIZE = 256

# The pitch asks for window length, units, dropout, category dropout,
# learning rate and loss. I keep the grid small on purpose: every extra
# combination is another full training run.
SEARCH_GRID = {
    "units": [32, 64],
    "dropout": [0.1, 0.2],
    "loss": ["huber", "mae"],
    "learning_rate": [0.001],
    "zone_dropout": [0.25],
}
SEARCH_COLUMNS = [
    "units",
    "dropout",
    "loss",
    "learning_rate",
    "zone_dropout",
    "val_mae",
    "test_mae",
    "test_skill",
    "test_top4",
    "epochs_run",
]
ABLATIONS = ("history", "metadata", "node_id")

# The window length changes the sample shape, so it gets its own phase
# instead of a slot in SEARCH_GRID.
WINDOW_DAYS = (3, 7, 14)
WINDOW_COLUMNS = [
    "input_days",
    "input_hours",
    "n_samples",
    "val_mae",
    "test_mae",
    "test_skill",
    "test_top4",
]

# Pre-2022 years come from a different market regime, so how far back to
# train is an experiment, not an assumption.
TRAINING_STARTS = (
    datetime.date(2016, 1, 29),
    datetime.date(2019, 1, 1),
    datetime.date(2022, 1, 1),
    datetime.date(2024, 1, 1),
)
TRAINING_COLUMNS = [
    "start",
    "train_years",
    "n_train",
    "val_mae",
    "test_mae",
    "test_skill",
    "test_top4",
]


def grid_combinations(grid):
    """Expand a parameter grid into a list of keyword dicts."""
    keys = sorted(grid)
    combos = []

    for values in itertools.product(*[grid[key] for key in keys]):
        combos.append(dict(zip(keys, values)))

    return combos


def score_split(ds, model, kind, positions, reference_pred=None):
    """Predict one split in MXN/MWh and summarize the result."""
    y_pred = models.predict_prices(
        model,
        train.model_inputs(ds, positions, kind),
        ds.arrays["mu"][positions],
        ds.arrays["sigma"][positions],
    )
    y_true = ds.y_prices(positions)
    is_spike = ds.index["is_spike_day"].to_numpy()[positions]
    summary = evaluate.summarize(y_true, y_pred, is_spike, reference_pred)

    return y_pred, summary


def run_search(ds, grid=None, epochs=SEARCH_EPOCHS, log=None):
    """Train one metadata model per grid point and rank them by val MAE."""
    log = log or config.setup_logging()
    combos = grid_combinations(grid or SEARCH_GRID)
    val = ds.positions("val")
    rows = []

    for i, params in enumerate(combos, 1):
        log.info("[%d/%d] search %s", i, len(combos), params)
        result = train.train_and_score(
            ds,
            kind="metadata",
            epochs=epochs,
            batch_size=BATCH_SIZE,
            patience=SEARCH_PATIENCE,
            verbose=0,
            **params,
        )
        _, val_summary = score_split(ds, result["model"], "metadata", val)
        row = dict(params)
        row["val_mae"] = val_summary["all"]["mae"]
        row["test_mae"] = result["summary"]["all"]["mae"]
        row["test_skill"] = result["summary"]["all"]["skill"]
        row["test_top4"] = result["summary"]["all"]["top4_overlap"]
        row["epochs_run"] = len(result["history"].history["loss"])
        rows.append(row)
        log.info(
            "    val MAE %.2f, test MAE %.2f, skill %.3f",
            row["val_mae"],
            row["test_mae"],
            row["test_skill"],
        )

    table = pd.DataFrame(rows, columns=SEARCH_COLUMNS)
    table = table.sort_values("val_mae").reset_index(drop=True)

    return table


def best_params(table):
    """Return the grid point with the lowest validation MAE.

    Values come out of a DataFrame, so they arrive as numpy scalars.
    Keras rejects a numpy integer where it wants a plain int (units // 2
    stops being an int), so everything is cast back to Python types.
    """
    row = table.iloc[0]
    casts = {
        "units": int,
        "dropout": float,
        "loss": str,
        "learning_rate": float,
        "zone_dropout": float,
    }

    return {key: cast(row[key]) for key, cast in casts.items()}


def train_on_fold(ds, kind, fold, params, epochs=FOLD_EPOCHS):
    """Train on a fold's train/val positions and score its held-out split."""
    models.set_seeds(config.SEED)
    model = train.build_for(
        ds,
        kind,
        int(params["units"]),
        float(params["dropout"]),
        params["loss"],
        float(params["learning_rate"]),
    )
    zone_dropout = 0.0
    if kind != "history":
        zone_dropout = float(params["zone_dropout"])

    models.train_model(
        model,
        train.model_inputs(ds, fold["train"], kind),
        ds.arrays["y"][fold["train"]],
        train.model_inputs(ds, fold["val"], kind),
        ds.arrays["y"][fold["val"]],
        epochs=epochs,
        patience=FOLD_PATIENCE,
        batch_size=BATCH_SIZE,
        zone_dropout=zone_dropout,
        verbose=0,
    )

    # A held-out zone was never seen in training, so it must be scored
    # with the UNKNOWN embedding (index 0).
    override = fold.get("zone_override")
    y_pred = models.predict_prices(
        model,
        train.model_inputs(ds, fold["test"], kind, zone_override=override),
        ds.arrays["mu"][fold["test"]],
        ds.arrays["sigma"][fold["test"]],
    )

    return y_pred


def run_holdout(
    ds, params, regime="node", max_folds=3, log=None, silver_df=None
):
    """Compare the three model kinds on held-out nodes or held-out zones."""
    log = log or config.setup_logging()
    if regime == "node":
        folds = dataset_module.leave_one_node_out(ds.index, max_folds)
    elif regime == "zone":
        folds = dataset_module.leave_one_zone_out(ds.index, max_folds)
    else:
        raise ValueError("regime must be 'node' or 'zone'")

    rows = []
    for fold in folds:
        if len(fold["test"]) == 0 or len(fold["train"]) == 0:
            log.info("skipping fold %s (not enough samples)", fold["held_out"])
            continue

        y_true = ds.y_prices(fold["test"])
        is_spike = ds.index["is_spike_day"].to_numpy()[fold["test"]]
        naive = train.naive168_reference(ds, fold["test"], silver_df)
        naive24 = baselines.Naive24().predict(ds, fold["test"])
        log.info(
            "%s hold-out %s: %d test samples",
            regime,
            fold["held_out"],
            len(fold["test"]),
        )

        for kind in ABLATIONS:
            y_pred = train_on_fold(ds, kind, fold, params)
            summary = evaluate.summarize(y_true, y_pred, is_spike, naive)
            rows.append(
                {
                    "regime": regime,
                    "held_out": fold["held_out"],
                    "model": kind,
                    "n_test": len(fold["test"]),
                    "mae": summary["all"]["mae"],
                    "skill": summary["all"]["skill"],
                    "top4_overlap": summary["all"]["top4_overlap"],
                    "mae_nospike": summary["no_spike"]["mae"],
                }
            )
            log.info(
                "    %-9s MAE %.2f skill %.3f top4 %.3f",
                kind,
                summary["all"]["mae"],
                summary["all"]["skill"],
                summary["all"]["top4_overlap"],
            )

        for name, pred in (("naive168", naive), ("naive24", naive24)):
            summary = evaluate.summarize(y_true, pred, is_spike, naive)
            rows.append(
                {
                    "regime": regime,
                    "held_out": fold["held_out"],
                    "model": name,
                    "n_test": len(fold["test"]),
                    "mae": summary["all"]["mae"],
                    "skill": summary["all"]["skill"],
                    "top4_overlap": summary["all"]["top4_overlap"],
                    "mae_nospike": summary["no_spike"]["mae"],
                }
            )

    return pd.DataFrame(rows)


def ablation_table(holdout):
    """Average the hold-out folds per regime and model."""
    if holdout.empty:
        return holdout

    grouped = holdout.groupby(["regime", "model"], as_index=False).agg(
        folds=("held_out", "nunique"),
        n_test=("n_test", "sum"),
        mae=("mae", "mean"),
        skill=("skill", "mean"),
        top4_overlap=("top4_overlap", "mean"),
        mae_nospike=("mae_nospike", "mean"),
    )

    return grouped.sort_values(["regime", "mae"]).reset_index(drop=True)


def run_training_window(
    ds, params, starts=TRAINING_STARTS, epochs=SEARCH_EPOCHS, log=None
):
    """Compare how far back the training data should reach.

    Older years come from a different market regime, so more history is
    not automatically better. Each start date needs its own dataset: the
    split boundaries stay the same, only the amount of training data
    before them changes.
    """
    log = log or config.setup_logging()
    silver = storage.read_silver(
        config.SILVER_DIR, ds.meta["market"], nodes=ds.meta["nodes"]
    )
    catalog = catalog_module.load_catalog(
        os.path.join(config.CATALOG_DIR, "nodes.parquet")
    )
    first, last = ds.meta["silver_range"]
    earliest = datetime.date.fromisoformat(first)
    rows = []

    for start in starts:
        if start < earliest:
            log.info("skipping %s, silver starts at %s", start, earliest)
            continue

        log.info("training data from %s", start)
        variant = dataset_module.build_dataset(
            silver,
            catalog,
            ds.meta["nodes"],
            "{}_from{}".format(ds.meta["name"], start.year),
            start=start,
            end=datetime.date.fromisoformat(last),
            volatility=ds.meta["scaling"]["volatility_feature"],
            market=ds.meta["market"],
        )
        result = train.train_and_score(
            variant,
            kind="metadata",
            epochs=epochs,
            batch_size=BATCH_SIZE,
            patience=SEARCH_PATIENCE,
            verbose=0,
            silver_df=silver,
            **params,
        )
        _, val_summary = score_split(
            variant, result["model"], "metadata", variant.positions("val")
        )
        rows.append(
            {
                "start": start.isoformat(),
                "train_years": round(
                    (variant.index["target_date"].max() - start).days / 365.25,
                    1,
                ),
                "n_train": variant.meta["n_samples"]["train"],
                "val_mae": val_summary["all"]["mae"],
                "test_mae": result["summary"]["all"]["mae"],
                "test_skill": result["summary"]["all"]["skill"],
                "test_top4": result["summary"]["all"]["top4_overlap"],
            }
        )
        log.info(
            "    val MAE %.2f, test MAE %.2f, skill %.3f",
            rows[-1]["val_mae"],
            rows[-1]["test_mae"],
            rows[-1]["test_skill"],
        )

    table = pd.DataFrame(rows, columns=TRAINING_COLUMNS)

    return table.sort_values("val_mae").reset_index(drop=True)


def metadata_verdict(ablation):
    """Say whether the metadata model earned its place in both regimes."""
    lines = []
    ships = True

    for regime in sorted(ablation["regime"].unique()):
        block = ablation[ablation["regime"] == regime].set_index("model")
        if "metadata" not in block.index or "history" not in block.index:
            continue
        meta_mae = block.loc["metadata", "mae"]
        history_mae = block.loc["history", "mae"]
        better = meta_mae <= history_mae
        ships = ships and better
        lines.append(
            "- {} hold-out: metadata MAE {:.2f} vs history-only {:.2f} "
            "-> metadata {}".format(
                regime,
                meta_mae,
                history_mae,
                "wins" if better else "loses",
            )
        )

    if ships:
        lines.append("")
        lines.append(
            "The metadata model beats history-only in every regime, so it "
            "is the one that ships."
        )
    else:
        lines.append("")
        lines.append(
            "The metadata features did not earn their place in every "
            "regime, so the history-only model ships (pitch section 8)."
        )

    return "\n".join(lines)


def write_report(
    search,
    holdout,
    ablation,
    out_md,
    dataset_name,
    windows=None,
    training=None,
):
    """Write the search and generalization tables as markdown."""
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    lines = ["# Model selection - dataset {}".format(dataset_name), ""]

    if search is not None and not search.empty:
        lines.append("## Hyperparameter search (task 2.2)")
        lines.append("")
        lines.append(
            "{} configurations of the metadata model, ranked by validation "
            "MAE. Test metrics are shown for context only; the choice is "
            "made on validation.".format(len(search))
        )
        lines.append("")
        lines.append(search.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        lines.append(
            "Chosen configuration: {}".format(
                ", ".join(
                    "{}={}".format(key, value)
                    for key, value in best_params(search).items()
                )
            )
        )
        lines.append("")

    if windows is not None and not windows.empty:
        lines.append("## Input window length (task 2.2)")
        lines.append("")
        lines.append(
            "One dataset per window length, same nodes and date range, "
            "ranked by validation MAE. A longer window means fewer usable "
            "samples, because a sample needs its whole window clean."
        )
        lines.append("")
        lines.append(windows.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        lines.append(
            "Chosen window: {} days ({} hours).".format(
                int(windows.iloc[0]["input_days"]),
                int(windows.iloc[0]["input_hours"]),
            )
        )
        lines.append("")

    if training is not None and not training.empty:
        lines.append("## How far back to train")
        lines.append("")
        lines.append(
            "One dataset per training start date, same validation and test "
            "periods, ranked by validation MAE. The pitch treats this as an "
            "experiment because the pre-2022 years come from a different "
            "market regime."
        )
        lines.append("")
        lines.append(training.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        lines.append(
            "Chosen start: {} ({} years of training data).".format(
                training.iloc[0]["start"], training.iloc[0]["train_years"]
            )
        )
        lines.append("")

    if not holdout.empty:
        lines.append("## Generalization to unseen nodes and zones (task 2.3)")
        lines.append("")
        lines.append(
            "Leave-one-node-out holds out a node whose load zone keeps other "
            "nodes. Leave-one-zone-out holds out every node of a zone, so the "
            "held-out samples are scored with the UNKNOWN zone embedding. "
            "Skill is relative to Naive-168h on the same samples."
        )
        lines.append("")
        lines.append("### Average per regime and model")
        lines.append("")
        lines.append(ablation.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        lines.append("### Per fold")
        lines.append("")
        lines.append(holdout.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        lines.append("### Verdict")
        lines.append("")
        lines.append(metadata_verdict(ablation))
        lines.append("")

    text = "\n".join(lines)
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write(text)

    return text


def run_window_search(
    ds, params, windows=WINDOW_DAYS, epochs=SEARCH_EPOCHS, log=None
):
    """Compare input window lengths, rebuilding the dataset for each one.

    The window length changes the shape of every sample, so it cannot be
    swept like the other hyperparameters: each length needs its own
    dataset artifact built from the same silver rows and node list.
    """
    log = log or config.setup_logging()
    silver = storage.read_silver(
        config.SILVER_DIR, ds.meta["market"], nodes=ds.meta["nodes"]
    )
    catalog = catalog_module.load_catalog(
        os.path.join(config.CATALOG_DIR, "nodes.parquet")
    )
    first, last = ds.meta["silver_range"]
    rows = []

    for days in windows:
        log.info("window %d days", days)
        variant = dataset_module.build_dataset(
            silver,
            catalog,
            ds.meta["nodes"],
            "{}_w{}".format(ds.meta["name"], days),
            start=datetime.date.fromisoformat(first),
            end=datetime.date.fromisoformat(last),
            volatility=ds.meta["scaling"]["volatility_feature"],
            market=ds.meta["market"],
            input_days=days,
        )
        result = train.train_and_score(
            variant,
            kind="metadata",
            epochs=epochs,
            batch_size=BATCH_SIZE,
            patience=SEARCH_PATIENCE,
            verbose=0,
            silver_df=silver,
            **params,
        )
        _, val_summary = score_split(
            variant, result["model"], "metadata", variant.positions("val")
        )
        rows.append(
            {
                "input_days": days,
                "input_hours": days * 24,
                "n_samples": len(variant),
                "val_mae": val_summary["all"]["mae"],
                "test_mae": result["summary"]["all"]["mae"],
                "test_skill": result["summary"]["all"]["skill"],
                "test_top4": result["summary"]["all"]["top4_overlap"],
            }
        )
        log.info(
            "    val MAE %.2f, test MAE %.2f, skill %.3f",
            rows[-1]["val_mae"],
            rows[-1]["test_mae"],
            rows[-1]["test_skill"],
        )

    table = pd.DataFrame(rows, columns=WINDOW_COLUMNS)

    return table.sort_values("val_mae").reset_index(drop=True)


def main():
    """Run the search, the hold-out folds and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-windows", action="store_true")
    parser.add_argument("--skip-training-window", action="store_true")
    parser.add_argument("--units", type=int, default=models.DEFAULT_UNITS)
    parser.add_argument(
        "--dropout", type=float, default=models.DEFAULT_DROPOUT
    )
    parser.add_argument("--loss", default="huber", choices=("huber", "mae"))
    parser.add_argument("--epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--gold-dir", default=config.GOLD_DIR)
    parser.add_argument("--docs-dir", default=config.DOCS_DIR)
    args = parser.parse_args()

    log = config.setup_logging()
    ds = dataset_module.load_dataset(
        dataset_module.dataset_dir(args.dataset, args.gold_dir)
    )
    log.info(
        "dataset %s: %s samples, %d nodes, %d zones",
        args.dataset,
        ds.meta["n_samples"],
        len(ds.meta["nodes"]),
        len(ds.meta["zone_vocab"]),
    )

    search = None
    if args.skip_search:
        params = {
            "units": args.units,
            "dropout": args.dropout,
            "loss": args.loss,
            "learning_rate": models.DEFAULT_LEARNING_RATE,
            "zone_dropout": 0.25,
        }
        log.info("search skipped, using %s", params)
    else:
        search = run_search(ds, epochs=args.epochs, log=log)
        params = best_params(search)
        log.info("best configuration %s", params)

    windows = None
    if not args.skip_windows:
        windows = run_window_search(ds, params, epochs=args.epochs, log=log)
        best_window = int(windows.iloc[0]["input_days"])
        log.info("best window %d days", best_window)

    silver = storage.read_silver(
        config.SILVER_DIR, ds.meta["market"], nodes=ds.meta["nodes"]
    )
    training = None
    if not args.skip_training_window:
        training = run_training_window(ds, params, epochs=args.epochs, log=log)
        log.info("best training start %s", training.iloc[0]["start"])

    node_folds = run_holdout(ds, params, "node", args.max_folds, log, silver)
    zone_folds = run_holdout(ds, params, "zone", args.max_folds, log, silver)
    holdout = pd.concat([node_folds, zone_folds], ignore_index=True)
    ablation = ablation_table(holdout)

    out_md = os.path.join(args.docs_dir, "model_selection.md")
    write_report(
        search, holdout, ablation, out_md, args.dataset, windows, training
    )
    log.info("report -> %s", out_md)
    print(ablation.to_string(index=False, float_format="{:.3f}".format))
    print("")
    print(metadata_verdict(ablation))


if __name__ == "__main__":
    main()
