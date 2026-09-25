#!/usr/bin/env python3
"""Train an LSTM forecaster on a dataset artifact and score the test split."""

import argparse
import datetime
import os

import mlflow
import numpy as np

import pmlcast
from pmlcast import baselines
from pmlcast import config
from pmlcast import dataset as dataset_module
from pmlcast import evaluate
from pmlcast import models
from pmlcast import storage

EXPERIMENT = "pmlcast-lstm"
MODEL_KINDS = ("history", "metadata", "node_id")
MODELS_DIR = os.path.join(config.DATA_DIR, "models")


def naive168_reference(ds, positions, silver_df=None):
    """Return the Naive-168h forecast used as the skill reference."""
    return (
        baselines.Naive168()
        .fit(ds, positions, silver_df=silver_df)
        .predict(ds, positions)
    )


def model_inputs(ds, positions, kind, zone_override=None):
    """Return the input dict a model kind expects for positions."""
    with_meta = kind in ("metadata", "node_id")
    return dataset_module.inputs_dict(
        ds,
        positions,
        use_meta=with_meta,
        use_zone=with_meta,
        use_node=kind == "node_id",
        zone_override=zone_override,
    )


def build_for(ds, kind, units, dropout, loss, learning_rate):
    """Build the model of one kind with shapes taken from the dataset."""
    if kind not in MODEL_KINDS:
        raise ValueError("kind must be one of {}".format(MODEL_KINDS))
    kwargs = {
        "seq_shape": ds.arrays["X_seq"].shape[1:],
        "n_cal": ds.arrays["X_cal"].shape[1],
        "n_level": ds.arrays["X_level"].shape[1],
        "units": units,
        "dropout": dropout,
        "loss": loss,
        "learning_rate": learning_rate,
    }
    if kind in ("metadata", "node_id"):
        kwargs["n_meta"] = ds.arrays["X_meta"].shape[1]
        kwargs["n_zones"] = len(ds.meta["zone_vocab"])
    if kind == "node_id":
        kwargs["n_nodes"] = len(ds.meta["nodes"])
    return models.build_model(**kwargs)


def train_and_score(
    ds,
    kind="history",
    epochs=50,
    batch_size=256,
    units=models.DEFAULT_UNITS,
    dropout=models.DEFAULT_DROPOUT,
    loss="huber",
    learning_rate=models.DEFAULT_LEARNING_RATE,
    patience=5,
    seed=config.SEED,
    zone_dropout=0.25,
    checkpoint_path=None,
    verbose=1,
    silver_df=None,
):
    """Train one model kind and score it on the test split.

    Returns:
        Dict with ``model``, ``history``, ``y_pred``, ``y_true``,
        ``summary`` (vs Naive-168h) and ``references`` (naive forecasts).
    """
    train = ds.positions("train")
    val = ds.positions("val")
    test = ds.positions("test")
    models.set_seeds(seed)
    model = build_for(ds, kind, units, dropout, loss, learning_rate)
    history = models.train_model(
        model,
        model_inputs(ds, train, kind),
        ds.arrays["y"][train],
        model_inputs(ds, val, kind),
        ds.arrays["y"][val],
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        seed=seed,
        zone_dropout=zone_dropout if kind != "history" else 0.0,
        verbose=verbose,
    )
    y_pred = models.predict_prices(
        model,
        model_inputs(ds, test, kind),
        ds.arrays["mu"][test],
        ds.arrays["sigma"][test],
    )
    y_true = ds.y_prices(test)
    references = {
        "naive24": baselines.Naive24().predict(ds, test),
        "naive168": naive168_reference(ds, test, silver_df),
    }
    is_spike = ds.index["is_spike_day"].to_numpy()[test]
    summary = evaluate.summarize(
        y_true, y_pred, is_spike, references["naive168"]
    )
    return {
        "model": model,
        "history": history,
        "y_pred": y_pred,
        "y_true": y_true,
        "summary": summary,
        "references": references,
        "test_positions": test,
    }


def append_summary(docs_path, name, dataset_name, summary, references_mae):
    """Append a headline section for a trained model to a markdown file."""
    os.makedirs(os.path.dirname(os.path.abspath(docs_path)), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    row = summary["all"]
    lines = [
        "",
        "## {} on dataset {} ({})".format(name, dataset_name, stamp),
        "",
        "| model | MAE | RMSE | skill vs naive168 | "
        "top-4 overlap | captured value | MAE no-spike |",
        "|---|---|---|---|---|---|---|",
        "| {} | {:.2f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
            name,
            row["mae"],
            row["rmse"],
            row["skill"],
            row["top4_overlap"],
            row["captured_value"],
            summary["no_spike"]["mae"],
        ),
        "",
        "Naive-24h MAE {:.2f}, Naive-168h MAE {:.2f} "
        "on the same samples.".format(
            references_mae["naive24"], references_mae["naive168"]
        ),
        "",
    ]
    with open(docs_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    """Train, score, store and log one LSTM model from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="history", choices=MODEL_KINDS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--units", type=int, default=models.DEFAULT_UNITS)
    parser.add_argument(
        "--dropout", type=float, default=models.DEFAULT_DROPOUT
    )
    parser.add_argument("--loss", default="huber", choices=("huber", "mae"))
    parser.add_argument(
        "--learning-rate", type=float, default=models.DEFAULT_LEARNING_RATE
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--zone-dropout", type=float, default=0.25)
    parser.add_argument("--gold-dir", default=config.GOLD_DIR)
    parser.add_argument("--docs-dir", default=config.DOCS_DIR)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument("--tracking-uri", default=config.MLRUNS_URI)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    log = config.setup_logging()
    ds = dataset_module.load_dataset(
        dataset_module.dataset_dir(args.dataset, args.gold_dir)
    )
    name = "lstm_{}".format(args.model)
    os.makedirs(args.models_dir, exist_ok=True)
    checkpoint = os.path.join(
        args.models_dir, "{}_{}.keras".format(name, args.dataset)
    )
    log.info(
        "training %s on %s (%s samples)",
        name,
        args.dataset,
        ds.meta["n_samples"],
    )

    result = train_and_score(
        ds,
        kind=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        units=args.units,
        dropout=args.dropout,
        loss=args.loss,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        zone_dropout=args.zone_dropout,
        checkpoint_path=checkpoint,
    )
    result["model"].save(checkpoint)
    test = result["test_positions"]
    frame = baselines.predictions_frame(ds, test, result["y_pred"], name)
    predictions_path = storage.write_predictions(args.gold_dir, name, frame)

    references_mae = {
        key: evaluate.mae(result["y_true"], pred)
        for key, pred in result["references"].items()
    }
    summary = result["summary"]
    append_summary(
        os.path.join(args.docs_dir, "baselines.md"),
        name,
        args.dataset,
        summary,
        references_mae,
    )

    if not args.no_mlflow:
        baselines.setup_mlflow(args.tracking_uri, EXPERIMENT)
        with mlflow.start_run(run_name="{}_{}".format(name, args.dataset)):
            params = {
                "model": name,
                "dataset": args.dataset,
                "epochs": args.epochs,
                "epochs_run": len(result["history"].history["loss"]),
                "batch_size": args.batch_size,
                "units": args.units,
                "dropout": args.dropout,
                "loss": args.loss,
                "learning_rate": args.learning_rate,
                "patience": args.patience,
                "seed": args.seed,
                "zone_dropout": args.zone_dropout,
                "n_train": ds.meta["n_samples"]["train"],
                "n_val": ds.meta["n_samples"]["val"],
                "n_test": ds.meta["n_samples"]["test"],
                "pmlcast_version": pmlcast.__version__,
            }
            mlflow.log_params(params)
            mlflow.log_metrics(baselines.flat_metrics(summary))
            mlflow.log_metrics(
                {
                    "naive24_mae": references_mae["naive24"],
                    "naive168_mae": references_mae["naive168"],
                }
            )
            for epoch, (loss, val_loss) in enumerate(
                zip(
                    result["history"].history["loss"],
                    result["history"].history["val_loss"],
                )
            ):
                mlflow.log_metrics(
                    {"train_loss": loss, "val_loss": val_loss}, step=epoch
                )
            mlflow.log_artifact(checkpoint)
            mlflow.log_artifact(predictions_path)

    beats = summary["all"]["mae"] < references_mae["naive24"]
    log.info(
        "%s test MAE %.2f (naive24 %.2f, naive168 %.2f, skill %.3f) -> %s",
        name,
        summary["all"]["mae"],
        references_mae["naive24"],
        references_mae["naive168"],
        summary["all"]["skill"],
        "beats Naive-24h" if beats else "does NOT beat Naive-24h",
    )
    print(np.round([summary["all"]["mae"], references_mae["naive24"]], 2))


if __name__ == "__main__":
    main()
