#!/usr/bin/env python3
"""Fit, score and log the four baseline forecasters."""

import argparse
import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import pmlcast
from pmlcast import config
from pmlcast import dataset as dataset_module
from pmlcast import evaluate
from pmlcast import preprocess
from pmlcast import storage

EXPERIMENT = "pmlcast-baselines"
REFERENCE = "naive168"
LEVEL_MIN_ABS = 10.0
N_SEASONS = 4
METRIC_KEYS = (
    "mae",
    "rmse",
    "smape",
    "skill",
    "top4_overlap",
    "captured_value",
)
SUBSET_SUFFIX = {"all": "", "no_spike": "_nospike", "spike": "_spike"}


def raw_input_prices(ds, positions):
    """Return the 168 input prices in MXN/MWh, shape (n, 168)."""
    return preprocess.invert(
        ds.arrays["X_seq"][positions, :, 0],
        ds.arrays["mu"][positions],
        ds.arrays["sigma"][positions],
    )


def season_of(month):
    """Map a month number to a meteorological season index 0..3."""
    return (np.asarray(month) % 12) // 3


class Naive24:
    """Predict every hour with the same hour of the last input day D."""

    name = "naive24"

    def params(self):
        """Return the loggable hyperparameters."""
        return {}

    def fit(self, ds, train_positions, val_positions=None, silver_df=None):
        """Nothing to fit."""
        return self

    def predict(self, ds, positions):
        """Return prices of day D, shape (n, 24)."""
        return raw_input_prices(ds, positions)[:, -24:]


class Naive168:
    """Predict every hour with the same hour one week earlier.

    This is the reference the skill score is measured against, so it is a
    property of the problem and not of the model: it always reads the day
    7 days before the target from silver, even when the model's input
    window is shorter than a week.
    """

    name = "naive168"

    def __init__(self):
        self.history = {}

    def params(self):
        """Return the loggable hyperparameters."""
        return {}

    def fit(self, ds, train_positions, val_positions=None, silver_df=None):
        """Index the hourly history per node, when silver is available."""
        if silver_df is None:
            return self

        for node, group in silver_df.groupby("node"):
            pivot = group.pivot_table(
                index="fecha", columns="hora", values="pml"
            )
            self.history[node] = pivot

        return self

    def predict(self, ds, positions):
        """Return the prices of the day 7 days before each target."""
        rows = ds.index.iloc[positions]
        window_days = ds.arrays["X_seq"].shape[1] // 24

        if not self.history and window_days >= 7:
            prices = raw_input_prices(ds, positions)
            start = prices.shape[1] - 7 * 24
            return prices[:, start : start + 24]

        if not self.history:
            raise ValueError(
                "Naive-168h needs silver_df when the input window is "
                "shorter than 7 days"
            )

        out = np.empty((len(rows), 24))
        for i, (node, target) in enumerate(
            zip(rows["node"], rows["target_date"])
        ):
            day = pd.Timestamp(target) - pd.Timedelta(days=7)
            out[i] = self.history[node].loc[day.date()].to_numpy()

        return out


class SeasonalMA:
    """Seasonal moving average: circular day-of-year level x hourly shape.

    The daily level of the target day is the mean of the daily means of
    the training history within ``half_window`` days of the same day of
    year (circular distance). The hourly shape is the mean ratio
    ``price / daily mean`` by (hour, weekday, season). Zero and negative
    prices are kept; nothing is clipped.
    """

    name = "seasonal_ma"

    def __init__(self, half_window=15, history_years=2):
        self.half_window = half_window
        self.history_years = history_years
        self.levels = {}
        self.doy = {}
        self.profiles = {}
        self.global_mean = {}

    def params(self):
        """Return the loggable hyperparameters."""
        return {
            "half_window": self.half_window,
            "history_years": self.history_years,
        }

    def fit(self, ds, train_positions, val_positions=None, silver_df=None):
        """Learn daily levels and hourly profiles per node from silver.

        Only rows up to the last target date of the fit samples and inside
        the last ``history_years`` years are used.
        """
        if silver_df is None:
            raise ValueError("SeasonalMA needs silver_df to fit")
        positions = np.asarray(train_positions)
        if val_positions is not None:
            positions = np.concatenate([positions, np.asarray(val_positions)])
        rows = ds.index.iloc[positions]
        cutoff = pd.to_datetime(rows["target_date"]).max()
        earliest = cutoff - pd.DateOffset(years=self.history_years)

        for node in sorted(rows["node"].unique()):
            history = silver_df[silver_df["node"] == node]
            dates = pd.to_datetime(history["fecha"])
            history = history[(dates <= cutoff) & (dates > earliest)]
            pivot = history.pivot_table(
                index="fecha", columns="hora", values="pml"
            ).dropna()
            if pivot.empty:
                raise ValueError("no full days to fit node {}".format(node))
            daily = pivot.mean(axis=1)
            self.levels[node] = daily.to_numpy(dtype=float)
            self.doy[node] = pd.to_datetime(pd.Index(daily.index)).dayofyear
            self.global_mean[node] = float(daily.mean())

            profile = np.ones((24, 7, N_SEASONS))
            usable = daily.abs() >= LEVEL_MIN_ABS
            ratios = pivot[usable].div(daily[usable], axis=0)
            stamps = pd.to_datetime(pd.Index(ratios.index))
            keys = pd.DataFrame(
                {"dow": stamps.dayofweek, "season": season_of(stamps.month)},
                index=ratios.index,
            )
            grouped = ratios.groupby([keys["dow"], keys["season"]]).mean()
            for (dow, season), shape in grouped.iterrows():
                profile[:, dow, season] = shape.to_numpy(dtype=float)
            self.profiles[node] = profile
        return self

    def predict(self, ds, positions):
        """Return level x shape forecasts, shape (n, 24)."""
        rows = ds.index.iloc[positions]
        out = np.empty((len(rows), 24))
        for i, (node, target) in enumerate(
            zip(rows["node"], rows["target_date"])
        ):
            target = pd.Timestamp(target)
            distance = np.abs(self.doy[node] - target.dayofyear)
            distance = np.minimum(distance, 365 - distance)
            mask = distance <= self.half_window
            if mask.any():
                level = float(self.levels[node][mask].mean())
            else:
                level = self.global_mean[node]
            shape = self.profiles[node][
                :, target.dayofweek, int(season_of(target.month))
            ]
            out[i] = level * shape
        return out


class LinearLags:
    """Ridge regression on lagged z-scores plus calendar and level inputs."""

    name = "linear_lags"

    def __init__(self, alphas=(0.1, 1.0, 10.0)):
        self.alphas = tuple(alphas)
        self.alpha = None
        self.model = None

    def params(self):
        """Return the loggable hyperparameters."""
        return {"alphas": str(self.alphas), "alpha": self.alpha}

    @staticmethod
    def design(ds, positions):
        """Build the design matrix in z-space: lags, daily means, features.

        The lags are picked relative to the end of the window, so the
        matrix follows whatever input length the dataset was built with.
        """
        z = ds.arrays["X_seq"][positions, :, 0]
        n_days = z.shape[1] // 24
        daily = z.reshape(len(positions), n_days, 24)
        week_ago = max(n_days - 7, 0)
        return np.column_stack(
            [
                daily[:, -1],  # day D
                daily[:, -2],  # day D-1
                daily[:, week_ago],  # same weekday, one week back
                daily.mean(axis=2),  # one mean per input day
                ds.arrays["X_cal"][positions],
                ds.arrays["X_level"][positions],
            ]
        ).astype(np.float64)

    def fit(self, ds, train_positions, val_positions=None, silver_df=None):
        """Choose alpha on validation MAE (MXN) and refit on train + val."""
        train_positions = np.asarray(train_positions)
        x_train = self.design(ds, train_positions)
        y_train = ds.arrays["y"][train_positions].astype(np.float64)
        x_fit, y_fit = x_train, y_train
        self.alpha = self.alphas[len(self.alphas) // 2]

        if val_positions is not None and len(val_positions):
            val_positions = np.asarray(val_positions)
            x_val = self.design(ds, val_positions)
            y_val = ds.y_prices(val_positions)
            best = None
            for alpha in self.alphas:
                model = Ridge(alpha=alpha).fit(x_train, y_train)
                pred = preprocess.invert(
                    model.predict(x_val),
                    ds.arrays["mu"][val_positions],
                    ds.arrays["sigma"][val_positions],
                )
                score = evaluate.mae(y_val, pred)
                if best is None or score < best[0]:
                    best = (score, alpha)
            self.alpha = best[1]
            x_fit = np.vstack([x_train, x_val])
            y_fit = np.vstack([y_train, ds.arrays["y"][val_positions]])

        self.model = Ridge(alpha=self.alpha).fit(x_fit, y_fit)
        return self

    def predict(self, ds, positions):
        """Return inverted predictions in MXN/MWh, shape (n, 24)."""
        z_hat = self.model.predict(self.design(ds, positions))
        return preprocess.invert(
            z_hat, ds.arrays["mu"][positions], ds.arrays["sigma"][positions]
        )


BASELINES = (Naive24, Naive168, SeasonalMA, LinearLags)


# ---------------------------------------------------------------------------
# Running, storing and logging
# ---------------------------------------------------------------------------


def predictions_frame(ds, positions, y_pred, model_name, model_version=None):
    """Turn (n, 24) predictions into the gold predictions table."""
    rows = ds.index.iloc[positions]
    y_true = ds.y_prices(positions)
    n = len(positions)
    origin = pd.to_datetime(rows["origin_date"])
    return pd.DataFrame(
        {
            "node": np.repeat(rows["node"].to_numpy(), 24),
            "market": ds.meta.get("market", "MDA"),
            "origin_date": np.repeat(rows["origin_date"].to_numpy(), 24),
            "target_date": np.repeat(rows["target_date"].to_numpy(), 24),
            "hora": np.tile(np.arange(1, 25), n),
            "pml_pred": np.asarray(y_pred, dtype=float).reshape(-1),
            "pml_actual": y_true.reshape(-1),
            "model_name": model_name,
            "model_version": model_version or pmlcast.__version__,
            "issued_at": np.repeat(
                (origin + pd.Timedelta(hours=6)).to_numpy(), 24
            ),
            "run_at": pd.Timestamp.now(tz="UTC"),
            "split": np.repeat(rows["split"].to_numpy(), 24),
            "sample_id": np.repeat(rows["sample_id"].to_numpy(), 24),
        }
    )


def flat_metrics(summary):
    """Flatten a ``summarize`` dict into MLflow metric names."""
    out = {}
    for subset, suffix in SUBSET_SUFFIX.items():
        for key in METRIC_KEYS:
            value = summary[subset].get(key)
            if value is not None and np.isfinite(value):
                out["test_{}{}".format(key, suffix)] = float(value)
        out["n_test{}".format(suffix)] = int(summary[subset]["n_samples"])
    return out


def artifact_root_for(tracking_uri):
    """Return the artifact folder next to a SQLite tracking database."""
    prefix = "sqlite:///"
    if tracking_uri.startswith(prefix):
        db_dir = os.path.dirname(tracking_uri[len(prefix) :])
        return os.path.join(db_dir, "mlruns")
    return None


def setup_mlflow(tracking_uri, experiment):
    """Point MLflow at a tracking store and select the experiment.

    A missing experiment is created with its artifacts stored next to
    the SQLite database, so tests and the repo stay self-contained.
    """
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=artifact_root_for(tracking_uri)
        )
    mlflow.set_experiment(experiment)


def log_run(
    model,
    ds,
    summary,
    y_true,
    y_pred,
    positions,
    tracking_uri,
    predictions_path=None,
    experiment=EXPERIMENT,
):
    """Log parameters, metrics and artifacts of one model to MLflow."""
    setup_mlflow(tracking_uri, experiment)
    with mlflow.start_run(run_name=model.name) as run:
        params = {
            "model": model.name,
            "dataset": ds.meta["name"],
            "n_train": ds.meta["n_samples"]["train"],
            "n_val": ds.meta["n_samples"]["val"],
            "n_test": ds.meta["n_samples"]["test"],
            "test_start": ds.meta["splits"]["test_start"],
            "test_end": ds.meta["splits"]["test_end"],
            "pmlcast_version": pmlcast.__version__,
        }
        params.update(model.params())
        mlflow.log_params(params)
        mlflow.log_metrics(flat_metrics(summary))
        with tempfile.TemporaryDirectory() as tmp:
            per_hour = pd.DataFrame(
                {
                    "hora": np.arange(1, 25),
                    "mae": evaluate.per_hour_mae(y_true, y_pred),
                }
            )
            per_hour.to_csv(os.path.join(tmp, "per_hour.csv"), index=False)
            regions = ds.index["region"].to_numpy()[positions]
            evaluate.summarize_by(y_true, y_pred, regions).to_csv(
                os.path.join(tmp, "by_region.csv"), index=False
            )
            mlflow.log_artifacts(tmp)
        if predictions_path and os.path.exists(predictions_path):
            mlflow.log_artifact(predictions_path)
        return run.info.run_id


def run_baseline(
    model,
    ds,
    silver_df=None,
    reference_pred=None,
    tracking_uri=config.MLRUNS_URI,
    gold_dir=config.GOLD_DIR,
    log_to_mlflow=True,
):
    """Fit a baseline on train (+val), score the test split and store it.

    Baselines are fitted once before the test period; there is no refit
    inside it, so scoring all test samples at once equals walking forward
    day by day.

    Args:
        model: A baseline instance with ``fit``/``predict``/``params``.
        ds: Dataset artifact.
        silver_df: Silver rows (needed by ``SeasonalMA``).
        reference_pred: Optional reference forecast for the skill score.
        tracking_uri: MLflow tracking URI.
        gold_dir: Root of the gold layer for the predictions table.
        log_to_mlflow: Log the run to MLflow when True.

    Returns:
        Tuple ``(y_pred, summary)`` for the test split.
    """
    train = ds.positions("train")
    val = ds.positions("val")
    test = ds.positions("test")
    model.fit(ds, train, val_positions=val, silver_df=silver_df)
    y_pred = np.asarray(model.predict(ds, test), dtype=float)
    y_true = ds.y_prices(test)
    is_spike = ds.index["is_spike_day"].to_numpy()[test]
    summary = evaluate.summarize(y_true, y_pred, is_spike, reference_pred)

    frame = predictions_frame(ds, test, y_pred, model.name)
    path = storage.write_predictions(gold_dir, model.name, frame)
    if log_to_mlflow:
        log_run(model, ds, summary, y_true, y_pred, test, tracking_uri, path)
    return y_pred, summary


def run_all(ds, silver_df, tracking_uri, gold_dir, log_to_mlflow=True):
    """Run the four baselines; the reference model runs first.

    Returns:
        Tuple ``(predictions, summaries)`` keyed by model name, in report
        order.
    """
    reference = Naive168()
    ref_pred, ref_summary = run_baseline(
        reference, ds, silver_df, None, tracking_uri, gold_dir, log_to_mlflow
    )
    predictions = {"naive24": None, REFERENCE: ref_pred}
    summaries = {REFERENCE: ref_summary}
    for cls in (Naive24, SeasonalMA, LinearLags):
        model = cls()
        pred, summary = run_baseline(
            model,
            ds,
            silver_df,
            ref_pred,
            tracking_uri,
            gold_dir,
            log_to_mlflow,
        )
        predictions[model.name] = pred
        summaries[model.name] = summary
    return predictions, summaries


def summary_table(summaries):
    """Return a compact DataFrame of the headline test metrics per model."""
    rows = []
    for name, summary in summaries.items():
        row = {"model": name}
        for key in ("mae", "rmse", "skill", "top4_overlap", "captured_value"):
            row[key] = summary["all"][key]
        row["mae_nospike"] = summary["no_spike"]["mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Run the baselines on a dataset and write the baseline report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="dataset name")
    parser.add_argument("--gold-dir", default=config.GOLD_DIR)
    parser.add_argument("--silver-dir", default=config.SILVER_DIR)
    parser.add_argument("--docs-dir", default=config.DOCS_DIR)
    parser.add_argument("--tracking-uri", default=config.MLRUNS_URI)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    log = config.setup_logging()
    ds = dataset_module.load_dataset(
        dataset_module.dataset_dir(args.dataset, args.gold_dir)
    )
    silver = storage.read_silver(
        args.silver_dir, ds.meta["market"], nodes=ds.meta["nodes"]
    )
    predictions, summaries = run_all(
        ds, silver, args.tracking_uri, args.gold_dir, not args.no_mlflow
    )
    test = ds.positions("test")
    report_path = os.path.join(args.docs_dir, "baselines.md")
    evaluate.write_report(
        ds.y_prices(test),
        predictions,
        ds.index.iloc[test].reset_index(drop=True),
        report_path,
        os.path.join(args.docs_dir, "figures"),
        reference=REFERENCE,
        title="Baselines - dataset {}".format(args.dataset),
    )
    table = summary_table(summaries)
    log.info("report -> %s", report_path)
    print(table.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
