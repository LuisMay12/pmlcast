#!/usr/bin/env python3
"""Turn a node key and a target date into a 24-hour price forecast."""

import datetime
import json
import logging
import os

import numpy as np
import onnxruntime as ort
import pandas as pd

import pmlcast
from pmlcast import catalog as catalog_module
from pmlcast import cenace
from pmlcast import config
from pmlcast import dataset as dataset_module
from pmlcast import preprocess
from pmlcast import storage

# The service runs the model through ONNX Runtime: the same weights and
# the same predictions, but ~40 MB of dependency instead of TensorFlow's
# ~600 MB, which is what makes the container deployable anywhere. The
# Keras file still works if it is the one handed over.
MODELS_DIR = os.path.join(config.DATA_DIR, "models")
DEFAULT_MODEL = os.path.join(MODELS_DIR, "lstm_history_final.onnx")
KERAS_MODEL = os.path.join(MODELS_DIR, "lstm_history_final.keras")
DEFAULT_META = os.path.join(config.GOLD_DIR, "dataset_final", "meta.json")
DEFAULT_EVALUATED = os.path.join(config.SNAPSHOT_DIR, "nodes_stage2.csv")
ISSUE_HOUR = 6
TOP_HOURS = 4
CHEAP_HOURS = 4


class ForecastError(Exception):
    """A request the service cannot answer, with the reason for the caller."""


class Forecaster:
    """Hold the model and the metadata needed to answer a request."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        meta_path=DEFAULT_META,
        silver_dir=config.SILVER_DIR,
        catalog_path=None,
        backfill=False,
        bronze_dir=config.BRONZE_DIR,
        evaluated_path=DEFAULT_EVALUATED,
    ):
        if not os.path.exists(model_path):
            raise ForecastError("model not found: {}".format(model_path))
        if not os.path.exists(meta_path):
            raise ForecastError(
                "dataset metadata not found: {}".format(meta_path)
            )

        self.runtime = "onnx" if model_path.endswith(".onnx") else "keras"
        if self.runtime == "onnx":
            self.model = ort.InferenceSession(model_path)
        else:  # only pulls TensorFlow in when a Keras file is served
            import tensorflow.keras as keras_api

            self.model = keras_api.models.load_model(model_path)
        with open(meta_path, encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.silver_dir = silver_dir
        self.backfill = backfill
        self.bronze_dir = bronze_dir
        self.input_days = self.meta["input_hours"] // 24
        self.market = self.meta.get("market", "MDA")
        self.model_version = "{}@{}".format(
            os.path.basename(model_path), pmlcast.__version__
        )

        path = catalog_path or os.path.join(
            config.CATALOG_DIR, "nodes.parquet"
        )
        self.catalog = (
            catalog_module.load_catalog(path)
            if os.path.exists(path)
            else pd.DataFrame(columns=["node"])
        )
        self.holidays = preprocess.mx_holidays(range(2016, 2036))

        # The model answers for any SIN node, but the published error
        # was measured on the evaluation set only. Callers are told
        # which of the two they are getting rather than having to
        # assume the headline number applies everywhere.
        self.evaluated = (
            set(cenace.read_nodes_file(evaluated_path))
            if evaluated_path and os.path.exists(evaluated_path)
            else set()
        )

    # -- scope checks ----------------------------------------------------

    def node_row(self, node):
        """Return the catalog row of a node, or None when not catalogued."""
        rows = self.catalog[self.catalog["node"] == node]

        return None if rows.empty else rows.iloc[0]

    def check_scope(self, node):
        """Raise ForecastError when a node is outside the supported scope.

        The pitch promises an explicit error instead of an extrapolated
        curve: a node of an unsupported system or voltage level, or one
        with too little published history, is rejected here.
        """
        row = self.node_row(node)
        if row is None:
            return  # not catalogued: silver history is the only requirement

        if row["sistema"] != "SIN":
            raise ForecastError(
                "node {} belongs to {}, which is out of scope "
                "(SIN only)".format(node, row["sistema"])
            )

    # -- the forecast ----------------------------------------------------

    def load_history(self, node, origin):
        """Read the rows a forecast needs, up to the day after the origin.

        The window the model sees ends at 23:00 of the origin, but the
        sample builder pairs each origin with its target day, so the
        target has to be present in the frame for the pair to exist. Its
        prices are the label, never an input: for a future target they
        are NaN and the sample is still built.

        With ``backfill`` on, a gap at the recent end is filled from
        CENACE before giving up: a packaged demo goes stale every day,
        and asking for the few missing days costs a wait rather than an
        error.
        """
        needed = max(self.input_days, config.STATS_DAYS) + 1
        start = origin - datetime.timedelta(days=needed)
        end = origin + datetime.timedelta(days=1)
        history = self._read_history(node, start, end)
        if self.backfill and self._missing_tail(history, start, end):
            self.fetch_missing(node, start, end)
            history = self._read_history(node, start, end)
        if history.empty:
            raise ForecastError(
                "no published history for node {} up to {}".format(
                    node, origin
                )
            )

        return history

    def _read_history(self, node, start, end):
        """Read one node's silver rows over a date range."""
        return storage.read_silver(
            self.silver_dir, self.market, nodes=[node], start=start, end=end
        )

    def _missing_tail(self, history, start, end):
        """Say whether the window is short of days at its recent end.

        Only the tail matters: CENACE publishes day by day, so a hole in
        the middle is a genuine gap, while a short tail is just a
        snapshot that stopped being current. The end of the window is
        the target day, whose row has to exist for the sample builder to
        pair it with its origin, even while its prices are still NaN.
        """
        if history.empty:
            return True

        last = history["fecha"].max()
        if isinstance(last, pd.Timestamp):
            last = last.date()

        return last < end

    def fetch_missing(self, node, start, end):
        """Ask CENACE for the days this node is missing, and ignore failures.

        A backfill is a convenience, not a contract: when the upstream
        service is down or slow the forecast should still be attempted
        with whatever is already stored, and fail with the ordinary
        not-enough-history message if that is not enough.
        """
        row = self.node_row(node)
        sistema = row["sistema"] if row is not None else "SIN"
        try:
            cenace.collect(
                [node],
                sistema,
                self.market,
                start,
                end,
                self.bronze_dir,
                self.silver_dir,
            )
        except Exception:  # noqa: BLE001 - upstream is best-effort here
            logging.getLogger("pmlcast").warning(
                "backfill failed for %s (%s -> %s)", node, start, end
            )

    def build_sample(self, node, target_date):
        """Build the single model input for one node and target day.

        Everything is resolved from the target date: the origin is the
        day before, the input window ends at 23:00 of that origin, and
        the scaling statistics are frozen there. Nothing published after
        the origin can reach the model, so a past date replays exactly
        what the service would have seen that morning.
        """
        origin = target_date - datetime.timedelta(days=1)
        history = self.load_history(node, origin)
        meta = dataset_module.node_metadata(self.catalog, node)
        arrays, index = dataset_module.build_node_samples(
            history,
            meta,
            self.holidays,
            self.meta["scaling"]["volatility_feature"],
            self.input_days,
        )
        if arrays is None or index.empty:
            raise ForecastError(
                "node {} has too little clean history before {} "
                "(needs {} days of prices and {} days for the scaling "
                "statistics)".format(
                    node, target_date, self.input_days, config.STATS_DAYS
                )
            )

        wanted = index.index[index["target_date"] == target_date]
        if len(wanted) == 0:
            last = index["target_date"].max()
            raise ForecastError(
                "cannot forecast {} for node {}: the latest target the "
                "published history supports is {}".format(
                    target_date, node, last
                )
            )

        return arrays, index, int(wanted[0])

    def predict(self, inputs, mu, sigma):
        """Run the model on prepared inputs and return prices in MXN/MWh."""
        if self.runtime == "onnx":
            feeds = {
                name: np.asarray(inputs[name], dtype=np.float32)
                for name in ("seq", "cal", "level")
            }
            z = self.model.run(None, feeds)[0]

            return preprocess.invert(z, mu, sigma)

        # importing here keeps TensorFlow out of the ONNX path
        from pmlcast import models

        return models.predict_prices(self.model, inputs, mu, sigma)

    def forecast(self, node, target_date=None):
        """Return the 24 hourly prices of the target day for one node.

        Args:
            node: Node key from the CENACE catalog.
            target_date: Day to forecast; tomorrow when omitted.

        Returns:
            Dict with the prices, the injection windows and the metadata
            a caller needs to judge the forecast.
        """
        today = config.today_local()
        target_date = target_date or today + datetime.timedelta(days=1)
        if target_date > today + datetime.timedelta(days=1):
            raise ForecastError(
                "PMLcast is a single-horizon model: it forecasts at most "
                "one day ahead, so {} is out of range".format(target_date)
            )

        self.check_scope(node)
        arrays, index, position = self.build_sample(node, target_date)
        row = index.iloc[position]
        inputs = {
            "seq": arrays["X_seq"][position : position + 1],
            "cal": arrays["X_cal"][position : position + 1],
            "level": arrays["X_level"][position : position + 1],
        }
        prices = self.predict(
            inputs,
            arrays["mu"][position : position + 1],
            arrays["sigma"][position : position + 1],
        )[0]

        origin = row["origin_date"]
        issued_at = datetime.datetime.combine(
            origin + datetime.timedelta(days=1), datetime.time(ISSUE_HOUR)
        )

        return {
            "node": node,
            "system": row["sistema"],
            "market": self.market,
            "target_date": target_date.isoformat(),
            "issued_at": issued_at.isoformat(),
            "history_end": "{}T23:00:00".format(origin.isoformat()),
            "unit": "MXN/MWh",
            "hours": [
                {"hora": hour + 1, "pml": round(float(price), 2)}
                for hour, price in enumerate(prices)
            ],
            "best_injection_window": window_of(
                prices, TOP_HOURS, expensive=True
            ),
            "cheapest_window": window_of(prices, CHEAP_HOURS, expensive=False),
            "model_version": self.model_version,
            "backtest": target_date <= today,
            "evaluated_node": node in self.evaluated,
        }


def window_of(prices, size, expensive=True):
    """Return the contiguous window of `size` hours worth acting on.

    The metrics score the top-4 hours as a set, but an operator needs a
    block: this picks the contiguous run with the highest (or lowest)
    total, which is what a battery can actually charge or discharge in.
    """
    prices = np.asarray(prices, dtype=float)
    totals = [
        prices[i : i + size].sum() for i in range(len(prices) - size + 1)
    ]
    best = int(np.argmax(totals) if expensive else np.argmin(totals))

    return {"start_hour": best + 1, "end_hour": best + size}
