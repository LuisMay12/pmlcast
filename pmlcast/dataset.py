#!/usr/bin/env python3
"""Build model-ready samples: 168-hour windows, splits and hold-out folds."""

import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd

import pmlcast
from pmlcast import catalog as catalog_module
from pmlcast import cenace
from pmlcast import config
from pmlcast import preprocess
from pmlcast import storage

INPUT_DAYS = config.INPUT_HOURS // 24
SEQ_FEATURES = (
    "z_pml",
    "z_ene",
    "z_per",
    "z_cng",
) + preprocess.SEQ_CAL_FEATURES
META_FEATURES = tuple("region_" + r for r in config.REGIONS) + ("voltage_z",)
INDEX_COLUMNS = [
    "sample_id",
    "node",
    "sistema",
    "region",
    "zone",
    "kv",
    "origin_date",
    "target_date",
    "split",
    "is_spike_day",
    "n_interp_input",
    "mu",
    "sigma",
    "std_raw",
    "mu_7d",
]
ARRAY_KEYS = (
    "X_seq",
    "X_cal",
    "X_level",
    "X_meta",
    "zone_idx",
    "node_idx",
    "y",
    "mu",
    "sigma",
    "comp_mu",
    "sample_id",
)
SPLITS = ("train", "val", "test")


class Dataset:
    """Hold the sample arrays, the per-sample index and the metadata."""

    def __init__(self, arrays, index, meta):
        self.arrays = arrays
        self.index = index.reset_index(drop=True)
        self.meta = meta

    def __len__(self):
        return len(self.index)

    def positions(self, split=None, nodes=None):
        """Return the sample positions of a split and/or a set of nodes."""
        mask = np.ones(len(self.index), dtype=bool)
        if split is not None:
            mask &= (self.index["split"] == split).to_numpy()
        if nodes is not None:
            mask &= self.index["node"].isin(list(nodes)).to_numpy()
        return np.flatnonzero(mask)

    def y_prices(self, positions=None):
        """Return the actual target prices in MXN/MWh for positions."""
        if positions is None:
            positions = np.arange(len(self.index))
        return preprocess.invert(
            self.arrays["y"][positions],
            self.arrays["mu"][positions],
            self.arrays["sigma"][positions],
        )


# ---------------------------------------------------------------------------
# Per-node sample construction
# ---------------------------------------------------------------------------


def _day_windows(array, window):
    """Stack ``window`` consecutive days: (n_days, 24, F) -> (n, 24*w, F)."""
    view = np.lib.stride_tricks.sliding_window_view(array, window, axis=0)
    view = np.moveaxis(view, -1, 1)
    return view.reshape(view.shape[0], window * array.shape[1], array.shape[2])


def enumerate_origins(
    day_missing,
    day_interp,
    stats_valid,
    max_interp_input,
    input_days=INPUT_DAYS,
):
    """Return the boolean mask of valid origin days D.

    D is valid when its statistics are valid, the input days
    D-(input_days-1)..D and the target day D+1 contain no missing hour,
    the target day has no interpolated hour and the input holds at most
    ``max_interp_input`` interpolated hours.

    Args:
        day_missing: Bool array (n_days,) - day has a missing hour.
        day_interp: Int array (n_days,) - interpolated hours in the day.
        stats_valid: Bool array (n_days,) - scaling stats valid at D.
        max_interp_input: Cap on interpolated hours inside the input.
        input_days: Length of the input window in whole days.

    Returns:
        Bool array (n_days,).
    """
    n_days = len(day_missing)
    valid = np.zeros(n_days, dtype=bool)
    if n_days < input_days + 1:
        return valid
    span = input_days + 1  # input days plus the target day
    missing_span = np.convolve(
        day_missing.astype(int), np.ones(span, int), "valid"
    )
    interp_input = np.convolve(
        day_interp.astype(int), np.ones(input_days, int), "valid"
    )
    origins = np.arange(input_days - 1, n_days - 1)
    ok = (
        stats_valid[origins]
        & (missing_span[origins - (input_days - 1)] == 0)
        & (day_interp[origins + 1] == 0)
        & (interp_input[origins - (input_days - 1)] <= max_interp_input)
    )
    valid[origins[ok]] = True
    return valid


def build_node_samples(
    node_df,
    node_meta,
    holiday_set,
    volatility=config.VOLATILITY_FEATURE,
    input_days=INPUT_DAYS,
):
    """Build every valid sample of one node.

    Args:
        node_df: Silver rows of one node (continuous hourly grid).
        node_meta: Dict with ``sistema``, ``region``, ``zone`` and ``kv``.
        holiday_set: Set of holiday dates.
        volatility: Volatility level feature (see ``preprocess``).
        input_days: Length of the input window in whole days.

    Returns:
        Tuple ``(arrays, index)``; ``arrays`` holds ``X_seq``
        (n, input_days * 24, 9),
        ``X_cal`` (n, 10), ``X_level`` (n, 3), ``y`` (n, 24) in z-units,
        ``mu``, ``sigma`` (n,), ``comp_mu`` (n, 3), ``region_onehot``
        (n, 8) and ``kv`` (n,); ``index`` is a DataFrame of sample facts.
    """
    df = node_df.sort_values("ts_local").reset_index(drop=True)
    if len(df) % 24 != 0 or df["ts_local"].iloc[0].hour != 0:
        raise ValueError("node_df must be a continuous grid of whole days")
    n_days = len(df) // 24
    days = np.array(df["fecha"].to_numpy()[::24])
    prices = df[preprocess.PRICE_COLUMNS].to_numpy(dtype=float)
    prices = prices.reshape(n_days, 24, len(preprocess.PRICE_COLUMNS))
    quality = df["quality"].to_numpy().reshape(n_days, 24)
    day_missing = (quality == "missing").any(axis=1)
    day_interp = (quality == "interp").sum(axis=1)

    stats = preprocess.rolling_stats(df, input_hours=input_days * 24).reindex(
        days
    )
    valid = enumerate_origins(
        day_missing,
        day_interp,
        stats["valid"].fillna(False).to_numpy(dtype=bool),
        config.MAX_INTERP_INPUT,
        input_days,
    )
    origins = np.flatnonzero(valid)
    if len(origins) == 0:
        return None, pd.DataFrame(columns=INDEX_COLUMNS[1:])

    price_windows = _day_windows(prices, input_days)[
        origins - (input_days - 1)
    ]
    calendar = preprocess.sequence_calendar(df["ts_local"], holiday_set)
    calendar = calendar.reshape(n_days, 24, len(preprocess.SEQ_CAL_FEATURES))
    cal_windows = _day_windows(calendar, input_days)[
        origins - (input_days - 1)
    ]

    picked = stats.iloc[origins]
    mu = picked["mu"].to_numpy(dtype=float)
    sigma = picked["sigma"].to_numpy(dtype=float)
    comp_mu = picked[["mu_ene", "mu_per", "mu_cng"]].to_numpy(dtype=float)

    z_pml = preprocess.standardize(price_windows[..., 0], mu, sigma)
    z_comp = (price_windows[..., 1:] - comp_mu[:, None, :]) / sigma[
        :, None, None
    ]
    x_seq = np.concatenate([z_pml[..., None], z_comp, cal_windows], axis=2)
    y = preprocess.standardize(prices[origins + 1, :, 0], mu, sigma)

    targets = days[origins + 1]
    x_cal = np.stack(
        [preprocess.target_day_features(t, holiday_set) for t in targets]
    )
    x_level = preprocess.level_features(picked, volatility)
    region_onehot = np.tile(
        preprocess.encode_region(node_meta["region"]), (len(origins), 1)
    )
    interp_input = np.convolve(
        day_interp.astype(int), np.ones(input_days, int), "valid"
    )

    index = pd.DataFrame(
        {
            "node": node_meta["node"],
            "sistema": node_meta["sistema"],
            "region": node_meta["region"],
            "zone": node_meta["zone"],
            "kv": float(node_meta["kv"]),
            "origin_date": days[origins],
            "target_date": targets,
            "n_interp_input": interp_input[origins - (input_days - 1)],
            "mu": mu,
            "sigma": sigma,
            "std_raw": picked["std_raw"].to_numpy(dtype=float),
            "mu_7d": picked["mu_7d"].to_numpy(dtype=float),
        }
    )
    arrays = {
        "X_seq": x_seq.astype(np.float32),
        "X_cal": x_cal.astype(np.float32),
        "X_level": x_level.astype(np.float32),
        "y": y.astype(np.float32),
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "comp_mu": comp_mu.astype(np.float32),
        "region_onehot": region_onehot.astype(np.float32),
        "kv": np.full(len(origins), float(node_meta["kv"]), dtype=np.float32),
    }
    return arrays, index


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def assign_splits(index, test_months=6, val_days=90):
    """Label samples train/val/test by target date.

    Test = the last ``test_months`` months of target dates; val = the
    ``val_days`` before the test period; train = everything earlier.

    Returns:
        Tuple ``(labels, boundaries)`` with a Series of labels and a dict
        of ISO date strings ``train_end``, ``val_start``, ``val_end``,
        ``test_start``, ``test_end``.
    """
    targets = pd.to_datetime(index["target_date"])
    last = targets.max()
    test_start = (last - pd.DateOffset(months=test_months)) + pd.Timedelta(
        days=1
    )
    val_start = test_start - pd.Timedelta(days=val_days)
    labels = pd.Series("train", index=index.index, dtype=object)
    labels[(targets >= val_start) & (targets < test_start)] = "val"
    labels[targets >= test_start] = "test"
    boundaries = {
        "train_end": (val_start - pd.Timedelta(days=1)).date().isoformat(),
        "val_start": val_start.date().isoformat(),
        "val_end": (test_start - pd.Timedelta(days=1)).date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": last.date().isoformat(),
    }
    return labels, boundaries


def node_metadata(catalog, node):
    """Return the metadata dict of a node (UNKNOWN when not catalogued)."""
    rows = catalog[catalog["node"] == node]
    if rows.empty:
        return {
            "node": node,
            "sistema": "SIN",
            "region": config.UNKNOWN,
            "zone": config.UNKNOWN,
            "kv": float("nan"),
        }
    row = rows.iloc[0]
    return {
        "node": node,
        "sistema": row["sistema"],
        "region": row["region"],
        "zone": row["zone"],
        "kv": float(row["kv"]),
    }


def build_dataset(
    silver_df,
    catalog,
    nodes,
    name,
    start=None,
    end=None,
    test_months=6,
    val_days=90,
    volatility=config.VOLATILITY_FEATURE,
    market="MDA",
    input_days=INPUT_DAYS,
):
    """Assemble the dataset artifact for a list of nodes.

    Args:
        silver_df: Silver rows (continuous grids) of the nodes.
        catalog: Node dimension table.
        nodes: Node keys to include.
        name: Dataset name (used for the artifact folder).
        start: Optional first operation date to use.
        end: Optional last operation date to use.
        test_months: Length of the test period in months.
        val_days: Length of the validation period in days.
        volatility: Volatility level feature.
        market: Market label recorded in the metadata.
        input_days: Length of the input window in whole days. The window
            length is a hyperparameter, so a dataset is built per length.

    Returns:
        A ``Dataset``. Zone vocabulary and voltage scaler are fitted on
        the nodes that have training samples.
    """
    years = set()
    frames = []
    for node in sorted(set(nodes)):
        node_df = silver_df[silver_df["node"] == node]
        if start is not None:
            node_df = node_df[node_df["fecha"] >= start]
        if end is not None:
            node_df = node_df[node_df["fecha"] <= end]
        if node_df.empty:
            continue
        years.update(
            pd.to_datetime(node_df["fecha"]).dt.year.unique().tolist()
        )
        frames.append((node, node_df))
    if not frames:
        raise ValueError("no silver rows for the requested nodes")
    holiday_set = preprocess.mx_holidays(range(min(years), max(years) + 2))

    parts, indexes = [], []
    for node, node_df in frames:
        arrays, index = build_node_samples(
            node_df,
            node_metadata(catalog, node),
            holiday_set,
            volatility,
            input_days,
        )
        if arrays is None:
            continue
        parts.append(arrays)
        indexes.append(index)
    if not parts:
        raise ValueError("no valid samples could be built")

    index = pd.concat(indexes, ignore_index=True)
    arrays = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    labels, boundaries = assign_splits(index, test_months, val_days)
    index["split"] = labels.to_numpy()
    index["sample_id"] = np.arange(len(index), dtype=np.int64)
    # imported here so the serving container can build an input window
    # without carrying the evaluation stack (matplotlib, scikit-learn)
    from pmlcast import evaluate

    index["is_spike_day"] = evaluate.spike_days(
        preprocess.invert(arrays["y"], arrays["mu"], arrays["sigma"]),
        arrays["mu"],
        arrays["sigma"],
    )

    train_nodes = sorted(index.loc[index["split"] == "train", "node"].unique())
    train_meta = index[index["node"].isin(train_nodes)].drop_duplicates("node")
    zone_vocab = preprocess.build_zone_vocab(train_meta["zone"])
    log_mean, log_std = preprocess.fit_voltage_scaler(train_meta["kv"])
    voltage_z = np.array(
        [
            preprocess.voltage_feature(kv, log_mean, log_std)
            for kv in arrays["kv"]
        ]
    )
    node_list = sorted(index["node"].unique())
    node_pos = {node: i for i, node in enumerate(node_list)}

    final = {
        "X_seq": arrays["X_seq"],
        "X_cal": arrays["X_cal"],
        "X_level": arrays["X_level"],
        "X_meta": np.column_stack([arrays["region_onehot"], voltage_z]).astype(
            np.float32
        ),
        "zone_idx": index["zone"]
        .map(lambda z: preprocess.encode_zone(z, zone_vocab))
        .to_numpy(np.int32),
        "node_idx": index["node"].map(node_pos).to_numpy(np.int32),
        "y": arrays["y"],
        "mu": arrays["mu"],
        "sigma": arrays["sigma"],
        "comp_mu": arrays["comp_mu"],
        "sample_id": index["sample_id"].to_numpy(),
    }
    meta = {
        "name": name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pmlcast_version": pmlcast.__version__,
        "market": market,
        "input_hours": input_days * 24,
        "target_hours": config.TARGET_HOURS,
        "seq_features": list(SEQ_FEATURES),
        "cal_features": list(preprocess.CAL_FEATURES),
        "level_features": list(preprocess.LEVEL_FEATURES),
        "meta_features": list(META_FEATURES),
        "scaling": {
            "window_days": config.STATS_DAYS,
            "sigma_floor_abs": config.SIGMA_FLOOR_ABS,
            "sigma_floor_rel": config.SIGMA_FLOOR_REL,
            "ddof": 0,
            "min_coverage": config.MIN_STATS_COVERAGE,
            "volatility_feature": volatility,
        },
        "spike_rule": {"k_sigma": config.SPIKE_K_SIGMA},
        "regions": list(config.REGIONS),
        "zone_vocab": zone_vocab,
        "nodes": node_list,
        "train_nodes": train_nodes,
        "voltage_log_mean": log_mean,
        "voltage_log_std": log_std,
        "splits": boundaries,
        "catalog_version": (
            str(catalog["catalog_version"].iloc[0]) if len(catalog) else None
        ),
        "silver_range": [
            str(index["origin_date"].min()),
            str(index["target_date"].max()),
        ],
        "n_samples": {s: int((index["split"] == s).sum()) for s in SPLITS},
    }
    return Dataset(final, index[INDEX_COLUMNS], meta)


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------


def walk_forward(index, split="test"):
    """Yield ``(target_date, positions)`` for each target day, in order."""
    mask = (index["split"] == split).to_numpy()
    positions = np.flatnonzero(mask)
    targets = pd.to_datetime(index["target_date"].iloc[positions])
    frame = pd.DataFrame({"pos": positions, "target": targets.to_numpy()})
    for target, group in frame.groupby("target", sort=True):
        yield pd.Timestamp(target).date(), group["pos"].to_numpy()


def leave_one_node_out(index, max_folds=None):
    """Yield hold-out folds for nodes whose zone keeps other nodes.

    Each fold has ``held_out`` (node), ``zone``, ``train`` and ``val``
    positions of the other nodes and ``test`` positions of the held-out
    node inside the test period.
    """
    nodes_per_zone = (
        index.drop_duplicates("node").groupby("zone")["node"].nunique()
    )
    node_zone = index.drop_duplicates("node").set_index("node")["zone"]
    eligible = sorted(
        n for n, z in node_zone.items() if nodes_per_zone[z] >= 2
    )
    if max_folds is not None:
        eligible = eligible[:max_folds]
    node_col = index["node"].to_numpy()
    split_col = index["split"].to_numpy()
    for node in eligible:
        others = node_col != node
        yield {
            "held_out": node,
            "zone": node_zone[node],
            "train": np.flatnonzero(others & (split_col == "train")),
            "val": np.flatnonzero(others & (split_col == "val")),
            "test": np.flatnonzero((node_col == node) & (split_col == "test")),
        }


def leave_one_zone_out(index, max_folds=3):
    """Yield hold-out folds where a whole load zone is unseen.

    Zones with the most nodes come first. Held-out samples must be fed
    with ``zone_override=0`` (UNKNOWN) at inference time.
    """
    per_zone = index.drop_duplicates("node").groupby("zone")["node"].nunique()
    per_zone = per_zone.drop(config.UNKNOWN, errors="ignore")
    order = sorted(per_zone.index, key=lambda z: (-per_zone[z], z))
    if max_folds is not None:
        order = order[:max_folds]
    zone_col = index["zone"].to_numpy()
    split_col = index["split"].to_numpy()
    for zone in order:
        inside = zone_col == zone
        yield {
            "held_out": zone,
            "nodes": sorted(index.loc[inside, "node"].unique()),
            "train": np.flatnonzero(~inside & (split_col == "train")),
            "val": np.flatnonzero(~inside & (split_col == "val")),
            "test": np.flatnonzero(inside & (split_col == "test")),
            "zone_override": 0,
        }


def inputs_dict(
    ds,
    positions,
    use_meta=False,
    use_zone=False,
    use_node=False,
    zone_override=None,
):
    """Return the model inputs for positions, keyed as the model expects."""
    inputs = {
        "seq": ds.arrays["X_seq"][positions],
        "cal": ds.arrays["X_cal"][positions],
        "level": ds.arrays["X_level"][positions],
    }
    if use_meta:
        inputs["meta"] = ds.arrays["X_meta"][positions]
    if use_zone:
        zone = ds.arrays["zone_idx"][positions]
        if zone_override is not None:
            zone = np.full_like(zone, zone_override)
        inputs["zone"] = zone
    if use_node:
        inputs["node"] = ds.arrays["node_idx"][positions]
    return inputs


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return value


def save_dataset(ds, out_dir):
    """Write ``arrays.npz``, ``index.parquet`` and ``meta.json``."""
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "arrays.npz"), **ds.arrays)
    ds.index.to_parquet(os.path.join(out_dir, "index.parquet"), index=False)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(_json_ready(ds.meta), fh, indent=2, ensure_ascii=False)
    return out_dir


def load_dataset(out_dir):
    """Read a dataset artifact written by ``save_dataset``."""
    with np.load(os.path.join(out_dir, "arrays.npz")) as data:
        arrays = {key: data[key] for key in data.files}
    index = pd.read_parquet(os.path.join(out_dir, "index.parquet"))
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return Dataset(arrays, index, meta)


def dataset_dir(name, gold_dir=config.GOLD_DIR):
    """Return the artifact folder of a named dataset."""
    return os.path.join(gold_dir, "dataset_{}".format(name))


def main():
    """Build a dataset artifact from silver and the catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--nodes-file", required=True)
    parser.add_argument("--start", type=datetime.date.fromisoformat)
    parser.add_argument("--end", type=datetime.date.fromisoformat)
    parser.add_argument("--market", default="MDA", choices=config.MARKETS)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--val-days", type=int, default=90)
    parser.add_argument("--volatility", default=config.VOLATILITY_FEATURE)
    parser.add_argument("--input-days", type=int, default=INPUT_DAYS)
    parser.add_argument("--silver-dir", default=config.SILVER_DIR)
    parser.add_argument(
        "--catalog", default=os.path.join(config.CATALOG_DIR, "nodes.parquet")
    )
    parser.add_argument("--gold-dir", default=config.GOLD_DIR)
    args = parser.parse_args()

    log = config.setup_logging()
    nodes = cenace.read_nodes_file(args.nodes_file)
    silver = storage.read_silver(args.silver_dir, args.market, nodes=nodes)
    catalog = catalog_module.load_catalog(args.catalog)
    ds = build_dataset(
        silver,
        catalog,
        nodes,
        args.name,
        start=args.start,
        end=args.end,
        test_months=args.test_months,
        val_days=args.val_days,
        volatility=args.volatility,
        market=args.market,
        input_days=args.input_days,
    )
    out = save_dataset(ds, dataset_dir(args.name, args.gold_dir))
    log.info(
        "dataset %s: %d samples %s from %d nodes -> %s",
        args.name,
        len(ds),
        ds.meta["n_samples"],
        len(ds.meta["nodes"]),
        out,
    )
    log.info(
        "X_seq %s, X_cal %s, X_level %s, X_meta %s, y %s",
        ds.arrays["X_seq"].shape,
        ds.arrays["X_cal"].shape,
        ds.arrays["X_level"].shape,
        ds.arrays["X_meta"].shape,
        ds.arrays["y"].shape,
    )


if __name__ == "__main__":
    main()
