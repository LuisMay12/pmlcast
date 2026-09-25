#!/usr/bin/env python3
"""Turn raw CENACE records into clean hourly series and model features."""

import datetime
import functools
import math
import zoneinfo

import holidays
import numpy as np
import pandas as pd

from pmlcast import config

SILVER_COLUMNS = [
    "node",
    "market",
    "sistema",
    "ts_local",
    "fecha",
    "hora",
    "pml",
    "pml_ene",
    "pml_per",
    "pml_cng",
    "quality",
    "source_hora",
    "ingested_at",
]
PRICE_COLUMNS = list(config.PRICE_COLUMNS)
COMPONENT_COLUMNS = ["pml_ene", "pml_per", "pml_cng"]
DERIVED_QUALITIES = ("interp", "missing")
SEQ_CAL_FEATURES = ("hod_sin", "hod_cos", "dow_sin", "dow_cos", "is_holiday")
CAL_FEATURES = tuple("dow_{}".format(i) for i in range(7)) + (
    "month_sin",
    "month_cos",
    "is_holiday",
)
LEVEL_FEATURES = ("mu_over_1000", "volatility", "trend_7d")
VOLATILITY_OPTIONS = ("sigma_over_1000", "log1p_sigma")


# ---------------------------------------------------------------------------
# Daylight saving time and hourly alignment
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _dst_days_year(year):
    """Return the DST transition days of one year as {date: kind}."""
    tz = zoneinfo.ZoneInfo(config.TZ_NAME)
    found = {}
    day = datetime.date(year, 1, 1)
    while day.year == year:
        nxt = day + datetime.timedelta(days=1)
        start = datetime.datetime.combine(day, datetime.time(), tzinfo=tz)
        end = datetime.datetime.combine(nxt, datetime.time(), tzinfo=tz)
        offset_shift = start.utcoffset() - end.utcoffset()
        hours = 24 + offset_shift.total_seconds() / 3600.0
        if hours < 24:
            found[day] = "spring"
        elif hours > 24:
            found[day] = "fall"
        day = nxt
    return found


def dst_days(years):
    """Return the DST transition days for the given years.

    Args:
        years: Iterable of calendar years.

    Returns:
        Dict mapping each transition date to ``"spring"`` (23-hour day)
        or ``"fall"`` (25-hour day), in the CENACE system time zone.
    """
    out = {}
    for year in years:
        out.update(_dst_days_year(int(year)))
    return out


def align_day(day_df, kind="normal"):
    """Map the records of one operation day onto 24 clock slots.

    CENACE numbers the hours of a day sequentially (``hora`` 1..N), so a
    spring-forward day has 23 records and a fall-back day has 25. The
    23-hour day gets a synthetic 02:00 slot (mean of the hours around the
    jump, quality ``dst_fill``); the 25-hour day merges its two 01:00
    records into one slot (quality ``dst_merge``). Any other record count
    is kept as is and the grid builder treats absent slots as gaps.

    Args:
        day_df: Records of one node and one ``fecha`` with columns
            ``node``, ``fecha``, ``hora`` and the four price columns.
        kind: ``"normal"``, ``"spring"`` or ``"fall"``.

    Returns:
        DataFrame with one row per clock slot present, columns ``node``,
        ``fecha``, ``hora`` (slot 1-24), prices, ``quality``,
        ``source_hora`` and ``ts_local``.
    """
    day = day_df.sort_values("hora").drop_duplicates("hora", keep="last")
    horas = day["hora"].to_numpy().astype(int)
    prices = day[PRICE_COLUMNS].to_numpy(dtype=float)
    count = len(day)

    if kind == "spring" and count == 23 and horas.max() == 23:
        # clock 00,01,03..23 -> shift everything from hora 3 up one slot
        slots = np.where(horas >= 3, horas + 1, horas)
        quality = np.array(["ok"] * count, dtype=object)
        source = horas.astype(float)
        around = (horas == 2) | (horas == 3)
        fill = prices[around].mean(axis=0, keepdims=True)
        prices = np.vstack([prices, fill])
        slots = np.append(slots, 3)
        quality = np.append(quality, "dst_fill")
        source = np.append(source, np.nan)
    elif kind == "fall" and count == 25 and horas.max() == 25:
        # clock 00,01,01,02..23 -> hora 2 and 3 share the 01:00 slot
        merged = prices[(horas == 2) | (horas == 3)].mean(axis=0)
        keep = horas != 3
        slots = np.where(horas >= 4, horas - 1, horas)[keep]
        prices = prices[keep].copy()
        quality = np.array(["ok"] * int(keep.sum()), dtype=object)
        source = horas[keep].astype(float)
        row = int(np.flatnonzero(slots == 2)[0])
        prices[row] = merged
        quality[row] = "dst_merge"
    else:
        keep = horas <= 24
        slots = horas[keep]
        prices = prices[keep]
        quality = np.array(["ok"] * int(keep.sum()), dtype=object)
        source = horas[keep].astype(float)

    order = np.argsort(slots)
    out = pd.DataFrame(prices[order], columns=PRICE_COLUMNS)
    out.insert(0, "hora", slots[order].astype(int))
    out.insert(0, "fecha", day["fecha"].iloc[0])
    out.insert(0, "node", day["node"].iloc[0])
    out["quality"] = quality[order]
    out["source_hora"] = source[order]
    out["ts_local"] = pd.Timestamp(out["fecha"].iloc[0]) + pd.to_timedelta(
        out["hora"] - 1, unit="h"
    )
    return out


def records_to_hourly(records, sistema="SIN", market="MDA"):
    """Convert parsed CENACE records into observed silver rows.

    Args:
        records: List of dicts with ``node``, ``fecha``, ``hora`` and the
            four price fields, as returned by ``cenace.parse_response``.
        sistema: Power system label stored with the rows.
        market: Market label (``MDA`` or ``MTR``) stored with the rows.

    Returns:
        DataFrame with the silver columns and only observed rows
        (qualities ``ok``, ``dst_fill`` or ``dst_merge``).
    """
    if not records:
        return pd.DataFrame(columns=SILVER_COLUMNS)

    df = pd.DataFrame(records)
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
    df["hora"] = pd.to_numeric(df["hora"]).astype(int)
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    years = sorted({d.year for d in df["fecha"].unique()})
    transitions = dst_days(years)
    is_dst = df["fecha"].map(lambda d: d in transitions).to_numpy()

    normal = df[~is_dst & (df["hora"] <= 24)]
    normal = normal.drop_duplicates(["node", "fecha", "hora"], keep="last")
    normal = normal.assign(
        quality="ok", source_hora=normal["hora"].astype(float)
    )
    normal["ts_local"] = pd.to_datetime(normal["fecha"]) + pd.to_timedelta(
        normal["hora"] - 1, unit="h"
    )
    parts = [normal]
    for (_, fecha), group in df[is_dst].groupby(["node", "fecha"]):
        parts.append(align_day(group, transitions[fecha]))

    out = pd.concat(parts, ignore_index=True)
    out["sistema"] = sistema
    out["market"] = market
    out["ingested_at"] = pd.Timestamp.now(tz="UTC")
    out = out.sort_values(["node", "ts_local"]).reset_index(drop=True)
    return out[SILVER_COLUMNS]


def build_hourly_grid(observed_df, max_interp_hours=config.MAX_INTERP_HOURS):
    """Expand observed rows of one node into a continuous hourly grid.

    Args:
        observed_df: Silver rows of a single node and market.
        max_interp_hours: Longest run of absent hours filled by linear
            interpolation (quality ``interp``); longer runs stay NaN with
            quality ``missing``.

    Returns:
        DataFrame with the silver columns, one row per hour from the first
        observed day at 00:00 to the last observed day at 23:00.
    """
    if observed_df.empty:
        raise ValueError("observed_df must not be empty")
    nodes = observed_df["node"].unique()
    if len(nodes) != 1:
        raise ValueError("build_hourly_grid expects rows of a single node")

    obs = observed_df[observed_df["quality"].isin(config.OBSERVED_QUALITIES)]
    obs = obs.sort_values("ts_local").drop_duplicates("ts_local", keep="first")
    start = obs["ts_local"].min().normalize()
    end = obs["ts_local"].max().normalize() + pd.Timedelta(hours=23)
    index = pd.date_range(start, end, freq="h", name="ts_local")

    full = obs.set_index("ts_local").reindex(index)
    absent = full["quality"].isna().to_numpy()
    full["node"] = nodes[0]
    full["market"] = obs["market"].iloc[0]
    full["sistema"] = obs["sistema"].iloc[0]
    full["fecha"] = index.date
    full["hora"] = (index.hour + 1).astype(int)

    if absent.any():
        change = np.r_[True, absent[1:] != absent[:-1]]
        run_id = np.cumsum(change)
        run_len = pd.Series(absent).groupby(run_id).transform("size")
        short = absent & (run_len.to_numpy() <= max_interp_hours)
        filled = full[PRICE_COLUMNS].interpolate(
            method="linear", limit_area="inside"
        )
        short = short & filled["pml"].notna().to_numpy()
        full.loc[short, PRICE_COLUMNS] = filled.loc[short, PRICE_COLUMNS]
        full.loc[short, "quality"] = "interp"
        full.loc[absent & ~short, "quality"] = "missing"
        full.loc[absent, "source_hora"] = np.nan

    full = full.reset_index()
    return full[SILVER_COLUMNS]


def gap_table(silver_df):
    """List the runs of derived (interpolated or missing) hours per node.

    Args:
        silver_df: Silver rows of one or more nodes.

    Returns:
        DataFrame with columns ``node``, ``start``, ``end``, ``n_hours``
        and ``kind`` (``interp`` or ``missing``).
    """
    rows = []
    for node, group in silver_df.groupby("node"):
        group = group.sort_values("ts_local")
        derived = group["quality"].isin(DERIVED_QUALITIES).to_numpy()
        if not derived.any():
            continue
        change = np.r_[True, derived[1:] != derived[:-1]]
        run_id = np.cumsum(change)
        for gid in np.unique(run_id[derived]):
            block = group[run_id == gid]
            rows.append(
                {
                    "node": node,
                    "start": block["ts_local"].iloc[0],
                    "end": block["ts_local"].iloc[-1],
                    "n_hours": len(block),
                    "kind": block["quality"].iloc[0],
                }
            )
    columns = ["node", "start", "end", "n_hours", "kind"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Per-node scaling (pitch section 5.3)
# ---------------------------------------------------------------------------


def rolling_stats(
    node_df,
    stats_hours=config.STATS_HOURS,
    min_coverage=config.MIN_STATS_COVERAGE,
    input_hours=config.INPUT_HOURS,
):
    """Compute the frozen scaling statistics for every origin day D.

    The statistics of day D use the trailing ``stats_hours`` hours ending
    at 23:00 of D (672 hours = 28 days by default). Missing hours are NaN
    in silver, so they are excluded from the means automatically.

    Args:
        node_df: Silver rows of a single node (continuous hourly grid).
        stats_hours: Length of the trailing window in hours.
        min_coverage: Minimum share of observed hours for a valid window.
        input_hours: Length of the model input window (for ``mu_7d``).

    Returns:
        DataFrame indexed by origin date with columns ``mu``, ``std_raw``,
        ``sigma``, ``n_obs``, ``mu_7d``, ``mu_ene``, ``mu_per``, ``mu_cng``
        and ``valid``.
    """
    df = node_df.sort_values("ts_local").set_index("ts_local")
    min_periods = int(math.ceil(stats_hours * min_coverage))
    price = df["pml"].astype(float)
    roll = price.rolling(stats_hours, min_periods=min_periods)

    stats = pd.DataFrame(
        {
            "mu": roll.mean(),
            "std_raw": roll.std(ddof=0),
            "n_obs": roll.count(),
            "mu_7d": price.rolling(input_hours, min_periods=1).mean(),
        }
    )
    for col, name in zip(COMPONENT_COLUMNS, ("mu_ene", "mu_per", "mu_cng")):
        component = df[col].astype(float)
        stats[name] = component.rolling(
            stats_hours, min_periods=min_periods
        ).mean()

    stats = stats[df.index.hour == 23].copy()
    stats.index = pd.Index([ts.date() for ts in stats.index], name="origin")
    floor_rel = config.SIGMA_FLOOR_REL * stats["mu"].abs()
    sigma = np.fmax(
        np.fmax(stats["std_raw"], floor_rel), config.SIGMA_FLOOR_ABS
    )
    stats["valid"] = stats["mu"].notna() & (stats["n_obs"] >= min_periods)
    stats["sigma"] = np.where(stats["valid"], sigma, np.nan)
    return stats


def _as_column(values, ndim):
    """Reshape a 1-D array so it broadcasts along the first axis."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0 or ndim == 1:
        return values
    return values.reshape((-1,) + (1,) * (ndim - 1))


def standardize(prices, mu, sigma):
    """Standardize prices with frozen statistics.

    Args:
        prices: Array of prices in MXN/MWh; the first axis is the sample.
        mu: Scalar or per-sample means.
        sigma: Scalar or per-sample (floored) standard deviations.

    Returns:
        Array of z-scores with the same shape as ``prices``.
    """
    prices = np.asarray(prices, dtype=float)
    return (prices - _as_column(mu, prices.ndim)) / _as_column(
        sigma, prices.ndim
    )


def invert(z, mu, sigma):
    """Undo ``standardize`` and return prices in MXN/MWh."""
    z = np.asarray(z, dtype=float)
    return _as_column(mu, z.ndim) + _as_column(sigma, z.ndim) * z


def level_features(stats, volatility=config.VOLATILITY_FEATURE):
    """Build the three static level features from scaling statistics.

    Args:
        stats: DataFrame (or row) with ``mu``, ``sigma`` and ``mu_7d``.
        volatility: ``"sigma_over_1000"`` or ``"log1p_sigma"``.

    Returns:
        Array of shape (n, 3): ``mu / 1000``, the volatility feature and
        ``(mu_7d - mu) / sigma``. No ratio has ``mu`` in the denominator.
    """
    if volatility not in VOLATILITY_OPTIONS:
        raise ValueError(
            "volatility must be one of {}".format(VOLATILITY_OPTIONS)
        )
    frame = (
        pd.DataFrame(stats) if not isinstance(stats, pd.DataFrame) else stats
    )
    mu = frame["mu"].to_numpy(dtype=float)
    sigma = frame["sigma"].to_numpy(dtype=float)
    mu_7d = frame["mu_7d"].to_numpy(dtype=float)
    if volatility == "sigma_over_1000":
        vol = sigma / 1000.0
    else:
        vol = np.log1p(sigma)
    return np.column_stack([mu / 1000.0, vol, (mu_7d - mu) / sigma])


# ---------------------------------------------------------------------------
# Calendar features and metadata encoders
# ---------------------------------------------------------------------------


def mx_holidays(years):
    """Return the set of Mexican public holidays for the given years."""
    return set(holidays.country_holidays("MX", years=list(years)).keys())


def sequence_calendar(ts_index, holiday_set):
    """Build per-hour calendar features for an input window.

    Args:
        ts_index: DatetimeIndex (or array of timestamps) of the hours.
        holiday_set: Set of ``datetime.date`` holidays.

    Returns:
        Array of shape (n, 5): hour-of-day sine/cosine, day-of-week
        sine/cosine and a holiday flag.
    """
    ts = pd.DatetimeIndex(ts_index)
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    is_holiday = np.array([d in holiday_set for d in ts.date], dtype=float)
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            is_holiday,
        ]
    )


def target_day_features(target_date, holiday_set):
    """Build the calendar features of the day being forecast.

    Args:
        target_date: ``datetime.date`` of the target day.
        holiday_set: Set of ``datetime.date`` holidays.

    Returns:
        Array of shape (10,): weekday one-hot (Monday first), month
        sine/cosine and a holiday flag.
    """
    features = np.zeros(len(CAL_FEATURES), dtype=float)
    features[target_date.weekday()] = 1.0
    angle = 2 * np.pi * (target_date.month - 1) / 12.0
    features[7] = np.sin(angle)
    features[8] = np.cos(angle)
    features[9] = float(target_date in holiday_set)
    return features


def encode_region(region):
    """One-hot encode a regional control center over ``config.REGIONS``."""
    out = np.zeros(len(config.REGIONS), dtype=float)
    label = region if region in config.REGIONS else config.UNKNOWN
    out[config.REGIONS.index(label)] = 1.0
    return out


def fit_voltage_scaler(kv_values):
    """Return the mean and std of ``log(kv)`` over training nodes."""
    logs = np.log(np.asarray(kv_values, dtype=float))
    std = float(logs.std())
    return float(logs.mean()), max(std, 1e-6)


def voltage_feature(kv, log_mean, log_std):
    """Return the standardized log voltage of a node."""
    return (math.log(float(kv)) - log_mean) / log_std


def build_zone_vocab(zones):
    """Map load zones to embedding indices, with UNKNOWN at index 0."""
    vocab = {config.UNKNOWN: 0}
    for zone in sorted(set(zones)):
        if zone != config.UNKNOWN and zone not in vocab:
            vocab[zone] = len(vocab)
    return vocab


def encode_zone(zone, vocab):
    """Return the embedding index of a load zone (0 when unseen)."""
    return int(vocab.get(zone, 0))
