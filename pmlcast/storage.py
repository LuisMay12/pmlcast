#!/usr/bin/env python3
"""Store bronze envelopes, silver hourly tables and gold predictions."""

import argparse
import datetime
import glob
import gzip
import hashlib
import json
import os

import pandas as pd

from pmlcast import config
from pmlcast import preprocess

INDEX_NAME = "index.jsonl"
# Raw CENACE JSON repeats a lot, so it gzips about 12 to 1. The reader
# still accepts the plain files written before this changed.
BRONZE_SUFFIX = ".json.gz"


# ---------------------------------------------------------------------------
# Bronze: one JSON envelope per CENACE request, plus a small line index
# ---------------------------------------------------------------------------


def request_key(request):
    """Return a short, stable file key for a request."""
    digest = hashlib.sha1(",".join(request.nodes).encode("utf-8"))
    return "{:%Y%m%d}-{:%Y%m%d}__{}n_{}".format(
        request.start, request.end, len(request.nodes), digest.hexdigest()[:12]
    )


def bronze_path(bronze_dir, request):
    """Return the envelope path of a request inside the bronze layer."""
    return os.path.join(
        bronze_dir,
        request.sistema,
        request.proceso,
        str(request.start.year),
        request_key(request) + BRONZE_SUFFIX,
    )


def _index_path(bronze_dir, sistema, proceso):
    return os.path.join(bronze_dir, sistema, proceso, INDEX_NAME)


def _open_envelope(path, mode="r"):
    """Open an envelope, transparently handling the gzipped ones."""
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def find_bronze(bronze_dir, request):
    """Return the stored envelope path of a request, or None.

    Looks for the gzipped name first and falls back to the plain one, so
    a half-finished collection keeps working after the format changed.
    """
    path = bronze_path(bronze_dir, request)
    if os.path.exists(path):
        return path
    plain = path[: -len(BRONZE_SUFFIX)] + ".json"
    if os.path.exists(plain):
        return plain
    return None


def write_bronze(path, request, response):
    """Write a request/response envelope and append it to the index.

    Args:
        path: Destination path (see ``bronze_path``).
        request: The ``Request`` namedtuple that was sent.
        response: Dict returned by ``cenace.fetch``.
    """
    envelope = {
        "url": response.get("url"),
        "sistema": request.sistema,
        "proceso": request.proceso,
        "nodes": list(request.nodes),
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "fetched_at": response.get("fetched_at"),
        "status_code": response.get("status_code"),
        "elapsed_s": response.get("elapsed_s"),
        "attempts": response.get("attempts"),
        "error": response.get("error"),
        "user_agent": response.get("user_agent"),
        "body": response.get("text") or None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _open_envelope(path, "w") as handle:
        json.dump(envelope, handle, ensure_ascii=False)

    line = {key: envelope[key] for key in envelope if key != "body"}
    line["key"] = os.path.basename(path)
    line["has_body"] = envelope["body"] is not None
    market_dir = os.path.dirname(os.path.dirname(path))
    index_path = os.path.join(market_dir, INDEX_NAME)
    with open(index_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_bronze(path):
    """Read an envelope and split it into request metadata and response.

    Returns:
        Tuple ``(meta, response)`` where ``meta`` has ``sistema``,
        ``proceso``, ``nodes`` (tuple), ``start`` and ``end`` (dates) and
        ``response`` mirrors the dict produced by ``cenace.fetch``.
    """
    with _open_envelope(path) as handle:
        envelope = json.load(handle)
    meta = {
        "sistema": envelope["sistema"],
        "proceso": envelope["proceso"],
        "nodes": tuple(envelope["nodes"]),
        "start": datetime.date.fromisoformat(envelope["start"]),
        "end": datetime.date.fromisoformat(envelope["end"]),
    }
    response = {
        "url": envelope.get("url"),
        "status_code": envelope.get("status_code"),
        "text": envelope.get("body") or "",
        "elapsed_s": envelope.get("elapsed_s"),
        "fetched_at": envelope.get("fetched_at"),
        "attempts": envelope.get("attempts"),
        "error": envelope.get("error"),
        "user_agent": envelope.get("user_agent"),
    }
    return meta, response


def iter_bronze(bronze_dir, sistema=None, proceso=None):
    """Yield envelope paths, sorted, optionally filtered by system/market."""
    root = os.path.join(bronze_dir, sistema or "*", proceso or "*", "*")
    paths = glob.glob(os.path.join(root, "*.json"))
    paths += glob.glob(os.path.join(root, "*" + BRONZE_SUFFIX))
    for path in sorted(paths):
        yield path


def _window_dates(start, end):
    days = (end - start).days + 1
    return [start + datetime.timedelta(days=i) for i in range(days)]


def _index_lines(bronze_dir, proceso):
    pattern = os.path.join(bronze_dir, "*", proceso, INDEX_NAME)
    lines = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if raw:
                    lines.append(json.loads(raw))
    return lines


def rebuild_index(bronze_dir, sistema, proceso):
    """Recreate the line index of one system/market from its envelopes."""
    index_path = _index_path(bronze_dir, sistema, proceso)
    lines = []
    for path in iter_bronze(bronze_dir, sistema, proceso):
        meta, response = read_bronze(path)
        lines.append(
            {
                "sistema": sistema,
                "proceso": proceso,
                "nodes": list(meta["nodes"]),
                "start": meta["start"].isoformat(),
                "end": meta["end"].isoformat(),
                "status_code": response["status_code"],
                "key": os.path.basename(path),
                "has_body": bool(response["text"]),
            }
        )
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    return lines


def bronze_requested(bronze_dir, proceso):
    """Return every ``(node, date)`` pair already requested from CENACE.

    Envelopes with any status count, so a window that returned no data is
    not requested again unless the caller asks for it explicitly.
    """
    lines = _index_lines(bronze_dir, proceso)
    if not lines:
        for sistema in config.SYSTEMS:
            if glob.glob(os.path.join(bronze_dir, sistema, proceso, "*")):
                lines.extend(rebuild_index(bronze_dir, sistema, proceso))
    requested = set()
    for line in lines:
        start = datetime.date.fromisoformat(line["start"])
        end = datetime.date.fromisoformat(line["end"])
        for node in line["nodes"]:
            for day in _window_dates(start, end):
                requested.add((node, day))
    return requested


# ---------------------------------------------------------------------------
# Silver: one Parquet file per (market, node) with a continuous hourly grid
# ---------------------------------------------------------------------------


def silver_path(silver_dir, market, node):
    """Return the Parquet path of one node's hourly series."""
    return os.path.join(
        silver_dir, "market={}".format(market), "node={}.parquet".format(node)
    )


def upsert_silver(silver_dir, market, node, observed_df):
    """Merge observed rows into a node's silver file and rebuild its grid.

    Existing observed rows win over incoming duplicates, so applying the
    same rows twice leaves the file unchanged. Derived rows (interpolated
    or missing) are always recomputed from the observed ones.

    Args:
        silver_dir: Root of the silver layer.
        market: ``MDA`` or ``MTR``.
        node: Node key.
        observed_df: Silver rows of this node (observed qualities only).

    Returns:
        The full hourly DataFrame that was written.
    """
    path = silver_path(silver_dir, market, node)
    frames = []
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        observed_mask = existing["quality"].isin(config.OBSERVED_QUALITIES)
        frames.append(existing[observed_mask])
    incoming = observed_df[observed_df["node"] == node]
    incoming = incoming[incoming["quality"].isin(config.OBSERVED_QUALITIES)]
    frames.append(incoming)

    observed = pd.concat(frames, ignore_index=True)
    observed = observed.sort_values("ts_local")
    observed = observed.drop_duplicates("ts_local", keep="first")
    if observed.empty:
        raise ValueError("no observed rows to write for node {}".format(node))

    full = preprocess.build_hourly_grid(observed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    full.to_parquet(path, index=False)
    return full


def read_silver(
    silver_dir, market="MDA", nodes=None, start=None, end=None, columns=None
):
    """Read silver rows for one market, optionally filtered.

    Args:
        silver_dir: Root of the silver layer.
        market: ``MDA`` or ``MTR``.
        nodes: Optional iterable of node keys.
        start: Optional first operation date (inclusive).
        end: Optional last operation date (inclusive).
        columns: Optional list of columns to read.

    Returns:
        DataFrame sorted by ``node`` and ``ts_local``; empty when no file
        matches.
    """
    pattern = os.path.join(
        silver_dir, "market={}".format(market), "node=*.parquet"
    )
    paths = sorted(glob.glob(pattern))
    if nodes is not None:
        wanted = {silver_path(silver_dir, market, node) for node in nodes}
        paths = [path for path in paths if path in wanted]
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    if not frames:
        return pd.DataFrame(columns=columns or preprocess.SILVER_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if start is not None:
        df = df[df["fecha"] >= start]
    if end is not None:
        df = df[df["fecha"] <= end]
    order = [c for c in ("node", "ts_local") if c in df.columns]
    return df.sort_values(order).reset_index(drop=True)


def silver_coverage(silver_dir, market):
    """Return the ``(node, date)`` pairs with 24 observed hours in silver."""
    df = read_silver(
        silver_dir, market=market, columns=["node", "fecha", "quality"]
    )
    if df.empty:
        return set()
    observed = df[df["quality"].isin(config.OBSERVED_QUALITIES)]
    counts = observed.groupby(["node", "fecha"]).size()
    return set(counts[counts >= config.TARGET_HOURS].index)


def coverage_table(silver_dir, market):
    """Summarize per-node coverage: first day, last day, days, share."""
    df = read_silver(
        silver_dir, market=market, columns=["node", "fecha", "quality"]
    )
    if df.empty:
        return pd.DataFrame(
            columns=["node", "first", "last", "days", "full_days", "share"]
        )
    observed = df["quality"].isin(config.OBSERVED_QUALITIES)
    per_day = df[observed].groupby(["node", "fecha"]).size().reset_index()
    per_day.columns = ["node", "fecha", "n"]
    rows = []
    for node, group in per_day.groupby("node"):
        first, last = group["fecha"].min(), group["fecha"].max()
        span = (last - first).days + 1
        full_days = int((group["n"] >= config.TARGET_HOURS).sum())
        rows.append(
            {
                "node": node,
                "first": first,
                "last": last,
                "days": span,
                "full_days": full_days,
                "share": full_days / span if span else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gold: model predictions
# ---------------------------------------------------------------------------


def predictions_path(gold_dir, model_name):
    """Return the Parquet path of a model's predictions."""
    return os.path.join(
        gold_dir, "predictions", "{}.parquet".format(model_name)
    )


def write_predictions(gold_dir, model_name, df):
    """Write a predictions table to the gold layer."""
    path = predictions_path(gold_dir, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_predictions(gold_dir, model_name):
    """Read a model's predictions table from the gold layer."""
    return pd.read_parquet(predictions_path(gold_dir, model_name))


def main():
    """Print the silver coverage table for one market."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["coverage"])
    parser.add_argument("--market", default="MDA", choices=config.MARKETS)
    parser.add_argument("--silver-dir", default=config.SILVER_DIR)
    args = parser.parse_args()

    table = coverage_table(args.silver_dir, args.market)
    with pd.option_context("display.max_rows", 500, "display.width", 120):
        print(table.to_string(index=False))
    if not table.empty:
        print(
            "{} nodes, mean share of full days {:.3f}".format(
                len(table), table["share"].mean()
            )
        )


if __name__ == "__main__":
    main()
