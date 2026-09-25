#!/usr/bin/env python3
"""Pack the smallest data slice a public demo needs."""

import argparse
import datetime
import os
import shutil

from pmlcast import config
from pmlcast import serving
from pmlcast import storage

DEMO_DIR = os.path.join(config.DATA_DIR, "demo")
DEMO_DAYS = 365


def main():
    """Write a demo data folder: one year of prices, model and metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=DEMO_DAYS)
    parser.add_argument("--out", default=DEMO_DIR)
    args = parser.parse_args()

    log = config.setup_logging()
    silver_out = os.path.join(args.out, "silver", "market=MDA")
    os.makedirs(silver_out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "catalog"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "snapshot"), exist_ok=True)

    # The model only looks 28 days back, so old history is dead weight in
    # a demo: it was needed to train, not to forecast.
    frame = storage.read_silver(config.SILVER_DIR, "MDA")
    cutoff = frame["fecha"].max() - datetime.timedelta(days=args.days)
    frame = frame[frame["fecha"] > cutoff]

    for node, rows in frame.groupby("node"):
        rows.to_parquet(
            os.path.join(silver_out, "node={}.parquet".format(node)),
            index=False,
        )

    shutil.copy(
        serving.DEFAULT_MODEL,
        os.path.join(
            args.out, "models", os.path.basename(serving.DEFAULT_MODEL)
        ),
    )
    shutil.copytree(
        os.path.dirname(serving.DEFAULT_META),
        os.path.join(args.out, "gold", "dataset_final"),
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("arrays.npz", "index.parquet"),
    )
    for name in ("nodes_stage2.csv",):
        shutil.copy(
            os.path.join(config.SNAPSHOT_DIR, name),
            os.path.join(args.out, "snapshot", name),
        )
    shutil.copy(
        os.path.join(config.CATALOG_DIR, "nodes.parquet"),
        os.path.join(args.out, "catalog", "nodes.parquet"),
    )

    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(args.out)
        for f in files
    )
    log.info(
        "demo data: %d nodes, %s -> %s, %.1f MB",
        frame["node"].nunique(),
        frame["fecha"].min(),
        frame["fecha"].max(),
        total / 1e6,
    )


if __name__ == "__main__":
    main()
