#!/usr/bin/env python3
"""Load the CENACE node catalog and select the nodes to collect."""

import argparse
import collections
import datetime
import os
import re
import shutil
import unicodedata

import pandas as pd

from pmlcast import config

COLUMN_MAP = {
    "SISTEMA": "sistema",
    "CENTRO DE CONTROL REGIONAL": "region",
    "ZONA DE CARGA": "zone",
    "CLAVE": "node",
    "NOMBRE": "name",
    "NIVEL DE TENSION (KV)": "kv",
    "ZONA DE OPERACION DE TRANSMISION": "transmission_zone",
    "GERENCIA REGIONAL DE TRANSMISION": "transmission_mgmt",
    "ZONA DE DISTRIBUCION": "distribution_zone",
    "GERENCIA DIVISIONAL DE DISTRIBUCION": "distribution_division",
    "CLAVE DE ENTIDAD FEDERATIVA (INEGI)": "state_code",
    "ENTIDAD FEDERATIVA (INEGI)": "state",
    "CLAVE DE MUNICIPIO (INEGI)": "municipality_code",
    "MUNICIPIO (INEGI)": "municipality",
    "REGION DE TRANSMISION": "transmission_region",
}
FLAG_LABELS = ("DIRECTAMENTE MODELADA", "INDIRECTAMENTE MODELADA")
FLAG_COLUMNS = ("load_direct", "load_indirect", "gen_direct", "gen_indirect")
CATALOG_COLUMNS = [
    "node",
    "name",
    "sistema",
    "region",
    "zone",
    "kv",
    "load_direct",
    "load_indirect",
    "gen_direct",
    "gen_indirect",
    "transmission_zone",
    "transmission_mgmt",
    "distribution_zone",
    "distribution_division",
    "state_code",
    "state",
    "municipality_code",
    "municipality",
    "transmission_region",
    "catalog_version",
    "loaded_at",
]
TEXT_COLUMNS = (
    "node",
    "name",
    "sistema",
    "region",
    "zone",
    "transmission_zone",
    "transmission_mgmt",
    "distribution_zone",
    "distribution_division",
    "state",
    "municipality",
    "transmission_region",
)
NO_APLICA = "NO APLICA"
STAGE_QUOTAS = {
    1: {400: "all", 230: "all", 115: 15},
    2: {400: "all", 230: 30, 115: 17},
}
STAGE_REGION = {1: "PENINSULAR", 2: None}
NODE_LIST_COLUMNS = ["node", "sistema", "region", "zone", "kv"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def normalize_label(label):
    """Upper-case a header label, drop accents and collapse spaces."""
    text = unicodedata.normalize("NFKD", str(label))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().upper()


def parse_version(path):
    """Extract a ``YYYYMMDD`` catalog version from a file name, or None."""
    name = os.path.basename(path)
    match = re.search(r"[Vv](\d{8})", name)
    if match:
        return match.group(1)
    match = re.search(r"[Vv](\d{4})-(\d{2})-(\d{2})", name)
    if match:
        return "".join(match.groups())
    return None


def _read_raw(path):
    """Read a catalog file into a raw string DataFrame with header labels."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if suffix in (".xlsx", ".xlsm", ".xls"):
        sheet = pd.read_excel(path, header=None, dtype=str)
        labels = sheet.map(normalize_label)
        header_rows = labels.index[(labels == "CLAVE").any(axis=1)]
        if len(header_rows) == 0:
            raise ValueError(
                "no header row with CLAVE found in {}".format(path)
            )
        header = header_rows[0]
        raw = sheet.iloc[header + 1 :].copy()
        raw.columns = [str(c) for c in sheet.iloc[header].tolist()]
        raw = raw.dropna(how="all")
        return raw.reset_index(drop=True)
    raise ValueError("unsupported catalog format: {}".format(path))


def _flag(series):
    text = series.fillna("").astype(str).str.strip().str.upper()
    return ~text.isin(["", "0", "NO", "NAN", "NONE"])


def _from_raw(raw, version):
    """Map raw catalog columns to the dimension-table schema."""
    normalized = [normalize_label(c) for c in raw.columns]
    out = pd.DataFrame(index=raw.index)
    for label, target in COLUMN_MAP.items():
        matches = [i for i, c in enumerate(normalized) if c == label]
        if not matches:
            raise ValueError("catalog column missing: {}".format(label))
        out[target] = raw.iloc[:, matches[0]]

    # Two DIRECTAMENTE/INDIRECTAMENTE pairs: the first is load, the second
    # generation (pandas suffixes duplicated CSV headers with .1).
    positions = {label: [] for label in FLAG_LABELS}
    for i, column in enumerate(normalized):
        base = re.sub(r"\.\d+$", "", column)
        if base in positions:
            positions[base].append(i)
    for label, targets in zip(
        FLAG_LABELS,
        (("load_direct", "gen_direct"), ("load_indirect", "gen_indirect")),
    ):
        for target, position in zip(targets, positions[label]):
            out[target] = _flag(raw.iloc[:, position])
    for target in FLAG_COLUMNS:
        if target not in out.columns:
            out[target] = False

    for column in TEXT_COLUMNS:
        out[column] = out[column].fillna("").astype(str).str.strip()
    out = out[out["node"] != ""].copy()
    out.loc[out["region"].str.upper() == NO_APLICA, "region"] = config.UNKNOWN
    out.loc[out["region"] == "", "region"] = config.UNKNOWN
    out.loc[out["zone"] == "", "zone"] = config.UNKNOWN
    out["kv"] = pd.to_numeric(out["kv"], errors="coerce")
    for column in ("state_code", "municipality_code"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(
            "Int64"
        )
    out["catalog_version"] = version
    out["loaded_at"] = pd.Timestamp.now(tz="UTC")
    out = out.drop_duplicates("node").sort_values("node")
    return out[CATALOG_COLUMNS].reset_index(drop=True)


def load_catalog(path, version=None):
    """Load a Catalogo NodosP file (csv or xlsx) as the dimension table.

    Args:
        path: Path of the catalog file.
        version: Optional ``YYYYMMDD`` version; parsed from the file name
            when omitted.

    Returns:
        DataFrame with ``CATALOG_COLUMNS``, one row per node, with labels
        stripped and ``No Aplica`` regions mapped to UNKNOWN.
    """
    if os.path.splitext(path)[1].lower() == ".parquet":
        return pd.read_parquet(path)
    version = version or parse_version(path)
    if not version:
        raise ValueError(
            "cannot infer the catalog version from {}; pass version".format(
                path
            )
        )
    return _from_raw(_read_raw(path), version)


def save_catalog(df, catalog_dir, snapshot_dir=None):
    """Write the versioned dimension table and refresh ``nodes.parquet``.

    Returns:
        Path of the versioned Parquet file.
    """
    version = str(df["catalog_version"].iloc[0])
    os.makedirs(catalog_dir, exist_ok=True)
    versioned = os.path.join(catalog_dir, "nodes_v{}.parquet".format(version))
    df.to_parquet(versioned, index=False)
    df.to_parquet(os.path.join(catalog_dir, "nodes.parquet"), index=False)
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)
        shutil.copyfile(
            versioned,
            os.path.join(snapshot_dir, "catalog_v{}.parquet".format(version)),
        )
    return versioned


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------


def select_nodes(catalog, quotas, sistema="SIN", region=None):
    """Pick nodes per voltage level, spreading them over regions and zones.

    For each voltage level (highest first) the quota is filled greedily:
    the next node is the one whose region has the fewest picks at that
    level, then whose load zone has the fewest picks overall, then the
    smallest key. ``"all"`` takes every eligible node of that level. The
    result therefore honours "at least two per region per level where
    available" without hard-coding it.

    Args:
        catalog: Dimension table from ``load_catalog``.
        quotas: Dict mapping kV level to an int or ``"all"``.
        sistema: Power system to draw from.
        region: Optional regional control center to restrict to.

    Returns:
        Subset of ``catalog`` sorted by kV (desc), region and node.
    """
    eligible = catalog[
        (catalog["sistema"] == sistema) & (catalog["region"] != config.UNKNOWN)
    ]
    if region is not None:
        eligible = eligible[eligible["region"] == region]

    picked = []
    per_region_kv = collections.Counter()
    per_zone = collections.Counter()
    for kv in sorted(quotas, key=float, reverse=True):
        pool = eligible[eligible["kv"] == float(kv)].sort_values("node")
        rows = pool[["node", "region", "zone"]].to_dict("records")
        quota = quotas[kv]
        limit = len(rows) if quota == "all" else int(quota)
        for _ in range(min(limit, len(rows))):
            best = min(
                rows,
                key=lambda r: (
                    per_region_kv[(r["region"], kv)],
                    per_zone[r["zone"]],
                    r["node"],
                ),
            )
            rows.remove(best)
            picked.append(best["node"])
            per_region_kv[(best["region"], kv)] += 1
            per_zone[best["zone"]] += 1

    selected = catalog[catalog["node"].isin(picked)]
    return selected.sort_values(
        ["kv", "region", "node"], ascending=[False, True, True]
    ).reset_index(drop=True)


def coverage_report(selected, catalog, title="Node coverage"):
    """Describe how the selected nodes cover regions, levels and zones.

    Returns:
        Markdown text with a region x kV table and the zone coverage per
        region, including the zones left uncovered.
    """
    lines = ["# {}".format(title), ""]
    lines.append(
        "{} nodes selected out of {} in catalog v{}.".format(
            len(selected), len(catalog), catalog["catalog_version"].iloc[0]
        )
    )
    lines.append("")

    pivot = pd.crosstab(selected["region"], selected["kv"])
    pivot.columns = ["{:g} kV".format(c) for c in pivot.columns]
    pivot["total"] = pivot.sum(axis=1)
    pivot.loc["total"] = pivot.sum()
    lines.append("## Nodes per region and voltage level")
    lines.append("")
    lines.append(pivot.to_markdown())
    lines.append("")

    lines.append("## Load zones per region")
    lines.append("")
    lines.append("| region | zones in catalog | zones covered | uncovered |")
    lines.append("|---|---|---|---|")
    scope = catalog[catalog["sistema"].isin(selected["sistema"].unique())]
    for region, group in scope.groupby("region"):
        if region == config.UNKNOWN:
            continue
        zones = set(group["zone"])
        covered = set(selected.loc[selected["region"] == region, "zone"])
        uncovered = sorted(zones - covered)
        lines.append(
            "| {} | {} | {} | {} |".format(
                region,
                len(zones),
                len(covered & zones),
                ", ".join(uncovered) if uncovered else "-",
            )
        )
    lines.append("")

    per_zone = selected.groupby("zone").size().sort_values(ascending=False)
    lines.append("## Nodes per selected zone")
    lines.append("")
    lines.append(
        ", ".join("{} ({})".format(z, n) for z, n in per_zone.items())
    )
    lines.append("")
    return "\n".join(lines)


def write_node_list(selected, path):
    """Write the node list CSV used by the collector."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    selected[NODE_LIST_COLUMNS].to_csv(path, index=False)
    return path


def main():
    """Load a catalog, select a stage's nodes and write list + report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", required=True, help="csv, xlsx or parquet"
    )
    parser.add_argument("--version", help="YYYYMMDD when not in the file name")
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--out")
    parser.add_argument("--report")
    parser.add_argument("--catalog-dir", default=config.CATALOG_DIR)
    parser.add_argument("--snapshot-dir", default=config.SNAPSHOT_DIR)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog, version=args.version)
    if not args.catalog.lower().endswith(".parquet"):
        save_catalog(catalog, args.catalog_dir, args.snapshot_dir)

    selected = select_nodes(
        catalog, STAGE_QUOTAS[args.stage], region=STAGE_REGION[args.stage]
    )
    out = args.out or os.path.join(
        args.snapshot_dir, "nodes_stage{}.csv".format(args.stage)
    )
    write_node_list(selected, out)
    report = args.report or os.path.join(
        config.DOCS_DIR,
        "coverage.md" if args.stage == 2 else "coverage_stage1.md",
    )
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", encoding="utf-8") as handle:
        handle.write(
            coverage_report(
                selected, catalog, "Stage {} node coverage".format(args.stage)
            )
        )
    print(
        "stage {}: {} nodes -> {} ; report -> {}".format(
            args.stage, len(selected), out, report
        )
    )
    print(pd.crosstab(selected["region"], selected["kv"]).to_string())
    _ = datetime  # keep import explicit for future date filters


if __name__ == "__main__":
    main()
