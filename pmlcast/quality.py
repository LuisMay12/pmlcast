#!/usr/bin/env python3
"""Build the data quality report of the silver layer."""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

from pmlcast import catalog as catalog_module
from pmlcast import cenace
from pmlcast import config
from pmlcast import preprocess
from pmlcast import storage

matplotlib.use("Agg")

SIGMA_TAIL_RATIO = 5.0  # p99 / p50 above this -> use log1p(sigma)


def coverage_summary(silver_df):
    """Summarize per node: span, observed share and derived-row counts."""
    rows = []
    for node, group in silver_df.groupby("node"):
        quality = group["quality"]
        rows.append(
            {
                "node": node,
                "first": group["fecha"].min(),
                "last": group["fecha"].max(),
                "hours": len(group),
                "share_observed": quality.isin(
                    config.OBSERVED_QUALITIES
                ).mean(),
                "n_interp": int((quality == "interp").sum()),
                "n_missing": int((quality == "missing").sum()),
                "n_dst_fill": int((quality == "dst_fill").sum()),
                "n_dst_merge": int((quality == "dst_merge").sum()),
            }
        )
    return pd.DataFrame(rows)


def price_level_table(silver_df):
    """Summarize the price distribution of every node in MXN/MWh."""
    rows = []
    for node, group in silver_df.groupby("node"):
        price = group["pml"].dropna()
        rows.append(
            {
                "node": node,
                "mean": price.mean(),
                "median": price.median(),
                "std": price.std(ddof=0),
                "p01": price.quantile(0.01),
                "p99": price.quantile(0.99),
                "max": price.max(),
                "share_nonpositive": float((price <= 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def stats_per_node(silver_df):
    """Return ``{node: rolling_stats DataFrame}``."""
    return {
        node: preprocess.rolling_stats(group)
        for node, group in silver_df.groupby("node")
    }


def spike_summary(silver_df, stats_by_node, k=config.SPIKE_K_SIGMA):
    """Count spike days per node and year.

    A day is a spike day when its highest hourly price exceeds
    ``mu + k * sigma`` of the statistics frozen the day before.
    """
    rows = []
    for node, group in silver_df.groupby("node"):
        daily_max = group.groupby("fecha")["pml"].max()
        stats = stats_by_node[node]
        previous = stats.shift(1).reindex(daily_max.index)
        threshold = previous["mu"] + k * previous["sigma"]
        spike = (daily_max > threshold) & previous["valid"].eq(True)
        years = pd.to_datetime(pd.Index(daily_max.index)).year
        counts = spike.groupby(years).sum()
        for year, count in counts.items():
            rows.append(
                {"node": node, "year": int(year), "spike_days": int(count)}
            )
    return pd.DataFrame(rows, columns=["node", "year", "spike_days"])


def dst_check(silver_df):
    """Count DST-derived rows per node and year (expected 1 + 1 <= 2022)."""
    frame = silver_df[silver_df["quality"].isin(["dst_fill", "dst_merge"])]
    if frame.empty:
        return pd.DataFrame(columns=["node", "year", "dst_fill", "dst_merge"])
    years = pd.to_datetime(frame["fecha"]).dt.year
    table = pd.crosstab([frame["node"], years], frame["quality"])
    table = table.reindex(columns=["dst_fill", "dst_merge"], fill_value=0)
    table.index.names = ["node", "year"]
    return table.reset_index()


def sigma_distribution(stats_by_node):
    """Describe the spread of sigma_D across nodes and days.

    Returns:
        Dict with quantiles, the ``p99 / p50`` ratio and the recommended
        volatility feature (``sigma_over_1000`` or ``log1p_sigma``).
    """
    sigmas = np.concatenate(
        [
            s.loc[s["valid"], "sigma"].to_numpy(dtype=float)
            for s in stats_by_node.values()
        ]
    )
    if len(sigmas) == 0:
        return {"n": 0, "recommendation": config.VOLATILITY_FEATURE}
    quantiles = {
        q: float(np.quantile(sigmas, q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)
    }
    ratio = (
        quantiles[0.99] / quantiles[0.5]
        if quantiles[0.5] > 0
        else float("inf")
    )
    return {
        "n": int(len(sigmas)),
        "quantiles": quantiles,
        "p99_over_p50": float(ratio),
        "share_at_floor": float((sigmas <= config.SIGMA_FLOOR_ABS).mean()),
        "recommendation": (
            "log1p_sigma" if ratio > SIGMA_TAIL_RATIO else "sigma_over_1000"
        ),
        "sigmas": sigmas,
    }


def _plot_price_levels(levels, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    order = levels.sort_values("median")
    ax.bar(order["node"], order["median"], yerr=order["std"], capsize=2)
    ax.set_ylabel("median PML (MXN/MWh)")
    ax.set_title("Price level per node (bar = median, whisker = std)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_sigma_hist(sigmas, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sigmas, bins=60)
    ax.set_xlabel("sigma_D (MXN/MWh)")
    ax.set_ylabel("count of (node, day)")
    ax.set_title("Distribution of the 28-day sigma_D")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_quality_report(silver_df, out_md, fig_dir, k=config.SPIKE_K_SIGMA):
    """Write ``docs/data_quality.md`` and its figures from silver rows.

    Returns:
        Tuple ``(markdown, sigma_info)``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    stats_by_node = stats_per_node(silver_df)
    coverage = coverage_summary(silver_df)
    levels = price_level_table(silver_df)
    gaps = preprocess.gap_table(silver_df)
    spikes = spike_summary(silver_df, stats_by_node, k)
    dst = dst_check(silver_df)
    sigma_info = sigma_distribution(stats_by_node)

    lines = ["# Data quality report", ""]
    lines.append(
        "{} nodes, {} to {}, {} hourly rows (silver layer).".format(
            silver_df["node"].nunique(),
            silver_df["fecha"].min(),
            silver_df["fecha"].max(),
            len(silver_df),
        )
    )
    lines.append("")
    lines.append("## Coverage and gaps")
    lines.append("")
    lines.append(coverage.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append(
        "{} gap runs; longest missing run {} h; {} interpolated runs.".format(
            len(gaps),
            (
                int(gaps.loc[gaps["kind"] == "missing", "n_hours"].max())
                if (gaps["kind"] == "missing").any()
                else 0
            ),
            int((gaps["kind"] == "interp").sum()),
        )
    )
    lines.append("")
    lines.append("## Price level per node (MXN/MWh)")
    lines.append("")
    lines.append(levels.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append("## Zero and negative prices")
    lines.append("")
    lines.append(
        "Non-positive hours share: mean {:.4f}, max {:.4f} ({}).".format(
            levels["share_nonpositive"].mean(),
            levels["share_nonpositive"].max(),
            levels.loc[levels["share_nonpositive"].idxmax(), "node"],
        )
    )
    lines.append("")
    lines.append(
        "## Spike days per node-year (max > mu + {:g} sigma)".format(k)
    )
    lines.append("")
    if not spikes.empty:
        pivot = (
            spikes.pivot(index="node", columns="year", values="spike_days")
            .fillna(0)
            .astype(int)
        )
        lines.append(pivot.to_markdown())
    lines.append("")
    lines.append("## Daylight saving days")
    lines.append("")
    if dst.empty:
        lines.append("No DST-derived rows (data starts after October 2022).")
    else:
        lines.append(dst.to_markdown(index=False))
        lines.append("")
        lines.append(
            "Expected: one dst_fill and one dst_merge per node-year to 2022."
        )
    lines.append("")
    lines.append("## Distribution of sigma_D (28-day scaling statistic)")
    lines.append("")
    if sigma_info["n"]:
        q = sigma_info["quantiles"]
        lines.append(
            "{} (node, day) pairs. p01 {:.1f}, p05 {:.1f}, p50 {:.1f}, "
            "p95 {:.1f}, p99 {:.1f}; "
            "p99/p50 = {:.2f}; share at the floor {:.4f}.".format(
                sigma_info["n"],
                q[0.01],
                q[0.05],
                q[0.5],
                q[0.95],
                q[0.99],
                sigma_info["p99_over_p50"],
                sigma_info["share_at_floor"],
            )
        )
        lines.append("")
        lines.append(
            "**Decision:** volatility feature = `{}` "
            "(log1p when p99/p50 > {:g}).".format(
                sigma_info["recommendation"], SIGMA_TAIL_RATIO
            )
        )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    figures = {
        "price_level_by_node.png": lambda p: _plot_price_levels(levels, p)
    }
    if sigma_info["n"]:
        sigmas = sigma_info["sigmas"]
        figures["sigma_hist.png"] = lambda p: _plot_sigma_hist(sigmas, p)
    for name, plot in figures.items():
        path = os.path.join(fig_dir, name)
        plot(path)
        rel = os.path.relpath(path, os.path.dirname(os.path.abspath(out_md)))
        lines.append("![{}]({})".format(name, rel))
        lines.append("")

    text = "\n".join(lines)
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write(text)
    sigma_info.pop("sigmas", None)
    return text, sigma_info


def main():
    """Generate the data quality report for one market."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="MDA", choices=config.MARKETS)
    parser.add_argument("--nodes-file")
    parser.add_argument("--silver-dir", default=config.SILVER_DIR)
    parser.add_argument(
        "--out", default=os.path.join(config.DOCS_DIR, "data_quality.md")
    )
    parser.add_argument("--fig-dir", default=config.FIGURES_DIR)
    args = parser.parse_args()

    nodes = (
        cenace.read_nodes_file(args.nodes_file) if args.nodes_file else None
    )
    silver = storage.read_silver(args.silver_dir, args.market, nodes=nodes)
    if silver.empty:
        raise SystemExit("no silver rows found")
    _, info = write_quality_report(silver, args.out, args.fig_dir)
    print("report -> {}".format(args.out))
    print(
        "sigma_D: {}".format(
            {k: v for k, v in info.items() if k != "quantiles"}
        )
    )
    _ = catalog_module  # catalog labels are joined by the dataset builder


if __name__ == "__main__":
    main()
