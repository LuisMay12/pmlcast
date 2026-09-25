#!/usr/bin/env python3
"""Render the dashboard page and the data it needs."""

import datetime
import json
import os

import pandas as pd

import pmlcast
from pmlcast import storage

# `daily` imports the evaluation stack (matplotlib, scikit-learn), which
# the serving container does not carry. The dashboard only needs the score
# table name and the recent-skill summary, so it imports them on use.
SCORE_TABLE = "daily_scores"

PAGE_TITLE = "PMLcast"


def node_options(nodes_file):
    """Return the nodes the dashboard offers, grouped by region."""
    if not os.path.exists(nodes_file):
        return []

    frame = pd.read_csv(nodes_file)
    frame = frame.sort_values(["region", "node"])

    return frame[["node", "region", "zone", "kv"]].to_dict("records")


def latest_scored(gold_dir, node=None):
    """Return the most recent scored forecast, for the accuracy panel."""
    path = storage.predictions_path(gold_dir, SCORE_TABLE)
    if not os.path.exists(path):
        return None

    scores = storage.read_predictions(gold_dir, SCORE_TABLE)
    if node is not None:
        scores = scores[scores["node"] == node]
    if scores.empty:
        return None

    row = scores.sort_values("target_date").iloc[-1]

    return {
        "node": row["node"],
        "target_date": str(row["target_date"]),
        "mae": float(row["mae"]),
        "top4_overlap": float(row["top4_overlap"]),
        "captured_value": float(row["captured_value"]),
        "skill_vs_naive24": float(row["skill_vs_naive24"]),
    }


def data_freshness(silver_dir, market="MDA"):
    """Return the last operation day present in silver."""
    table = storage.coverage_table(silver_dir, market)
    if table.empty:
        return None

    return str(table["last"].max())


def recent_skill(gold_dir, days=30):
    """Summarize how the model has done over the last N scored days."""
    path = storage.predictions_path(gold_dir, SCORE_TABLE)
    if not os.path.exists(path):
        return {}

    scores = storage.read_predictions(gold_dir, SCORE_TABLE)
    if scores.empty:
        return {}

    cutoff = max(scores["target_date"]) - datetime.timedelta(days=days)
    recent = scores[scores["target_date"] > cutoff]

    return {
        "days": int(recent["target_date"].nunique()),
        "nodes": int(recent["node"].nunique()),
        "mae": float(recent["mae"].mean()),
        "top4_overlap": float(recent["top4_overlap"].mean()),
        "captured_value": float(recent["captured_value"].mean()),
        "skill_vs_naive24": float(recent["skill_vs_naive24"].mean()),
    }


def model_card_panel(meta_path, gold_dir):
    """Collect what the model card panel shows next to a forecast."""
    panel = {"version": pmlcast.__version__}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        panel.update(
            {
                "dataset": meta["name"],
                "nodes": len(meta["nodes"]),
                "input_hours": meta["input_hours"],
                "trained_from": meta["silver_range"][0],
                "scaling_days": meta["scaling"]["window_days"],
                "splits": meta["splits"],
            }
        )
    panel["recent"] = recent_skill(gold_dir)

    return panel


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0;
        background: #f6f7f9; color: #1a1d21; }}
 header {{ background: #12263f; color: #fff; padding: 18px 24px; }}
 header h1 {{ margin: 0; font-size: 20px; }}
 header p {{ margin: 4px 0 0; opacity: .75; font-size: 13px; }}
 main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
 .row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }}
 .card {{ background: #fff; border: 1px solid #e3e6ea; border-radius: 8px;
         padding: 18px; margin-bottom: 18px; }}
 label {{ display: block; font-size: 13px; color: #5b6472;
         margin-bottom: 4px; }}
 select, input, button {{ font: inherit; padding: 7px 10px;
         border: 1px solid #ccd2d9; border-radius: 6px; }}
 button {{ background: #12263f; color: #fff; border-color: #12263f;
          cursor: pointer; }}
 button:disabled {{ opacity: .5; cursor: default; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ text-align: right; padding: 5px 8px;
          border-bottom: 1px solid #eef0f3; }}
 th:first-child, td:first-child {{ text-align: left; }}
 .bar {{ background: #d7e3f4; height: 14px; border-radius: 3px; }}
 .bar.peak {{ background: #e8833a; }}
 .bar.cheap {{ background: #58a55c; }}
 .muted {{ color: #5b6472; font-size: 13px; }}
 .error {{ color: #b3261e; }}
 .pill {{ display: inline-block; background: #eef1f5; border-radius: 999px;
         padding: 3px 10px; margin-right: 6px; font-size: 12px; }}
</style>
<header>
  <h1>PMLcast</h1>
  <p>Day-ahead hourly prices for Mexican grid nodes. Informational only,
     not financial advice.</p>
</header>
<main>
  <div class="card">
    <div class="row">
      <div>
        <label for="node">Node</label>
        <select id="node"></select>
      </div>
      <div>
        <label for="date">Target date (blank = tomorrow)</label>
        <input type="date" id="date">
      </div>
      <div><button id="go">Forecast</button></div>
    </div>
    <p class="muted" id="status"></p>
  </div>

  <div class="card" id="result" hidden>
    <div id="windows"></div>
    <table id="hours"></table>
  </div>

  <div class="card">
    <h3 style="margin-top:0">Model</h3>
    <div id="card"></div>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);

async function boot() {{
  const meta = await (await fetch("/dashboard/data")).json();
  $("node").innerHTML = meta.nodes
    .map((n) => `<option value="${{n.node}}">` +
                `${{n.node}} - ${{n.region}}</option>`)
    .join("");
  const card = meta.model;
  const recent = card.recent || {{}};
  $("card").innerHTML = `
    <span class="pill">version ${{card.version}}</span>
    <span class="pill">${{card.nodes || "?"}} nodes</span>
    <span class="pill">${{card.input_hours || "?"}} h input</span>
    <span class="pill">trained from ${{card.trained_from || "?"}}</span>
    <span class="pill">prices through ${{meta.freshness || "?"}}</span>
    ${{recent.days
      ? `<p class="muted">Last ${{recent.days}} scored days:
         MAE ${{recent.mae.toFixed(1)}} MXN/MWh,
         top-4 overlap ${{(recent.top4_overlap * 100).toFixed(0)}}%,
         captured value ${{(recent.captured_value * 100).toFixed(0)}}%,
         skill vs yesterday
         ${{(recent.skill_vs_naive24 * 100).toFixed(0)}}%.</p>`
      : `<p class="muted">No scored forecasts yet: the daily job publishes
         accuracy here once CENACE publishes the real prices.</p>`}}`;
}}

async function run() {{
  $("go").disabled = true;
  $("status").textContent = "Forecasting...";
  $("status").className = "muted";
  const body = {{ node: $("node").value }};
  if ($("date").value) body.target_date = $("date").value;

  // A node whose stored prices stopped short is filled from CENACE
  // before it can be forecast, and a free instance may also be waking
  // up. Both are ordinary waits, so say what is happening instead of
  // leaving the page still.
  const waited = [
    setTimeout(() => {{
      $("status").textContent =
        "Asking CENACE for the prices this node is missing...";
    }}, 1500),
    setTimeout(() => {{
      $("status").textContent =
        "Still waiting on CENACE. A sleeping free instance can take " +
        "up to a minute to wake up.";
    }}, 8000),
  ];

  let response;
  let data;
  try {{
    response = await fetch("/forecast", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(body),
    }});
    data = await response.json();
  }} catch (error) {{
    waited.forEach(clearTimeout);
    $("go").disabled = false;
    $("status").textContent =
      "Could not reach the service. It may still be waking up: retry " +
      "in a moment.";
    $("status").className = "error";
    $("result").hidden = true;
    return;
  }}
  waited.forEach(clearTimeout);
  $("go").disabled = false;

  if (!response.ok) {{
    $("status").textContent = data.detail || "Could not forecast.";
    $("status").className = "error";
    $("result").hidden = true;
    return;
  }}

  const prices = data.hours.map((h) => h.pml);
  const max = Math.max(...prices, 1);
  const peak = data.best_injection_window;
  const cheap = data.cheapest_window;

  $("status").textContent =
    `Issued ${{data.issued_at}}, using prices through ${{data.history_end}}.` +
    (data.backtest ? " Backtest: this date is already published." : "") +
    (data.evaluated_node
      ? ""
      : " This node is outside the evaluation set, so the reported error" +
        " does not describe it.");
  $("windows").innerHTML = `
    <p><strong>Inject</strong> between hour ${{peak.start_hour}} and
       ${{peak.end_hour}}. <strong>Charge</strong> between hour
       ${{cheap.start_hour}} and ${{cheap.end_hour}}.</p>`;
  $("hours").innerHTML =
    "<tr><th>Hour</th><th>MXN/MWh</th><th style='width:60%'></th></tr>" +
    data.hours
      .map((h) => {{
        const cls =
          h.hora >= peak.start_hour && h.hora <= peak.end_hour
            ? "peak"
            : h.hora >= cheap.start_hour && h.hora <= cheap.end_hour
            ? "cheap"
            : "";
        const width = Math.max(1, (h.pml / max) * 100);
        return `<tr><td>${{h.hora}}</td>` +
               `<td>${{h.pml.toFixed(2)}}</td><td>` +
               `<div class="bar ${{cls}}" style="width:${{width}}%">` +
               `</div></td></tr>`;
      }})
      .join("");
  $("result").hidden = false;
}}

$("go").addEventListener("click", run);
boot();
</script>
"""


def render_page():
    """Return the dashboard HTML."""
    return PAGE.format(title=PAGE_TITLE)


def dashboard_data(nodes_file, meta_path, gold_dir, silver_dir):
    """Return everything the page needs on load."""
    return {
        "nodes": node_options(nodes_file),
        "model": model_card_panel(meta_path, gold_dir),
        "freshness": data_freshness(silver_dir),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
