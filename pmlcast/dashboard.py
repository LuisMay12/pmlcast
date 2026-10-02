#!/usr/bin/env python3
"""Render the dashboard page and the data it needs."""

import datetime
import glob
import json
import os
import time

import pandas as pd

import pmlcast
from pmlcast import config
from pmlcast import storage

# `daily` imports the evaluation stack (matplotlib, scikit-learn), which
# the serving container does not carry. The dashboard only needs the score
# table name and the recent-skill summary, so it imports them on use.
SCORE_TABLE = "daily_scores"

PAGE_TITLE = "PMLcast"


def node_options(nodes_file, catalog_path=None):
    """Return the nodes the dashboard offers, grouped by region.

    A node key says nothing about where it is, so the catalogue's name and
    municipality are joined in when it is available: picking a node is
    easier from "Tula, Hidalgo" than from ``01TUL-400``.
    """
    if not os.path.exists(nodes_file):
        return []

    frame = pd.read_csv(nodes_file)
    columns = ["node", "region", "zone", "kv"]

    path = catalog_path or os.path.join(config.CATALOG_DIR, "nodes.parquet")
    if os.path.exists(path):
        catalog = pd.read_parquet(path)
        keep = ["node", "name", "municipality", "state"]
        available = [c for c in keep if c in catalog.columns]
        frame = frame.merge(catalog[available], on="node", how="left")
        columns += [c for c in available if c != "node"]

    frame = frame.sort_values(["region", "node"])

    return frame[columns].to_dict("records")


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


FRESHNESS_TTL_S = 300
_freshness_cache = {}


def data_freshness(silver_dir, market="MDA"):
    """Return the last operation day present in silver.

    Only the date column is read, and the answer is kept for a few minutes.
    The full coverage table loads every price row, about 175 MB on the demo
    slice, and building it on each page load ran a 512 MB instance out of
    memory as soon as two visitors opened the page together.
    """
    key = (silver_dir, market)
    now = time.monotonic()
    cached = _freshness_cache.get(key)
    if cached and now - cached[0] < FRESHNESS_TTL_S:
        return cached[1]

    pattern = os.path.join(
        silver_dir, "market={}".format(market), "node=*.parquet"
    )
    last = None
    for path in glob.glob(pattern):
        day = pd.read_parquet(path, columns=["fecha"])["fecha"].max()
        if pd.notna(day) and (last is None or day > last):
            last = day

    value = None if last is None else str(last)
    _freshness_cache[key] = (now, value)

    return value


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
 :root {{
   --night: #111c2e; --night-soft: #1c2b44; --amber: #f2a007;
   --amber-lt: #ffcf5c; --cream: #f7f5f0; --slate: #5a6b84;
   --ink: #16202e; --line: #e4e6ea; --green: #3fa07a; --red: #c4553b;
 }}
 * {{ box-sizing: border-box; }}
 body {{ font: 15px/1.55 -apple-system, system-ui, sans-serif; margin: 0;
        background: var(--cream); color: var(--ink); }}
 header {{ background: var(--night); color: #fff; padding: 22px 24px; }}
 header .wrap {{ max-width: 980px; margin: 0 auto; }}
 header h1 {{ margin: 0; font-size: 22px; letter-spacing: -.2px; }}
 header h1 span {{ color: var(--amber); }}
 header p {{ margin: 5px 0 0; color: #9fb0c9; font-size: 13px; }}
 main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
 .card {{ background: #fff; border: 1px solid var(--line); border-radius: 10px;
         padding: 20px; margin-bottom: 18px; }}
 .card h3 {{ margin: 0 0 14px; font-size: 15px; }}

 /* The picker: a labelled field and its button. There is no date: the
    service forecasts the day after the last one CENACE has published,
    the only day a forecast is still news for. */
 .picker {{ display: grid; gap: 14px;
           grid-template-columns: minmax(0, 1fr) auto;
           align-items: end; }}
 @media (max-width: 680px) {{ .picker {{ grid-template-columns: 1fr; }} }}
 .field label {{ display: block; font-size: 11px; font-weight: 600;
       letter-spacing: .07em; text-transform: uppercase;
       color: var(--slate); margin-bottom: 6px; }}
 .field input {{ font: inherit; width: 100%; padding: 11px 13px;
       border: 1px solid #cfd5dd; border-radius: 8px;
       background: #fff; color: var(--ink); }}
 .field input::placeholder {{ color: #9aa5b4; }}
 .field input:hover {{ border-color: #b3bcc8; }}
 .field input:focus {{ outline: none; border-color: var(--amber);
       box-shadow: 0 0 0 3px rgba(242, 160, 7, .18); }}
 .field .hint {{ font-size: 11px; color: var(--slate); margin: 5px 0 0;
       min-height: 15px; }}
 .field .hint.warn {{ color: #9a6400; }}
 button {{ font: inherit; font-weight: 600; padding: 11px 22px;
       background: var(--night); color: #fff; border-radius: 8px;
       border: 1px solid var(--night); cursor: pointer; }}
 button:hover:not(:disabled) {{ background: var(--night-soft);
       border-color: var(--night-soft); }}
 button:focus-visible {{ outline: none;
       box-shadow: 0 0 0 3px rgba(242, 160, 7, .35); }}
 button:disabled {{ opacity: .45; cursor: default; }}

 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ text-align: right; padding: 6px 8px;
          border-bottom: 1px solid #eef0f3; }}
 th {{ font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
      color: var(--slate); font-weight: 600; }}
 th:first-child, td:first-child {{ text-align: left; }}
 .bar {{ background: #dde3ec; height: 14px; border-radius: 3px; }}
 .bar.peak {{ background: var(--amber); }}
 .bar.cheap {{ background: var(--green); }}
 .where {{ margin: 0 0 12px; font-size: 14px; color: var(--slate); }}
 .where strong {{ color: var(--ink); }}
 .muted {{ color: var(--slate); font-size: 13px; }}
 .error {{ color: var(--red); font-size: 13px; }}
 .pill {{ display: inline-block; background: var(--cream);
         border: 1px solid var(--line); border-radius: 999px;
         padding: 4px 11px; margin: 0 6px 6px 0; font-size: 12px;
         color: var(--ink); }}
</style>
<header>
  <div class="wrap">
    <h1>PML<span>cast</span></h1>
    <p>Day-ahead hourly prices for Mexican grid nodes. Informational only,
       not financial advice.</p>
  </div>
</header>
<main>
  <div class="card">
    <div class="picker">
      <div class="field">
        <label for="node">Node</label>
        <input id="node" list="nodelist" autocomplete="off"
               placeholder="Type a node key, or pick one">
        <datalist id="nodelist"></datalist>
        <p class="hint" id="nodehint"></p>
      </div>
      <div class="field"><button id="go">Forecast</button></div>
    </div>
    <p class="muted" id="status"></p>
  </div>

  <div class="card" id="result" hidden>
    <p class="where" id="where"></p>
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
let NODES = [];
let BY_KEY = {{}};

function placeOf(n) {{
  // Where the node is, in the order a reader scans: town, state, then the
  // voltage level that distinguishes two nodes in the same town.
  const town = n.municipality
    ? n.municipality.charAt(0) + n.municipality.slice(1).toLowerCase()
    : null;
  const state = n.state
    ? n.state.charAt(0) + n.state.slice(1).toLowerCase()
    : null;
  const where = [town, state].filter(Boolean).join(", ");
  const kv = n.kv ? `${{Math.round(n.kv)}} kV` : null;
  return [n.name, where, kv].filter(Boolean).join(" \u00b7 ")
    || n.region || n.node;
}}

function describeNode() {{
  // The hint carries the node's place when it is known, and otherwise
  // warns that the published error does not describe it.
  const key = $("node").value.trim().toUpperCase();
  const n = BY_KEY[key];
  if (n) {{
    $("nodehint").textContent = placeOf(n);
    $("nodehint").className = "hint";
  }} else if (key) {{
    $("nodehint").textContent =
      "Not in the evaluation set \u2014 it will forecast, but the " +
      "reported error does not describe it";
    $("nodehint").className = "hint warn";
  }} else {{
    $("nodehint").textContent =
      `${{NODES.length}} evaluated nodes \u00b7 any SIN node works`;
    $("nodehint").className = "hint";
  }}
}}

async function boot() {{
  const meta = await (await fetch("/dashboard/data")).json();
  NODES = meta.nodes;
  BY_KEY = {{}};
  NODES.forEach((n) => {{ BY_KEY[n.node] = n; }});
  $("nodelist").innerHTML = NODES
    .map((n) => `<option value="${{n.node}}">${{placeOf(n)}}</option>`)
    .join("");
  if (!$("node").value && meta.nodes.length) {{
    $("node").value = meta.nodes[0].node;
  }}
  describeNode();
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
  const node = $("node").value.trim().toUpperCase();
  if (!node) {{
    $("status").textContent = "Type or pick a node first.";
    $("status").className = "error";
    $("go").disabled = false;
    return;
  }}
  const body = {{ node: node }};

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
    // The API already says why in plain words; the prefix tells the
    // three cases apart at a glance: a key that does not exist, a real
    // node this model does not cover, and one CENACE has not published
    // enough history for yet.
    const why = data.detail || "Could not forecast.";
    let lead = "Rejected";
    if (/not in the CENACE catalogue/.test(why)) {{
      lead = "No such node";
    }} else if (/out of scope/.test(why)) {{
      lead = "Out of scope";
    }} else if (/history/.test(why)) {{
      lead = "Not enough published history";
    }}
    $("status").textContent = `${{lead}}: ${{why}}`;
    $("status").className = "error";
    $("result").hidden = true;
    return;
  }}

  const prices = data.hours.map((h) => h.pml);
  const max = Math.max(...prices, 1);
  const peak = data.best_injection_window;
  const cheap = data.cheapest_window;

  $("status").textContent =
    `Forecast for ${{data.target_date}}, issued ${{data.issued_at}} from ` +
    `prices through ${{data.history_end}}.` +
    (data.backtest ? " Backtest: this date is already published." : "") +
    (data.evaluated_node
      ? ""
      : " This node is outside the evaluation set, so the reported error" +
        " does not describe it.");
  // Name the node and where it is, so a printed or shared forecast still
  // says which place it describes.
  const known = BY_KEY[data.node];
  $("where").innerHTML = known
    ? `<strong>${{data.node}}</strong> \u2014 ${{placeOf(known)}}`
    : `<strong>${{data.node}}</strong> \u2014 ${{data.system}}, ` +
      `${{data.market}}`;
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
$("node").addEventListener("input", describeNode);
$("node").addEventListener("change", describeNode);
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
