# Databricks notebook source
# MAGIC %md
# MAGIC # PMLcast — day-ahead electricity price forecasting for Mexican grid nodes
# MAGIC
# MAGIC Given the last 14 days of hourly prices at a node of the Mexican grid,
# MAGIC PMLcast predicts the **24 hourly prices of the next day** so a solar plant
# MAGIC with a battery can decide when to charge and when to sell.
# MAGIC
# MAGIC Holberton School — Machine Learning Portfolio Project. Luis May.
# MAGIC Code and full reports: https://github.com/LuisMay12/pmlcast
# MAGIC
# MAGIC This notebook is the showcase: it loads the trained model and the stored
# MAGIC results, and shows what the project found. The model was trained locally
# MAGIC (262k samples, ~20 minutes on a laptop) and is registered here in MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The problem
# MAGIC
# MAGIC CENACE sets a different price at every node of the grid, for every hour.
# MAGIC In the day-ahead market, offers close *before* anyone knows the clearing
# MAGIC price. Between the cheapest and most expensive hour of the same day at the
# MAGIC same node the difference is routinely 2–3×, so knowing *which hours* will
# MAGIC be expensive is worth money to anyone who can choose when to inject.
# MAGIC
# MAGIC **The forecast is issued the morning of day D, for day D+1**, before offers
# MAGIC close. Every input is public by then; nothing from the target day leaks in.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup
# MAGIC
# MAGIC Serverless compute ships without TensorFlow, so the notebook installs the
# MAGIC version the model was trained with. It must sit **alone in its own cell**:
# MAGIC `restartPython` restarts the interpreter, so anything below it in the same
# MAGIC cell would never run. Takes two or three minutes.

# COMMAND ----------

# MAGIC %pip install --quiet "tensorflow==2.21.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

BUNDLE = "/Volumes/workspace/pmlcast/bundle"

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open(f"{BUNDLE}/manifest.json") as handle:
    manifest = json.load(handle)

print("dataset:      ", manifest["dataset"])
print("trained from: ", manifest["trained_from"])
print("input window: ", manifest["input_hours"], "hours")
print("test period:  ", manifest["splits"]["test_start"], "->", manifest["splits"]["test_end"])
print("sample nodes: ", ", ".join(manifest["sample_nodes"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## The data
# MAGIC
# MAGIC 112 nodes collected straight from CENACE's public service, hourly, from
# MAGIC January 2016 to today: about 10.6 years per node, 99.9 % coverage, no gaps.
# MAGIC The sample below is 12 of those nodes, spread across the country's regions.
# MAGIC
# MAGIC The collector is polite by design: sequential requests, exponential
# MAGIC backoff, every raw response cached so nothing is ever fetched twice.

# COMMAND ----------

silver = pd.read_parquet(f"{BUNDLE}/silver_sample.parquet")
catalog = pd.read_csv(f"{BUNDLE}/nodes_stage2.csv")

print(f"{len(silver):,} hourly rows | {silver['node'].nunique()} nodes")
print(f"{silver['fecha'].min()} -> {silver['fecha'].max()}")
display(
    silver.merge(catalog[["node", "region", "kv"]], on="node", how="left")
    .groupby(["region", "node"])
    .agg(rows=("pml", "size"), median_price=("pml", "median"), p99=("pml", lambda s: s.quantile(0.99)))
    .round(1)
    .reset_index()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why one model can serve every node
# MAGIC
# MAGIC Prices differ wildly between nodes: a node in Yucatán may sit near 900
# MAGIC MXN/MWh while one in Monterrey sits near 2,500. Feeding raw pesos to the
# MAGIC network would force it to memorise each node, and it could say nothing
# MAGIC about a node it never saw.
# MAGIC
# MAGIC Instead every price is standardised against **that node's own last 28
# MAGIC days**, frozen at the forecast origin:
# MAGIC
# MAGIC ```
# MAGIC z = (price − mean of the last 28 days) / std of the last 28 days
# MAGIC ```
# MAGIC
# MAGIC The model stops seeing pesos and starts seeing *shapes*. That is what lets
# MAGIC it forecast a node that was never in training: it needs the node's recent
# MAGIC history, not its identity.

# COMMAND ----------

sample = silver[silver["node"] == manifest["sample_nodes"][0]].tail(24 * 14).copy()
window = silver[silver["node"] == manifest["sample_nodes"][0]].tail(24 * 28)
mu, sigma = window["pml"].mean(), window["pml"].std()

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
axes[0].plot(range(len(sample)), sample["pml"], color="#12263f")
axes[0].set_title(f"{manifest['sample_nodes'][0]} — last 14 days in MXN/MWh")
axes[0].set_xlabel("hour")
axes[1].plot(range(len(sample)), (sample["pml"] - mu) / sigma, color="#e8833a")
axes[1].axhline(0, color="#999", lw=0.8)
axes[1].set_title("the same days in z-units — what the model sees")
axes[1].set_xlabel("hour")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How it is judged
# MAGIC
# MAGIC Four honest baselines, from dumb to serious:
# MAGIC
# MAGIC | baseline | idea |
# MAGIC |---|---|
# MAGIC | Naive-24h | tomorrow costs what today cost |
# MAGIC | Naive-168h | tomorrow costs what the same weekday cost last week |
# MAGIC | Seasonal MA | the historical average for this time of year |
# MAGIC | Ridge on lags | linear regression on past prices and the calendar |
# MAGIC
# MAGIC And four metrics. Two are ordinary error measures; two are about the
# MAGIC decision the user actually makes:
# MAGIC
# MAGIC - **Top-4 overlap**: of the four hours the model called most expensive,
# MAGIC   how many really were. A *set* overlap, not an exact match: calling
# MAGIC   {17,19,20,21} when the truth was {18,19,20,21} scores 0.75, not 0.
# MAGIC - **Captured value**: the revenue of discharging in the four predicted
# MAGIC   hours over the best possible. Being one hour off costs little when the
# MAGIC   price was nearly the same.

# COMMAND ----------

metrics = pd.read_parquet(f"{BUNDLE}/metrics.parquet").sort_values("mae")
display(metrics)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
colors = ["#e8833a" if m == "lstm_history" else "#b9c3cf" for m in metrics["model"]]
axes[0].barh(metrics["model"], metrics["mae"], color=colors)
axes[0].set_title("Mean absolute error (MXN/MWh) — lower is better")
axes[1].barh(metrics["model"], metrics["top4_overlap"], color=colors)
axes[1].axvline(4 / 24, color="#b3261e", ls="--", lw=1)
axes[1].set_title("Top-4 overlap — dashed line is random guessing")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The LSTM wins on absolute error: **243 MXN/MWh against 280** for the best
# MAGIC baseline, a 26 % skill score over the weekly naive. On days without a
# MAGIC price spike it drops to 197.
# MAGIC
# MAGIC Worth saying plainly: the ridge regression lands within 35 MXN/MWh of the
# MAGIC LSTM and edges it on the two product metrics. A good part of this problem
# MAGIC is linear, and reporting that is more useful than claiming a clean sweep.

# COMMAND ----------

# MAGIC %md
# MAGIC ## One day, forecast against reality

# COMMAND ----------

preds = pd.read_parquet(f"{BUNDLE}/predictions.parquet")
lstm = preds[preds["model_name"] == "lstm_history"]
pick = lstm.groupby(["node", "target_date"]).size().reset_index().iloc[len(lstm) // 48]
day = lstm[(lstm["node"] == pick["node"]) & (lstm["target_date"] == pick["target_date"])]
day = day.sort_values("hora")

top_pred = set(day.nlargest(4, "pml_pred")["hora"])
top_real = set(day.nlargest(4, "pml_actual")["hora"])

plt.figure(figsize=(11, 4))
plt.plot(day["hora"], day["pml_actual"], "o-", color="#12263f", label="actual")
plt.plot(day["hora"], day["pml_pred"], "o--", color="#e8833a", label="forecast")
for hour in sorted(top_pred):
    plt.axvspan(hour - 0.4, hour + 0.4, color="#e8833a", alpha=0.12)
plt.title(f"{pick['node']} — {pick['target_date']}   (shaded: the 4 hours the model called most expensive)")
plt.xlabel("hour of day")
plt.ylabel("MXN/MWh")
plt.legend()
plt.tight_layout()
plt.show()

print(f"most expensive hours — forecast {sorted(top_pred)}, actual {sorted(top_real)}")
print(f"hit {len(top_pred & top_real)} of 4")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the experiments found
# MAGIC
# MAGIC Four results, three of which contradicted what I expected going in:
# MAGIC
# MAGIC **Node metadata does not help.** Region, voltage and load zone were meant
# MAGIC to let the model specialise. Across two hold-out regimes and two datasets
# MAGIC the sign of the difference kept flipping, with gaps of ~2 MXN/MWh on errors
# MAGIC of 180. That is noise, not a small effect. The per-node scaling already
# MAGIC carries the information the metadata was supposed to add, so the simpler
# MAGIC model ships.
# MAGIC
# MAGIC **Node identity does not help either.** A node-ID embedding never beat the
# MAGIC history-only model. Knowing *which* node it is adds nothing over knowing
# MAGIC what that node did last week — which is exactly why the model generalises
# MAGIC to nodes it has never seen.
# MAGIC
# MAGIC **Deep history does help**, which I had predicted wrong. Training from 2019
# MAGIC beat 2022 and 2024 consistently, even though the pre-2022 market was a
# MAGIC different regime. More shapes beat fresher shapes.
# MAGIC
# MAGIC **The robust loss lost.** Huber was chosen in the pitch for its resistance
# MAGIC to price spikes, but MAE beat it in all eight search configurations: the
# MAGIC error is *measured* in MAE, and Huber smooths exactly the days that weigh
# MAGIC most in it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model in MLflow
# MAGIC
# MAGIC Training happened on a laptop; the registry lives here. That is the point
# MAGIC of MLflow: train where you can, version and serve where the team looks.

# COMMAND ----------

import keras
import mlflow

mlflow.set_registry_uri("databricks-uc")
model = keras.models.load_model(f"{BUNDLE}/model.keras")
model.summary()

# COMMAND ----------

REGISTERED_NAME = "workspace.pmlcast.pmlcast_lstm"
row = metrics[metrics["model"] == "lstm_history"].iloc[0]

# Unity Catalog refuses a model without a signature. Passing one real
# example lets MLflow infer it: three named inputs in, 24 hourly prices
# out. The shapes come from the dataset metadata, so they cannot drift
# from what the model was trained with.
example = {
    "seq": np.zeros((1, manifest["input_hours"], 9), dtype=np.float32),
    "cal": np.zeros((1, 10), dtype=np.float32),
    "level": np.zeros((1, 3), dtype=np.float32),
}
signature = mlflow.models.infer_signature(example, model.predict(example))
print(signature)

with mlflow.start_run(run_name="lstm_history_final") as run:
    mlflow.log_params(
        {
            "architecture": "LSTM(64) -> LSTM(32) -> Dense(64) -> Dense(24)",
            "input_hours": manifest["input_hours"],
            "loss": "mae",
            "dropout": 0.2,
            "trained_from": manifest["trained_from"],
            "n_nodes": len(pd.read_csv(f"{BUNDLE}/nodes_stage2.csv")),
            "scaling": "28-day z-score frozen at the forecast origin",
            "trained_on": "local laptop, TensorFlow 2.21",
        }
    )
    mlflow.log_metrics(
        {
            "test_mae": float(row["mae"]),
            "test_rmse": float(row["rmse"]),
            "test_skill_vs_naive168": float(row["skill"]),
            "test_top4_overlap": float(row["top4_overlap"]),
            "test_captured_value": float(row["captured_value"]),
        }
    )
    mlflow.tensorflow.log_model(
        model,
        name="model",
        signature=signature,
        input_example=example,
        registered_model_name=REGISTERED_NAME,
    )
    print("run:", run.info.run_id)

print("registered as:", REGISTERED_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Limits, stated plainly
# MAGIC
# MAGIC - It forecasts the shape of a **normal** day. It cannot predict congestion
# MAGIC   or fuel-supply spikes, so spike days are reported separately instead of
# MAGIC   hidden inside an average.
# MAGIC - **One day ahead only.** Further horizons would need recursive forecasting.
# MAGIC - A node needs **28 days of published history**. Nodes outside the
# MAGIC   supported scope get an explicit error, never an extrapolated curve.
# MAGIC - It is informational. It is not financial advice, and the product shows
# MAGIC   its recent accuracy next to every forecast so nobody reads a single
# MAGIC   number as certainty.
# MAGIC
# MAGIC All inputs are public market data published by CENACE for transparency. No
# MAGIC personal data of any kind is involved.
