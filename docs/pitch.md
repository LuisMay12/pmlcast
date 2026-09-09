# PMLcast — Project Pitch

**Day-ahead hourly electricity price forecasting for Mexican grid nodes**

| | |
|---|---|
| Program | Holberton School — Machine Learning Specialization, Portfolio Project |
| Author | Luis May ([@LuisMay12](https://github.com/LuisMay12)) |
| Repository | `github.com/LuisMay12/pmlcast` |
| Pitch version | 0.5 — September 2, 2026 |
| Build window | August 31 – September 24, 2026 (3½ weeks, no slack) |
| Mentor / reviewer | **TBD** (see §11) |

---

## 1. Project Name

**PMLcast** — *PML* is the *Precio Marginal Local*, the hourly wholesale electricity price at each node of the Mexican grid; *cast* as in fore**cast**. The name says exactly what the product does to anyone in the Mexican energy sector.

## 2. Authors and Responsibilities

Solo project. Luis May owns every area:

| Area | Responsibilities |
|---|---|
| Data engineering | CENACE data collection, cleaning, storage (Delta / Parquet), daily refresh job |
| Modeling | Problem framing, baselines, LSTM model, hyperparameter search, evaluation methodology |
| MLOps & deployment | MLflow tracking and registry, Databricks Model Serving endpoint, scheduled batch forecasts |
| Product | Dashboard, REST API contract, "best injection window" feature |
| Quality | Unit tests, CI, reproducibility (seeds, pinned dependencies) |
| Communication | README, model card, blog post, presentation, demo |

## 3. Introduction

### The problem

Mexico's wholesale electricity market (operated by CENACE) sets a different price at every node of the grid, for every hour of the day, in two markets. In the **day-ahead market (MDA)**, participants submit offers *before* knowing what the clearing price will be; the resulting price is binding for every MWh scheduled there. The **real-time market (MTR)** then settles deviations from the day-ahead schedule at a second price that reflects what actually happened (outages, renewable swings, actual demand) and is published only seven days later. For anyone who owns a solar plant with a battery (BESS), or any distributed generator that can choose *when* to inject energy, the key operational question is:

> *At which hours tomorrow will electricity at my node be most expensive, so that I store energy during cheap hours and sell during expensive ones?*

Getting this wrong is costly. Price differences between the cheapest and most expensive hour of the same day at the same node are routinely 2–3×, and congestion events push individual hours to 3–5× the daily mean.

### The goal

Build **PMLcast**: a machine-learning model that, given the recent price history of a node, forecasts the **24 hourly MDA prices for the next day** as an array, and serves it through an API and a dashboard that highlights the best hours to inject energy. The day-ahead price is the one a participant can still act on when the forecast is issued — offers close the morning before the operation day. A planned second phase adds the real-time view: once the day-ahead prices are published, how far the MTR prices are likely to deviate from them.

Concretely, the project will:

1. Collect and clean multi-year hourly PML data from CENACE's public API for nodes across every regional control area of the national interconnected system (SIN).
2. Frame day-ahead forecasting as a supervised sequence-to-vector problem (168 hourly inputs → 24 hourly outputs).
3. Establish honest baselines (naive daily, naive weekly, seasonal moving average, linear regression) and train an **LSTM** conditioned on the node's recent history and its stable CENACE metadata (regional control center as one-hot, voltage level as a number, load zone as an embedding with an UNKNOWN fallback) that must beat them.
4. Validate with time-based splits, walk-forward backtesting on the last six months, and two held-out regimes — leave-one-node-out (a new node in a load zone seen in training) and leave-one-zone-out (a new node in a zone never seen) — to measure, separately, how well the model generalizes to nodes it has never seen.
5. Deploy the model on **Databricks** (MLflow registry → Model Serving endpoint), with a daily job that publishes tomorrow's forecast for every tracked node.
6. Phase 2, after the MVP: a second model that predicts the 24 hourly **MTR** prices of the next day given the published MDA prices for that day, evaluated against the naive rule *MTR = MDA*.

### Why this project

I work at Sunwise, a software platform for solar + storage companies in Mexico, where I previously built a *long-term* (25-year) node price simulation for ROI analysis using a seasonal moving average with Monte Carlo uncertainty bands. That work taught me the data (CENACE quirks, spikes, gaps, isolated grids) and showed me the limits of statistical averaging. PMLcast is a **new problem** — short-horizon, operational, genuinely learnable — where deep learning has a real chance to add value, and where my prior seasonal model becomes one of the baselines the LSTM must beat.

## 4. Description — What the User Experiences

### Personas

- **Plant operator / energy analyst** at a solar+BESS company: needs tomorrow's hourly price curve for their node to schedule charging and injection.
- **Developer** integrating price forecasts into a dispatch optimizer or a platform like Sunwise: needs a stable JSON endpoint.

### Dashboard flow

1. The user opens the PMLcast dashboard in a browser and selects a node (e.g. `01TUL-400`, Mexico City area) from a list grouped by regional control area.
2. They see **tomorrow's forecast**: a 24-bar/line chart of predicted MXN/MWh per hour, with the top-N most expensive hours highlighted and a suggested **injection window** (e.g. *18:00–21:00*), plus the cheapest hours for charging.
3. Below it, **latest forecast vs. published MDA** for the same node, with the day's MAE, so the user always sees how the model has been performing — not just its predictions.
4. A **model card panel** shows the model version, training data range, data freshness ("prices through 2026-09-22 23:00"), and the model's skill versus the naive baseline on the last 30 days.

Phase 2 (after the MVP) adds a panel with tomorrow's published MDA next to the predicted MTR deviation, scored weekly when CENACE publishes the MTR prices.

### API flow

```http
POST /forecast
{ "node": "01TUL-400", "target_date": "2026-09-23" }
```

`target_date` is optional; `{ "node": "01TUL-400" }` forecasts tomorrow.

```json
{
  "node": "01TUL-400",
  "system": "SIN",
  "market": "MDA",
  "target_date": "2026-09-23",
  "issued_at": "2026-09-22T06:00:00-06:00",
  "history_end": "2026-09-22T23:00:00-06:00",
  "unit": "MXN/MWh",
  "hours": [ {"hora": 1, "pml": 812.3}, {"hora": 2, "pml": 790.1}, "…", {"hora": 24, "pml": 905.7} ],
  "best_injection_window": {"start_hour": 18, "end_hour": 21},
  "cheapest_window": {"start_hour": 2, "end_hour": 5},
  "model_version": "pmlcast-lstm@3",
  "skill_vs_naive_30d": 0.18
}
```

**How the target date works.** The date is a parameter of the service, never something the network has to infer. If `target_date` is omitted, the service forecasts tomorrow in America/Mexico_City. The service resolves the date into the model's inputs: the 168 hourly MDA prices of the seven days ending at 23:00 the day before the target (all already published, because CENACE publishes MDA prices the afternoon before the operation day), the target day's calendar features (weekday, month, holiday) and the node's catalog metadata. A past `target_date` runs in **backtest mode**, using only data that had been published before that day's cutoff, so demos and historical evaluation cannot leak. The model is single-horizon: it predicts exactly the next day, and a `target_date` more than one day ahead is rejected (multi-day horizons would need recursive forecasting and are out of scope). A node outside the supported scope (§5.2) — voltage level not covered in training, BCA/BCS, or fewer than 28 days of published history — gets an explicit error instead of an extrapolated curve. Phase 2 adds an `mtr` block to the same response once the MDA prices for the target date have been published.

The array of 24 hourly values is the core deliverable: it is what a dispatch optimizer consumes to decide when to sell.

### What the user will *not* get

PMLcast forecasts the **shape and level of a normal day** well; it cannot predict congestion or fuel-supply spikes. The UI and API always expose recent accuracy and never present a single number as certain.

## 5. Data

### 5.1 What data is needed

| Data | Granularity | Role | Source |
|---|---|---|---|
| MDA PML per node (`pml`) and its components (`pml_ene` energy, `pml_per` losses, `pml_cng` congestion) | Hourly | Target and main input features | CENACE public web service |
| Calendar features: hour, day of week, month, Mexican public holidays | Hourly | Input features | Derived; `holidays` Python package (MX) |
| Node metadata — confirmed in catalog v2026-08-19 (2,609 nodes: 2,457 SIN, 121 BCA, 31 BCS): system, regional control center (7 in SIN), load zone (109), voltage level (8 levels, 34.5–400 kV), transmission operation zone (38), transmission region (53), distribution zone, state and municipality (INEGI codes), load/generation modelling flags | Static | **Model conditioning inputs** (regional control center, voltage level, load zone); state/municipality kept for grouping, per-region evaluation and a later lat/lon experiment | CENACE *Catálogo NodosP* — monthly `.xlsx` at [cenace.gob.mx/Paginas/SIM/NodosP.aspx](https://www.cenace.gob.mx/Paginas/SIM/NodosP.aspx), with a change-log sheet for renamed/added nodes |
| MTR (real-time market) PML — published **7 days after** the operation day, so at forecast time the latest known MTR is a week old | Hourly | **Phase 2 target**: predict the 24 MTR prices of *D+1* given the published MDA prices for *D+1*, the MDA history and the MTR history through *D−7*; naive baseline *MTR = MDA*; scored a week later. Collected 2022 → today for the Stage 2 nodes | CENACE public web service |
| Exogenous drivers: CENACE demand forecast, natural gas price | Hourly / daily | **Stretch**: extra features | CENACE, public sources |

Target horizon and framing: early on day *D*, using MDA prices through day *D* (published during *D−1*), predict the 24 hourly prices of day *D+1* — **before** offers close and before CENACE publishes the *D+1* MDA results later on day *D*. Those published results then become the ground truth, so every forecast is scored the same afternoon. Input window: the last 168 hours (7 days) plus calendar features for the target day. The window therefore ends at 23:00 of day *D* and the target starts at 00:00 of *D+1*, with no gap; training windows are built the same way (input days *D−6…D*, target *D+1*). The MTR publication lag rules out the mirror-image design for real-time prices — 168 hours of MTR history would end a week before the target — which is why the Phase 2 MTR model conditions on the published MDA instead (§8).

### 5.2 How it will be collected

CENACE exposes PML data through a public, unauthenticated REST endpoint (verified working on 2026-09-01):

```
https://ws01.cenace.gob.mx:8082/SWPML/SIM/{SISTEMA}/{PROCESO}/{NODOS}/{YYYY}/{MM}/{DD}/{YYYY}/{MM}/{DD}/JSON
                                          SIN|BCA|BCS  MDA|MTR  comma-separated node keys   start date  end date
```

Each response returns, per node, a list of `{fecha, hora (1–24), pml, pml_ene, pml_per, pml_cng}` records. CENACE's technical manual for the service (*Manual Técnico SW-PML*, 2022-06-24, [PDF](https://www.cenace.gob.mx/DocsMEM/2022-06-24%20Manual%20T%C3%A9cnico%20SW-PML.pdf)) confirms the contract: `GET` only; **1–20 nodes per call**; **1–7 operation days per call** (verified: longer ranges return *"No se pueden mostrar datos con un lapso mayor a 7 dias"*); XML by default, JSON on request; a request may return `204 No Content`, `404`, or a `200` carrying a `Message` field instead of data. **MDA history is available from January 29, 2016 (SIN)** and MTR from January 27, 2017, so up to ~10.6 years of hourly prices can be collected per node. The collector will:

- iterate date windows and node batches, with retries and exponential backoff;
- store raw responses as-is (bronze layer) before any transformation, so any cleaning step can be re-run;
- run idempotently (upsert by `node, fecha, hora`) so a daily scheduled job can top up the dataset without duplicates;
- treat `204`, `404` and `Message` payloads as data-quality events to log, not as crashes.

**Measured collection cost (2026-09-01).** A 7-day request for 20 nodes takes **2.4–3 s when the service is quiet, but 6–19 s at other moments** (both observed within the same hour). Covering Feb 2022 → today is ~240 weekly windows (240 calls ≈ **10–40 minutes** per batch of 20 nodes, sequentially); the full history from January 2016 is ~550 windows (≈ **25–90 minutes** per batch). **Parallel requests do not increase throughput:** 5 concurrent calls finished in 12.4 s wall time (≈0.4 calls/s), the same rate as sequential calls, with every individual request slower — the service appears to process requests serially.

**Collection policy (be a good citizen of a shared public service).** The manual states the terms: the service is public, for downloading PML information only, and **CENACE may disable it without notice in case of improper use** — a shutdown that would hit every user of the service, not just the offender. It publishes no numeric rate limit; the 7-day / 20-node caps are its load-control mechanism, and bursts of parallel requests degrade it for everyone without helping us. The collector therefore runs **sequentially (at most 2 workers)**, with exponential backoff on errors and timeouts, an identifiable `User-Agent` with contact information, raw responses cached so nothing is fetched twice, and long runs scheduled overnight. Failures and outages are reported through the support contact named in the manual instead of being retried aggressively.

**Staged collection (de-risks the timeline).**

| Stage | Nodes | Purpose | API time |
|---|---|---|---|
| 1 (week 1, Sep 2–3) | Peninsular region: one batch of ~20 nodes covering its 8 load zones and 400/230/115 kV levels (the region has 115 nodes); 2022 → today | End-to-end smoke test: collector → bronze → silver → quality report | 240 calls, 10–40 min |
| 2 (week 1, nights of Sep 3–5) | **~100 SIN nodes**: all 53 × 400 kV nodes (the grid backbone), ~30 × 230 kV and ~17 × 115 kV nodes, with **at least two nodes per regional control center at each of the three voltage levels** and as many load zones as possible; **full MDA history 2016 → today** plus **MTR 2022 → today** | MVP training set; the voltage feature spans 400/230/115 kV instead of extrapolating; MTR ready for Phase 2 | ~2,800 MDA calls + ~1,200 MTR calls, 3–21 h (two or three nights) |
| 3 (optional, only if Stage 2 is done early) | Remaining 230 kV nodes (~180) and more 115 kV nodes, including **whole load zones absent from Stage 2** (held out later for the leave-one-zone-out test); 2022 → today | Better coverage of load zones / voltage levels; full MTR history back to 2017 for the Stage 2 nodes if Phase 2 is reached | 2,400–4,800 calls, 2–25 h |

The minimum viable training set is ~30 nodes (3–5 per region); Stage 2 aims higher because collection is cheap and more nodes make the metadata features learnable. Even so, ~100 nodes cannot cover the 109 load zones: roughly half the zones will have no training node and most of the rest one to three, which shapes how load zone is encoded (§8).

**Fallback scope.** If the pipeline is not reliable for Stage 2 by Sunday, September 6 (end of week 1), the MVP narrows to the **Peninsular region** and SIN-wide becomes a stretch goal. The go/no-go is recorded in task P5.

**Starting point.** I already hold ~4 years (Feb 2022 → Feb 2026) of hourly prices for 9 representative nodes covering all 7 SIN regional control areas plus the two isolated systems (BCA, BCS). These are re-collected directly from CENACE so that the project depends only on public data.

**Node metadata.** CENACE publishes a monthly *Catálogo NodosP* (`.xlsx`) describing every node of the national grid: system, regional control center, load zone, voltage level, state, municipality. It is downloaded once per refresh, versioned, and stored as a dimension table keyed by node key (`clv_nodo`). The model reads its conditioning features from there: the user asks for a node key, the system looks up its metadata and fetches its history. **Supported scope:** any SIN node at a voltage level covered in training (400, 230 and 115 kV) with at least 28 days of published MDA history (§5.3), whether or not its load zone appeared in training — unseen zones fall back to a trained UNKNOWN embedding. Nodes at lower voltage levels (≤ 85 kV) and the BCA/BCS systems are outside the supported scope until data for them is collected; the API says so instead of extrapolating silently.

**Volume.** 100 nodes × ~10.6 years × 8,760 h ≈ 9 M MDA rows, plus ≈ 4 M MTR rows for 2022 → today — small by ML standards (≈100 MB as Parquet), trainable on CPU. Whether pre-2022 years help or hurt the model (different market regimes) is a modeling experiment, not an assumption. The full set lives in Delta tables; the repo snapshot holds the exact evaluation subset.

### 5.3 Preprocessing (known data quirks)

- **Zero / negative prices** are real MDA outcomes (curtailment) and are kept as targets; the model must learn them, unlike a long-term ROI model where they are dropped.
- **Spikes** (3–5× the daily mean) are kept in the data but handled with a robust loss (Huber / MAE) and reported separately in evaluation so a handful of spike days does not dominate the metrics.
- **Daylight saving time.** Mexico abolished DST in October 2022; every year from 2016 to 2022 contains one 23-hour and one 25-hour day that must be aligned before windowing.
- **Gaps** from API outages: short gaps interpolated, long gaps excluded from training windows and flagged.
- **Catalog quirks.** The catalog has a two-row header (the load/generation modelling flags share column names), one region label duplicated by a trailing space, and 9 nodes whose regional control center is *No Aplica*; the dimension-table loader normalizes these, maps *No Aplica* to UNKNOWN, and keys everything by `CLAVE`.
- **Per-node scaling.** Defined exactly below; it is the central piece of generalization to unseen nodes.

#### Per-node scaling — exact definition

Every training sample and every forecast is one `(node, forecast origin D)` pair, where *D* is the last day of the 168-hour input window and *D+1* the target day. Prices are standardized with statistics **frozen at the forecast origin**:

```
z_t = (P_t − μ_D) / σ_D

μ_D, σ_D = mean and standard deviation of the node's hourly MDA price over the trailing
           28 days [D−27, D]: 672 hours ending at 23:00 of day D, the last hour of the input window
σ_D      = max(σ_D, 0.1·|μ_D|, 10 MXN/MWh)          floor against flat weeks and curtailment-heavy nodes
```

The same `μ_D, σ_D` standardize the 168 input hours, the 24 target hours during training, and invert the prediction at serving time (`P̂ = μ_D + σ_D · ẑ`), so each sample is internally consistent and every metric is computed in MXN/MWh after inversion.

| Decision | Choice | Why |
|---|---|---|
| Window | **28 days** (4 complete weeks) | `μ_D` is not biased by which weekdays a calendar month over-represents; 672 points keep `σ_D` stable even when the window contains a spike day; short enough to follow regime changes (gas price jumps, seasonal transitions) within a month |
| Rejected: the 168-hour input window alone | — | One spiky day dominates `σ`; a 7-day `μ` carries a weekday bias; it is the option that most amplifies noise |
| Rejected: 90 days | — | Too slow after a regime shift and demands 90 days of node history before a node can be served |
| Rejected: expanding all-history statistics | — | Nominal prices and market regimes drift over 2016 → 2026, so z-scores would drift with them — the opposite of what one model over ten years and a hundred nodes needs |
| Indexing | Frozen at day *D*, not rolling hour by hour | With `μ_t, σ_t` changing inside the sample, the target would be expressed in units that change from hour to hour |
| Causality | Only prices published before the forecast is issued enter `μ_D, σ_D` | For MDA that is everything through 23:00 of day *D*; backtests use the same cutoff |
| Components | `pml_ene`, `pml_per`, `pml_cng`: own trailing mean removed, divided by the **same** `σ_D` as the total price | Relative magnitudes and the identity `pml = ene + per + cng` survive scaling. Phase 2 scales MTR with the MDA statistics of the same node and day, so the spread stays in the same units |
| Level fed back | Three scalars re-enter as static inputs: `μ_D / 1000` (price level), `σ_D / μ_D` (relative volatility), `(μ_7d − μ_D) / σ_D` (short-term trend) | Standardization removes the level; the model still needs to know the regime, without the targets depending on it |
| Requirement | A node needs **≥ 28 days of published MDA history** | Younger nodes are rejected explicitly by the API (§4); listed in the supported scope (§5.2) |
| Tested, not assumed | Scaling window (7 / 28 / 90 days) and robust statistics (median / IQR) are hyperparameters in task 2.2 | 28 days with mean / std is the default and the first setting frozen if the search is cut (scope guard) |

This is the main mechanism that lets the model forecast nodes it has never seen: it needs their recent history, not their identity. The catalog metadata is a second-order refinement that must earn its place in the ablation (§8).

### 5.4 How it will be stored

| Layer | Store | Contents |
|---|---|---|
| Bronze | Delta table in Databricks Unity Catalog | Raw CENACE JSON records, append-only, with ingestion timestamp |
| Silver | Delta table | Clean hourly series: one row per `(node, market, timestamp)` with `market ∈ {MDA, MTR}`, DST-aligned, quality flags |
| Gold | Delta table | Model-ready windows and daily forecasts (`node, market, target_date, hora, pml_pred, model_version, issued_at`) |
| Reproducibility snapshot | Parquet files versioned in the GitHub repo (`data/`) | The exact training/validation/test data used for the reported results |
| Models & metrics | MLflow (Databricks) | Every experiment run, parameters, metrics, artifacts; registered model versions |

Storing the snapshot in the repo means anyone can reproduce the reported numbers without a Databricks account.

## 6. Ethics and Fairness

**Privacy.** PMLcast uses no personal data of any kind. All inputs are public market prices published by CENACE for transparency purposes; the project will credit CENACE as the source and comply with its terms of use.

**Financial harm and overconfidence.** The main ethical risk is a user treating a forecast as a certainty and losing money. Mitigations: (1) accuracy versus a naive baseline is reported *inside* the product, continuously; (2) the documentation, README and UI state plainly that the model cannot predict congestion or fuel-driven spikes; (3) the output is labelled informational, not financial advice; (4) a stretch goal adds prediction intervals so uncertainty is visible, not implied.

**Fairness across users and regions.** A model trained only on large industrial hubs would serve big players well and small rural nodes poorly. PMLcast trains on nodes from every SIN regional control area and at three voltage levels (400, 230, 115 kV), reports metrics **per node and per region**, and states its supported scope explicitly, publishing where it works and where it does not instead of a single flattering average. The isolated systems (BCA, BCS) behave differently and are explicitly out of the MVP rather than silently mis-served.

**Market integrity.** Forecasting public prices from public data is standard practice and does not manipulate the market; PMLcast produces independent information for one participant, not coordination between participants. It will not be used to recommend withholding capacity.

**Openness.** The code, the data snapshot, the evaluation methodology and the model card are public under an open-source license, so results can be audited and reproduced by anyone — including people without access to paid tooling.

**Terms of the data source.** CENACE's manual allows public use of the service for downloading PML information and reserves the right to disable it for misuse. PMLcast uses it only for that purpose, keeps its request rate modest and sequential, caches every response, and reports outages through CENACE's support channel — protecting a shared public resource that other users, including Sunwise's production systems, depend on.

**Environment.** Better storage dispatch increases the use of solar energy that would otherwise be curtailed. The model itself is small and CPU-trainable; no GPU cluster is needed.

**Dependence on an external source.** If CENACE's service is down, the product must degrade gracefully: show data freshness, fall back to the last available forecast, and never fabricate values.

## 7. Platform

| Layer | Choice |
|---|---|
| End users | Web browser on desktop (responsive layout); no installation |
| API consumers | HTTPS REST endpoint returning JSON |
| Serving & MLOps | **Databricks Free Edition**: serverless compute, Unity Catalog (Delta tables), MLflow tracking & model registry, **Model Serving** endpoint, scheduled Jobs for daily ingestion and batch forecasts, Databricks dashboard (or a Streamlit Databricks App) for the UI |
| Fallback (risk mitigation) | FastAPI + Docker container serving the same model, deployable on any Linux host, if Model Serving proves unavailable or too limited on the free tier |
| Development | macOS, Python 3.11, TensorFlow/Keras, NumPy, pandas, scikit-learn (baselines), pytest, GitHub Actions CI, Black |

## 8. Model and Evaluation Plan

**Framing.** Sequence-to-vector regression: input `(168 hours × F features)` → output `(24,)` — the 24 hourly prices of the next day predicted jointly. Sequence features: `pml`, `pml_ene`, `pml_per`, `pml_cng`, standardized per node with the 28-day statistics frozen at the forecast origin (§5.3); hour-of-day and day-of-week encodings; holiday flag; target-day calendar features (weekday, month, holiday). The target date itself is never a raw input: the serving layer resolves it into the input window and these calendar features (§4). **Single horizon:** the model predicts *D+1* only.

**Static conditioning inputs** — the node's catalog metadata, concatenated with the sequence encoding before the output head:

| Attribute | Encoding | Why |
|---|---|---|
| Regional control center | One-hot (7; all present in training; *No Aplica* → UNKNOWN) | Small, closed vocabulary |
| Voltage level | `log(kV)`, standardized | Ordinal with physical meaning; a number interpolates between levels instead of needing every level in a vocabulary. Training must span 400/230/115 kV (Stage 2); the feature is not trusted outside that range |
| Load zone | Embedding + **UNKNOWN** token, with **20–30 % category dropout** during training (the zone is randomly replaced by UNKNOWN) so the fallback vector is actually learned | 109 zones vs. ~100 training nodes: half the zones have no node, most of the rest one to three. A plain embedding would have no vector for unseen zones and, for seen ones, would act as a node identity in disguise |
| State / municipality | **Not an input** in the MVP: redundant with region + zone and with the same sparsity problem. Kept in the dimension table; municipality lat/lon is a later experiment as a vocabulary-free replacement for load zone | Avoids a third sparse categorical |
| Node identity | **Never** in the main model; ablation only | Keeps the model usable on unseen nodes |
| Price level (from the scaling statistics, §5.3) | Three scalars: `μ_D / 1000`, `σ_D / μ_D`, `(μ_7d − μ_D) / σ_D` | Standardization removes the level; these give it back as inputs without making the targets depend on it |

**Timing.** The forecast for day *D+1* is issued early on day *D*, before offers close and before CENACE publishes the *D+1* MDA results later that day; the published results are the ground truth, so each forecast is scored within hours. **Training window** (2016 → vs 2022 →) is treated as a hyperparameter, since older years come from a different market regime.

**Phase 2 — real-time prices (after the MVP).** A second model with the same architecture predicts the 24 hourly **MTR** prices of *D+1*. Inputs: the published MDA prices for *D+1* (available the afternoon of *D*), the MDA history, the MTR history through *D−7* (the latest published), and the same calendar and static features. Baselines: *MTR = MDA* and *MTR = MDA + mean hourly spread*. Beating *MTR = MDA* is the honest measure of whether the model knows something the day-ahead market did not. Ground truth arrives seven days later, so Phase 2 forecasts are scored weekly, not daily.

**Models, in order of complexity.**

| Model | Purpose |
|---|---|
| Naive-24h (same hour yesterday) | Absolute floor |
| Naive-168h (same hour last week) | Captures the weekly pattern; the baseline that matters |
| Seasonal moving average (ported from my previous work) | Strong seasonal baseline |
| Linear regression on lags + calendar | Classic electricity-price-forecasting benchmark |
| LSTM, history only (`LSTM → Dropout → LSTM → Dense(24)`) | Deep baseline: does sequence learning beat the classical baselines at all? |
| **LSTM + node metadata** (history encoder ⊕ region one-hot ⊕ voltage as a number ⊕ load-zone embedding with UNKNOWN) | **Main model**: one SIN model, queried by node key; supports any SIN node at a voltage level covered in training, load zone seen or unseen |
| LSTM + node-ID embedding | Ablation: does knowing the node's identity add skill beyond history + metadata? Only for training nodes |
| LSTM for MTR given the published MDA (Phase 2) | Real-time view: does the model beat *MTR = MDA*? Only after the MVP |

**Validation.** Strict time ordering: train on the oldest data, validate on the next block, test on the **last 6 months** with walk-forward (re-forecast each day using only past data). **Two held-out regimes, reported separately.** *Leave-one-node-out*: hold out one node whose load zone still has other nodes in training — a new node in a known zone. *Leave-one-zone-out*: hold out every node of a load zone, so the held-out nodes are scored with the UNKNOWN zone — a new node in a zone the model has never seen. Together they measure the generalization claim the product actually makes (§5.2, supported scope), instead of one average that mixes both cases.

**Metrics.** MAE and RMSE in MXN/MWh, sMAPE, **skill score vs. Naive-168h** (`1 − MAE_model / MAE_naive`), error by hour of day, and a **top-4-hours hit rate** (how often the model's four most expensive predicted hours match the actual four most expensive hours — the metric that maps directly to "when should I inject"). All reported overall and excluding spike days.

**Success criteria (targets, to be reported honestly either way).**

- Skill ≥ 15 % vs. Naive-168h on the 6-month test set for training nodes.
- Skill ≥ 5 % vs. Naive-168h on held-out nodes in **both** regimes (zone seen, zone unseen), reported separately.
- Top-4-hours hit rate ≥ 60 %.
- LSTM + metadata ≥ LSTM history-only in both held-out regimes (the metadata must earn its place; otherwise the history-only model ships).
- Phase 2, only if reached: skill ≥ 5 % vs. the *MTR = MDA* baseline on the 6-month test set.
- Endpoint responds in < 2 s; daily job success rate ≥ 95 % over the demo period.

## 9. Scope

### MVP (what will be complete, tested and demoed)

1. CENACE collector + preprocessing pipeline, with tests, producing the silver Delta table and the Parquet snapshot.
2. Dataset builder (windowing, per-node scaling, time-based splits).
3. Four baselines and the LSTM variants (history-only, + metadata, + node-ID ablation), all logged in MLflow, with the full evaluation report (tables + plots, both held-out regimes) for the Stage 2 node set (~100 SIN nodes; minimum ~30).
4. One **SIN-wide model conditioned on node metadata** (region one-hot, voltage as a number, load zone with UNKNOWN), queried by node key, with a documented supported scope, registered in MLflow and deployed as a Databricks Model Serving endpoint. Fallback scope if Stage 2 collection lags: the Peninsular region only.
5. Daily scheduled job: fetch latest prices → forecast tomorrow for all tracked nodes → write to the gold table.
6. Dashboard: tomorrow's 24-hour forecast per node with injection/charging windows, latest forecast vs. published MDA, model card panel.
7. README with architecture, results, limitations and how to reproduce; model card; blog post; presentation; live demo.

### Stretch goals (only after the MVP is done)

- **Phase 2, first in line:** MTR forecast for *D+1* given the published MDA for *D+1*, baseline *MTR = MDA*, weekly scoring, `mtr` block in the API response and a dashboard panel (§8).
- Prediction intervals (quantile loss or residual-based bands).
- BCA and BCS isolated systems (separate models or system feature).
- Exogenous features (demand forecast, gas price).
- Attention/Transformer or gradient-boosting challenger.
- Public FastAPI mirror of the endpoint.

### Out of scope

- Long-term (multi-year) price simulation — solved by my earlier work, different problem.
- Bidding automation or any trading execution.
- Real-time (intra-hour) forecasting.
- Multi-day horizons (*D+2* onward): they would require recursive forecasting.
- Forecasting MTR without the published MDA: the 7-day publication lag leaves no recent MTR history at forecast time.

## 10. Schedule

Build window: **Monday, August 31 – Thursday, September 24, 2026** (3½ weeks). There is no separate pre-window any more: data collection and setup are week 1, and the six-week plan of the earlier drafts is compressed into 25 days, so it carries **no slack** — the scope guard after the risk table fixes in advance what gets cut first if a milestone slips. Every task below becomes an issue on the GitHub Project board `LuisMay12/pmlcast`, grouped by milestone.

### M0 — Data & setup (Mon Aug 31 – Sun Sep 6)

| # | Task | Done when |
|---|---|---|
| P1 | Submit this pitch; secure mentor/alumni reviewer | Pitch approved, reviewer named |
| P2 | Create repo, GitHub Project board, CI skeleton (pytest, Black) | Green CI on empty test suite |
| P3 | Databricks Free Edition workspace; **go/no-go check that Model Serving is available** | Toy model served through an endpoint; else fallback decided |
| P4 | CENACE collector (7-day windows, 20-node batches, retries, idempotent upsert, bronze table) with tests | **Stage 1**: Peninsular nodes collected 2022→present; pipeline smoke-tested end to end |
| P5 | Node metadata dimension table from the CENACE *Catálogo NodosP*; select the **Stage 2** set (~100 nodes: all 400 kV plus 230 kV and 115 kV nodes, ≥ 2 per regional control center per voltage level) and collect it over the nights of Sep 3–5: MDA 2016 →, MTR 2022 → | Dimension + silver tables populated; coverage report by region × voltage × load zone; **go/no-go: SIN-wide vs Peninsular MVP** recorded |
| P6 | Data quality report: gaps, zeros, spikes, DST days, per-node price level | Notebook + summary in `docs/data_quality.md` |

### M1 — Dataset & baselines (Mon Sep 7 – Thu Sep 10)

| # | Task | Done when |
|---|---|---|
| 1.1 | Preprocessing: DST alignment, gap handling, per-node scaling (28-day trailing z-score frozen at the forecast origin, §5.3), holiday features, metadata encoders (region one-hot, `log(kV)`, load-zone vocabulary + UNKNOWN) | Tested module |
| 1.2 | Dataset builder: 168→24 windows aligned to the MDA publication schedule, target-day calendar features, time-based train/val/test split, walk-forward iterator, held-out-node and held-out-zone splits | Tested module; shapes documented |
| 1.3 | Baselines: Naive-24h, Naive-168h, seasonal MA, linear regression | Results logged in MLflow |
| 1.4 | Evaluation module: MAE/RMSE/sMAPE, skill score, per-hour error, top-4-hours hit rate, spike-aware split | Tested; baseline report generated |
| **M1** | **Milestone: baseline report published in README** | |

### M2 — LSTM (Fri Sep 11 – Tue Sep 15)

| # | Task | Done when |
|---|---|---|
| 2.1 | LSTM history-only model + training script (early stopping, checkpoints, seeds), MLflow autologging | First run beats Naive-24h |
| 2.2 | LSTM + node metadata (region one-hot, voltage as a number, load-zone embedding + UNKNOWN with category dropout); hyperparameter search: window length, units, dropout, category-dropout rate, learning rate, loss (MAE vs Huber), scaling window (7 / 28 / 90 d) and robust vs mean/std statistics | Search table in MLflow; best config chosen |
| 2.3 | Leave-one-node-out and leave-one-zone-out experiments + node-ID embedding ablation | Generalization table per held-out node and per held-out zone; ablation table (history-only vs +metadata vs +node-ID) in both regimes |
| 2.4 | Model selection write-up: LSTM vs baselines, where it wins/loses, per region | `docs/model_card.md` draft |
| **M2** | **Milestone: chosen model registered in MLflow registry** | |

### M3 — Deployment (Wed Sep 16 – Sun Sep 20)

| # | Task | Done when |
|---|---|---|
| 3.1 | Deploy Model Serving endpoint; request/response contract (`target_date` optional → tomorrow; past dates → backtest mode with as-of cutoff; more than one day ahead → rejected; supported-scope check on the node); contract tests | `curl` returns 24 prices for a node, with and without `target_date` |
| 3.2 | Daily jobs: 06:00 America/Mexico_City — ingest latest MDA → features → forecast *D+1* for all nodes → gold table; afternoon — score the forecast against the published MDA | Jobs run on schedule 2 consecutive days |
| 3.3 | Dashboard: forecast chart, injection/charging windows, latest forecast vs published MDA, model card panel | Usable end-to-end in browser |
| 3.4 | Monitoring: log daily skill vs naive; data-freshness indicator; graceful degradation | Visible in dashboard |
| **M3** | **Milestone: end-to-end demo works for any tracked node** | |

### M4 — Polish & deliverables (Mon Sep 21 – Thu Sep 24)

| # | Task | Done when |
|---|---|---|
| 4.1 | Final backtest report with plots (forecast vs actual, error by hour, per-region table) | In README and `docs/` |
| 4.2 | README complete: problem, architecture diagram, results, limitations, reproduce-in-10-minutes guide | Reviewed by mentor |
| 4.3 | Blog post draft (methodology, what worked, what didn't) | Published or ready to publish |
| 4.4 | Presentation slides + recorded demo video | Rehearsed |
| 4.5 | Mentor review of finished product; fix findings | Sign-off |
| 4.6 | No buffer remains inside the window; Sep 23 is reserved for fixing review findings and Sep 24 for the submission itself. Stretch goals — Phase 2 (MTR given the published MDA) first — happen only after the September 24 submission | |
| **M4** | **Milestone: Portfolio Project submitted** | |

### Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Databricks Free Edition lacks or limits Model Serving | Medium | Go/no-go in P3; fallback FastAPI + Docker with the same MLflow model |
| CENACE API slowness / instability / blocking | Medium | Measured 2.4–19 s per 20-node week; parallelism does not help, so the Stage 2 set (~2,800 calls) runs over one or two nights, sequentially, with backoff. The manual lets CENACE disable the service for misuse, which is one more reason to stay polite. Bronze layer + retries; staged collection starts day 1; repo snapshot means training never depends on the live API. Fallback scope: Peninsular region |
| Node catalog changes (renamed nodes, new columns) | Low | Catalog is versioned monthly; store the version used; key everything by `clv_nodo` |
| Metadata features do not generalize to unseen load zones | Medium | UNKNOWN token trained with category dropout; both held-out regimes reported; per-node scaling carries most of the signal, so the history-only model ships if metadata does not earn its place |
| LSTM does not beat baselines | Medium | That is a valid result; the comparison and analysis *are* the deliverable. Try Huber loss, longer window, per-region models before concluding |
| Compressed window (25 days instead of six weeks) | High | Stage 2 collection starts the night of Sep 3 so modeling never waits for data; milestones are short and each ends demoable; the scope guard below fixes the cut order in advance so slipping never turns into improvising |
| Scope creep toward stretch goals | High | Stretch goals gated behind the September 24 submission; there is no buffer week to spend on them |
| Solo project, no redundancy | — | Small, tested increments; every milestone ends demoable |

### Scope guard (cut order if a milestone slips)

1. Hyperparameter search narrowed to window length and loss; scaling frozen at 28 days with mean / std, everything else at sensible defaults.
2. Leave-one-zone-out limited to three held-out zones; node-ID ablation dropped.
3. Blog post reduced to an outline inside the window, finished after submission.
4. Dashboard limited to Databricks' built-in dashboard; no separate Streamlit app.
5. Peninsular-only model (the M0 go/no-go on September 6 already covers this).

Never cut: tests, the evaluation report with both held-out regimes, the model card, the live endpoint, the README.

## 11. Mentor / Reviewer

**TBD.** 

## 12. Definition of Done

PMLcast is done when a reviewer can: open the dashboard, pick any tracked SIN node, see tomorrow's 24 hourly prices with a recommended injection window and the latest scored forecast's accuracy; call the endpoint and get the same array as JSON; read in the README exactly how much better (or not) the LSTM is than a naive forecast, per region, and reproduce that number from the repo snapshot; and read the blog post and slides that explain every technical choice above.
