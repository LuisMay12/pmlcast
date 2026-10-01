# PMLcast

**Day-ahead hourly electricity price forecasting for Mexican grid nodes.**

Live demo: **[pmlcast.onrender.com](https://pmlcast.onrender.com)** · A free
instance sleeps after 15 minutes, so the first request can take a minute.

Given the last 14 days of hourly prices at a node of the Mexican national
grid, PMLcast predicts the **24 hourly prices of the next day** so a solar
plant with a battery can decide when to charge and when to sell.

Holberton School — Machine Learning Specialization, Portfolio Project.
Author: Luis May.

---

## The problem

CENACE sets a different price at every node of the grid, for every hour. In
the day-ahead market (MDA) offers close *before* anyone knows the clearing
price. Between the cheapest and the most expensive hour of the same day at
the same node the difference is routinely 2–3×, so knowing *which hours*
will be expensive is worth money to anyone who can choose when to inject.

The forecast is issued the morning of day D, for day D+1, before offers
close. Every input is public by then; nothing from the target day leaks in.

**Why the day-ahead market and not real time.** CENACE runs two markets:
day-ahead (MDA), whose price is binding for every MWh scheduled there, and
real time (MTR), which settles deviations from that schedule. MDA is the
one an operator can still act on when the forecast is issued — offers close
the morning before the operation day. MTR is also the harder target for a
structural reason: CENACE publishes it **seven days after** the operation
day, so at forecast time the most recent MTR price is already a week old,
which rules out the mirror-image design of feeding it its own recent
history. Forecasting MTR is [planned as a second
phase](docs/pitch.md) that conditions on the published MDA instead.

## Results

Test split: 18,216 forecasts over the six months ending September 2026,
99 SIN nodes.

| Model | MAE | RMSE | Skill | Top-4 overlap |
|---|---|---|---|---|
| Naive-24h | 280 | 756 | 0.141 | 0.512 |
| Naive-168h | 327 | 876 | 0.000 | 0.517 |
| Seasonal MA | 653 | 1045 | -1.001 | 0.600 |
| Ridge on lags | 278 | 659 | 0.148 | **0.608** |
| **LSTM history-only** | **243** | **659** | **0.256** | 0.574 |

MAE on days without a price spike drops to 197 MXN/MWh.

Two results worth stating plainly. The LSTM wins on absolute error, but a
ridge regression on lags lands within 35 MXN/MWh of it, and **beats it on
top-4 overlap** — the metric the pitch set as a success criterion at ≥ 60 %,
which the LSTM did not reach. Much of day-ahead pricing is linear.

Full numbers, per-region tables and the hold-out experiments are in
[docs/model_card.md](docs/model_card.md) and
[docs/model_selection.md](docs/model_selection.md). The original plan,
schedule and success criteria are in [docs/pitch.md](docs/pitch.md).

## How it works

```
seq   (336, 9)  ─┐
cal   (10,)     ─┤
level (3,)      ─┼─> LSTM(64) → Dropout → LSTM(32) → Dropout
meta  (9,)      ─┤        → Dense(64, relu) → Dense(24)
zone  (1,)      ─┘
```

Prices are standardized **per node** against the mean and standard
deviation of that node's trailing 28 days, frozen at the forecast origin.
That is what lets the model serve a node it never saw in training: it needs
the node's recent history, not its identity.

Trained on 261,842 samples over 7.7 years of history. Loss is MAE on
z-units; Huber was the pitch's choice and lost in all eight search runs.

## Repository layout

The project has **two dependency sets**, and that split is the reason the
public demo fits in a free tier at all.

### What trains the model

Everything here runs on your machine, never in the deployed container.

| Path | Role |
|---|---|
| `pmlcast/cenace.py` | Plan, fetch, parse and store CENACE requests |
| `pmlcast/catalog.py` | Load the node catalogue, select the node set |
| `pmlcast/preprocess.py` | Clean hourly series, per-node scaling, features |
| `pmlcast/dataset.py` | Build 336→24 windows, splits, hold-out generators |
| `pmlcast/models.py`, `train.py` | Define and train the LSTM |
| `pmlcast/baselines.py` | The four reference forecasters |
| `pmlcast/evaluate.py` | Error, skill and product metrics |
| `pmlcast/experiments.py` | Hyperparameter search, generalization runs |
| `pmlcast/quality.py` | Data quality report of the silver layer |
| `requirements.in` / `.txt` | TensorFlow, MLflow, scikit-learn, matplotlib |

### What gets deployed

| Path | Role |
|---|---|
| `pmlcast/serving.py` | Node + date → 24 prices, through ONNX Runtime |
| `pmlcast/api.py` | FastAPI app: `/forecast`, `/health`, the dashboard |
| `pmlcast/dashboard.py` | The single-page UI, rendered server-side |
| `pmlcast/storage.py`, `config.py` | Shared paths and table access |
| `deploy/Dockerfile` | The image Render builds |
| `data/demo/` | One year of prices, the ONNX model, the catalogue (29 MB) |
| `requirements-serve.in` / `.txt` | ONNX Runtime, FastAPI, pandas — no TensorFlow |

**ONNX Runtime instead of TensorFlow** is what makes this deployable: the
same weights and the same predictions in ~40 MB of dependency instead of
~600 MB. The final image is 795 MB and runs in 228 MB of RAM.

`pmlcast/daily.py` sits between the two: it forecasts every node each
morning and scores yesterday's forecasts once CENACE publishes the real
prices. It needs the evaluation stack, so it runs outside the container.

### Everything else

| Path | Role |
|---|---|
| `tests/` | 53 tests, run on every push |
| `docs/` | Model card, pitch, coverage, baselines, model selection |
| `scripts/build_demo_data.py` | Pack `data/demo/` from the full data tree |
| `scripts/export_for_databricks.py` | Pack the model and metrics for the notebook |
| `notebooks/pmlcast_databricks.py` | The Databricks showcase notebook |
| `.github/workflows/ci.yml` | Black, flake8 and pytest on every push |

## The demo keeps itself current

The packaged data ends the day the slice was built, so a frozen demo would
be stale within a week. Instead, when a requested node's stored prices stop
short, the service **asks CENACE for the missing days at request time**,
then forecasts. A cold node costs a few seconds; the next request for it is
instant. Enabled with `PMLCAST_BACKFILL=1`, which `deploy/Dockerfile` sets.

## Running it

### The service, locally

```bash
make venv install
PMLCAST_DATA_DIR=$PWD/data/demo PMLCAST_BACKFILL=1 make serve
```

Then open <http://localhost:8000>.

### The deployed image

```bash
docker build -f deploy/Dockerfile -t pmlcast-demo .
docker run --rm -p 7860:7860 pmlcast-demo
```

### The API

```bash
curl -X POST https://pmlcast.onrender.com/forecast \
  -H "Content-Type: application/json" \
  -d '{"node": "01PIT-400"}'
```

`target_date` is optional and defaults to tomorrow. A past date replays
what the service would have seen that morning. More than one day ahead is
rejected: this is a single-horizon model. `evaluated_node` in the response
says whether the node is one of the 99 the reported error describes — the
service answers for any of the ~2,444 SIN nodes, but the metrics were
measured on the evaluation set.

### Reproducing the model

```bash
make lint test
make dataset NAME=final START=2019-01-01
make baselines NAME=final
make train NAME=final
```

Runs are tracked in MLflow (`sqlite:///mlflow.db`). Seeds are fixed for
numpy, TensorFlow and the `tf.data` shuffle.

## Scope and limitations

- It forecasts the shape and level of a normal day. It **cannot** predict
  congestion events or fuel-supply spikes, which is why spike days are
  reported separately rather than hidden in an average.
- Single horizon: D+1 only.
- SIN nodes at a voltage level present in training, with at least 28 days
  of published history. The isolated BCA and BCS systems are out of scope
  and rejected with an explicit error rather than an extrapolated curve.
- Real-time (MTR) prices are a separate, later model: CENACE publishes them
  seven days after the operation day.

## Ethics

No personal data of any kind. Every input is a public market price
published by CENACE for transparency, credited as the source and used
within the terms of its technical manual: public use, a modest and
sequential request rate, every response cached.

The main risk is a user treating a forecast as a certainty and losing
money. The accuracy against a naive baseline is reported inside the
product, the documentation states plainly what the model cannot predict,
and the output is labelled informational. Fairness across regions is
measured, not assumed: metrics are reported per node and per region, so it
is visible where the model works and where it does not.

**Informational only. Not financial advice.**

## Data source

Prices come from the [CENACE](https://www.cenace.gob.mx/) public SW-PML web
service. The node catalogue is the published Catálogo NodosP, snapshot
`v20260218`.
