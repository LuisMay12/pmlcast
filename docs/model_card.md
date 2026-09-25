# Model card - PMLcast

The numbers below come from the final model: 99 SIN nodes, trained from
January 2019, scored on the six months ending September 2026.

## What the model does

Given the last 336 hourly day-ahead prices (MDA) at a node of the Mexican
grid, PMLcast predicts the 24 hourly prices of the next day as a single
vector.

The forecast for day D+1 is issued early on day D, before offers close and
before CENACE publishes the D+1 results later that day. Those published
results are the ground truth, so every forecast is scored within hours.

Intended use: deciding when to charge and when to inject energy from a
battery during the next day. It is informational, not financial advice.

## Inputs and outputs

| Input | Shape | Content |
|---|---|---|
| `seq` | (336, 9) | `pml`, `pml_ene`, `pml_per`, `pml_cng` standardized per node, plus hour-of-day and day-of-week sine/cosine and a holiday flag |
| `cal` | (10,) | Target-day weekday one-hot, month sine/cosine, holiday flag |
| `level` | (3,) | `mu_D / 1000`, `sigma_D / 1000`, `(mu_7d - mu_D) / sigma_D` |
| `meta` | (9,) | Regional control center one-hot (7 regions + UNKNOWN) and standardized `log(kV)` |
| `zone` | (1,) | Load-zone index, 0 = UNKNOWN |

Output: 24 values in z-units, inverted to MXN/MWh with the same `mu_D` and
`sigma_D` that scaled the inputs.

The target date is never a raw input. The serving layer turns it into the
input window and the calendar features, so a past date can be replayed in
backtest mode with the same cutoff the model had on the day.

## Scaling

```
z = (P - mu_D) / sigma_D
mu_D, sigma_D = mean and std of the node's hourly MDA price over the 28 days
                ending at 23:00 of day D (672 hours)
sigma_D       = max(sigma_D, 0.05 * |mu_D|, 10 MXN/MWh)
```

The statistics are frozen at the forecast origin, so one sample is
internally consistent and the same numbers invert the prediction. This is
the main reason the model can serve a node it never saw during training: it
needs that node's recent history, not its identity.

A node needs at least 28 days of published history to be served.

## Architecture

```
seq -> LSTM(64, return_sequences) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2)
concat[encoder, cal, level] -> Dense(64, relu) -> Dense(24)
```

Loss: **MAE** on z-units. Huber was the pitch's choice for robustness
against spikes, but it lost to MAE in every one of the eight search
configurations: the error is measured in MAE, and on series this
heavy-tailed Huber smooths exactly the days that weigh most. Adam, early
stopping on validation loss with weight restoration.

The shipped model takes no node metadata. The ablation could not separate
it from the history-only model: it won one hold-out regime and lost the
other, and the sign flipped between datasets, so the differences are noise
rather than a small effect. The pitch's rule (metadata must win in both
regimes) then selects the simpler model.

The node identity is likewise not an input. Its embedding exists only as an
ablation, and it never added skill beyond the node's own recent history.

## Training data

| | |
|---|---|
| Source | CENACE public SW-PML web service, day-ahead market (MDA) |
| Nodes | 99 SIN nodes, 69 load zones, 115/230/400 kV |
| Range | 2019-01-01 to 2026-09-19, hourly |
| Samples | 261,842 (train 234,716 / val 8,910 / test 18,216) |
| Splits | By target date: test = the last six months, val = the 90 days before, train = everything earlier |
| Node catalog | *Catálogo NodosP* v2026-02-18 |

How far back to train was measured, not assumed: 7.7 years of history beat
4.7 and 2.7 consistently, so the model trains from 2019. The input window
was measured the same way: 14 days beat 7 and 3.

Zero and negative prices are kept as targets; they are real curtailment
outcomes. Spikes are kept too, and reported separately instead of being
clipped. The 23-hour and 25-hour days that daylight saving produced until
October 2022 are aligned onto 24 clock slots and flagged.

## Evaluation

Metrics in MXN/MWh on the test split, all reported for every day, for
no-spike days and for spike days separately:

- MAE, RMSE, sMAPE
- Skill vs Naive-168h: `1 - MAE_model / MAE_naive`
- Top-4 overlap: `|PredTop4 ∩ ActualTop4| / 4`, a set overlap, not an exact
  match. Picking four hours at random scores 0.167
- Captured value: actual revenue of the four predicted hours over the
  revenue of the four best hours

Generalization is measured in two regimes, reported apart: leave-one-node-out
(a new node in a load zone that keeps other nodes) and leave-one-zone-out
(a new node in a zone the model never saw, scored with the UNKNOWN
embedding).

### Results

Test split: 18,216 forecasts over the last six months, 99 nodes.

| model | MAE | RMSE | skill | top-4 overlap | captured value |
|---|---|---|---|---|---|
| Naive-24h | 280 | 756 | 0.141 | 0.512 | 0.865 |
| Naive-168h | 327 | 876 | 0.000 | 0.517 | 0.871 |
| Seasonal MA | 653 | 1045 | -1.001 | 0.600 | 0.914 |
| Ridge on lags | 278 | 659 | 0.148 | 0.608 | 0.917 |
| **LSTM history-only** | **243** | **659** | **0.256** | 0.574 | 0.901 |

MAE on days without a price spike drops to 197.

Hold-out and metadata-ablation numbers are in
[model_selection.md](model_selection.md).

Three observations worth keeping. Naive-24h is a far stronger reference
than Naive-168h, which the pitch had assumed to be the baseline that
matters. The seasonal moving average, ported from my earlier long-horizon
work, loses badly at one day ahead: it was tuned for an annual arc, not for
tomorrow. And the ridge regression on lags gets within 35 MXN/MWh of the
LSTM, which is a fair reminder of how much of this problem is linear.

On the two product metrics the ridge model edges ahead. The LSTM wins on
absolute error; picking the right block of expensive hours is nearly a tie.
Both are reported rather than cherry-picked.

## Limitations

- It forecasts the shape and level of a normal day. It cannot predict
  congestion events or fuel-supply spikes, which is why spike days are
  reported separately rather than hidden in an average.
- Single horizon. It predicts D+1 only; D+2 would need recursive
  forecasting and is out of scope.
- Supported scope is SIN nodes at a voltage level present in training, with
  at least 28 days of published history. The isolated systems BCA and BCS
  behave differently and are out of scope. Nodes outside the scope get an
  explicit error instead of an extrapolated curve.
- Prices come from a single public service. If it is down the product must
  show its data freshness and fall back to the last forecast, never
  fabricate values.
- Real-time (MTR) prices are a separate, later model: they are published
  seven days after the operation day, so at forecast time the most recent
  MTR is a week old.

## Ethics

No personal data of any kind. All inputs are public market prices published
by CENACE for transparency, credited as the source and used within the terms
of its technical manual: public use, modest and sequential request rate,
every response cached.

The main risk is a user treating a forecast as a certainty and losing money.
Mitigations: the accuracy against a naive baseline is reported inside the
product and updated daily, the documentation states plainly what the model
cannot predict, and the output is labelled informational.

Fairness across regions is measured, not assumed: metrics are reported per
node and per region so it is visible where the model works and where it does
not, instead of a single flattering average.

## Reproducing

```bash
make lint test
make dataset NAME=final START=2019-01-01
make baselines NAME=final
make train NAME=final
make serve          # dashboard and API on http://localhost:8000
```

Runs are tracked in MLflow (`sqlite:///mlflow.db`, artifacts under
`mlruns/`). Seeds are fixed for numpy, TensorFlow and the `tf.data` shuffle.
