# PMLcast — public demo

Day-ahead hourly electricity price forecasting for Mexican grid nodes.

Pick a node and a date, and the model predicts the 24 hourly prices of the
next day, highlighting when to inject energy and when to charge a battery.

Trained on 112 nodes of the Mexican national grid with prices published by
[CENACE](https://www.cenace.gob.mx/), from 2019 to 2026. On the six-month
test split it reaches a mean absolute error of 243 MXN/MWh, a 26 % skill
score over forecasting "the same hour last week".

The image carries one year of prices so it stays small, and asks CENACE for
the days it is missing when a node is requested, so the demo forecasts
tomorrow rather than a frozen date. The full project, with ten years of
history, the collector, the experiments and the reports, is at
[github.com/LuisMay12/pmlcast](https://github.com/LuisMay12/pmlcast).

Informational only. Not financial advice: the model forecasts the shape of
a normal day and cannot predict congestion or fuel-supply spikes.

Holberton School — Machine Learning Portfolio Project, by Luis May.
