# Model selection - dataset stage2full

## How far back to train

One dataset per training start date, same validation and test periods, ranked by validation MAE. The pitch treats this as an experiment because the pre-2022 years come from a different market regime.

| start      |   train_years |   n_train |   val_mae |   test_mae |   test_skill |   test_top4 |
|:-----------|--------------:|----------:|----------:|-----------:|-------------:|------------:|
| 2019-01-01 |         7.700 |    234716 |   172.412 |    240.651 |        0.263 |       0.598 |
| 2022-01-01 |         4.700 |    137595 |   176.069 |    245.633 |        0.248 |       0.606 |
| 2024-01-01 |         2.700 |     67808 |   181.309 |    248.215 |        0.240 |       0.601 |

Chosen start: 2019-01-01 (7.7 years of training data).

## Generalization to unseen nodes and zones (task 2.3)

Leave-one-node-out holds out a node whose load zone keeps other nodes. Leave-one-zone-out holds out every node of a zone, so the held-out samples are scored with the UNKNOWN zone embedding. Skill is relative to Naive-168h on the same samples.

### Average per regime and model

| regime   | model    |   folds |   n_test |     mae |   skill |   top4_overlap |   mae_nospike |
|:---------|:---------|--------:|---------:|--------:|--------:|---------------:|--------------:|
| node     | node_id  |       2 |      368 | 177.288 |   0.220 |          0.610 |       141.086 |
| node     | history  |       2 |      368 | 178.122 |   0.217 |          0.584 |       140.062 |
| node     | metadata |       2 |      368 | 180.423 |   0.206 |          0.605 |       143.736 |
| node     | naive24  |       2 |      368 | 213.356 |   0.062 |          0.516 |       190.623 |
| node     | naive168 |       2 |      368 | 227.415 |   0.000 |          0.527 |       194.399 |
| zone     | node_id  |       2 |     2576 | 165.103 |   0.238 |          0.580 |       128.982 |
| zone     | metadata |       2 |     2576 | 165.525 |   0.236 |          0.587 |       131.602 |
| zone     | history  |       2 |     2576 | 166.077 |   0.234 |          0.586 |       133.076 |
| zone     | naive24  |       2 |     2576 | 200.267 |   0.076 |          0.523 |       179.175 |
| zone     | naive168 |       2 |     2576 | 216.700 |   0.000 |          0.530 |       184.218 |

### Per fold

| regime   | held_out   | model    |   n_test |     mae |   skill |   top4_overlap |   mae_nospike |
|:---------|:-----------|:---------|---------:|--------:|--------:|---------------:|--------------:|
| node     | 01ACO-230  | history  |      184 | 180.028 |   0.219 |          0.590 |       143.483 |
| node     | 01ACO-230  | metadata |      184 | 179.264 |   0.222 |          0.592 |       145.276 |
| node     | 01ACO-230  | node_id  |      184 | 180.373 |   0.218 |          0.610 |       145.780 |
| node     | 01ACO-230  | naive168 |      184 | 230.541 |   0.000 |          0.530 |       197.537 |
| node     | 01ACO-230  | naive24  |      184 | 215.917 |   0.063 |          0.514 |       193.322 |
| node     | 01ACS-230  | history  |      184 | 176.217 |   0.214 |          0.579 |       136.640 |
| node     | 01ACS-230  | metadata |      184 | 181.583 |   0.190 |          0.618 |       142.196 |
| node     | 01ACS-230  | node_id  |      184 | 174.202 |   0.223 |          0.610 |       136.392 |
| node     | 01ACS-230  | naive168 |      184 | 224.288 |   0.000 |          0.524 |       191.261 |
| node     | 01ACS-230  | naive24  |      184 | 210.796 |   0.060 |          0.519 |       187.924 |
| zone     | MONTERREY  | history  |     1288 | 170.542 |   0.229 |          0.572 |       133.821 |
| zone     | MONTERREY  | metadata |     1288 | 170.145 |   0.231 |          0.602 |       134.529 |
| zone     | MONTERREY  | node_id  |     1288 | 165.944 |   0.250 |          0.594 |       128.727 |
| zone     | MONTERREY  | naive168 |     1288 | 221.282 |   0.000 |          0.526 |       186.452 |
| zone     | MONTERREY  | naive24  |     1288 | 206.148 |   0.068 |          0.519 |       184.043 |
| zone     | TAMPICO    | history  |     1288 | 161.612 |   0.238 |          0.599 |       132.332 |
| zone     | TAMPICO    | metadata |     1288 | 160.906 |   0.241 |          0.571 |       128.675 |
| zone     | TAMPICO    | node_id  |     1288 | 164.263 |   0.226 |          0.566 |       129.236 |
| zone     | TAMPICO    | naive168 |     1288 | 212.117 |   0.000 |          0.534 |       181.985 |
| zone     | TAMPICO    | naive24  |     1288 | 194.386 |   0.084 |          0.528 |       174.307 |

### Verdict

- node hold-out: metadata MAE 180.42 vs history-only 178.12 -> metadata loses
- zone hold-out: metadata MAE 165.53 vs history-only 166.08 -> metadata wins

The metadata features did not earn their place in every regime, so the history-only model ships (pitch section 8).
