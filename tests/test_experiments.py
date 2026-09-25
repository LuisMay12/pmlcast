#!/usr/bin/env python3
"""Test the grid expansion, the ablation table and the metadata verdict."""

import pandas as pd

from pmlcast import experiments


def _holdout_rows(meta_mae, history_mae, regime="node"):
    return [
        {
            "regime": regime,
            "held_out": "A",
            "model": "metadata",
            "n_test": 10,
            "mae": meta_mae,
            "skill": 0.2,
            "top4_overlap": 0.6,
            "mae_nospike": meta_mae,
        },
        {
            "regime": regime,
            "held_out": "A",
            "model": "history",
            "n_test": 10,
            "mae": history_mae,
            "skill": 0.1,
            "top4_overlap": 0.5,
            "mae_nospike": history_mae,
        },
    ]


def test_grid_combinations_covers_every_point():
    grid = {"units": [32, 64], "loss": ["huber", "mae"], "dropout": [0.1]}
    combos = experiments.grid_combinations(grid)

    assert len(combos) == 4
    assert all(set(c) == set(grid) for c in combos)
    assert {(c["units"], c["loss"]) for c in combos} == {
        (32, "huber"),
        (32, "mae"),
        (64, "huber"),
        (64, "mae"),
    }
    assert experiments.grid_combinations({"units": [32]}) == [{"units": 32}]


def test_best_params_picks_lowest_val_mae():
    table = pd.DataFrame(
        {
            "units": [32, 64],
            "dropout": [0.1, 0.2],
            "loss": ["mae", "huber"],
            "learning_rate": [0.001, 0.001],
            "zone_dropout": [0.25, 0.25],
            "val_mae": [900.0, 800.0],
        }
    ).sort_values("val_mae")

    params = experiments.best_params(table)

    assert params["units"] == 64
    assert params["loss"] == "huber"
    assert "val_mae" not in params


def test_best_params_returns_python_types():
    # pandas hands back numpy scalars, and Keras rejects a numpy int
    # where it wants a plain int: units // 2 stops being an int.
    table = pd.DataFrame(
        {
            "units": [64],
            "dropout": [0.2],
            "loss": ["mae"],
            "learning_rate": [0.001],
            "zone_dropout": [0.25],
            "val_mae": [800.0],
        }
    )

    params = experiments.best_params(table)

    assert type(params["units"]) is int
    assert type(params["dropout"]) is float
    assert type(params["loss"]) is str
    assert params["units"] // 2 == 32


def test_ablation_table_averages_folds():
    rows = _holdout_rows(800.0, 900.0)
    rows += [dict(row, held_out="B", mae=row["mae"] + 100) for row in rows]
    table = experiments.ablation_table(pd.DataFrame(rows))

    assert set(table["model"]) == {"metadata", "history"}
    metadata = table[table["model"] == "metadata"].iloc[0]
    assert metadata["folds"] == 2
    assert metadata["n_test"] == 20
    assert metadata["mae"] == 850.0
    # the lower MAE is listed first inside a regime
    assert table.iloc[0]["model"] == "metadata"


def test_verdict_ships_metadata_only_when_it_wins_everywhere():
    node = _holdout_rows(800.0, 900.0, "node")
    zone_win = _holdout_rows(700.0, 750.0, "zone")
    ablation = experiments.ablation_table(pd.DataFrame(node + zone_win))
    verdict = experiments.metadata_verdict(ablation)

    assert "metadata wins" in verdict
    assert "it is the one that ships" in verdict

    zone_loss = _holdout_rows(800.0, 700.0, "zone")
    ablation = experiments.ablation_table(pd.DataFrame(node + zone_loss))
    verdict = experiments.metadata_verdict(ablation)

    assert "metadata loses" in verdict
    assert "history-only model ships" in verdict


def test_write_report_has_both_sections(tmp_path):
    search = pd.DataFrame(
        {
            "units": [64],
            "dropout": [0.2],
            "loss": ["huber"],
            "learning_rate": [0.001],
            "zone_dropout": [0.25],
            "val_mae": [800.0],
            "test_mae": [850.0],
            "test_skill": [0.2],
            "test_top4": [0.6],
            "epochs_run": [12],
        }
    )
    holdout = pd.DataFrame(
        _holdout_rows(800.0, 900.0, "node")
        + _holdout_rows(700.0, 750.0, "zone")
    )
    ablation = experiments.ablation_table(holdout)
    out_md = tmp_path / "model_selection.md"

    text = experiments.write_report(
        search, holdout, ablation, str(out_md), "stage1"
    )

    assert out_md.exists()
    assert "## Hyperparameter search" in text
    assert "Chosen configuration: units=64" in text
    assert "## Generalization to unseen nodes and zones" in text
    assert "### Verdict" in text
