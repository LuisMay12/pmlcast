#!/usr/bin/env python3
"""Test bronze envelopes and silver upserts."""

import datetime

import pandas as pd

from pmlcast import cenace
from pmlcast import preprocess
from pmlcast import storage
from tests.conftest import make_records
from tests.conftest import parsed_records

D = datetime.date


def _request():
    return cenace.Request(
        "SIN", "MDA", ("01TUL-400", "02LAV-400"), D(2024, 3, 1), D(2024, 3, 7)
    )


def test_upsert_silver_is_idempotent_and_merges(tmp_path):
    silver = str(tmp_path / "silver")
    node = "07PNC-115"
    week1 = preprocess.records_to_hourly(
        parsed_records(node, make_records(D(2024, 3, 1), 7))
    )
    week3 = preprocess.records_to_hourly(
        parsed_records(node, make_records(D(2024, 3, 15), 7, seed=3))
    )

    first = storage.upsert_silver(silver, "MDA", node, week1)
    again = storage.upsert_silver(silver, "MDA", node, week1)
    pd.testing.assert_frame_equal(first, again)
    assert len(first) == 7 * 24

    merged = storage.upsert_silver(silver, "MDA", node, week3)
    assert len(merged) == 21 * 24  # continuous grid across the hole
    quality = merged.set_index("ts_local")["quality"]
    assert (quality["2024-03-08":"2024-03-14"] == "missing").all()
    assert (quality["2024-03-01":"2024-03-07"] == "ok").all()

    coverage = storage.silver_coverage(silver, "MDA")
    assert len(coverage) == 14
    assert (node, D(2024, 3, 10)) not in coverage

    table = storage.coverage_table(silver, "MDA")
    assert table.loc[0, "full_days"] == 14 and table.loc[0, "days"] == 21
