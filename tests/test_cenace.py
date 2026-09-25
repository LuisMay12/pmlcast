#!/usr/bin/env python3
"""Test request planning, fetching with retries, parsing and collection."""

import datetime

import pytest

from pmlcast import cenace
from pmlcast import storage
from tests.conftest import FakeSession
from tests.conftest import cenace_payload
from tests.conftest import make_records

D = datetime.date


def test_build_url_format():
    request = cenace.Request(
        "SIN",
        "MDA",
        ("01TUL-400", "02LAV-400"),
        D(2026, 8, 25),
        D(2026, 8, 31),
    )
    assert cenace.build_url(request) == (
        "https://ws01.cenace.gob.mx:8082/SWPML/SIM/SIN/MDA/"
        "01TUL-400,02LAV-400/2026/08/25/2026/08/31/JSON"
    )


@pytest.mark.parametrize(
    "request_",
    [
        cenace.Request(
            "SIN",
            "MDA",
            tuple("N{}".format(i) for i in range(21)),
            D(2026, 8, 25),
            D(2026, 8, 31),
        ),
        cenace.Request("SIN", "MDA", ("A",), D(2026, 8, 25), D(2026, 9, 1)),
        cenace.Request("SIN", "MDA", ("A",), D(2026, 8, 25), D(2026, 8, 24)),
        cenace.Request("XXX", "MDA", ("A",), D(2026, 8, 25), D(2026, 8, 25)),
        cenace.Request("SIN", "PML", ("A",), D(2026, 8, 25), D(2026, 8, 25)),
        cenace.Request("SIN", "MDA", (), D(2026, 8, 25), D(2026, 8, 25)),
    ],
)
def test_build_url_guards(request_):
    with pytest.raises(ValueError):
        cenace.build_url(request_)


def test_epoch_windows_anchor_and_clip():
    windows = cenace.epoch_windows("MDA", D(2016, 2, 10), D(2016, 2, 28))
    assert windows[0] == (D(2016, 2, 5), D(2016, 2, 11))
    assert windows[-1] == (D(2016, 2, 26), D(2016, 2, 28))
    assert all((e - s).days <= 6 for s, e in windows)
    assert all(e <= D(2016, 2, 28) for _, e in windows)
    # a fresh run with a later start reuses the same boundaries
    again = cenace.epoch_windows("MDA", D(2016, 2, 13), D(2016, 2, 28))
    assert again[0] == (D(2016, 2, 12), D(2016, 2, 18))


def test_plan_requests_chunks_and_have():
    nodes = ["N{:02d}".format(i) for i in range(45)]
    plan = cenace.plan_requests(
        nodes, "SIN", "MDA", D(2024, 1, 5), D(2024, 1, 11)
    )
    assert [len(r.nodes) for r in plan] == [20, 20, 5]
    assert plan[0].nodes[0] == "N00" and plan[2].nodes[-1] == "N44"

    days = cenace.window_dates(plan[0].start, plan[0].end)
    have = {(node, day) for node in nodes[:30] for day in days}
    plan = cenace.plan_requests(
        nodes, "SIN", "MDA", D(2024, 1, 5), D(2024, 1, 11), have=have
    )
    assert [len(r.nodes) for r in plan] == [15]
    # a node with one day missing is still requested
    have.discard(("N00", days[3]))
    plan = cenace.plan_requests(
        nodes, "SIN", "MDA", D(2024, 1, 5), D(2024, 1, 11), have=have
    )
    assert [len(r.nodes) for r in plan] == [16]

    have = {(node, day) for node in nodes for day in days}
    assert (
        cenace.plan_requests(
            nodes, "SIN", "MDA", D(2024, 1, 5), D(2024, 1, 11), have=have
        )
        == []
    )


def _request(days=7):
    return cenace.Request(
        "SIN",
        "MDA",
        ("01TUL-400",),
        D(2026, 8, 25),
        D(2026, 8, 25) + datetime.timedelta(days=days - 1),
    )


def test_parse_real_shape(fixture_text):
    records, message = cenace.parse_response(fixture_text("cenace_ok.json"))
    assert message is None
    assert len(records) == 4
    first = records[0]
    assert first["node"] == "01TUL-400"
    assert first["fecha"] == "2026-08-30"
    assert first["hora"] == 1 and isinstance(first["hora"], int)
    assert first["pml"] == 787.13 and first["pml_cng"] == -0.12
    assert records[3]["pml"] is None  # empty string -> None


def test_collect_end_to_end(tmp_path):
    bronze = str(tmp_path / "bronze")
    silver = str(tmp_path / "silver")
    nodes = ["07PNC-115", "07TIZ-400"]
    start, end = D(2024, 3, 2), D(2024, 3, 15)
    windows = cenace.epoch_windows("MDA", start, end)
    assert len(windows) == 3  # anchored windows: partial, full, partial

    script = []
    for win_start, win_end in windows:
        days = (win_end - win_start).days + 1
        values = {
            n: make_records(win_start, days, seed=i)
            for i, n in enumerate(nodes)
        }
        script.append(("ok", cenace_payload(values)))
    session = FakeSession(script)
    sleeps = []
    summary = cenace.collect(
        nodes,
        "SIN",
        "MDA",
        start,
        end,
        bronze,
        silver,
        session=session,
        sleep=sleeps.append,
    )
    assert summary["n_planned"] == 3
    assert summary["n_fetched"] == 3 and summary["n_ok"] == 3
    assert len(session.calls) == 3
    assert len(list(storage.iter_bronze(bronze))) == 3

    frame = storage.read_silver(silver, "MDA")
    assert set(frame["node"]) == set(nodes)
    per_node = frame.groupby("node").size()
    assert (per_node == per_node.iloc[0]).all()
    assert (frame["quality"] == "ok").all()
    first_day = frame["fecha"].min()
    assert first_day <= start  # anchored window starts before `start`

    # second run: everything is covered, no HTTP at all
    second = cenace.collect(
        nodes,
        "SIN",
        "MDA",
        start,
        end,
        bronze,
        silver,
        session=FakeSession([]),
        sleep=sleeps.append,
    )
    assert second["n_planned"] == 0

    # an empty window is remembered and only re-requested on demand
    extra_start, extra_end = D(2024, 3, 16), D(2024, 3, 21)
    session = FakeSession([("status", 204)])
    third = cenace.collect(
        nodes,
        "SIN",
        "MDA",
        extra_start,
        extra_end,
        bronze,
        silver,
        session=session,
        sleep=sleeps.append,
    )
    assert third["n_empty"] == 1 and len(session.calls) == 1
    fourth = cenace.collect(
        nodes,
        "SIN",
        "MDA",
        extra_start,
        extra_end,
        bronze,
        silver,
        session=FakeSession([]),
        sleep=sleeps.append,
    )
    assert fourth["n_planned"] == 0
    values = {n: make_records(D(2024, 3, 15), 7, seed=9) for n in nodes}
    session = FakeSession([("ok", cenace_payload(values))])
    fifth = cenace.collect(
        nodes,
        "SIN",
        "MDA",
        extra_start,
        extra_end,
        bronze,
        silver,
        session=session,
        refetch_empty=True,
        sleep=sleeps.append,
    )
    assert fifth["n_fetched"] == 1 and fifth["n_ok"] == 1
    frame = storage.read_silver(silver, "MDA", nodes=["07PNC-115"])
    assert frame["fecha"].max() == extra_end
