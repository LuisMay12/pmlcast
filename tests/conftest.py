#!/usr/bin/env python3
"""Provide shared fixtures: no network, fake HTTP and synthetic CENACE data."""

import datetime
import json
import math
import os

import numpy as np
import pandas as pd
import pytest
import requests

from pmlcast import cenace
from pmlcast import dataset
from pmlcast import preprocess

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SILVER_NODES = ("07PNC-115", "07MRD-230", "07TIZ-400")
SILVER_START = datetime.date(2022, 3, 15)
SILVER_DAYS = 240


def _no_network(*args, **kwargs):
    raise RuntimeError("network disabled in tests")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test tries to reach the network."""
    monkeypatch.setattr(requests.Session, "get", _no_network)
    monkeypatch.setattr(requests.Session, "request", _no_network)
    monkeypatch.setattr(requests, "get", _no_network)


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


class FakeSession:
    """Replay scripted responses and record every call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        """Return the next scripted item or raise the scripted error."""
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        if not self.script:
            raise AssertionError("unexpected HTTP call: {}".format(url))
        kind, value = self.script.pop(0)
        if kind == "ok":
            return FakeResponse(200, value)
        if kind == "status":
            return FakeResponse(value, "")
        if kind == "raise":
            raise value
        raise ValueError("unknown script item {}".format(kind))


def make_records(start, days, dst=True, seed=0, drop=None):
    """Build CENACE-shaped ``Valores`` (all strings) with an evening peak.

    Args:
        start: First operation date.
        days: Number of days.
        dst: Emit 23/25 records on DST transition days when True.
        seed: Random seed for the noise.
        drop: Optional set of ``(date, hora)`` pairs to leave out.

    Returns:
        List of dicts with ``fecha``, ``hora`` and the four price strings.
    """
    rng = np.random.default_rng(seed)
    last = start + datetime.timedelta(days=days)
    transitions = preprocess.dst_days(range(start.year, last.year + 1))
    drop = drop or set()
    values = []
    for i in range(days):
        day = start + datetime.timedelta(days=i)
        kind = transitions.get(day) if dst else None
        n_hours = {"spring": 23, "fall": 25}.get(kind, 24)
        for hora in range(1, n_hours + 1):
            if (day, hora) in drop:
                continue
            hour = (hora - 1) % 24
            peak = 300.0 * math.exp(-((hour - 19) ** 2) / 8.0)
            pml = 800.0 + peak + 50.0 * rng.standard_normal()
            ene = 0.9 * pml
            per = 0.05 * pml
            values.append(
                {
                    "fecha": day.isoformat(),
                    "hora": str(hora),
                    "pml": "{:.2f}".format(pml),
                    "pml_ene": "{:.2f}".format(ene),
                    "pml_per": "{:.2f}".format(per),
                    "pml_cng": "{:.2f}".format(pml - ene - per),
                }
            )
    return values


def cenace_payload(values_by_node, proceso="MDA", sistema="SIN"):
    """Wrap per-node ``Valores`` lists in the real SW-PML JSON envelope."""
    payload = {
        "nombre": "PML",
        "proceso": proceso,
        "sistema": sistema,
        "area": "PÚBLICA",
        "Resultados": [
            {"clv_nodo": node, "Valores": values}
            for node, values in values_by_node.items()
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def parsed_records(node, values):
    """Run ``Valores`` through the real parser for one node."""
    records, message = cenace.parse_response(cenace_payload({node: values}))
    assert message is None
    return records


@pytest.fixture
def fixture_text():
    """Return a loader for files under ``tests/fixtures``."""

    def _load(name):
        with open(os.path.join(FIXTURES_DIR, name), encoding="utf-8") as fh:
            return fh.read()

    return _load


@pytest.fixture(scope="session")
def silver_small():
    """Three synthetic nodes, 240 days covering both 2022 DST days.

    Node ``07PNC-115`` carries a 2-hour gap (interpolated) and a 30-hour
    gap (missing).
    """
    frames = []
    for i, node in enumerate(SILVER_NODES):
        drop = set()
        if i == 0:
            drop |= {
                (datetime.date(2022, 5, 10), 10),
                (datetime.date(2022, 5, 10), 11),
            }
            drop |= {(datetime.date(2022, 6, 1), h) for h in range(5, 25)}
            drop |= {(datetime.date(2022, 6, 2), h) for h in range(1, 11)}
        values = make_records(SILVER_START, SILVER_DAYS, seed=i, drop=drop)
        observed = preprocess.records_to_hourly(parsed_records(node, values))
        frames.append(preprocess.build_hourly_grid(observed))
    return pd.concat(frames, ignore_index=True)


def mini_catalog():
    """Catalog rows for the three synthetic silver nodes."""
    return pd.DataFrame(
        {
            "node": list(SILVER_NODES),
            "sistema": ["SIN"] * 3,
            "region": ["PENINSULAR"] * 3,
            "zone": ["MERIDA", "MERIDA", "CANCUN"],
            "kv": [115.0, 230.0, 400.0],
            "catalog_version": ["20260218"] * 3,
        }
    )


@pytest.fixture(scope="session")
def dataset_small(silver_small):
    """Dataset built from ``silver_small`` with short val/test periods."""
    return dataset.build_dataset(
        silver_small,
        mini_catalog(),
        SILVER_NODES,
        "small",
        test_months=1,
        val_days=14,
    )
