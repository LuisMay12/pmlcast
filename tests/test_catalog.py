#!/usr/bin/env python3
"""Test the catalog loader, node selection and coverage report."""

import os

import openpyxl
import pandas as pd
import pytest

from pmlcast import catalog
from pmlcast import config

HEADER = [
    "SISTEMA",
    "CENTRO DE CONTROL REGIONAL",
    "ZONA DE CARGA",
    "CLAVE",
    "NOMBRE",
    "NIVEL DE TENSIÓN (kV)",
    "DIRECTAMENTE MODELADA",
    "INDIRECTAMENTE MODELADA",
    "DIRECTAMENTE MODELADA",
    "INDIRECTAMENTE MODELADA",
    "ZONA DE OPERACIÓN DE TRANSMISIÓN",
    "GERENCIA REGIONAL DE TRANSMISIÓN",
    "ZONA DE DISTRIBUCIÓN",
    "GERENCIA DIVISIONAL DE DISTRIBUCIÓN",
    "CLAVE DE ENTIDAD FEDERATIVA (INEGI)",
    "ENTIDAD FEDERATIVA (INEGI)",
    "CLAVE DE MUNICIPIO (INEGI)",
    "MUNICIPIO (INEGI)",
    "REGION DE TRANSMISION",
]
REGIONS = [r for r in config.REGIONS if r != config.UNKNOWN]


def mini_rows():
    """Build ~40 catalog rows: 7 regions x (1x400, 2x230, 2x115) + quirks."""
    rows = []

    def row(sistema, region, zone, node, kv, load="X", gen=""):
        return [
            sistema,
            region,
            zone,
            node,
            "NAME " + node,
            str(kv),
            load,
            "",
            gen,
            "",
            "ZOT",
            "GRT",
            "ZD",
            "GDD",
            "9",
            "STATE",
            "9001",
            "MUNI",
            "RT",
        ]

    for i, region in enumerate(REGIONS):
        prefix = "{:02d}".format(i + 1)
        zones = ["{}-Z1".format(region[:3]), "{}-Z2".format(region[:3])]
        rows.append(
            row("SIN", region, zones[0], prefix + "AAA-400", 400, gen="X")
        )
        rows.append(row("SIN", region, zones[0], prefix + "BBB-230", 230))
        rows.append(row("SIN", region, zones[1], prefix + "CCC-230", 230))
        rows.append(row("SIN", region, zones[0], prefix + "DDD-115", 115))
        rows.append(row("SIN", region, zones[1], prefix + "EEE-115", 115))
    rows.append(row("SIN", "No Aplica", "PEN-Z1", "09NAP-230", 230))
    rows.append(row("SIN", "PENINSULAR ", "PEN-Z2", "07TRL-115", 115))
    rows.append(row("SIN", "CENTRAL", "CEN-Z1", "01LOW-34", 34.5))
    rows.append(row("BCA", "BAJA CALIFORNIA", "TIJUANA", "07PJZ-230", 230))
    rows.append(row("BCS", "BAJA CALIFORNIA SUR", "LA PAZ", "07OLA-115", 115))
    return rows


@pytest.fixture
def mini_csv(tmp_path):
    path = tmp_path / "Catalogo de NodosP V20260218-Table 1.csv"
    frame = pd.DataFrame(mini_rows(), columns=HEADER)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


@pytest.fixture
def mini_xlsx(tmp_path):
    path = tmp_path / "Catálogo NodosP v2026-08-19.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Catálogo de NodosP"] + [""] * (len(HEADER) - 1))
    group = [""] * len(HEADER)
    group[6], group[8] = "CARGA", "GENERACIÓN"
    sheet.append(group)
    sheet.append(HEADER)
    for values in mini_rows():
        sheet.append(values)
    book.save(path)
    return str(path)


def test_load_csv_normalizes(mini_csv):
    df = catalog.load_catalog(mini_csv)
    assert list(df.columns) == catalog.CATALOG_COLUMNS
    assert df["catalog_version"].iloc[0] == "20260218"
    assert len(df) == len(mini_rows())
    assert df["kv"].dtype.kind == "f"
    assert (
        df.loc[df["node"] == "09NAP-230", "region"].iloc[0] == config.UNKNOWN
    )
    assert df.loc[df["node"] == "07TRL-115", "region"].iloc[0] == "PENINSULAR"
    assert df.loc[df["node"] == "01AAA-400", "gen_direct"].iloc[0]
    assert not df.loc[df["node"] == "01BBB-230", "gen_direct"].iloc[0]
    assert df["load_direct"].all()
    assert df["state_code"].dtype.name == "Int64"
    assert (df["sistema"] == "SIN").sum() == len(mini_rows()) - 2


def test_load_xlsx_matches_csv(mini_csv, mini_xlsx):
    a = catalog.load_catalog(mini_csv).drop(
        columns=["catalog_version", "loaded_at"]
    )
    b = catalog.load_catalog(mini_xlsx).drop(
        columns=["catalog_version", "loaded_at"]
    )
    pd.testing.assert_frame_equal(a, b)
    assert catalog.parse_version(mini_xlsx) == "20260819"
    assert catalog.parse_version("nodes.csv") is None
    with pytest.raises(ValueError):
        catalog.load_catalog(
            os.path.join(os.path.dirname(mini_csv), "nodes.csv")
        )


def test_select_spreads_over_regions_and_zones(mini_csv):
    df = catalog.load_catalog(mini_csv)
    sel = catalog.select_nodes(df, {400: "all", 230: 14, 115: 7})
    assert (sel["kv"] == 400).sum() == 7  # one per region, 34.5 kV excluded
    per_region_230 = sel[sel["kv"] == 230].groupby("region").size()
    assert (per_region_230 == 2).all() and len(per_region_230) == 7
    per_region_115 = sel[sel["kv"] == 115].groupby("region").size()
    assert (per_region_115 == 1).all() and len(per_region_115) == 7
    assert config.UNKNOWN not in set(sel["region"])
    assert set(sel["sistema"]) == {"SIN"}
    assert "01LOW-34" not in set(sel["node"])
    # quota above availability returns everything available
    sel = catalog.select_nodes(df, {115: 99})
    assert len(sel) == 15
    # zones get spread: with quota 7 at 115 kV every pick is a first zone
    sel = catalog.select_nodes(df, {115: 7})
    assert sel.groupby("region").size().eq(1).all()
