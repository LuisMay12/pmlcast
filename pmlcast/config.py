#!/usr/bin/env python3
"""Hold the paths, constants and logging setup shared by every module."""

import datetime
import logging
import os
import sys
import zoneinfo

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("PMLCAST_DATA_DIR", os.path.join(REPO_DIR, "data"))
BRONZE_DIR = os.path.join(DATA_DIR, "bronze")
SILVER_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DIR = os.path.join(DATA_DIR, "gold")
CATALOG_DIR = os.path.join(DATA_DIR, "catalog")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshot")
DOCS_DIR = os.path.join(REPO_DIR, "docs")
FIGURES_DIR = os.path.join(DOCS_DIR, "figures")
MLRUNS_URI = "sqlite:///" + os.path.join(REPO_DIR, "mlflow.db")
MLFLOW_ARTIFACTS = os.path.join(REPO_DIR, "mlruns")

# CENACE market facts
TZ_NAME = "America/Mexico_City"
SYSTEMS = ("SIN", "BCA", "BCS")
MARKETS = ("MDA", "MTR")
MDA_EPOCH = datetime.date(2016, 1, 29)
MTR_EPOCH = datetime.date(2017, 1, 27)
EPOCHS = {"MDA": MDA_EPOCH, "MTR": MTR_EPOCH}
MTR_LAG_DAYS = 8

# Windowing and scaling (pitch section 5.3 and 8)
INPUT_HOURS = 168
TARGET_HOURS = 24
STATS_DAYS = 28
STATS_HOURS = STATS_DAYS * 24
MIN_STATS_COVERAGE = 0.9
SIGMA_FLOOR_ABS = 10.0
SIGMA_FLOOR_REL = 0.05
MAX_INTERP_HOURS = 3
MAX_INTERP_INPUT = 12
SPIKE_K_SIGMA = 4.0
VOLATILITY_FEATURE = "sigma_over_1000"

# Node metadata vocabularies
UNKNOWN = "UNKNOWN"
REGIONS = (
    "CENTRAL",
    "NORESTE",
    "NOROESTE",
    "NORTE",
    "OCCIDENTAL",
    "ORIENTAL",
    "PENINSULAR",
    UNKNOWN,
)
PRICE_COLUMNS = ("pml", "pml_ene", "pml_per", "pml_cng")
OBSERVED_QUALITIES = ("ok", "dst_fill", "dst_merge")
SEED = 0


def today_local():
    """Return today's date in the CENACE system time zone."""
    return datetime.datetime.now(zoneinfo.ZoneInfo(TZ_NAME)).date()


def setup_logging(log_file=None, level=logging.INFO):
    """Configure logging to stdout and, optionally, to a file.

    Args:
        log_file: Optional path of a file that also receives the log.
        level: Logging level for the ``pmlcast`` logger.

    Returns:
        The configured ``pmlcast`` logger.
    """
    logger = logging.getLogger("pmlcast")
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
