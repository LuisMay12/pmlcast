#!/usr/bin/env python3
"""Plan, fetch, parse and store CENACE SW-PML price requests."""

import argparse
import collections
import csv
import datetime
import json
import logging
import os
import random
import time

import requests

import pmlcast
from pmlcast import config
from pmlcast import preprocess
from pmlcast import storage

BASE_URL = "https://ws01.cenace.gob.mx:8082/SWPML/SIM"
REPO_URL = "https://github.com/LuisMay12/pmlcast"
MAX_NODES_PER_REQUEST = 20
MAX_DAYS_PER_REQUEST = 7
TIMEOUT_S = 60
MAX_RETRIES = 5
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0
PAUSE_S = 0.5
RETRY_STATUSES = (429, 500, 502, 503, 504)
SECONDS_PER_CALL = (3.0, 19.0)
DEFAULT_STAGE1_START = datetime.date(2022, 1, 1)
EVENTS = ("ok", "empty", "not_found", "message", "error")

Request = collections.namedtuple("Request", "sistema proceso nodes start end")


# ---------------------------------------------------------------------------
# Request planning
# ---------------------------------------------------------------------------


def user_agent():
    """Return an identifiable User-Agent with optional contact info."""
    contact = os.environ.get("PMLCAST_CONTACT", "").strip()
    suffix = "; {}".format(contact) if contact else ""
    return "pmlcast/{} (+{}{})".format(pmlcast.__version__, REPO_URL, suffix)


def validate_request(request):
    """Raise ``ValueError`` when a request violates the service limits."""
    if request.sistema not in config.SYSTEMS:
        raise ValueError("sistema must be one of {}".format(config.SYSTEMS))
    if request.proceso not in config.MARKETS:
        raise ValueError("proceso must be one of {}".format(config.MARKETS))
    if not request.nodes or len(request.nodes) > MAX_NODES_PER_REQUEST:
        raise ValueError(
            "a request needs 1 to {} nodes".format(MAX_NODES_PER_REQUEST)
        )
    if request.end < request.start:
        raise ValueError("end date must not be before start date")
    days = (request.end - request.start).days + 1
    if days > MAX_DAYS_PER_REQUEST:
        raise ValueError(
            "a request spans at most {} days".format(MAX_DAYS_PER_REQUEST)
        )


def build_url(request):
    """Return the SW-PML URL of a request.

    Args:
        request: A ``Request`` namedtuple.

    Returns:
        URL string ending in ``/JSON``.
    """
    validate_request(request)
    return "{}/{}/{}/{}/{:%Y/%m/%d}/{:%Y/%m/%d}/JSON".format(
        BASE_URL,
        request.sistema,
        request.proceso,
        ",".join(request.nodes),
        request.start,
        request.end,
    )


def epoch_windows(proceso, start, end):
    """Split a date range into 7-day windows anchored at the market epoch.

    Anchoring at the epoch keeps window boundaries identical across runs
    and stages, so cached envelopes are reusable. The first window may
    begin a few days before ``start``; the last one is clipped to ``end``.

    Args:
        proceso: ``MDA`` or ``MTR``.
        start: First date wanted.
        end: Last date wanted.

    Returns:
        List of ``(window_start, window_end)`` date tuples.
    """
    epoch = config.EPOCHS[proceso]
    start = max(start, epoch)
    if end < start:
        return []
    first = (start - epoch).days // MAX_DAYS_PER_REQUEST
    window_start = epoch + datetime.timedelta(
        days=first * MAX_DAYS_PER_REQUEST
    )
    windows = []
    while window_start <= end:
        window_end = min(
            window_start + datetime.timedelta(days=MAX_DAYS_PER_REQUEST - 1),
            end,
        )
        windows.append((window_start, window_end))
        window_start += datetime.timedelta(days=MAX_DAYS_PER_REQUEST)
    return windows


def window_dates(window_start, window_end):
    """Return the list of dates inside a window."""
    days = (window_end - window_start).days + 1
    return [window_start + datetime.timedelta(days=i) for i in range(days)]


def plan_requests(nodes, sistema, proceso, start, end, have=None):
    """Plan the requests needed to cover nodes over a date range.

    Args:
        nodes: Iterable of node keys.
        sistema: ``SIN``, ``BCA`` or ``BCS``.
        proceso: ``MDA`` or ``MTR``.
        start: First date wanted.
        end: Last date wanted.
        have: Optional set of ``(node, date)`` pairs already available;
            nodes fully covered inside a window are not requested again.

    Returns:
        List of ``Request`` namedtuples, ordered by window then node.
    """
    have = have or set()
    nodes = sorted(set(nodes))
    planned = []
    for window_start, window_end in epoch_windows(proceso, start, end):
        days = window_dates(window_start, window_end)
        pending = [
            node
            for node in nodes
            if any((node, day) not in have for day in days)
        ]
        for i in range(0, len(pending), MAX_NODES_PER_REQUEST):
            chunk = tuple(pending[i : i + MAX_NODES_PER_REQUEST])
            planned.append(
                Request(sistema, proceso, chunk, window_start, window_end)
            )
    return planned


# ---------------------------------------------------------------------------
# HTTP and parsing
# ---------------------------------------------------------------------------


def fetch(request, session, sleep=time.sleep, jitter=random.random):
    """Send one request with retries and exponential backoff.

    Args:
        request: A ``Request`` namedtuple.
        session: Object with a ``get(url, headers=, timeout=)`` method.
        sleep: Function used to wait between attempts (injectable).
        jitter: Function returning a float in [0, 1) added to each wait.

    Returns:
        Dict with ``url``, ``status_code`` (None on network failure),
        ``text``, ``elapsed_s``, ``fetched_at``, ``attempts``, ``error``
        and ``user_agent``.
    """
    url = build_url(request)
    headers = {"User-Agent": user_agent(), "Accept": "application/json"}
    attempts = 0
    while True:
        attempts += 1
        started = time.monotonic()
        error = None
        try:
            response = session.get(url, headers=headers, timeout=TIMEOUT_S)
            status = response.status_code
            text = response.text
        except (requests.Timeout, requests.ConnectionError) as exc:
            status = None
            text = ""
            error = repr(exc)
        elapsed = time.monotonic() - started

        retry = error is not None or status in RETRY_STATUSES
        if not retry or attempts > MAX_RETRIES:
            break
        delay = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (attempts - 1))
        sleep(delay + jitter())

    return {
        "url": url,
        "status_code": status,
        "text": text,
        "elapsed_s": round(elapsed, 3),
        "fetched_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "attempts": attempts,
        "error": error,
        "user_agent": headers["User-Agent"],
    }


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _find_message(payload):
    for key in ("Message", "message", "mensaje", "Mensaje"):
        if key in payload and payload[key]:
            return str(payload[key])
    for value in payload.values():
        if isinstance(value, str) and "no se pueden" in value.lower():
            return value
    return None


def parse_response(text):
    """Parse a SW-PML JSON body into flat price records.

    Args:
        text: Raw response body (may be empty).

    Returns:
        Tuple ``(records, message)``. ``records`` is a list of dicts with
        ``node``, ``fecha`` (ISO string), ``hora`` (int) and the four price
        fields as floats (None when absent). ``message`` is the service
        message when the body carried one instead of data, else None.
    """
    if not text or not text.strip():
        return [], None
    try:
        payload = json.loads(text)
    except ValueError:
        return [], "invalid json"

    if isinstance(payload, dict):
        message = _find_message(payload)
        if message:
            return [], message
        results = payload.get("Resultados", payload.get("resultados"))
    elif isinstance(payload, list):
        results = payload
    else:
        return [], "unexpected payload"

    records = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        node = item.get("clv_nodo") or item.get("nodo") or item.get("clave")
        values = item.get("Valores", item.get("valores")) or []
        for value in values:
            hora = _to_int(value.get("hora"))
            fecha = value.get("fecha")
            if node is None or hora is None or not fecha:
                continue
            records.append(
                {
                    "node": str(node).strip(),
                    "fecha": str(fecha)[:10],
                    "hora": hora,
                    "pml": _to_float(value.get("pml")),
                    "pml_ene": _to_float(value.get("pml_ene")),
                    "pml_per": _to_float(value.get("pml_per")),
                    "pml_cng": _to_float(value.get("pml_cng")),
                }
            )
    return records, None


def classify(response, records, message):
    """Label the outcome of a request as a data-quality event.

    Returns:
        One of ``ok``, ``empty``, ``not_found``, ``message`` or ``error``.
    """
    status = response.get("status_code")
    if status is None or status in RETRY_STATUSES or status >= 500:
        return "error"
    if status == 404:
        return "not_found"
    if message:
        return "message"
    if status == 204 or not records:
        return "empty"
    if status == 200:
        return "ok"
    return "error"


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------


def _store_records(records, sistema, proceso, silver_dir):
    observed = preprocess.records_to_hourly(
        records, sistema=sistema, market=proceso
    )
    for node, node_df in observed.groupby("node"):
        storage.upsert_silver(silver_dir, proceso, node, node_df)
    return len(observed)


def collect(
    nodes,
    sistema,
    proceso,
    start,
    end,
    bronze_dir,
    silver_dir,
    session=None,
    refetch_empty=False,
    dry_run=False,
    log=None,
    sleep=time.sleep,
):
    """Collect prices for nodes over a date range into bronze and silver.

    Every ``(node, date)`` already present in silver, or already requested
    according to bronze, is skipped, so re-running is cheap and never asks
    CENACE twice for the same data. Requests run sequentially.

    Args:
        nodes: Iterable of node keys.
        sistema: ``SIN``, ``BCA`` or ``BCS``.
        proceso: ``MDA`` or ``MTR``.
        start: First operation date.
        end: Last operation date.
        bronze_dir: Root of the bronze layer.
        silver_dir: Root of the silver layer.
        session: Optional HTTP session (a ``requests.Session`` by default).
        refetch_empty: Re-request windows whose envelope carried no data.
        dry_run: Only plan and report, without any HTTP call.
        log: Optional logger.
        sleep: Function used to pause between requests (injectable).

    Returns:
        Summary dict with request and event counts.
    """
    log = log or logging.getLogger("pmlcast")
    have = storage.silver_coverage(silver_dir, proceso)
    if not refetch_empty:
        have |= storage.bronze_requested(bronze_dir, proceso)
    plan = plan_requests(nodes, sistema, proceso, start, end, have)

    summary = {"n_planned": len(plan), "n_fetched": 0, "n_cached": 0}
    summary.update({"n_" + event: 0 for event in EVENTS})
    summary.update({"n_records": 0, "elapsed_s": 0.0})
    low, high = SECONDS_PER_CALL
    log.info(
        "%s %s: %d requests planned for %d nodes (%s -> %s), ETA %d-%d min",
        sistema,
        proceso,
        len(plan),
        len(set(nodes)),
        start,
        end,
        len(plan) * low / 60,
        len(plan) * high / 60,
    )
    if dry_run or not plan:
        return summary

    session = session or requests.Session()
    started = time.monotonic()
    for i, request in enumerate(plan, 1):
        path = storage.bronze_path(bronze_dir, request)
        stored = storage.find_bronze(bronze_dir, request)
        response = None
        if stored:
            _, cached = storage.read_bronze(stored)
            if not (refetch_empty and not cached.get("text")):
                response = cached
                summary["n_cached"] += 1
        if response is None:
            response = fetch(request, session, sleep=sleep)
            storage.write_bronze(path, request, response)
            summary["n_fetched"] += 1
            sleep(PAUSE_S)

        records, message = parse_response(response.get("text") or "")
        event = classify(response, records, message)
        summary["n_" + event] += 1
        if records:
            summary["n_records"] += _store_records(
                records, sistema, proceso, silver_dir
            )
        log.info(
            "[%d/%d] %s %s..%s %d nodes -> %s status=%s records=%d %.1fs%s",
            i,
            len(plan),
            request.proceso,
            request.start,
            request.end,
            len(request.nodes),
            event,
            response.get("status_code"),
            len(records),
            response.get("elapsed_s") or 0.0,
            " message={}".format(message) if message else "",
        )

    summary["elapsed_s"] = round(time.monotonic() - started, 1)
    log.info("done: %s", summary)
    return summary


def rebuild_silver(bronze_dir, silver_dir, proceso, sistema=None, log=None):
    """Re-derive the silver layer of one market from every bronze envelope.

    Existing silver files of the touched nodes are replaced, so cleaning
    rules can be changed and re-applied without touching CENACE.
    """
    log = log or logging.getLogger("pmlcast")
    by_node = collections.defaultdict(list)
    for path in storage.iter_bronze(bronze_dir, sistema, proceso):
        meta, response = storage.read_bronze(path)
        records, _ = parse_response(response.get("text") or "")
        if not records:
            continue
        observed = preprocess.records_to_hourly(
            records, sistema=meta["sistema"], market=proceso
        )
        for node, node_df in observed.groupby("node"):
            by_node[node].append(node_df)

    for node, frames in sorted(by_node.items()):
        path = storage.silver_path(silver_dir, proceso, node)
        if os.path.exists(path):
            os.remove(path)
        merged = preprocess.pd.concat(frames, ignore_index=True)
        storage.upsert_silver(silver_dir, proceso, node, merged)
        log.info("rebuilt %s %s: %d observed rows", proceso, node, len(merged))
    return sorted(by_node)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def read_nodes_file(path):
    """Read node keys from a CSV with a ``node`` column (or one per line)."""
    with open(path, encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row and row[0].strip()]
    if not rows:
        return []
    header = [cell.strip().lower() for cell in rows[0]]
    if "node" in header:
        column = header.index("node")
        return [row[column].strip() for row in rows[1:] if len(row) > column]
    return [row[0].strip() for row in rows]


def _parse_date(text):
    return datetime.date.fromisoformat(text)


def main():
    """Run the collector from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", type=int, choices=(1, 2))
    group.add_argument("--nodes-file")
    group.add_argument("--nodes", help="comma-separated node keys")
    parser.add_argument("--sistema", default="SIN", choices=config.SYSTEMS)
    parser.add_argument("--proceso", default="MDA", choices=config.MARKETS)
    parser.add_argument("--start", type=_parse_date)
    parser.add_argument("--end", type=_parse_date)
    parser.add_argument("--refetch-empty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-silver", action="store_true")
    parser.add_argument("--bronze-dir", default=config.BRONZE_DIR)
    parser.add_argument("--silver-dir", default=config.SILVER_DIR)
    args = parser.parse_args()

    log = config.setup_logging(
        os.path.join(args.bronze_dir, "collect_{}.log".format(args.proceso))
    )
    if args.rebuild_silver:
        nodes = rebuild_silver(args.bronze_dir, args.silver_dir, args.proceso)
        log.info("rebuilt %d nodes", len(nodes))
        return

    if args.stage:
        nodes_file = os.path.join(
            config.SNAPSHOT_DIR, "nodes_stage{}.csv".format(args.stage)
        )
        nodes = read_nodes_file(nodes_file)
    elif args.nodes_file:
        nodes = read_nodes_file(args.nodes_file)
    else:
        nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    if not nodes:
        raise SystemExit("no nodes to collect")

    start = args.start
    if start is None:
        if args.stage == 2:
            start = config.EPOCHS[args.proceso]
        else:
            start = DEFAULT_STAGE1_START
    end = args.end
    if end is None:
        end = config.today_local()
        if args.proceso == "MTR":
            end -= datetime.timedelta(days=config.MTR_LAG_DAYS)

    collect(
        nodes,
        args.sistema,
        args.proceso,
        start,
        end,
        args.bronze_dir,
        args.silver_dir,
        refetch_empty=args.refetch_empty,
        dry_run=args.dry_run,
        log=log,
    )


if __name__ == "__main__":
    main()
