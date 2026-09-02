#!/usr/bin/env python3
"""
cg_capture.py — forward-snapshot capture for the CoinGlass heatmap study.

Runs on the Oracle VM under cron at 00/06/12/18 UTC. Pulls the configured
sweep, writes each raw payload to disk, and appends one manifest row per pull.

The 00/06/12/18 cadence is deliberate: it matches the in-house engine's
prediction-freeze schedule, so adjacent snapshots are exactly one 6h horizon
apart and are non-overlapping by construction.

Usage
-----
    export APIFY_TOKEN=...            # never commit this
    python3 cg_capture.py --tier standard
    python3 cg_capture.py --tier full --outdir /data/cgstudy

Crontab (VM, UTC):
    0 0,6,12,18 * * * cd ~/cgstudy && APIFY_TOKEN=$(cat .token) \\
        /usr/bin/python3 cg_capture.py --tier standard >> capture.log 2>&1
"""

import argparse, csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from urllib import request

ACTOR = "api_merge~coinglass-liquidation-heatmap"
ENDPOINT = f"https://api.apify.com/v2/acts/{ACTOR}/run-sync-get-dataset-items"

MODELS = ["model1", "model2", "model3"]
INTERVALS = ["12h", "24h", "48h", "3d", "1w", "2w", "1mo", "3mo"]

TIERS = {
    # Full: every model x every interval x 3 coins = 72 results, $0.72/cycle
    "full": [(c, m, i) for c in ("BTC", "ETH", "SOL") for m in MODELS for i in INTERVALS],
    # Standard: BTC complete (24) + ETH/SOL on three intervals (9 each) = 42,
    # $0.42/cycle, ~$50/month at 4 cycles/day. MEASURED 2026-09-02: mean payload
    # 2.7 MB, ~110 MB/cycle, ~440 MB/day, ~13.2 GB/month.
    "standard": (
        [("BTC", m, i) for m in MODELS for i in INTERVALS]
        + [(c, m, i) for c in ("ETH", "SOL") for m in MODELS for i in ("12h", "24h", "1w")]
    ),
    # Lean: the pre-registered primary config across coins, 3 results
    "lean": [(c, "model1", "24h") for c in ("BTC", "ETH", "SOL")],
    # Primary: BTC/model1/24h alone — the minimum that keeps the pre-registered
    # study alive if quota or budget becomes binding.
    "primary": [("BTC", "model1", "24h")],
}

MANIFEST_COLS = [
    "pull_utc", "symbol", "model", "interval", "update_time_ms", "update_utc",
    "lag_sec", "attempts", "n_y_levels", "y_lo", "y_hi", "y_step", "grid_rows",
    "payload_bytes", "sha256", "path", "status",
]

MAX_ATTEMPTS = 3
BACKOFF = [2.0, 6.0]        # seconds before retry 2 and retry 3


def pull(symbol, model, interval, token, timeout=120):
    """One attempt. Returns (raw_bytes, receipt_time_utc)."""
    body = json.dumps({"symbol": symbol, "model": model, "interval": interval}).encode()
    req = request.Request(
        f"{ENDPOINT}?token={token}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return raw, datetime.now(timezone.utc)


def pull_with_retry(symbol, model, interval, token):
    """Retry transient failures.

    ADDED 2026-09-02 after three HTTP 400s in the 18:00Z cycle (BTC/model1 at
    12h, 48h, 3d) — configs that had succeeded 5 minutes earlier, so transient.
    Without a retry a single 400 permanently destroys that window, and if it
    lands on the pre-registered primary config that window is unrecoverable:
    forward capture cannot be replayed.
    """
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw, recv = pull(symbol, model, interval, token)
            return raw, recv, attempt, None
        except Exception as e:                                # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF[attempt - 1])
    return None, None, MAX_ATTEMPTS, last


def open_manifest(path):
    """Append, rotating the file if its header predates the current columns."""
    if os.path.exists(path):
        with open(path, newline="") as f:
            existing = next(csv.reader(f), [])
        if [c.strip() for c in existing] != MANIFEST_COLS:
            old = path.replace(".csv", f"_v1_{int(time.time())}.csv")
            os.rename(path, old)
            print(f"manifest schema changed; previous file kept at {old}")
    new = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, lineterminator="\n")
    if new:
        w.writeheader()
    return f, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="standard", choices=sorted(TIERS))
    ap.add_argument("--outdir", default="./snapshots")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between pulls")
    args = ap.parse_args()

    token = os.environ.get("APIFY_TOKEN")
    if not token:
        sys.exit("APIFY_TOKEN not set. Never hard-code it; export it or read from a 0600 file.")

    configs = TIERS[args.tier]
    cycle = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = os.path.join(args.outdir, cycle)
    os.makedirs(outdir, exist_ok=True)
    mf, w = open_manifest(os.path.join(args.outdir, "manifest.csv"))

    ok = fail = retried = 0
    with mf:
        for symbol, model, interval in configs:
            row = {c: "" for c in MANIFEST_COLS}
            row.update(pull_utc=datetime.now(timezone.utc).isoformat(),
                       symbol=symbol, model=model, interval=interval)
            raw, recv, attempts, err = pull_with_retry(symbol, model, interval, token)
            row["attempts"] = attempts
            if attempts > 1 and raw is not None:
                retried += 1

            if raw is None:
                row["status"] = f"ERROR: {err}"[:200]
                fail += 1
            else:
                try:
                    items = json.loads(raw)
                    item = items[0] if isinstance(items, list) and items else items
                    if not item.get("success", False):
                        raise RuntimeError(item.get("message", "actor reported failure"))
                    y = item["y_axis"]
                    ut = item["updateTime"]
                    ut_dt = datetime.fromtimestamp(ut / 1000, timezone.utc)
                    path = os.path.join(outdir, f"{symbol}_{model}_{interval}.json")
                    with open(path, "wb") as f:
                        f.write(raw)
                    row.update(
                        update_time_ms=ut,
                        update_utc=ut_dt.isoformat(),
                        # FIXED 2026-09-02: lag is measured from RECEIPT, not
                        # from request start. The old version subtracted an
                        # updateTime stamped DURING the request from a clock
                        # read BEFORE it, so it measured request duration and
                        # went negative. Freshness is receipt minus updateTime.
                        lag_sec=round((recv - ut_dt).total_seconds(), 3),
                        n_y_levels=len(y),
                        y_lo=y[0], y_hi=y[-1],
                        y_step=round(y[1] - y[0], 6) if len(y) > 1 else "",
                        grid_rows=len(item.get("liquidation_leverage_data", [])),
                        payload_bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest()[:16],
                        path=path, status="OK",
                    )
                    ok += 1
                except Exception as e:                        # noqa: BLE001
                    # A failed pull is recorded, never silently skipped. Silent
                    # gaps in a time series masquerade as data.
                    row["status"] = f"ERROR: {type(e).__name__}: {e}"[:200]
                    fail += 1
            w.writerow(row)
            mf.flush()
            time.sleep(args.sleep)

    print(f"[{cycle}] tier={args.tier} ok={ok} fail={fail} retried={retried} -> {outdir}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
