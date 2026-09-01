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
    0 0,6,12,18 * * * cd /opt/cgstudy && APIFY_TOKEN=$(cat .token) \\
        /usr/bin/python3 cg_capture.py --tier standard >> capture.log 2>&1
"""

import argparse, hashlib, json, os, sys, time, csv
from datetime import datetime, timezone
from urllib import request, error

ACTOR = "api_merge~coinglass-liquidation-heatmap"
ENDPOINT = f"https://api.apify.com/v2/acts/{ACTOR}/run-sync-get-dataset-items"

MODELS = ["model1", "model2", "model3"]
INTERVALS = ["12h", "24h", "48h", "3d", "1w", "2w", "1mo", "3mo"]

TIERS = {
    # Full: every model x every interval x 3 coins = 72 results, $0.72/cycle
    "full": [(c, m, i) for c in ("BTC", "ETH", "SOL") for m in MODELS for i in INTERVALS],
    # Standard: BTC complete, ETH/SOL on three representative intervals
    "standard": (
        [("BTC", m, i) for m in MODELS for i in INTERVALS]
        + [(c, m, i) for c in ("ETH", "SOL") for m in MODELS for i in ("12h", "24h", "1w")]
    ),
    # Lean: the pre-registered primary config only, 3 results, $0.03/cycle
    "lean": [(c, "model1", "24h") for c in ("BTC", "ETH", "SOL")],
    # Primary: BTC/model1/24h alone -- the absolute minimum that keeps the
    # pre-registered study alive if quota or budget becomes binding.
    "primary": [("BTC", "model1", "24h")],
}

MANIFEST_COLS = [
    "pull_utc", "symbol", "model", "interval", "update_time_ms", "update_utc",
    "lag_sec", "n_y_levels", "y_lo", "y_hi", "y_step", "grid_rows",
    "payload_bytes", "sha256", "path", "status",
]


def pull(symbol, model, interval, token, timeout=90):
    body = json.dumps({"symbol": symbol, "model": model, "interval": interval}).encode()
    req = request.Request(
        f"{ENDPOINT}?token={token}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return raw, time.time() - t0


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
    manifest = os.path.join(args.outdir, "manifest.csv")
    new = not os.path.exists(manifest)

    ok = fail = 0
    with open(manifest, "a", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=MANIFEST_COLS)
        if new:
            w.writeheader()

        for symbol, model, interval in configs:
            pull_utc = datetime.now(timezone.utc)
            row = {c: "" for c in MANIFEST_COLS}
            row.update(pull_utc=pull_utc.isoformat(), symbol=symbol,
                       model=model, interval=interval)
            try:
                raw, _ = pull(symbol, model, interval, token)
                items = json.loads(raw)
                item = items[0] if isinstance(items, list) and items else items
                if not item.get("success", False):
                    raise RuntimeError(item.get("message", "actor reported failure"))

                y = item["y_axis"]
                ut = item["updateTime"]
                ut_dt = datetime.fromtimestamp(ut / 1000, timezone.utc)
                fn = f"{symbol}_{model}_{interval}.json"
                path = os.path.join(outdir, fn)
                with open(path, "wb") as f:
                    f.write(raw)

                row.update(
                    update_time_ms=ut,
                    update_utc=ut_dt.isoformat(),
                    lag_sec=round((pull_utc - ut_dt).total_seconds(), 3),
                    n_y_levels=len(y),
                    y_lo=y[0], y_hi=y[-1],
                    y_step=round(y[1] - y[0], 6) if len(y) > 1 else "",
                    grid_rows=len(item.get("liquidation_leverage_data", [])),
                    payload_bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest()[:16],
                    path=path, status="OK",
                )
                ok += 1
            except Exception as e:                       # noqa: BLE001
                # A failed pull is recorded, never silently skipped. Silent gaps
                # in a time series masquerade as data.
                row["status"] = f"ERROR: {type(e).__name__}: {e}"[:200]
                fail += 1
            w.writerow(row)
            mf.flush()
            time.sleep(args.sleep)

    print(f"[{cycle}] tier={args.tier} ok={ok} fail={fail} -> {outdir}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
