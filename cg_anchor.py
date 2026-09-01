#!/usr/bin/env python3
"""
cg_anchor.py -- is the heatmap's centre of mass a FORECAST or a LAGGING AVERAGE?

The 2026-09-01 sweep showed the map's centre of mass offset from spot varying
monotonically with the lookback interval: about +1.2% at 48h-3d, flipping to
-2.4% at 2w and -19% at 3mo. Three models agreed to within 0.2pp at 48h and 3d.

That pattern has an obvious mundane explanation: the mass simply sits near the
MEAN PRICE OVER THE LOOKBACK WINDOW. BTC rose over three months, so 3mo mass
sits far below spot; it fell this week, so short-window mass sits just above.

If true, the map's location carries no forward information at all -- it is a
moving average, and no amount of forward capture will make it predictive.

This script tests that directly. For each interval it computes the realised
mean and VWAP of BTC over exactly that lookback window, and regresses the map's
observed COM offset on it.

    R^2 > 0.90 and slope ~ 1  =>  COM IS the window mean. Lagging by
                                  construction. Stop the study; the instrument
                                  cannot forecast because it only summarises.
    R^2 0.50-0.90             =>  largely explained by the window mean, with a
                                  residual worth testing forward.
    R^2 < 0.50                =>  the offset is NOT just a lookback average;
                                  the forward study is justified.

The residual (COM minus window mean) is the only part that could carry signal,
so its size sets the ceiling on anything the forward study can find.

Usage:
    python3 cg_anchor.py snapshots/<cycle>_sweep/sweep.json --spot 77328.2
"""

import argparse, json, sys, time
from urllib import request
import numpy as np

HOURS = {"12h": 12, "24h": 24, "48h": 48, "3d": 72,
         "1w": 168, "2w": 336, "1mo": 720, "3mo": 2160}

BINANCE = "https://api.binance.com/api/v3/klines"


def klines(symbol, interval, limit):
    url = f"{BINANCE}?symbol={symbol}&interval={interval}&limit={limit}"
    with request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def series(symbol="BTCUSDT"):
    """Hourly for <= 41 days, daily beyond. Returns (ts_ms, close, volume)."""
    h = klines(symbol, "1h", 1000)
    time.sleep(0.3)
    d = klines(symbol, "1d", 200)
    hh = np.array([[float(k[0]), float(k[4]), float(k[5])] for k in h])
    dd = np.array([[float(k[0]), float(k[4]), float(k[5])] for k in d])
    return hh, dd


def window_stats(hh, dd, hours, now_ms):
    """Mean close and VWAP over the last `hours`, from whichever series covers it."""
    src = hh if hours <= 900 else dd            # 1000 hourly bars ~ 41 days
    cut = now_ms - hours * 3600 * 1000
    sel = src[src[:, 0] >= cut]
    if sel.shape[0] < 3:
        return None, None, 0
    mean = float(sel[:, 1].mean())
    vol = sel[:, 2]
    vwap = float((sel[:, 1] * vol).sum() / vol.sum()) if vol.sum() > 0 else mean
    return mean, vwap, int(sel.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("--spot", type=float, required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    args = ap.parse_args()

    rows = json.load(open(args.sweep_json))
    rows = [r for r in rows if r.get("coin", "BTC") == "BTC"]
    if not rows:
        sys.exit("no BTC rows in sweep.json")

    print("fetching Binance klines...")
    hh, dd = series(args.symbol)
    now_ms = float(max(hh[-1, 0], dd[-1, 0])) + 3600 * 1000

    hdr = (f"{'model':<8}{'intv':<6}{'COM%':>9}{'meanP%':>9}{'vwap%':>9}"
           f"{'resid%':>9}{'bars':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))

    com, pred, keep = [], [], []
    for r in sorted(rows, key=lambda x: (x["model"], HOURS.get(x["interval"], 0))):
        iv = r["interval"]
        if iv not in HOURS:
            continue
        mean, vwap, n = window_stats(hh, dd, HOURS[iv], now_ms)
        if mean is None:
            print(f"{r['model']:<8}{iv:<6}   insufficient klines")
            continue
        mp = (mean / args.spot - 1) * 100
        vp = (vwap / args.spot - 1) * 100
        resid = r["com_off_pct"] - mp
        com.append(r["com_off_pct"]); pred.append(mp); keep.append((r, mp, resid))
        print(f"{r['model']:<8}{iv:<6}{r['com_off_pct']:>+9.2f}{mp:>+9.2f}"
              f"{vp:>+9.2f}{resid:>+9.2f}{n:>6}")

    if len(com) < 4:
        sys.exit("\ntoo few points to regress")

    c, p = np.asarray(com), np.asarray(pred)
    slope, intercept = np.polyfit(p, c, 1)
    fit = slope * p + intercept
    ss_res = float(((c - fit) ** 2).sum())
    ss_tot = float(((c - c.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho = float(np.corrcoef(p, c)[0, 1])
    resid = c - fit

    print("\n=== COM OFFSET REGRESSED ON WINDOW MEAN PRICE ===")
    print(f"  n points        : {len(c)}")
    print(f"  slope           : {slope:.4f}   (1.0 = COM tracks the window mean exactly)")
    print(f"  intercept       : {intercept:+.4f}%")
    print(f"  Pearson r       : {rho:.4f}")
    print(f"  R^2             : {r2:.4f}")
    print(f"  residual sd     : {resid.std(ddof=1):.4f}%")
    print(f"  residual range  : {resid.min():+.2f}% .. {resid.max():+.2f}%")

    print("\n=== VERDICT ===")
    if r2 > 0.90 and abs(slope - 1) < 0.35:
        print(f"  LAGGING AVERAGE. R^2={r2:.3f}, slope={slope:.2f}: the centre of mass")
        print("  is essentially the mean price over the lookback window. The map")
        print("  SUMMARISES the past rather than forecasting. Forward capture cannot")
        print("  rescue a quantity that is a moving average by construction.")
        print(f"  Any real signal must live in the residual, sd {resid.std(ddof=1):.2f}%.")
    elif r2 > 0.50:
        print(f"  MOSTLY LAGGING. R^2={r2:.3f}. The window mean explains most of the")
        print("  offset, but a residual remains. Forward capture should score the")
        print("  RESIDUAL, not the raw location.")
    else:
        print(f"  NOT EXPLAINED. R^2={r2:.3f}: the offset is not merely a lookback")
        print("  average. The forward study is justified as designed.")

    print("\n  Caveat: one instant, one coin. This tests the map's LOCATION only --")
    print("  its SHAPE could still concentrate liquidations usefully even if its")
    print("  centre is a moving average. Shape needs forward data regardless.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
