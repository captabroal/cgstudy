#!/usr/bin/env python3
"""
cg_asym.py - does the heatmap's long/short imbalance predict DIRECTION?

This is the highest-value remaining test and the cheapest to run. Density
scoring asks "where will liquidations land" and needs ~100 windows. Direction
is a sign test: it needs far fewer, so it may report at the n=15 pilot rather
than late September. And if it works it is tradeable even though the map's
centre of mass is a moving average (Amendment 1), because a directional signal
does not depend on locating a level.

THE TRAP THIS SCRIPT EXISTS TO AVOID
------------------------------------
Raw asymmetry is contaminated by construction. The map centres on the window
mean, so whenever spot sits BELOW that mean, more mass mechanically sits above
spot and asymmetry prints positive -- with no liquidation information involved
at all. Raw asymmetry is therefore largely a restatement of "spot vs its own
moving average", i.e. a mean-reversion signal anyone can compute for free.

So two quantities are scored separately:

  raw       = (mass above spot - mass below spot) / total
  expected  = the same statistic computed on the MA-CENTRED NULL alone
  residual  = raw - expected

If RAW predicts and RESIDUAL does not, the signal is mean reversion and
CoinGlass adds nothing. Only RESIDUAL skill is evidence of a real edge.

Usage:
    python3 cg_asym.py --coin BTC --model model1 --interval 24h --horizon 6
    python3 cg_asym.py --coin BTC --model model1 --interval 24h --all-horizons
"""

import argparse, glob, json, os, sys, time
from urllib import request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cgscore as S
from cg_decode import load, classify, current_column

KL = "https://api.binance.com/api/v3/klines"
HOURS = {"12h": 12, "24h": 24, "48h": 48, "3d": 72,
         "1w": 168, "2w": 336, "1mo": 720, "3mo": 2160}
PAIR = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}


def klines(symbol, interval, start_ms, end_ms, limit=1000):
    url = (f"{KL}?symbol={symbol}&interval={interval}"
           f"&startTime={int(start_ms)}&endTime={int(end_ms)}&limit={limit}")
    with request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def price_series(symbol, t0_ms, t1_ms):
    """Hourly closes spanning [t0, t1], paged."""
    out, cur = [], t0_ms
    while cur < t1_ms:
        batch = klines(symbol, "1h", cur, t1_ms)
        if not batch:
            break
        out += batch
        nxt = batch[-1][0] + 3600_000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.25)
    if not out:
        return np.empty((0, 2))
    return np.array([[float(k[0]), float(k[4])] for k in out])


def at(series, ts_ms):
    """Close of the last bar at or before ts_ms."""
    if series.size == 0:
        return None
    i = np.searchsorted(series[:, 0], ts_ms, side="right") - 1
    return float(series[i, 1]) if i >= 0 else None


def asym_of(logp, widths, y, spot):
    """(mass above spot - mass below) / total, for any log-density."""
    mass = np.exp(logp) * widths
    above = mass[y > spot].sum()
    below = mass[y < spot].sum()
    tot = above + below
    return float((above - below) / tot) if tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="BTC")
    ap.add_argument("--model", default="model1")
    ap.add_argument("--interval", default="24h")
    ap.add_argument("--horizon", type=float, default=6.0, help="hours")
    ap.add_argument("--all-horizons", action="store_true")
    ap.add_argument("--snapdir", default="./snapshots")
    args = ap.parse_args()

    pat = os.path.join(args.snapdir, "*", f"{args.coin}_{args.model}_{args.interval}.json")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no snapshots matching {pat}")

    horizons = [6.0, 12.0, 24.0] if args.all_horizons else [args.horizon]
    lookback_h = HOURS.get(args.interval, 24)
    sym = PAIR.get(args.coin, args.coin + "USDT")

    # Parse every snapshot first, then fetch one price series covering all of
    # them plus the longest lookback and horizon.
    snaps = []
    for f in files:
        try:
            item = load(f)
            y = np.asarray(item["y_axis"], float)
            enc, _ = classify(item["liquidation_leverage_data"], y.size)
            prof = current_column(item["liquidation_leverage_data"], enc, y.size)
            if prof is None or prof.sum() <= 0:
                continue
            snaps.append({"ut": float(item["updateTime"]), "y": y, "prof": prof})
        except Exception:                                     # noqa: BLE001
            continue
    if not snaps:
        sys.exit("no parseable snapshots")
    snaps.sort(key=lambda s: s["ut"])

    t0 = snaps[0]["ut"] - (lookback_h + 2) * 3600_000
    t1 = snaps[-1]["ut"] + (max(horizons) + 2) * 3600_000
    print(f"{len(snaps)} snapshots; fetching {sym} 1h closes...")
    px = price_series(sym, t0, min(t1, time.time() * 1000))
    if px.size == 0:
        sys.exit("no price data")

    for H in horizons:
        rows = []
        for s in snaps:
            spot = at(px, s["ut"])
            fwd = at(px, s["ut"] + H * 3600_000)
            if spot is None or fwd is None or s["ut"] + H * 3600_000 > px[-1, 0]:
                continue    # horizon has not matured yet
            win = px[(px[:, 0] >= s["ut"] - lookback_h * 3600_000) & (px[:, 0] <= s["ut"])]
            if win.shape[0] < 3:
                continue
            wmean = float(win[:, 1].mean())

            lp, edges, widths = S.build_density(s["prof"], s["y"])
            raw = asym_of(lp, widths, s["y"], spot)
            # Expected asymmetry from the MA-centred null ALONE. Scale is set to
            # the map's own dispersion so the null is a like-for-like stand-in.
            mass = np.exp(lp) * widths
            com = float((mass * s["y"]).sum())
            sd = float(np.sqrt((mass * (s["y"] - com) ** 2).sum()))
            lp_ma, _, w_ma = S.ma_density(s["y"], wmean, max(sd / np.sqrt(2), 1e-6))
            exp = asym_of(lp_ma, w_ma, s["y"], spot)

            rows.append({"ut": s["ut"], "spot": spot,
                         "ret": (fwd / spot - 1) * 100,
                         "raw": raw, "exp": exp, "res": raw - exp})

        print(f"\n{'='*62}\nHORIZON {H:g}h  -  {args.coin}/{args.model}/{args.interval}\n{'='*62}")
        if len(rows) < 5:
            print(f"  {len(rows)} matured windows. Need >=5 to report, >=30 for a")
            print("  screening verdict. Capture is still accumulating.")
            continue

        ret = np.array([r["ret"] for r in rows])
        for label, key in (("RAW  (contaminated by mean reversion)", "raw"),
                           ("RESIDUAL (the only honest one)", "res")):
            a = np.array([r[key] for r in rows])
            ok = np.isfinite(a) & np.isfinite(ret) & (a != 0) & (ret != 0)
            if ok.sum() < 5:
                print(f"\n  {label}: too few usable points")
                continue
            agree = np.sign(a[ok]) == np.sign(ret[ok])
            p, pos, tot = S.sign_test(np.where(agree, 1.0, -1.0))
            rho = float(np.corrcoef(a[ok], ret[ok])[0, 1])
            mu, lo, hi, n = S.bootstrap_ci(np.where(agree, 1.0, 0.0))
            print(f"\n  {label}")
            print(f"    n windows        : {int(ok.sum())}")
            print(f"    direction hit    : {agree.mean()*100:.1f}%  "
                  f"95% CI [{lo*100:.1f}%, {hi*100:.1f}%]")
            print(f"    sign test p      : {p:.4f}  ({pos}/{tot} agree)")
            print(f"    corr with return : {rho:+.4f}")
            if lo > 0.5:
                print("    => directional skill, CI excludes coin-flip")
            elif hi < 0.5:
                print("    => INVERTED: fading this signal beat following it")
            else:
                print("    => not distinguishable from a coin flip at this n")

        print("\n  READ THIS BEFORE ACTING: if RAW shows skill and RESIDUAL does")
        print("  not, the signal is spot-versus-its-own-moving-average. That is")
        print("  free, public, and owes nothing to CoinGlass. Only RESIDUAL")
        print("  skill is evidence of an edge in the liquidation data itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
