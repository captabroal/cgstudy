#!/usr/bin/env python3
"""
cg_asym.py - does the heatmap's long/short imbalance predict DIRECTION?

Direction is a sign test, so it needs far fewer windows than density scoring.
It is also tradeable even though the map's centre of mass is a moving average
(Amendment 1), because a directional call does not depend on locating a level.

THE TRAP THIS SCRIPT EXISTS TO AVOID
------------------------------------
Raw asymmetry is contaminated by construction. The map centres on the window
mean, so whenever spot sits BELOW that mean, more mass mechanically sits above
spot and asymmetry prints positive -- with no liquidation information involved.
Raw asymmetry is therefore largely a restatement of "spot vs its own moving
average", a mean-reversion signal anyone can compute for free.

  raw       = (mass above spot - mass below spot) / total
  expected  = the same statistic on the MA-CENTRED NULL alone
  residual  = raw - expected

If RAW predicts and RESIDUAL does not, the signal is mean reversion and
CoinGlass adds nothing. Only RESIDUAL skill is evidence of a real edge.

STATISTICAL FIXES
-----------------
2026-09-02a
1. NON-OVERLAP. Snapshots are 6h apart, so at a 12h horizon adjacent windows
   share 6h of price path and are NOT independent. The first live run scored
   all of them and reported "0/6, p=0.031, INVERTED". Thinned that is 0/3,
   exact CI [0%, 70.8%] -- no finding at all.
2. EXACT BINOMIAL CI. Percentile bootstrap on a few binary points is
   unreliable, and on an all-identical sample returns a ZERO-WIDTH interval
   that reads as certainty but is degeneracy. Clopper-Pearson is exact.

2026-09-02b
3. GAP TOLERANCE. updateTime is CoinGlass's stamp, not the cron clock, so real
   6h gaps land either side of the mark -- measured 6.0023h, 6.0089h, 5.9894h.
   A strict >= horizon test discarded the 5.9894h window over 38 seconds of
   jitter, silently halving the sample and doubling every timeline. Tolerance
   is now 300s.
4. EXCLUDE _sweep DIRECTORIES. The one-off structural sweep and the manual
   --tier primary tests sit minutes apart and are not part of the scheduled
   forward series; mixing them in made the cadence look irregular.

Usage:
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

N_SCREEN = 30       # pre-registered screening n
N_CONFIRM = 100     # pre-registered confirmation n
TOL_SEC = 300       # gap tolerance: jitter must not cost a window


def exact_binomial_ci(successes, trials, alpha=0.05):
    """Clopper-Pearson interval. Exact at any n, including 0/n and n/n."""
    from scipy.stats import beta
    if trials == 0:
        return float("nan"), float("nan")
    lo = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, trials - successes + 1)
    hi = 1.0 if successes == trials else beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return float(lo), float(hi)


def klines(symbol, interval, start_ms, end_ms, limit=1000):
    url = (f"{KL}?symbol={symbol}&interval={interval}"
           f"&startTime={int(start_ms)}&endTime={int(end_ms)}&limit={limit}")
    with request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def price_series(symbol, t0_ms, t1_ms):
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
    if series.size == 0:
        return None
    i = np.searchsorted(series[:, 0], ts_ms, side="right") - 1
    return float(series[i, 1]) if i >= 0 else None


def asym_of(logp, widths, y, spot):
    mass = np.exp(logp) * widths
    above, below = mass[y > spot].sum(), mass[y < spot].sum()
    tot = above + below
    return float((above - below) / tot) if tot > 0 else np.nan


def thin_nonoverlapping(rows, horizon_h, tol_sec=TOL_SEC):
    """Greedy: keep windows at least one horizon apart, minus a jitter allowance."""
    need = horizon_h * 3600_000 - tol_sec * 1000
    kept, last = [], -np.inf
    for r in sorted(rows, key=lambda x: x["ut"]):
        if r["ut"] - last >= need:
            kept.append(r)
            last = r["ut"]
    return kept


def report(label, a, ret):
    ok = np.isfinite(a) & np.isfinite(ret) & (a != 0) & (ret != 0)
    print(f"\n  {label}")
    if ok.sum() < 5:
        print(f"    only {int(ok.sum())} usable points - nothing to report")
        return
    agree = np.sign(a[ok]) == np.sign(ret[ok])
    k, n = int(agree.sum()), int(ok.sum())
    p, _, _ = S.sign_test(np.where(agree, 1.0, -1.0))
    lo, hi = exact_binomial_ci(k, n)
    rho = float(np.corrcoef(a[ok], ret[ok])[0, 1])
    print(f"    independent n    : {n}")
    print(f"    direction hit    : {k}/{n} = {k/n*100:.1f}%   exact 95% CI "
          f"[{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"    sign test p      : {p:.4f}")
    print(f"    corr with return : {rho:+.4f}")
    if n < N_SCREEN:
        print(f"    => UNDERPOWERED (n={n} < {N_SCREEN}). Not a result at any p-value.")
    elif lo > 0.5:
        print("    => directional skill; exact CI excludes a coin flip")
    elif hi < 0.5:
        print("    => INVERTED; fading this beat following it")
    else:
        print("    => not distinguishable from a coin flip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="BTC")
    ap.add_argument("--model", default="model1")
    ap.add_argument("--interval", default="24h")
    ap.add_argument("--horizon", type=float, default=6.0)
    ap.add_argument("--all-horizons", action="store_true")
    ap.add_argument("--snapdir", default="./snapshots")
    ap.add_argument("--include-adhoc", action="store_true",
                    help="include _sweep and manual pulls (not the scheduled series)")
    args = ap.parse_args()

    pat = os.path.join(args.snapdir, "*", f"{args.coin}_{args.model}_{args.interval}.json")
    files = sorted(glob.glob(pat))
    if not args.include_adhoc:
        files = [f for f in files if "_sweep" not in os.path.basename(os.path.dirname(f))]
    if not files:
        sys.exit(f"no snapshots matching {pat}")

    horizons = [6.0, 12.0, 24.0] if args.all_horizons else [args.horizon]
    lookback_h = HOURS.get(args.interval, 24)
    sym = PAIR.get(args.coin, args.coin + "USDT")

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
                continue
            win = px[(px[:, 0] >= s["ut"] - lookback_h * 3600_000) & (px[:, 0] <= s["ut"])]
            if win.shape[0] < 3:
                continue
            wmean = float(win[:, 1].mean())
            lp, edges, widths = S.build_density(s["prof"], s["y"])
            raw = asym_of(lp, widths, s["y"], spot)
            mass = np.exp(lp) * widths
            com = float((mass * s["y"]).sum())
            sd = float(np.sqrt((mass * (s["y"] - com) ** 2).sum()))
            lp_ma, _, w_ma = S.ma_density(s["y"], wmean, max(sd / np.sqrt(2), 1e-6))
            exp = asym_of(lp_ma, w_ma, s["y"], spot)
            rows.append({"ut": s["ut"], "ret": (fwd / spot - 1) * 100,
                         "raw": raw, "res": raw - exp})

        n_all = len(rows)
        rows = thin_nonoverlapping(rows, H)
        print(f"\n{'='*64}\nHORIZON {H:g}h  -  {args.coin}/{args.model}/{args.interval}\n{'='*64}")
        print(f"  matured windows {n_all} -> {len(rows)} after non-overlap thinning")
        if len(rows) < 5:
            need_d = (N_SCREEN - len(rows)) * H / 24.0
            print(f"  Too few to report. Screening needs {N_SCREEN} (~{need_d:.0f} more days).")
            continue
        ret = np.array([r["ret"] for r in rows])
        report("RAW  (contaminated by mean reversion)",
               np.array([r["raw"] for r in rows]), ret)
        report("RESIDUAL (the only honest one)",
               np.array([r["res"] for r in rows]), ret)
        print(f"\n  Thresholds: {N_SCREEN} independent windows to screen, "
              f"{N_CONFIRM} to move sizing.")
        print("  If RAW shows skill and RESIDUAL does not, the signal is")
        print("  spot-versus-its-own-moving-average: free, public, and owing")
        print("  nothing to CoinGlass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
