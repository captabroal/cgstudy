#!/usr/bin/env python3
"""
cg_sweep.py -- capture all BTC configs at one instant and tabulate the
pre-flight geometry across them.

The question this answers: the first snapshot put the map's centre of mass
+1.08% ABOVE spot. Is that offset structural (present across every model and
interval, therefore a real property of the instrument) or noise (flipping sign
between configs, therefore meaningless)?

If it is structural and positive while price is falling, that is the anchoring
bias of Trap 1 showing up directly -- the map's mass sits where price HAS been,
not where it is going.

Cost: 24 results = $0.24 for BTC alone; add --coins BTC,ETH,SOL for 72 = $0.72.

Usage:
    export APIFY_TOKEN=$(cat .token)
    python3 cg_sweep.py --spot 77541.45
    python3 cg_sweep.py --spot 77541.45 --coins BTC,ETH,SOL --spots 77541.45,2840,142
    python3 cg_sweep.py --spot 77541.45 --reuse snapshots/20260901T172000Z
"""

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cgscore as S
from cg_capture import pull, MODELS, INTERVALS
from cg_decode import load, classify, current_column


def analyse(item, spot, eps=S.EPS_MIX):
    y = np.asarray(item["y_axis"], dtype=float)
    enc, _ = classify(item["liquidation_leverage_data"], y.size)
    prof = current_column(item["liquidation_leverage_data"], enc, y.size)
    if prof is None or prof.sum() <= 0:
        return None
    p_map, edges, cell = S.build_density(prof, y, eps=eps)
    m = p_map * cell
    H = float(-(m[m > 0] * np.log(m[m > 0])).sum())
    com = float((m * y).sum())
    peak = float(y[int(np.argmax(prof))])

    bs = np.geomspace(50, 8000, 60)
    kls = []
    for b in bs:
        p_ns, _, _ = S.nearspot_density(y, spot, b, 1.0)
        m_ns = p_ns * cell
        kls.append(float((m * (np.log(m) - np.log(m_ns))).sum()))
    kls = np.asarray(kls)
    i = int(np.argmin(kls))
    return {
        "n_y": y.size,
        "kl_uniform": float(np.log(y.size) - H),
        "com_off_pct": (com / spot - 1) * 100,
        "peak_off_pct": (peak / spot - 1) * 100,
        "best_b": float(bs[i]),
        "best_b_pct": float(bs[i]) / spot * 100,
        "min_kl": float(kls[i]),
        "spot_pos_pct": (spot - y[0]) / (y[-1] - y[0]) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", type=float, required=True, help="BTC spot now")
    ap.add_argument("--coins", default="BTC")
    ap.add_argument("--spots", default="", help="comma list matching --coins")
    ap.add_argument("--outdir", default="./snapshots")
    ap.add_argument("--reuse", default="", help="analyse an existing cycle dir; no pulls")
    args = ap.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    spots = ([float(s) for s in args.spots.split(",")] if args.spots
             else [args.spot] * len(coins))
    if len(spots) != len(coins):
        sys.exit("--spots must have one value per coin in --coins")
    spot_of = dict(zip(coins, spots))

    if args.reuse:
        cycdir = args.reuse
    else:
        token = os.environ.get("APIFY_TOKEN")
        if not token:
            sys.exit("APIFY_TOKEN not set")
        from datetime import datetime, timezone
        cycdir = os.path.join(args.outdir,
                              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_sweep")
        os.makedirs(cycdir, exist_ok=True)
        total = len(coins) * len(MODELS) * len(INTERVALS)
        print(f"pulling {total} configs (~${total*0.01:.2f}) -> {cycdir}\n")
        n = 0
        for c in coins:
            for m in MODELS:
                for iv in INTERVALS:
                    n += 1
                    try:
                        raw, _ = pull(c, m, iv, token)
                        with open(os.path.join(cycdir, f"{c}_{m}_{iv}.json"), "wb") as f:
                            f.write(raw)
                        print(f"  [{n}/{total}] {c} {m} {iv} ok")
                    except Exception as e:                    # noqa: BLE001
                        print(f"  [{n}/{total}] {c} {m} {iv} FAILED: {e}")
                    time.sleep(1.0)
        print()

    hdr = (f"{'coin':<5}{'model':<8}{'intv':<6}{'n_y':>5}{'KLuni':>8}"
           f"{'COM%':>8}{'peak%':>8}{'bestb%':>8}{'minKL':>8}{'spotpos':>9}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for c in coins:
        for m in MODELS:
            for iv in INTERVALS:
                p = os.path.join(cycdir, f"{c}_{m}_{iv}.json")
                if not os.path.exists(p):
                    continue
                try:
                    r = analyse(load(p), spot_of[c])
                except Exception as e:                        # noqa: BLE001
                    print(f"{c:<5}{m:<8}{iv:<6}  analyse failed: {e}")
                    continue
                if r is None:
                    print(f"{c:<5}{m:<8}{iv:<6}  no usable profile")
                    continue
                r.update(coin=c, model=m, interval=iv)
                rows.append(r)
                print(f"{c:<5}{m:<8}{iv:<6}{r['n_y']:>5}{r['kl_uniform']:>8.3f}"
                      f"{r['com_off_pct']:>+8.2f}{r['peak_off_pct']:>+8.2f}"
                      f"{r['best_b_pct']:>8.2f}{r['min_kl']:>8.3f}"
                      f"{r['spot_pos_pct']:>8.1f}%")

    if not rows:
        sys.exit("\nno rows analysed")

    com = np.array([r["com_off_pct"] for r in rows])
    print("\n=== IS THE OFFSET STRUCTURAL? ===")
    print(f"  configs           : {len(rows)}")
    print(f"  COM offset mean   : {com.mean():+.3f}%   sd {com.std(ddof=1):.3f}%")
    print(f"  sign agreement    : {max((com>0).mean(), (com<0).mean())*100:.0f}% "
          f"({int((com>0).sum())} positive, {int((com<0).sum())} negative)")
    if (com > 0).all() or (com < 0).all():
        print("  => STRUCTURAL. Every config agrees on the sign. The map's mass is")
        print("     systematically displaced from spot. Test next whether it LEADS")
        print("     price (predictive) or LAGS it (anchoring bias, Trap 1).")
    elif max((com > 0).mean(), (com < 0).mean()) >= 0.75:
        print("  => MOSTLY CONSISTENT. Sign holds in the majority but not all.")
    else:
        print("  => NOISE. The sign flips across configs; no systematic displacement.")

    out = os.path.join(cycdir, "sweep.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")
    print("\nNOTE: one instant in time. A displacement seen now may simply be where")
    print("price traded earlier today. Only forward capture separates lead from lag.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
