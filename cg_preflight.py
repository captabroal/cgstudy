#!/usr/bin/env python3
"""
cg_preflight.py -- best-case feasibility check. Run BEFORE committing to a month
of capture.

The decisive question is not "does the map beat uniform" -- it does, trivially,
by concentrating mass. It is "can the map beat a Laplace centred on spot".

This script answers the BEST CASE of that question using a single snapshot and
no forward data at all, via:

    KL(map || nearspot) = E_map[log p_map - log p_nearspot]

which is exactly the nll_gain_vs_nearspot you would measure IF realised events
followed the map's own density -- i.e. if the map were perfectly calibrated.
KL is non-negative, so this is an upper bound on what a CALIBRATED map can earn.
An under-confident map can beat it, but a well-calibrated one cannot.

Read it like this:

    min-over-b KL  >= 0.20   the map is genuinely distinct from near-spot;
                             the pre-registered Stage 1 threshold is reachable.
    min-over-b KL  0.05-0.20 marginal; lower the threshold and extend the sample
                             BEFORE capturing, not after.
    min-over-b KL  <  0.05   even a perfect map is nearly indistinguishable from
                             "price is near price". Do not spend a month.

The sweep is over the null's scale b in DOLLARS, so no ATR estimate is needed
and no assumption about k is smuggled in. The minimising b is the null's best
shot at impersonating the map.

Usage:
    python3 cg_preflight.py snapshots/<cycle>/BTC_model1_24h.json --spot 77541.45
"""

import argparse, json, sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import cgscore as S
from cg_decode import load, classify, current_column


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--spot", type=float, required=True,
                    help="spot price at the snapshot's updateTime")
    ap.add_argument("--eps", type=float, default=S.EPS_MIX)
    args = ap.parse_args()

    item = load(args.snapshot)
    y = np.asarray(item["y_axis"], dtype=float)
    enc, _ = classify(item["liquidation_leverage_data"], y.size)
    prof = current_column(item["liquidation_leverage_data"], enc, y.size)
    if prof is None or prof.sum() <= 0:
        sys.exit("no usable profile; run cg_decode.py first")

    p_map, edges, cell = S.build_density(prof, y, eps=args.eps)
    m_map = p_map * cell                                   # mass per cell
    spot = args.spot

    # --- geometry relative to spot -----------------------------------------
    lo_pct = (spot - y[0]) / spot * 100
    hi_pct = (y[-1] - spot) / spot * 100
    peak = y[int(np.argmax(prof))]
    com = float((m_map * y).sum())                          # centre of mass
    print("=== GEOMETRY vs SPOT ===")
    print(f"  spot            : {spot:,.2f}")
    print(f"  room below      : {spot - y[0]:,.2f}  ({lo_pct:.2f}%)")
    print(f"  room above      : {y[-1] - spot:,.2f}  ({hi_pct:.2f}%)")
    print(f"  spot at         : {(spot - y[0]) / (y[-1] - y[0]) * 100:.1f}% of range")
    print(f"  peak intensity  : {peak:,.2f}  ({(peak/spot - 1)*100:+.2f}% vs spot)")
    print(f"  centre of mass  : {com:,.2f}  ({(com/spot - 1)*100:+.2f}% vs spot)")

    # --- calibrated-case gain vs uniform -----------------------------------
    H = float(-(m_map[m_map > 0] * np.log(m_map[m_map > 0])).sum())
    kl_uni = float(np.log(y.size) - H)
    print("\n=== CALIBRATED-CASE GAIN vs UNIFORM ===")
    print(f"  entropy         : {H:.4f} nats   (uniform {np.log(y.size):.4f})")
    print(f"  KL(map||uniform): {kl_uni:.4f} nats")
    print(f"  effective cells : {np.exp(H):.1f} of {y.size}")
    print("  (this is the gain IF events follow the map exactly -- not a cap;")
    print("   an under-confident map can exceed it, a calibrated one cannot)")

    # --- the decisive sweep -------------------------------------------------
    print("\n=== CALIBRATED-CASE GAIN vs NEAR-SPOT, swept over null scale b ===")
    print(f"  {'b ($)':>9}  {'b as % spot':>12}  {'KL(map||ns)':>12}")
    bs = np.geomspace(50, 8000, 60)
    kls = []
    for b in bs:
        p_ns, _, _ = S.nearspot_density(y, spot, b, 1.0)     # atr=b, k=1
        m_ns = p_ns * cell
        kl = float((m_map * (np.log(m_map) - np.log(m_ns))).sum())
        kls.append(kl)
    kls = np.asarray(kls)
    for b, kl in list(zip(bs, kls))[::6]:
        print(f"  {b:9.0f}  {b/spot*100:11.2f}%  {kl:12.4f}")

    i = int(np.argmin(kls))
    b_star, kl_star = float(bs[i]), float(kls[i])
    print(f"\n  BEST NULL       : b = ${b_star:,.0f} ({b_star/spot*100:.2f}% of spot)")
    print(f"  MIN KL          : {kl_star:.4f} nats  <-- the number that decides this")

    print("\n=== VERDICT ===")
    if kl_star >= 0.20:
        print(f"  PROCEED. {kl_star:.4f} >= 0.20: the map is genuinely distinct from")
        print("  near-spot, and the pre-registered Stage 1 threshold is reachable.")
    elif kl_star >= 0.05:
        print(f"  MARGINAL. {kl_star:.4f} is in [0.05, 0.20). A perfect map barely")
        print("  clears the pre-registered +0.20 threshold, so that threshold is")
        print("  likely unreachable. Re-derive it and extend n BEFORE capturing.")
    else:
        print(f"  STOP. {kl_star:.4f} < 0.05: even a PERFECTLY calibrated map is")
        print("  nearly indistinguishable from 'price is near price'. A month of")
        print("  capture cannot produce a positive result. Do not spend it.")
    print("\n  Caveat: this is ONE snapshot at ONE interval. Re-run across models,")
    print("  intervals and coins before treating it as the study's conclusion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
