#!/usr/bin/env python3
"""
cg_book.py - order-book depth capture. THE ONLY INPUT THAT CANNOT BE BACKFILLED.

Everything else in this study is recoverable after the fact: heatmap snapshots
are archived on Apify, and 1-minute price history backfills from Binance
whenever we want it. Resting liquidity is not. The book at 03:47 today is gone
at 03:48 and no API will ever return it.

That matters because it carries the mechanism question. A CoinGlass magnet is
tradeable only if something real sits at that price. Three outcomes are
distinguishable only with book history:

  - magnets coincide with genuine resting liquidity  -> a real market feature
  - magnets sit where the book is THIN               -> they mark air, and
                                                        price passes through
  - no relationship                                  -> the map is decoration

Uses Binance USD-M futures public depth. No API key, so no key/IP whitelist
exposure on the VM, and no interaction with the trading account whatsoever.
This script is READ-ONLY market data: it cannot place, modify or cancel
anything.

Usage:
    python3 cg_book.py                       # BTC, ETH, SOL
    python3 cg_book.py --symbols BTCUSDT

Crontab (VM, UTC) - every 15 minutes:
    */15 * * * * cd ~/cgstudy && /usr/bin/python3 cg_book.py >> book.log 2>&1
"""

import argparse, json, os, sys, time
from datetime import datetime, timezone
from urllib import request

DEPTH = "https://fapi.binance.com/fapi/v1/depth"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

BIN_BPS = 5.0          # bin width in basis points of mid
SPAN_PCT = 5.0         # bins span +/- this much around mid
BANDS = [0.25, 0.5, 1.0, 2.0]   # cumulative depth reported at these % from mid


def fetch_depth(symbol, limit=1000, timeout=20):
    url = f"{DEPTH}?symbol={symbol}&limit={limit}"
    with request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read()), datetime.now(timezone.utc)


def summarise(book, recv):
    bids = [(float(p), float(q)) for p, q in book["bids"] if float(q) > 0]
    asks = [(float(p), float(q)) for p, q in book["asks"] if float(q) > 0]
    if not bids or not asks:
        raise RuntimeError("empty book side")
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2.0

    bin_w = mid * BIN_BPS / 10000.0
    lo = mid * (1 - SPAN_PCT / 100.0)
    n_bins = int(round(2 * mid * SPAN_PCT / 100.0 / bin_w))

    bid_usd = [0.0] * n_bins
    ask_usd = [0.0] * n_bins
    for side, arr in ((bid_usd, bids), (ask_usd, asks)):
        for p, q in arr:
            i = int((p - lo) / bin_w)
            if 0 <= i < n_bins:
                side[i] += p * q

    def cum(arr, pct, upward):
        """USD notional within pct of mid on one side."""
        edge = mid * (1 + pct / 100.0) if upward else mid * (1 - pct / 100.0)
        i_mid = int((mid - lo) / bin_w)
        i_edge = int((edge - lo) / bin_w)
        a, b = (i_mid, min(i_edge + 1, len(arr))) if upward else (max(i_edge, 0), i_mid + 1)
        return round(sum(arr[a:b]), 2)

    def walls(arr, k=3):
        idx = sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)[:k]
        return [{"price": round(lo + (i + 0.5) * bin_w, 4),
                 "usd": round(arr[i], 2),
                 "pct_from_mid": round((lo + (i + 0.5) * bin_w) / mid * 100 - 100, 4)}
                for i in idx if arr[i] > 0]

    # How far the returned book ACTUALLY reaches. limit=1000 does not guarantee
    # +/-5% coverage; on a deep book it may span far less. Magnets beyond this
    # range are UNMEASURED, not measured-as-empty, and must never be scored as
    # "thin" on the strength of a truncated book.
    span_dn = (mid - bids[-1][0]) / mid * 100
    span_up = (asks[-1][0] - mid) / mid * 100

    return {
        "recv_utc": recv.isoformat(),
        "exchange_ts_ms": book.get("E"),
        "mid": round(mid, 6),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_bps": round((best_ask - best_bid) / mid * 10000, 4),
        "bin_lo": round(lo, 6),
        "bin_width": round(bin_w, 8),
        "n_bins": n_bins,
        "book_span_down_pct": round(span_dn, 4),
        "book_span_up_pct": round(span_up, 4),
        "n_levels_bid": len(bids),
        "n_levels_ask": len(asks),
        "cum_bid_usd": {str(p): cum(bid_usd, p, False) for p in BANDS},
        "cum_ask_usd": {str(p): cum(ask_usd, p, True) for p in BANDS},
        "walls_bid": walls(bid_usd),
        "walls_ask": walls(ask_usd),
        "bid_usd": [round(x, 2) for x in bid_usd],
        "ask_usd": [round(x, 2) for x in ask_usd],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--outdir", default="./books")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    ok = fail = 0

    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        rec = {"symbol": sym}
        try:
            book, recv = fetch_depth(sym)
            rec.update(summarise(book, recv))
            rec["status"] = "OK"
            ok += 1
        except Exception as e:                                # noqa: BLE001
            # Recorded, never skipped. A silent gap in a time series is
            # indistinguishable from data.
            rec.update(recv_utc=datetime.now(timezone.utc).isoformat(),
                       status=f"ERROR: {type(e).__name__}: {e}"[:200])
            fail += 1
        with open(os.path.join(args.outdir, f"{sym}_{day}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(0.3)

    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
          f"books ok={ok} fail={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
