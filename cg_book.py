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

MEASURED COVERAGE LIMIT (2026-09-02 01:06Z, limit=1000, Binance's maximum):

    BTCUSDT  book spans -0.16% / +0.18%
    ETHUSDT  book spans -0.44% / +0.45%
    SOLUSDT  book spans -10.04% / +10.02%

On BTC and ETH the visible book is FAR narrower than the magnet range, and no
parameter raises it -- 1000 is the API ceiling. SOL reaches +/-10% only because
its book is thin, so wide coverage there is a symptom of illiquidity, not a
bonus.

This is survivable because the question is about the book WHEN PRICE ARRIVES at
a magnet, and price must travel there to trigger anything. Sampling at 5-minute
cadence keeps the path covered: BTC typically moves less than the band width
between samples. Analysis must filter to samples where the level of interest
fell INSIDE book_span, and treat every other sample as unmeasured.

Uses Binance USD-M futures public depth. No API key, so no key/IP whitelist
exposure on the VM, and no interaction with the trading account whatsoever.
This script is READ-ONLY market data: it cannot place, modify or cancel
anything.

Usage:
    python3 cg_book.py                       # BTC, ETH, SOL
    python3 cg_book.py --symbols BTCUSDT

Crontab (VM, UTC) - every 5 minutes:
    */5 * * * * cd ~/cgstudy && /usr/bin/python3 cg_book.py >> book.log 2>&1
"""

import argparse, json, os, sys, time
from datetime import datetime, timezone
from urllib import request

DEPTH = "https://fapi.binance.com/fapi/v1/depth"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

BIN_BPS = 1.0          # bin width in basis points of mid
SPAN_PCT = 1.0         # bins span +/- this much around mid
BANDS = [0.05, 0.1, 0.25, 0.5, 1.0]   # cumulative depth at these % from mid


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

    # How far the returned book ACTUALLY reaches. limit=1000 is Binance's max
    # and on BTC covers only ~+/-0.17%, so most bands are UNMEASURABLE.
    span_dn = (mid - bids[-1][0]) / mid * 100
    span_up = (asks[-1][0] - mid) / mid * 100

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

    def cum(arr, pct, upward, span):
        """USD notional within pct of mid on one side, or None if UNMEASURED.

        FIXED 2026-09-02: previously returned a number for every band. On BTC,
        where the book spans 0.16%, the '0.5%' band summed only what was
        visible and reported it as 0.5% depth -- understating true depth by an
        unknown factor while looking authoritative. A band wider than the
        returned book is now None: unmeasured, not thin. Scoring a truncated
        book as empty would manufacture the exact finding being tested for.
        """
        if pct > span:
            return None
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

    # Sparse: only non-empty bins. At 1bp resolution over +/-1% most cells are
    # empty on any book, and storing 2000 zeros per coin per sample at 5-minute
    # cadence is pure waste.
    nz_bid = {i: round(v, 2) for i, v in enumerate(bid_usd) if v > 0}
    nz_ask = {i: round(v, 2) for i, v in enumerate(ask_usd) if v > 0}

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
        "cum_bid_usd": {str(p): cum(bid_usd, p, False, span_dn) for p in BANDS},
        "cum_ask_usd": {str(p): cum(ask_usd, p, True, span_up) for p in BANDS},
        "walls_bid": walls(bid_usd),
        "walls_ask": walls(ask_usd),
        "bid_bins": nz_bid,
        "ask_bins": nz_ask,
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
