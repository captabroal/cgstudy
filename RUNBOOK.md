# CoinGlass Heatmap Study — Runbook & Part 1 Status

**Session date:** 2026-09-01 · **Actor:** `api_merge/coinglass-liquidation-heatmap`
**Status:** scorer built and validated · capture pipeline built · **encoding gate not yet cleared**

---

## 1 · What was established this session

### Schema drift — the documented output is wrong

The actor's inferred output schema advertises `price_candlesticks`, `limit`, `usedToday` and `date`.
**A live run returns none of them.** Actual fields, verified on run `4886lGMxs3tyHQv9N`:

```
success · message · updateTime · y_axis · liquidation_leverage_data
```

Two consequences:

- **No `price_candlesticks`.** Ground truth price must come entirely from the exchange MCP. This is
  cleaner than the documented behaviour — it removes any temptation to score a snapshot against candles
  shipped inside the same payload, which are never forward data.
- **No `limit` / `usedToday`.** There is **no quota telemetry**. The upstream daily cap, if one exists,
  is invisible until a pull fails. `cg_capture.py` therefore records every failure to the manifest
  rather than skipping it, so quota exhaustion shows up as a visible run of errors at a consistent
  time of day rather than as a silent gap.

### Measured facts

| Quantity | Value | How |
| :--- | :--- | :--- |
| Run latency | 2.7 s | run `4886lGMxs3tyHQv9N` |
| `updateTime` → pull lag | **2.1 s** | 02:33:04.916Z vs 02:33:07.042Z |
| `y_axis` levels | **134** | uniform |
| `y_axis` step | **$59.72** | BTC, 24h |
| `y_axis` span | $74,444.38 – $82,387.14 | $7,942.76 |
| Span as % of midpoint | **10.13%**, i.e. **±5.06%** | midpoint $78,415.76 |
| Payload size | **672,127 bytes** | dataset `jiLhIUICYAyO4zMqL` |

**The 2.1 s lag is a good result** — the heatmap is effectively live at pull time, so no dead time is
built into a frozen forecast. It is a single measurement; `cg_capture.py` logs `lag_sec` on every pull
so this becomes a distribution rather than an anecdote.

**The ±5.06% support is a constraint to design around.** Events outside that band are unscoreable.
`cgscore.score_window` counts them as `n_events_out` and reports `out_of_support_rate` rather than
dropping them silently. If that rate is material at the 6h horizon the map's window is too narrow, and
that is itself a finding.

### Why the encoding gate could not be cleared from chat

`liquidation_leverage_data` remains **unclassified**. Three independent routes were tried and all are
closed from a chat session:

1. Pulling it into context — 672 KB is roughly 170k tokens.
2. `curl` from the sandbox — `api.apify.com` is not on the egress allow-list.
3. `web_fetch` on the signed dataset URL — refused: the URL did not originate from a search result.

**This is why the pipeline must live on the VM.** It is an architectural finding, not a setback: the
analysis was always going to outgrow a chat session at 672 KB × 36 configs × 4 cycles/day ≈ **97 MB/day**.

---

## 2 · What was built and proven

### `cgscore.py` — the scoring harness

Implements the pre-registered battery: NLL gain vs uniform, NLL gain vs truncated near-spot Laplace,
`sharp_lift`, coverage, bootstrap CI, exact sign test, BH-FDR, power arithmetic, and the
profile-maximised null.

### `test_cgscore.py` — 20 synthetic tests, all passing

Run it before every analysis session. Three findings came out of writing it:

**(a) A real bug, now fixed.** `sharp_lift` divided by the *nominal* 0.30 budget while greedy cell
selection *overshoots* it — the last cell is taken whole. On a peaked map this inflated the statistic
by **+7%**. That would have read as genuine lift.

**(b) `sharp_lift == 1.0` means CALIBRATED, not skill-less.** If events are drawn from the map's own
density, the top 30% of mass captures exactly 30% of events — an identity. So:

- `sharp_lift > 1` → the map is directionally right but **under-confident** in its hot cells.
- `sharp_lift < 1` → **over-confident**.
- Concentration relative to the full price range is measured by `nll_gain_vs_uniform`, never by this.

This matters for reading the in-house engine too: its `sharp_lift` of 3.033 is a statement that
realised liquidations are ~3× more concentrated than the map's own mass allocation claims. That is a
calibration defect as much as a skill claim, and it is worth re-reading the desk doctrine with that in
mind.

**(c) Two tests were flaky by construction.** Single-draw assertions on a bootstrap CI and on a sign
test fail ~5% of the time *when the estimator is correct*. Both now test the property that matters —
coverage rate (95.0% over 300 trials) and false-positive rate (4.3% over 300 trials).

### `cg_capture.py` — cron-driven capture, four tiers

| Tier | Configs | $/cycle | $/30d @ 4/day |
| :--- | ---: | ---: | ---: |
| `full` | 72 | $0.72 | ~$86 |
| `standard` *(recommended)* | 36 | $0.36 | ~$43 |
| `lean` | 3 | $0.03 | ~$3.60 |
| `primary` | 1 | $0.01 | ~$1.20 |

Writes raw payloads under `snapshots/<cycleUTC>/` and appends a manifest row per pull with
`lag_sec`, axis geometry, payload size, and a SHA-256 prefix. Failures are recorded, never skipped.

### `cg_decode.py` — the Phase A gate

Classifies the grid as `dense_ty`, `dense_yt`, or `sparse_triplet`, extracts the current-column
profile, and exits non-zero if it cannot classify confidently. **Smoke-tested against synthetic
payloads of all three layouts at realistic size — all three detected correctly.**

---

## 3 · What you need to do

### Step 1 — Deploy

```bash
ssh <oracle-vm>
sudo mkdir -p /opt/cgstudy && sudo chown $USER /opt/cgstudy
git clone https://github.com/captabroal/cgstudy.git /opt/cgstudy
cd /opt/cgstudy
python3 -m pip install --user numpy scipy
python3 test_cgscore.py          # must print ALL TESTS PASSED
```

### Step 2 — Token (never in chat, never committed; `.gitignore` covers `.token`)

```bash
printf '%s' 'YOUR_APIFY_TOKEN' > /opt/cgstudy/.token
chmod 600 /opt/cgstudy/.token
```

### Step 3 — Clear the encoding gate

```bash
cd /opt/cgstudy
APIFY_TOKEN=$(cat .token) python3 cg_capture.py --tier primary
python3 cg_decode.py snapshots/*/BTC_model1_24h.json
```

This is the one blocker on everything downstream — the scorer cannot be pointed at real data until the
encoding is known. If it prints `GATE FAILED`, stop; do not proceed by guessing.

### Step 4 — Start capture (do this even before Step 3 resolves)

```bash
crontab -e
```

```cron
0 0,6,12,18 * * * cd /opt/cgstudy && APIFY_TOKEN=$(cat .token) /usr/bin/python3 cg_capture.py --tier standard >> capture.log 2>&1
```

**Confirm the VM clock is UTC** (`timedatectl`). A capture running on Doha local time will not align
with the in-house engine's 00/06/12/18Z freeze schedule, and the non-overlap guarantee is lost.

Storage: `standard` tier ≈ **97 MB/day, ~2.9 GB/month**. Check `df -h` before committing to a month.

### Step 5 — Refresh cadence (5 minutes, whenever convenient)

Run `--tier primary` three times ~10 minutes apart and compare `update_time_ms` in the manifest. If it
does not advance between pulls, capturing 4×/day buys nothing over the source's own refresh rate and
the cadence should be reconsidered.

---

## 4 · Part 1 exit checklist

Part 1 is complete when all of these are true. Until then, Part 2 has nothing to stand on.

- [ ] `test_cgscore.py` passes on the VM
- [ ] Encoding gate cleared; the `cg_decode.py` block is recorded in the data dictionary
- [ ] Cron installed, VM clock confirmed UTC, first two cycles landed with `status=OK`
- [ ] `lag_sec` distribution sane across a full cycle (expect ~2 s)
- [ ] `out_of_support_rate` estimated on a first scored window
- [ ] Disk headroom confirmed for 30 days
- [ ] Refresh-cadence test done
- [ ] Pre-registration block (research prompt §10) frozen and dated **before** any scoring

---

## 5 · What Part 2 will and will not be able to say

Non-overlapping windows accumulate at the horizon rate, so one month does **not** buy three answers:

| Horizon | Windows/day | n=30 (screening) | n=100 (confirmation) |
| :--- | ---: | ---: | ---: |
| **6h** *(primary)* | 4 | ~8 days | **~25 days** |
| 12h | 2 | ~15 days | ~50 days |
| 24h | 1 | ~30 days | ~100 days |

**A month gives one confirmed answer at 6h and two screening reads at 12h/24h.** Not three verdicts.

The per-window SD of 0.50 driving those figures is **borrowed from the in-house engine — assumed, not
observed, for CoinGlass**. The n=15 internal pilot (~4 days) exists to replace it. If CoinGlass's
dispersion is higher, n=100 becomes n=200 and the calendar doubles. That is the single largest
schedule risk in the plan.
