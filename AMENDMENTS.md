# Pre-registration Amendments

The pre-registration in the research prompt (§10) was frozen 2026-09-01 before
any outcome was scored. This file logs every subsequent change, with date and
reason, as that protocol requires. Amendments are appended, never edited.

---

## AMENDMENT 1 — 2026-09-01, ~18:00 UTC

### Adds a third null: Laplace centred on the trailing window mean

**Status of the original block: UNCHANGED.** The primary config
(BTC / model1 / 24h / 6h horizon), the pre-declared k = 8, the two-stage
thresholds and the stopping rule all stand. This amendment ADDS a baseline; it
weakens nothing and relaxes no threshold.

### Why this is legitimate rather than result-shopping

The change follows from a **structural property of the instrument**, established
from a single instant with no forward data and no outcome scored. Nothing about
the map's predictive performance was known when it was made. Had it been derived
from scoring results, it would not be admissible.

### The finding

`cg_sweep.py` captured all 24 BTC configs at 2026-09-01 17:55Z; `cg_anchor.py`
regressed each map's centre-of-mass offset from spot on the realised mean BTC
price over that config's own lookback window (Binance klines).

Pooled across all 21 analysable configs: slope 1.0827, Pearson r 0.9464,
R² 0.8957, residual sd 2.30%.

But the pooled figure understates it. In the 24h–3d band — the only intervals
relevant to a 6h horizon — the residuals are:

| model | interval | COM offset | window mean | residual |
| :--- | :--- | ---: | ---: | ---: |
| model1 | 24h | +1.27% | +1.27% | **−0.00%** |
| model1 | 48h | +1.12% | +1.25% | −0.12% |
| model1 | 3d | +1.22% | +1.28% | −0.06% |
| model2 | 48h | +1.26% | +1.25% | +0.01% |
| model2 | 3d | +1.29% | +1.28% | +0.00% |
| model3 | 48h | +1.25% | +1.25% | +0.01% |
| model3 | 3d | +1.30% | +1.28% | +0.02% |

Residual sd across those seven ≈ **0.045%**, about fifty times tighter than the
pooled 2.30%. R² is held below 0.90 entirely by 2w/1mo/3mo, which are irrelevant
to a 6h horizon. In the band that matters this is not a correlation; it is an
identity — and it holds across all three models, which is what rules out
coincidence.

**On the pre-registered primary config the residual is −0.00%.**

### Why it must be so

Positions opened during the lookback window have an average entry near the
window's mean price. Liquidation levels sit a fixed leverage-distance from
entry. So the map's centre of mass restates average entry price, which is a
trailing average of price. Nothing in that construction can look forward.

### What it breaks

The pre-registered null is Laplace centred on **spot**. The map centres on the
**window mean**. So part of the map's measured advantage over near-spot is
nothing but that offset — and the offset is a moving average: free, public,
computable by anyone, in no way proprietary to CoinGlass.

Demonstrated in controlled conditions by `test_cgscore.py` #18: a map centred on
the window mean beats the near-spot null by **+0.077 nats** while tying the
MA-centred null to 3.6e-15.

### The amendment

Add `nll_gain_vs_ma` — the map scored against Laplace(window_mean, b), truncated
to the same support, with b defaulting to k×ATR and the profile-maximised b
reported beside it.

Decision rule, declared now:

- Map beats uniform, beats near-spot, **beats MA** → a real edge beyond a
  moving average. Proceed to sizing rules.
- Map beats uniform, beats near-spot, **loses to or ties MA** → **the edge was a
  moving average.** No sizing weight. Report it plainly.
- Map loses to near-spot → already fatal under the original block.

### Scope — what this does NOT settle

This tests the map's **location** only. `KL(map‖uniform)` of 0.4–0.6 says the map
genuinely concentrates mass, and its **shape** could still pick liquidation
clusters usefully even with a moving-average centre. Shape requires forward data
regardless, so the capture schedule is unchanged and continues as planned.

---

## AMENDMENT 2 — 2026-09-01 — cross-model comparability

Grid resolution differs by model: **model1 ≈ 132 price levels, model2 ≈ 119,
model3 ≈ 391** (measured 2026-09-01). More bins mechanically permit higher
entropy, so raw `KL(map‖uniform)` is **not comparable across models** and the
sweep's `KLuni` column should not be read as a cross-model ranking.

All cross-model comparisons now use `normalised_kl_uniform`, which divides by
log(n_cells) and lands on [0, 1].

This also partly answers the study's original question early: **the three models
are not renderings of one estimate.** They differ in price resolution by roughly
3×. Whether they differ in *information* still needs the Phase B correlation
work.

---

## Defect fixes (not amendments — no bearing on the pre-registration)

1. **`sharp_lift` normalisation.** Divided by the nominal 0.30 budget while
   greedy cell selection overshoots it; inflated by +7% on a peaked map. Now
   divides by mass actually kept.
2. **Non-uniform axes.** model2 returns non-uniform `y_axis` at 12h/24h/1mo,
   which the scorer refused outright (3 of 24 configs lost). Per-cell widths are
   now carried explicitly.
3. **Log underflow.** A tight Laplace over a wide support underflowed to exactly
   zero, producing divide-by-zero on model2/3mo. All densities are now computed
   and returned in log space.

## Open items

- **model1 12h reads `KLuni` 0.022** — essentially flat, against 0.4–0.6 at
  neighbouring intervals. Either the 12h map is genuinely uninformative or the
  current-column extraction is wrong at that interval. Unresolved.
- Storage estimate corrected: payloads measured at **944 KB**, not 672 KB.
  Standard tier ≈ **136 MB/day, ~4.1 GB/month**.
