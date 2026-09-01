"""
cgscore.py — scoring harness for the CoinGlass liquidation heatmap study.

Implements the metric battery pre-registered in the research prompt (§9, §10),
as amended 2026-09-01 to add the moving-average-centred null (AMENDMENT 1 in
AMENDMENTS.md).

Pure functions over numpy arrays; no I/O, no network. Validated by
test_cgscore.py — run it before every analysis session.

Conventions
-----------
All densities live on the SAME support, [edges[0], edges[-1]] derived from
y_axis. Comparing densities on different supports is invalid, so every Laplace
null is truncated to this support and renormalised.

Bin widths are NOT assumed uniform. CoinGlass model2 returns non-uniform axes on
some intervals (measured 2026-09-01 at 12h/24h/1mo), so every density carries an
explicit per-cell width array.

Densities are computed and returned in LOG space. A Laplace with a small scale
over a wide support underflows to exactly zero in linear space, which produced
divide-by-zero on model2/3mo. Log space removes the failure mode rather than
clamping around it.

All gains are in nats per event. Positive = the map beats the null.
"""

import numpy as np

EPS_MIX = 0.01  # pre-declared smoothing weight; see build_density


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def bin_edges(y_axis):
    """Cell edges and widths for bin CENTRES y_axis. Handles non-uniform axes.

    Interior edges are midpoints between adjacent centres; the outer two
    extrapolate by half the adjacent spacing.
    """
    y = np.asarray(y_axis, dtype=float)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("y_axis must be 1-D with >= 2 levels")
    if np.any(np.diff(y) <= 0):
        raise ValueError("y_axis must be strictly increasing")
    inner = (y[:-1] + y[1:]) / 2.0
    edges = np.concatenate([[y[0] - (y[1] - y[0]) / 2.0],
                            inner,
                            [y[-1] + (y[-1] - y[-2]) / 2.0]])
    return edges, np.diff(edges)


def _logsumexp(a):
    m = np.max(a)
    return m + np.log(np.exp(a - m).sum())


# ----------------------------------------------------------------------------
# Densities — all return (log_density, edges, widths)
# ----------------------------------------------------------------------------

def build_density(intensity, y_axis, eps=EPS_MIX):
    """Raw intensity profile -> log probability DENSITY over price.

    Smoothing is mandatory and pre-declared: p = (1-eps)*p_map + eps*uniform.
    Without it a single realised event on a zero-intensity cell yields log(0)
    and one event destroys the study. eps is a declared constant, never tuned.
    """
    w_raw = np.asarray(intensity, dtype=float)
    y = np.asarray(y_axis, dtype=float)
    if w_raw.shape != y.shape:
        raise ValueError("intensity and y_axis must have the same shape")
    if np.any(w_raw < 0):
        raise ValueError("negative intensity")
    edges, widths = bin_edges(y)
    total = w_raw.sum()
    if total <= 0:
        raise ValueError("intensity sums to zero")
    p_mass = w_raw / total
    u_mass = widths / widths.sum()      # uniform in DENSITY => mass ~ width
    mixed = (1.0 - eps) * p_mass + eps * u_mass
    with np.errstate(divide="ignore"):
        logp = np.log(mixed) - np.log(widths)
    return logp, edges, widths


def uniform_density(y_axis):
    edges, widths = bin_edges(y_axis)
    span = edges[-1] - edges[0]
    return np.full(np.asarray(y_axis).size, -np.log(span)), edges, widths


def laplace_density(y_axis, centre, scale):
    """Laplace(centre, b=scale), truncated to the support and renormalised.

    Computed entirely in log space: a tight scale over a wide support underflows
    to zero in linear space. Truncation matters — an untruncated Laplace leaks
    mass outside the support the map lives on, handing the map a free advantage.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    y = np.asarray(y_axis, dtype=float)
    edges, widths = bin_edges(y)
    logf = -np.abs(y - centre) / scale
    log_mass = logf + np.log(widths)
    log_mass -= _logsumexp(log_mass)
    return log_mass - np.log(widths), edges, widths


def nearspot_density(y_axis, spot, atr, k):
    """Pre-registered near-spot null: Laplace(spot, b=k*atr)."""
    if atr <= 0 or k <= 0:
        raise ValueError("atr and k must be positive")
    return laplace_density(y_axis, spot, k * atr)


def ma_density(y_axis, window_mean, scale):
    """AMENDMENT 1 (2026-09-01) — moving-average-centred null.

    Measured on the 2026-09-01 sweep: the map's centre of mass equals the mean
    price over its own lookback window with residual sd ~0.045% across the
    24h-3d band, and -0.00% on the pre-registered primary config
    (BTC/model1/24h).

    Mechanically it must: positions opened during the window have an average
    entry near the window mean, and liquidation levels sit a fixed
    leverage-distance from entry, so the map's centre restates average entry.

    A spot-centred null therefore credits the map for an offset that is just a
    trailing average — free, public, non-proprietary. This null removes that
    credit. If the map beats uniform, beats near-spot, then loses to THIS, the
    edge was a moving average.
    """
    return laplace_density(y_axis, window_mean, scale)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def _index_of(events, edges):
    """Bin index per event; -1 outside the support."""
    e = np.atleast_1d(np.asarray(events, dtype=float))
    idx = np.searchsorted(edges, e, side="right") - 1
    idx = np.where(idx >= edges.size - 1, edges.size - 2, idx)
    idx = np.where((e < edges[0]) | (e > edges[-1]), -1, idx)
    return idx


def mean_loglik(logp, edges, events):
    """Mean log-density at realised event prices -> (value, n_in, n_out)."""
    idx = _index_of(events, edges)
    inside = idx >= 0
    n_out = int((~inside).sum())
    if inside.sum() == 0:
        return np.nan, 0, n_out
    return float(np.mean(logp[idx[inside]])), int(inside.sum()), n_out


def score_window(intensity, y_axis, events, spot, atr, k=8.0, eps=EPS_MIX,
                 window_mean=None, ma_scale=None):
    """Score one frozen snapshot against its realised events.

    Out-of-support events are excluded from numerator and denominator and
    reported separately, never silently dropped: a high out-of-support rate
    means the map's ~+/-5% window is too narrow for the horizon, itself a
    finding.

    Pass window_mean (and optionally ma_scale, default k*atr) to also score the
    MA-centred null. Without it that field is NaN — and on the 24h-3d band it is
    the null that decides the question.
    """
    lp_map, edges, widths = build_density(intensity, y_axis, eps)
    lp_uni, _, _ = uniform_density(y_axis)
    lp_ns, _, _ = nearspot_density(y_axis, spot, atr, k)

    ll_map, n_in, n_out = mean_loglik(lp_map, edges, events)
    ll_uni, _, _ = mean_loglik(lp_uni, edges, events)
    ll_ns, _, _ = mean_loglik(lp_ns, edges, events)

    ll_ma = np.nan
    if window_mean is not None:
        lp_ma, _, _ = ma_density(y_axis, window_mean,
                                 ma_scale if ma_scale else k * atr)
        ll_ma, _, _ = mean_loglik(lp_ma, edges, events)

    return {
        "n_events_in": n_in,
        "n_events_out": n_out,
        "out_of_support_rate": n_out / max(n_in + n_out, 1),
        "ll_map": ll_map,
        "ll_uniform": ll_uni,
        "ll_nearspot": ll_ns,
        "ll_ma": ll_ma,
        "nll_gain_vs_uniform": ll_map - ll_uni,
        "nll_gain_vs_nearspot": ll_map - ll_ns,
        "nll_gain_vs_ma": ll_map - ll_ma,
        "coverage": coverage(lp_map, widths),
        "sharp_lift": sharp_lift(lp_map, edges, widths, events, budget=0.30),
        "k": k, "eps": eps, "window_mean": window_mean,
    }


def coverage(logp, widths, frac=0.99):
    """Fraction of cells holding the smallest set carrying `frac` of the mass.

    Quote beside every hit rate. A map covering 97% of the range shows a ~1.0
    hit rate by construction; coverage is what makes that visible.
    """
    mass = np.sort(np.exp(logp) * widths)[::-1]
    return float((np.searchsorted(np.cumsum(mass), frac) + 1) / mass.size)


def sharp_lift(logp, edges, widths, events, budget=0.30):
    """Share of realised events in the top-`budget` of mass, over mass kept.

    CRITICAL INTERPRETATION — a CALIBRATION metric, not a skill metric:

      If events are drawn from the map's own density, the top 30% of mass
      captures exactly 30% of events in expectation. sharp_lift == 1.0 is an
      IDENTITY for a perfectly calibrated map, not evidence of no skill.

      > 1  => events MORE concentrated in hot cells than the map claims: the
              map is directionally right but UNDER-CONFIDENT.
      < 1  => OVER-CONFIDENT.

    Concentration relative to the full range is nll_gain_vs_uniform, not this.
    Never quote sharp_lift as the skill claim.
    """
    mass = np.exp(logp) * widths
    order = np.argsort(mass)[::-1]
    keep, acc = np.zeros(mass.size, bool), 0.0
    for i in order:
        if acc >= budget:
            break
        keep[i] = True
        acc += mass[i]
    idx = _index_of(events, edges)
    inside = idx >= 0
    if inside.sum() == 0 or acc <= 0:
        return np.nan
    # Normalise by mass ACTUALLY kept, not the nominal budget. Greedy selection
    # overshoots (the last cell is taken whole); dividing by the nominal 0.30
    # inflated the statistic by +7% on a peaked map, which would read as lift.
    return float(keep[idx[inside]].mean() / acc)


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def bootstrap_ci(deltas, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI on the mean of per-window deltas."""
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=float)
    if d.size < 2:
        return (float(d.mean()) if d.size else np.nan), np.nan, np.nan, d.size
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi), d.size


def sign_test(deltas):
    """Two-sided exact sign test. Guards against a mean carried by a few tails."""
    from scipy.stats import binomtest
    d = np.asarray([x for x in deltas if np.isfinite(x) and x != 0], float)
    if d.size == 0:
        return np.nan, 0, 0
    pos = int((d > 0).sum())
    return float(binomtest(pos, d.size, 0.5).pvalue), pos, int(d.size)


def required_n(sd, effect, alpha=0.05):
    """Windows needed for a two-sided CI at `effect` to exclude zero."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return int(np.ceil((z * sd / effect) ** 2))


def benjamini_hochberg(pvals, q=0.10):
    """BH step-up. Boolean rejection mask in input order."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    rej = np.zeros(m, bool)
    if passed.any():
        rej[order[: np.max(np.where(passed)[0]) + 1]] = True
    return rej


def profile_max_scale(y_axis, events, centre, grid=None):
    """Best-fitting Laplace scale about `centre` — the null's best shot.

    Pre-declaring a scale protects against tuning the null to flatter the map,
    but also lets the null be weak by accident. The honest baseline is the BEST
    baseline, so report this beside the pre-declared one.
    """
    y = np.asarray(y_axis, dtype=float)
    span = y[-1] - y[0]
    grid = grid if grid is not None else np.geomspace(span / 500, span * 2, 80)
    best_s, best_ll = np.nan, -np.inf
    for s in grid:
        lp, edges, _ = laplace_density(y, centre, s)
        ll, n_in, _ = mean_loglik(lp, edges, events)
        if n_in and np.isfinite(ll) and ll > best_ll:
            best_ll, best_s = ll, float(s)
    return best_s, float(best_ll)


def normalised_kl_uniform(logp, widths):
    """KL(map||uniform) divided by log(n_cells) — comparable ACROSS grids.

    Raw KL is not comparable between models: model1 uses ~132 price levels and
    model3 ~391 (measured 2026-09-01), and more bins mechanically permit higher
    entropy. Dividing by log(n) puts every model on [0, 1].
    """
    mass = np.exp(logp) * widths
    mass = mass[mass > 0]
    H = float(-(mass * np.log(mass)).sum())
    n = logp.size
    return float((np.log(n) - H) / np.log(n))
