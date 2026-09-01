"""
cgscore.py — scoring harness for the CoinGlass liquidation heatmap study.

Implements the metric battery pre-registered in the research prompt (§9, §10).
Pure functions over numpy arrays; no I/O, no network. Validated by test_cgscore.py.

Conventions
-----------
All densities are defined over the SAME support: the closed interval spanned by
y_axis, [y_lo, y_hi]. Comparing densities on different supports is invalid, so
the near-spot Laplace is truncated to this support and renormalised.

All gains are in nats per event. Positive = the map beats the null.
"""

import numpy as np

EPS_MIX = 0.01  # pre-declared smoothing weight; see docstring of build_density


# ----------------------------------------------------------------------------
# Density construction
# ----------------------------------------------------------------------------

def bin_edges(y_axis):
    """Cell edges for a uniformly spaced y_axis of bin centres."""
    y = np.asarray(y_axis, dtype=float)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("y_axis must be 1-D with >= 2 levels")
    d = np.diff(y)
    if not np.allclose(d, d[0], rtol=1e-6):
        raise ValueError("y_axis is not uniformly spaced; scorer assumes uniform bins")
    w = d[0]
    return np.concatenate([[y[0] - w / 2], y + w / 2]), w


def build_density(intensity, y_axis, eps=EPS_MIX):
    """Turn a raw intensity profile into a probability DENSITY over price.

    Smoothing is mandatory and pre-declared: p = (1-eps)*p_map + eps*uniform.
    Without it a single realised event landing on a zero-intensity cell yields
    log(0) = -inf and one event destroys the entire study. eps is a declared
    constant, never tuned per result.
    """
    w_raw = np.asarray(intensity, dtype=float)
    if w_raw.shape != np.asarray(y_axis).shape:
        raise ValueError("intensity and y_axis must have the same shape")
    if np.any(w_raw < 0):
        raise ValueError("negative intensity")
    edges, cell = bin_edges(y_axis)
    total = w_raw.sum()
    if total <= 0:
        raise ValueError("intensity sums to zero")
    p_mass = w_raw / total                      # probability per cell
    u_mass = np.full_like(p_mass, 1.0 / p_mass.size)
    mixed = (1.0 - eps) * p_mass + eps * u_mass
    return mixed / cell, edges, cell            # density per unit price


def uniform_density(y_axis):
    edges, cell = bin_edges(y_axis)
    n = np.asarray(y_axis).size
    return np.full(n, 1.0 / (n * cell)), edges, cell


def nearspot_density(y_axis, spot, atr, k):
    """Laplace(spot, b=k*atr), truncated to the y_axis support and renormalised.

    Truncation matters: an untruncated Laplace leaks mass outside the support the
    map is defined on, which would hand the map a free advantage.
    """
    if atr <= 0 or k <= 0:
        raise ValueError("atr and k must be positive")
    y = np.asarray(y_axis, dtype=float)
    edges, cell = bin_edges(y)
    b = k * atr
    logf = -np.abs(y - spot) / b
    f = np.exp(logf - logf.max())               # stable
    mass = f / f.sum()
    return mass / cell, edges, cell


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def _index_of(events, edges):
    """Bin index per event; -1 for events outside the support."""
    e = np.asarray(events, dtype=float)
    idx = np.searchsorted(edges, e, side="right") - 1
    idx[(e < edges[0]) | (e > edges[-1])] = -1
    return idx


def mean_loglik(density, edges, events):
    """Mean log-density at realised event prices. Returns (value, n_in, n_out)."""
    idx = _index_of(events, edges)
    inside = idx >= 0
    n_out = int((~inside).sum())
    if inside.sum() == 0:
        return np.nan, 0, n_out
    return float(np.mean(np.log(density[idx[inside]]))), int(inside.sum()), n_out


def score_window(intensity, y_axis, events, spot, atr, k=8.0, eps=EPS_MIX):
    """Score one frozen snapshot against its realised events.

    Out-of-support events are EXCLUDED from both numerator and denominator and
    reported separately. They are not silently dropped: a high out-of-support
    rate means the map's +/-5% window is too narrow for the horizon, which is
    itself a finding.
    """
    p_map, edges, cell = build_density(intensity, y_axis, eps)
    p_uni, _, _ = uniform_density(y_axis)
    p_ns, _, _ = nearspot_density(y_axis, spot, atr, k)

    ll_map, n_in, n_out = mean_loglik(p_map, edges, events)
    ll_uni, _, _ = mean_loglik(p_uni, edges, events)
    ll_ns, _, _ = mean_loglik(p_ns, edges, events)

    return {
        "n_events_in": n_in,
        "n_events_out": n_out,
        "out_of_support_rate": n_out / max(n_in + n_out, 1),
        "ll_map": ll_map,
        "ll_uniform": ll_uni,
        "ll_nearspot": ll_ns,
        "nll_gain_vs_uniform": ll_map - ll_uni,
        "nll_gain_vs_nearspot": ll_map - ll_ns,
        "coverage": coverage(p_map, cell),
        "sharp_lift": sharp_lift(p_map, edges, events, budget=0.30),
        "k": k,
        "eps": eps,
    }


def coverage(density, cell, frac=0.99):
    """Fraction of cells holding the smallest set carrying `frac` of the mass.

    Quoted beside every hit rate. A map covering 97% of the range will show a
    ~1.0 hit rate by construction; coverage is what makes that visible.
    """
    mass = np.sort(density * cell)[::-1]
    c = np.cumsum(mass)
    return float((np.searchsorted(c, frac) + 1) / mass.size)


def sharp_lift(density, edges, events, budget=0.30):
    """Share of realised events in the top-`budget` of mass, divided by budget.

    CRITICAL INTERPRETATION — this is not a skill metric, it is a CALIBRATION
    metric, and reading it as skill is a trap:

      If events are drawn exactly from the map's own density, the top 30% of
      mass captures exactly 30% of events in expectation. sharp_lift == 1.0 is
      therefore an IDENTITY for a perfectly calibrated map, not evidence of no
      skill.

      sharp_lift > 1  =>  realised events are MORE concentrated in the hot cells
                          than the map's own mass allocation claims: the map is
                          directionally right but UNDER-CONFIDENT.
      sharp_lift < 1  =>  the map is OVER-CONFIDENT in its hot cells.

    Concentration relative to the full price range is measured by
    nll_gain_vs_uniform, NOT by this. Never quote sharp_lift as the skill claim.
    """
    cell = edges[1] - edges[0]
    mass = density * cell
    order = np.argsort(mass)[::-1]
    keep, acc = np.zeros(mass.size, bool), 0.0
    for i in order:
        if acc >= budget:
            break
        keep[i] = True
        acc += mass[i]
    idx = _index_of(events, edges)
    inside = idx >= 0
    if inside.sum() == 0:
        return np.nan
    hit = keep[idx[inside]].mean()
    # Normalise by the mass ACTUALLY kept, not the nominal budget. Greedy cell
    # selection overshoots the budget (the last cell is taken whole), and
    # dividing by the nominal 0.30 inflates the statistic by the overshoot --
    # measured at +7% on a peaked map, which would masquerade as real lift.
    return float(hit / acc) if acc > 0 else np.nan


# ----------------------------------------------------------------------------
# Aggregation across windows
# ----------------------------------------------------------------------------

def bootstrap_ci(deltas, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI on the mean of per-window deltas."""
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=float)
    if d.size < 2:
        return float(np.mean(d)) if d.size else np.nan, np.nan, np.nan, d.size
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
    """BH step-up. Returns a boolean rejection mask in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    rej = np.zeros(m, bool)
    if passed.any():
        cut = np.max(np.where(passed)[0])
        rej[order[: cut + 1]] = True
    return rej


def profile_max_k(intensity_unused, y_axis, events, spot, atr, grid=None):
    """Best-fitting near-spot null: the k maximising the BASELINE's own score.

    Pre-declaring k protects against tuning the null to flatter the map, but it
    also lets the null be weak by accident. The honest baseline is the best
    near-spot baseline, so report this alongside the pre-declared k=8.
    """
    grid = grid if grid is not None else np.geomspace(0.5, 40.0, 60)
    best_k, best_ll = np.nan, -np.inf
    for k in grid:
        p, edges, _ = nearspot_density(y_axis, spot, atr, k)
        ll, n_in, _ = mean_loglik(p, edges, events)
        if n_in and np.isfinite(ll) and ll > best_ll:
            best_ll, best_k = ll, float(k)
    return best_k, float(best_ll)
