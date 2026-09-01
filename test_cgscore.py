"""
test_cgscore.py — synthetic validation of the scoring harness.

Run this BEFORE any real snapshot exists. A normalisation bug discovered on
day 26 of forward capture costs the whole month; these tests cost 2 seconds.

Every test uses data with a KNOWN answer, so a pass is evidence the scorer is
correct, not merely that it runs.
"""

import numpy as np
import cgscore as S

Y = np.arange(74444.38, 74444.38 + 134 * 59.72, 59.72)[:134]  # real BTC geometry
SPOT = float(Y.mean())
ATR = 300.0
rng = np.random.default_rng(42)
FAILED = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


# 1. Densities integrate to 1 over the support.
for label, dens in [
    ("map", S.build_density(rng.random(134), Y)),
    ("uniform", S.uniform_density(Y)),
    ("nearspot", S.nearspot_density(Y, SPOT, ATR, 8.0)),
]:
    d, edges, cell = dens
    check(f"{label} density integrates to 1", abs(d.sum() * cell - 1) < 1e-9,
          f"{d.sum()*cell:.12f}")

# 2. THE CRITICAL TEST — feeding the null in as the map must return exactly zero.
#    Any non-zero here means a normalisation or support mismatch.
p_ns, edges, cell = S.nearspot_density(Y, SPOT, ATR, 8.0)
ev = rng.choice(Y, size=4000, p=p_ns * cell)
r = S.score_window(p_ns * cell, Y, ev, SPOT, ATR, k=8.0, eps=0.0)
check("null-as-map gives zero gain vs nearspot", abs(r["nll_gain_vs_nearspot"]) < 1e-9,
      f"{r['nll_gain_vs_nearspot']:.2e}")

# 3. Uniform-as-map vs uniform null must also be exactly zero.
p_u, _, cell_u = S.uniform_density(Y)
r = S.score_window(p_u * cell_u, Y, ev, SPOT, ATR, k=8.0, eps=0.0)
check("uniform-as-map gives zero gain vs uniform", abs(r["nll_gain_vs_uniform"]) < 1e-9,
      f"{r['nll_gain_vs_uniform']:.2e}")

# 4. Recover a KNOWN gain. A map concentrating all mass on m of n cells, scored
#    on events drawn only from those cells, must beat uniform by exactly log(n/m).
m, n = 20, 134
w = np.zeros(n); w[50:50 + m] = 1.0
ev_c = rng.choice(Y[50:50 + m], size=3000)
r = S.score_window(w, Y, ev_c, SPOT, ATR, eps=0.0)
check("recovers analytic gain log(n/m)", abs(r["nll_gain_vs_uniform"] - np.log(n / m)) < 1e-9,
      f"got {r['nll_gain_vs_uniform']:.6f} want {np.log(n/m):.6f}")

# 5. A concentrated map scored on events it did NOT predict must lose to uniform.
ev_bad = rng.choice(np.concatenate([Y[:50], Y[70:]]), size=3000)
r = S.score_window(w, Y, ev_bad, SPOT, ATR, eps=S.EPS_MIX)
check("wrong map loses to uniform", r["nll_gain_vs_uniform"] < 0,
      f"{r['nll_gain_vs_uniform']:.4f}")

# 6. Smoothing prevents -inf when an event lands on a zero-intensity cell.
r = S.score_window(w, Y, np.array([Y[0]]), SPOT, ATR, eps=S.EPS_MIX)
check("eps smoothing avoids -inf", np.isfinite(r["nll_gain_vs_uniform"]),
      f"{r['nll_gain_vs_uniform']:.4f}")

# 7. Out-of-support events are counted, not silently dropped.
r = S.score_window(np.ones(134), Y, np.array([Y[0], 1.0, 1e9, Y[5]]), SPOT, ATR)
check("out-of-support events counted", r["n_events_out"] == 2 and r["n_events_in"] == 2,
      f"in={r['n_events_in']} out={r['n_events_out']}")

# 8. sharp_lift ~ 1.0 for a uniform map (no concentration by construction).
r = S.score_window(np.ones(134), Y, rng.choice(Y, 5000), SPOT, ATR)
check("sharp_lift ~1.0 on uniform map", abs(r["sharp_lift"] - 1.0) < 0.10,
      f"{r['sharp_lift']:.3f}")

# 9. sharp_lift semantics. These encode the calibration identity, which the
#    first draft of this suite got wrong: drawing events from the map's OWN
#    density must give sharp_lift == 1.0 exactly (perfect calibration), and
#    only an UNDER-CONFIDENT map exceeds 1.0.
peak = np.exp(-0.5 * ((np.arange(134) - 67) / 12.0) ** 2)
p_peak, e_peak, c_peak = S.build_density(peak, Y, eps=0.0)
ev_cal = rng.choice(Y, size=40000, p=p_peak * c_peak)
r = S.score_window(peak, Y, ev_cal, SPOT, ATR, eps=0.0)
check("sharp_lift == 1.0 when map is perfectly calibrated",
      abs(r["sharp_lift"] - 1.0) < 0.05, f"{r['sharp_lift']:.3f}")

sharper = np.exp(-0.5 * ((np.arange(134) - 67) / 4.0) ** 2)
p_sh, e_sh, c_sh = S.build_density(sharper, Y, eps=0.0)
ev_sharp = rng.choice(Y, size=40000, p=p_sh * c_sh)
r = S.score_window(peak, Y, ev_sharp, SPOT, ATR, eps=0.0)
check("sharp_lift > 1.5 when map is under-confident", r["sharp_lift"] > 1.5,
      f"{r['sharp_lift']:.3f}")

flatter = np.exp(-0.5 * ((np.arange(134) - 67) / 40.0) ** 2)
p_fl, e_fl, c_fl = S.build_density(flatter, Y, eps=0.0)
ev_flat = rng.choice(Y, size=40000, p=p_fl * c_fl)
r = S.score_window(peak, Y, ev_flat, SPOT, ATR, eps=0.0)
check("sharp_lift < 1.0 when map is over-confident", r["sharp_lift"] < 1.0,
      f"{r['sharp_lift']:.3f}")

# 10. Truncated Laplace: a tighter k concentrates coverage.
c_tight = S.coverage(*S.nearspot_density(Y, SPOT, ATR, 2.0)[::2])
c_wide = S.coverage(*S.nearspot_density(Y, SPOT, ATR, 40.0)[::2])
check("tighter k gives lower coverage", c_tight < c_wide, f"{c_tight:.3f} < {c_wide:.3f}")

# 11. profile_max_k recovers the k that generated the events.
p_true, e_true, c_true = S.nearspot_density(Y, SPOT, ATR, 6.0)
ev_k = rng.choice(Y, size=20000, p=p_true * c_true)
khat, _ = S.profile_max_k(None, Y, ev_k, SPOT, ATR)
check("profile_max_k recovers true k", abs(khat - 6.0) / 6.0 < 0.25, f"khat={khat:.2f} true=6.0")

# 12. Power arithmetic matches the pre-registration.
check("required_n(sd=0.50, effect=0.10) ~ 96-100", 90 <= S.required_n(0.50, 0.10) <= 105,
      f"{S.required_n(0.50, 0.10)}")
check("required_n(sd=0.50, effect=0.20) ~ 24-25", 20 <= S.required_n(0.50, 0.20) <= 28,
      f"{S.required_n(0.50, 0.20)}")

# 13. Bootstrap CI COVERAGE RATE. A single-draw check is flaky by construction
#     (one unlucky sample fails a correct estimator ~5% of the time), so test
#     the property that actually matters: nominal 95% intervals must cover the
#     truth about 95% of the time across many independent samples.
covered = 0
TRIALS = 300
for t in range(TRIALS):
    samp = np.random.default_rng(1000 + t).normal(0.25, 0.50, 200)
    _, lo, hi, _ = S.bootstrap_ci(samp, n_boot=2000, seed=t)
    covered += (lo < 0.25 < hi)
rate = covered / TRIALS
check("bootstrap CI coverage ~95%", 0.90 <= rate <= 0.99, f"{rate:.1%} over {TRIALS} trials")

# 14. Sign test: check the FALSE-POSITIVE RATE, not one draw. Under the null a
#     correct test rejects at 5% by definition, so a single-draw assertion is
#     flaky exactly 5% of the time. Same failure class as the CI test above.
fp = 0
for t in range(300):
    g = np.random.default_rng(5000 + t)
    p, _, _ = S.sign_test(g.normal(0.0, 1.0, 200))
    fp += (p < 0.05)
fp_rate = fp / 300
p_real, pos, tot = S.sign_test(np.random.default_rng(7).normal(1.0, 1.0, 200))
check("sign test false-positive rate ~5%", 0.01 <= fp_rate <= 0.11, f"{fp_rate:.1%}")
check("sign test detects a real shift", p_real < 0.01, f"p={p_real:.2e} ({pos}/{tot} positive)")

# 15. BH controls: all-null p-values yield few rejections.
rej = S.benjamini_hochberg(rng.uniform(0, 1, 71), q=0.10)
check("BH rejects few under the null", rej.sum() <= 3, f"{int(rej.sum())}/71 rejected")

print("\n" + ("ALL TESTS PASSED — scorer is safe to run on real snapshots"
              if not FAILED else f"{len(FAILED)} FAILED: {FAILED}"))
raise SystemExit(1 if FAILED else 0)
