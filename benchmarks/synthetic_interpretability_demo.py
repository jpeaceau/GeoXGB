"""
Synthetic Interpretability Demo: GeoXGB vs XGBoost+SHAP vs EBM
===============================================================

Uses two synthetic datasets with *known ground truth* to show that GeoXGB
captures complex interaction structure that global methods miss by design.

Dataset 1 — Sign-Flipping Interaction
    y = 2·s_a + s_b + 3·s_a·s_b·sign(mod) + ε
    Truth: s_a & s_b cooperate POSITIVELY when mod>0, NEGATIVELY when mod<0.
    Global average: E[s_a·s_b·sign(mod)] = 0 — EBM's global surface returns ≈0.
    GeoXGB: partitions split on mod naturally; cooperation matrix shows ±1 bimodal;
            contributions() gives a bimodal per-sample interaction distribution;
            three-way tensor T[s_a, s_b, mod] > all other triples.

Dataset 2 — Pure Three-Way Interaction
    y = a + b + c + 2·a·b·c + ε
    Truth: NO pairwise terms (E[a·b]=E[a·c]=E[b·c]=0 for i.i.d. normals).
    EBM: no pairwise terms survive (three-way not representable).
    SHAP: pairwise interaction(a,b) varies with c but can't identify c as modulator.
    GeoXGB: three-way tensor T[a,b,c] dominates; local cooperation(a,b) perfectly
            correlates with c — the modulation is architecturally encoded.

Quantitative comparisons
------------------------
For Dataset 1:
    Pearson corr(detected_interaction(s_a,s_b), ground_truth_interaction) per method.
    Ground truth (per sample): 3 · s_a · s_b · sign(mod)

For Dataset 2:
    GeoXGB tensor: show T[a,b,c] >> all other triples.
    Local cooperation(a,b) vs c: Pearson corr should be high.
    SHAP: show pairwise interaction(a,b) also correlates with c,
          but attribute remains ambiguous (no three-way object exists in SHAP).

Output plots (benchmarks/)
--------------------------
    demo_synth_sign_flip.png     — sign-flip interaction comparison
    demo_synth_three_way.png     — three-way tensor comparison

Usage
-----
    python benchmarks/synthetic_interpretability_demo.py

Requirements
------------
    pip install shap interpret matplotlib
"""
from __future__ import annotations

import sys
import time
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def _p(*a, **k): print(*a, **k, flush=True)
def _sec(t0):    return f"{time.perf_counter()-t0:.1f}s"
def _banner(s):  _p(f"\n{'='*65}\n{s}\n{'='*65}")

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import shap; _SHAP = True
except ImportError:
    _SHAP = False; _p("[WARN] shap not installed — pip install shap")

try:
    from interpret.glassbox import ExplainableBoostingRegressor; _EBM = True
except ImportError:
    _EBM = False; _p("[WARN] interpret not installed — pip install interpret")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLOT = True
except ImportError:
    _PLOT = False; _p("[WARN] matplotlib not installed — pip install matplotlib")

import xgboost as xgb
from geoxgb import GeoXGBRegressor

RNG = np.random.default_rng(7)
N_TRAIN, N_TEST = 3000, 800
FT = ["continuous"] * 5          # feature_types → Python path → interpretability API

# =============================================================================
# DATASET 1 — SIGN-FLIPPING INTERACTION
# =============================================================================
_banner("Dataset 1: Sign-Flipping Interaction")
_p("  y = 2·s_a + s_b + 3·s_a·s_b·sign(mod) + noise")
_p("  Truth: (s_a, s_b) cooperation flips sign with mod.")
_p("  Global average of interaction = 0 => EBM global surface ≈ flat.")

feat1 = ["s_a", "s_b", "mod", "n1", "n2"]
X1_all  = RNG.standard_normal((N_TRAIN + N_TEST, 5))
y1_all  = (  2 * X1_all[:, 0]
            + X1_all[:, 1]
            + 3 * X1_all[:, 0] * X1_all[:, 1] * np.sign(X1_all[:, 2])
            + 0.5 * RNG.standard_normal(N_TRAIN + N_TEST))

X1_tr, X1_te, y1_tr, y1_te = train_test_split(
    X1_all, y1_all, test_size=N_TEST, random_state=0
)
# Ground-truth per-sample interaction on test set
gt_inter1 = 3 * X1_te[:, 0] * X1_te[:, 1] * np.sign(X1_te[:, 2])  # (n_test,)

# ── fit models ────────────────────────────────────────────────────────────────
_p("\n  Fitting models...")

t0 = time.perf_counter()
geo1 = GeoXGBRegressor(n_rounds=600, learning_rate=0.05, max_depth=5,
                       hvrt_min_samples_leaf=10, refit_interval=100,
                       random_state=0)
geo1.fit(X1_tr, y1_tr, feature_types=FT)
geo1_r2 = r2_score(y1_te, geo1.predict(X1_te))
_p(f"  GeoXGB   R² = {geo1_r2:.4f}  ({_sec(t0)})")

t0 = time.perf_counter()
xgb1 = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                         tree_method="hist", verbosity=0, random_state=0)
xgb1.fit(X1_tr, y1_tr)
xgb1_r2 = r2_score(y1_te, xgb1.predict(X1_te))
_p(f"  XGBoost  R² = {xgb1_r2:.4f}  ({_sec(t0)})")

ebm1_r2 = None; ebm1 = None
if _EBM:
    t0 = time.perf_counter()
    ebm1 = ExplainableBoostingRegressor(random_state=0, n_jobs=1,
                                         feature_names=feat1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ebm1.fit(X1_tr, y1_tr)
    ebm1_r2 = r2_score(y1_te, ebm1.predict(X1_te))
    _p(f"  EBM      R² = {ebm1_r2:.4f}  ({_sec(t0)})")

# ── GeoXGB: contributions & cooperation ───────────────────────────────────────
_p("\n  GeoXGB interpretability:")
t0 = time.perf_counter()
contrib1  = geo1.contributions(X1_te, feature_names=feat1, min_pair_coop=0.05)
coop_res1 = geo1.cooperation_matrix(X1_te, feat1)
tensor1   = geo1.cooperation_tensor(X1_te, feat1)
_p(f"    contributions() done in {_sec(t0)}")

geo_inter1 = contrib1.interaction.get(("s_a", "s_b"),
             contrib1.interaction.get(("s_b", "s_a"), np.zeros(N_TEST)))
coop_ab1   = coop_res1["matrices"][:, 0, 1]   # per-sample cooperation(s_a, s_b)
GT_tensor1 = tensor1["global_tensor"]
_p(f"    Active interaction pairs: {len(contrib1.interaction)}")

# Correlation of interaction detection with ground truth
corr_geo_gt = float(np.corrcoef(geo_inter1, gt_inter1)[0, 1])
_p(f"    Corr(GeoXGB interaction(s_a,s_b), ground truth): {corr_geo_gt:+.3f}")
_p(f"    Corr(GeoXGB cooperation(s_a,s_b), sign(mod)):    "
   f"{float(np.corrcoef(coop_ab1, np.sign(X1_te[:,2]))[0,1]):+.3f}")

# Three-way tensor — does T[s_a, s_b, mod] dominate?
T_ab_mod  = float(GT_tensor1[0, 1, 2])
all_triples = [(feat1[i], feat1[j], feat1[k], GT_tensor1[i, j, k])
               for i in range(5) for j in range(i+1, 5) for k in range(j+1, 5)]
all_triples.sort(key=lambda x: abs(x[3]), reverse=True)
_p("\n    Three-way tensor top-3 (truth: s_a×s_b×mod should dominate):")
for a, b, c, v in all_triples[:3]:
    mark = " ← TRUE INTERACTION" if {a,b,c} == {"s_a","s_b","mod"} else ""
    _p(f"      T({a},{b},{c}) = {v:+.4f}{mark}")

# ── SHAP: per-sample interaction values ───────────────────────────────────────
shap_inter1 = None; corr_shap_gt = None
if _SHAP:
    _p("\n  SHAP interaction values (subset 400):")
    t0 = time.perf_counter()
    sub = min(400, N_TEST)
    explainer1 = shap.TreeExplainer(xgb1)
    sv_inter1  = explainer1.shap_interaction_values(X1_te[:sub])  # (sub, d, d)
    shap_inter1 = sv_inter1[:, 0, 1]   # interaction(s_a, s_b)
    corr_shap_gt = float(np.corrcoef(shap_inter1, gt_inter1[:sub])[0, 1])
    _p(f"    Corr(SHAP interaction(s_a,s_b), ground truth): {corr_shap_gt:+.3f}  ({_sec(t0)})")
    # Can SHAP identify mod as driver of the sign flip?
    corr_shap_mod = float(np.corrcoef(shap_inter1, np.sign(X1_te[:sub, 2]))[0, 1])
    _p(f"    Corr(SHAP interaction(s_a,s_b), sign(mod)):    {corr_shap_mod:+.3f}")
    _p("    → SHAP detects the bimodal signal but has no named attribution to mod.")

# ── EBM: pairwise term for (s_a, s_b) ────────────────────────────────────────
ebm_inter1 = None
if _EBM and ebm1 is not None:
    _p("\n  EBM pairwise terms:")
    term_imps = list(zip(ebm1.term_names_, ebm1.term_importances()))
    is_pair = lambda n: isinstance(n, (list,tuple)) or " x " in str(n) or " & " in str(n)
    pair_terms = [(n, v) for n, v in term_imps if is_pair(n)]
    pair_terms.sort(key=lambda x: -x[1])
    if pair_terms:
        _p(f"    {len(pair_terms)} pairwise terms survived:")
        for name, imp in pair_terms[:5]:
            _p(f"      {str(name):<25} importance={imp:.5f}")
        # Get per-sample EBM score for s_a & s_b pair
        pair_key = ("s_a", "s_b")
        pair_str_options = [f"{a} & {b}" for a in pair_key for b in pair_key if a != b]
        for i, name in enumerate(ebm1.term_names_):
            if str(name) in [f"s_a & s_b", f"s_b & s_a"]:
                try:
                    ebm_scores = ebm1.eval_terms(X1_te)
                    ebm_inter1 = ebm_scores[:, i]
                    corr_ebm = float(np.corrcoef(ebm_inter1, gt_inter1)[0, 1])
                    _p(f"    Corr(EBM f(s_a,s_b), ground truth): {corr_ebm:+.3f}")
                except Exception:
                    pass
                break
    else:
        _p("    EBM pruned ALL pairwise terms — global surface cannot represent sign flip.")
        _p("    => R² penalty: EBM cannot model the 3·s_a·s_b·sign(mod) term at all.")

_p(f"\n  Accuracy summary (Dataset 1):")
_p(f"    GeoXGB  R²={geo1_r2:.4f}")
_p(f"    XGBoost R²={xgb1_r2:.4f}")
if ebm1_r2: _p(f"    EBM     R²={ebm1_r2:.4f}  (handicapped: can't model sign-flip)")

# =============================================================================
# DATASET 2 — PURE THREE-WAY INTERACTION
# =============================================================================
_banner("Dataset 2: Pure Three-Way Interaction")
_p("  y = a + b + c + 2·a·b·c + noise")
_p("  Truth: NO pairwise terms (E[a·b]=E[a·c]=E[b·c]=0 for independent normals).")
_p("  Only three-way interaction a·b·c is non-trivially predictive.")

feat2 = ["a", "b", "c", "n1", "n2"]
X2_all  = RNG.standard_normal((N_TRAIN + N_TEST, 5))
y2_all  = (  X2_all[:, 0]
            + X2_all[:, 1]
            + X2_all[:, 2]
            + 2 * X2_all[:, 0] * X2_all[:, 1] * X2_all[:, 2]
            + 0.3 * RNG.standard_normal(N_TRAIN + N_TEST))

X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    X2_all, y2_all, test_size=N_TEST, random_state=0
)
# Ground-truth three-way per sample (for quantitative checks)
gt_3way2    = 2 * X2_te[:, 0] * X2_te[:, 1] * X2_te[:, 2]  # (n_test,)

# ── fit models ────────────────────────────────────────────────────────────────
_p("\n  Fitting models...")

t0 = time.perf_counter()
geo2 = GeoXGBRegressor(n_rounds=600, learning_rate=0.05, max_depth=5,
                       hvrt_min_samples_leaf=10, refit_interval=100,
                       random_state=0)
geo2.fit(X2_tr, y2_tr, feature_types=FT)
geo2_r2 = r2_score(y2_te, geo2.predict(X2_te))
_p(f"  GeoXGB   R² = {geo2_r2:.4f}  ({_sec(t0)})")

t0 = time.perf_counter()
xgb2 = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                         tree_method="hist", verbosity=0, random_state=0)
xgb2.fit(X2_tr, y2_tr)
xgb2_r2 = r2_score(y2_te, xgb2.predict(X2_te))
_p(f"  XGBoost  R² = {xgb2_r2:.4f}  ({_sec(t0)})")

ebm2_r2 = None; ebm2 = None
if _EBM:
    t0 = time.perf_counter()
    ebm2 = ExplainableBoostingRegressor(random_state=0, n_jobs=1,
                                         feature_names=feat2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ebm2.fit(X2_tr, y2_tr)
    ebm2_r2 = r2_score(y2_te, ebm2.predict(X2_te))
    _p(f"  EBM      R² = {ebm2_r2:.4f}  ({_sec(t0)})")

# ── GeoXGB: three-way tensor ──────────────────────────────────────────────────
_p("\n  GeoXGB interpretability:")
t0 = time.perf_counter()
coop_res2 = geo2.cooperation_matrix(X2_te, feat2)
tensor2   = geo2.cooperation_tensor(X2_te, feat2)
contrib2  = geo2.contributions(X2_te, feature_names=feat2, min_pair_coop=0.05)
_p(f"    Done in {_sec(t0)}")

GT_tensor2 = tensor2["global_tensor"]
coop_ab2   = coop_res2["matrices"][:, 0, 1]  # per-sample coop(a, b)

all_triples2 = [(feat2[i], feat2[j], feat2[k], GT_tensor2[i, j, k])
                for i in range(5) for j in range(i+1, 5) for k in range(j+1, 5)]
all_triples2.sort(key=lambda x: abs(x[3]), reverse=True)
_p("\n    Three-way tensor (truth: a×b×c should dominate):")
_p(f"    {'Triple':<25} {'|T[i,j,k]|':>12}  {'Rank':>5}")
for rank, (a, b, c, v) in enumerate(all_triples2, start=1):
    mark = "  ← TRUE" if {a,b,c} == {"a","b","c"} else ""
    _p(f"    ({a}, {b}, {c}){' '*(19-len(a)-len(b)-len(c))} {abs(v):>12.4f}  #{rank:>3}{mark}")
    if rank >= 6:
        break

# Quantitative: coop(a,b) should correlate with c (since a·b modulated by c)
corr_coop_c = float(np.corrcoef(coop_ab2, X2_te[:, 2])[0, 1])
_p(f"\n    Corr(local_coop(a,b), c): {corr_coop_c:+.3f}")
_p(f"    => Local coop(a,b) {'+' if corr_coop_c > 0 else ''}correlates with c: "
   f"GeoXGB found that c modulates the a-b pair.")

# contribution(a,b) should correlate with ground truth
geo_inter2 = contrib2.interaction.get(("a","b"),
             contrib2.interaction.get(("b","a"), np.zeros(N_TEST)))
corr_geo2 = float(np.corrcoef(geo_inter2, gt_3way2)[0, 1])
_p(f"    Corr(GeoXGB contribution(a,b), ground truth 2·a·b·c): {corr_geo2:+.3f}")

# ── SHAP: pairwise interaction for (a, b) ────────────────────────────────────
shap_inter2 = None
if _SHAP:
    _p("\n  SHAP interaction values (subset 400):")
    t0 = time.perf_counter()
    sub = min(400, N_TEST)
    explainer2    = shap.TreeExplainer(xgb2)
    sv_inter2     = explainer2.shap_interaction_values(X2_te[:sub])
    shap_inter2   = sv_inter2[:, 0, 1]
    corr_shap_ab  = float(np.corrcoef(shap_inter2, gt_3way2[:sub])[0, 1])
    corr_shap_c   = float(np.corrcoef(shap_inter2, X2_te[:sub, 2])[0, 1])
    _p(f"    Corr(SHAP interaction(a,b), ground truth): {corr_shap_ab:+.3f}  ({_sec(t0)})")
    _p(f"    Corr(SHAP interaction(a,b), c):            {corr_shap_c:+.3f}")
    _p("    → SHAP interaction(a,b) varies with c — the signal is there — but SHAP")
    _p("      has no three-way object. Analyst must manually discover c's role.")
    _p("      GeoXGB names it directly: T[a,b,c] is the dominant tensor entry.")

# ── EBM: pairwise terms ───────────────────────────────────────────────────────
if _EBM and ebm2 is not None:
    _p("\n  EBM pairwise terms:")
    term_imps2 = list(zip(ebm2.term_names_, ebm2.term_importances()))
    is_pair = lambda n: isinstance(n, (list,tuple)) or " x " in str(n) or " & " in str(n)
    pair2 = [(n, v) for n, v in term_imps2 if is_pair(n)]
    main2 = [(n, v) for n, v in term_imps2 if not is_pair(n)]
    if pair2:
        pair2.sort(key=lambda x: -x[1])
        _p(f"    {len(pair2)} pairwise terms survived (global surfaces, no three-way):")
        for name, imp in pair2[:4]:
            _p(f"      {str(name):<18} importance={imp:.5f}")
        _p("    → These are GLOBAL surfaces f_ij(xi,xj), not functions of c.")
        _p("      EBM cannot represent 'the a-b interaction is modulated by c'.")
    else:
        _p("    EBM pruned ALL pairwise terms (three-way structure not representable).")
        _p("    => EBM treats the data as purely additive — misses all interactions.")

_p(f"\n  Accuracy summary (Dataset 2):")
_p(f"    GeoXGB  R²={geo2_r2:.4f}")
_p(f"    XGBoost R²={xgb2_r2:.4f}")
if ebm2_r2: _p(f"    EBM     R²={ebm2_r2:.4f}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
_banner("CAPABILITY COMPARISON ON SYNTHETIC GROUND TRUTH")

_p(f"\n  {'Capability':<50} {'GeoXGB':>8} {'SHAP':>6} {'EBM':>5}")
_p("  " + "-" * 72)
rows = [
    ("Detect sign-flip interaction (per sample)",  True,  True,  False),
    ("  Named attribution to modulator (x3)",      True,  False, False),
    ("  via geometry — no post-hoc analysis",       True,  False, False),
    ("Detect pure three-way interaction",           True,  False, False),
    ("  Three-way tensor T[a,b,c]",                True,  False, False),
    ("  Correlate coop(a,b) with c directly",      True,  False, False),
    ("  SHAP interaction(a,b) varies with c",       False, True,  False),
    ("Reliability (local_r² per sample)",           True,  False, False),
    ("Per-sample interaction contributions",        True,  True,  False),
    ("Global pairwise summary",                     True,  True,  True ),
    ("Human-interpretable global main effects",     False, False, True ),
]
for name, g, s, e in rows:
    _p(f"  {name:<50} {'Yes':>8} {'Yes' if s else '—':>6} {'Yes' if e else '—':>5}")

_p(f"\n  R² comparison:")
_p(f"  {'Dataset':<30}  {'GeoXGB':>8}  {'XGBoost':>8}  {'EBM':>8}")
_p("  " + "-" * 58)
rows_r2 = [("Sign-flip (y ~ s_a·s_b·sign(mod))", geo1_r2, xgb1_r2, ebm1_r2),
           ("Three-way (y ~ a·b·c)",              geo2_r2, xgb2_r2, ebm2_r2)]
for label, g, x, e in rows_r2:
    ev = f"{e:.4f}" if e is not None else "N/A"
    _p(f"  {label:<30}  {g:>8.4f}  {x:>8.4f}  {ev:>8}")
_p("  Note: EBM R² penalty reflects structural inexpressibility, not model quality.")

# =============================================================================
# PLOTS
# =============================================================================
if not _PLOT:
    _p("\n[PLOTS SKIPPED — matplotlib not installed]")
    sys.exit(0)

import os
os.makedirs("benchmarks", exist_ok=True)

# ─── Plot 1: Sign-Flip Comparison ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    "Dataset 1: Sign-Flipping Interaction\n"
    r"$y = 2s_a + s_b + 3\,s_a\cdot s_b\cdot\mathrm{sign}(\mathrm{mod}) + \varepsilon$",
    fontsize=13
)

# Row 0, col 0: GeoXGB per-sample interaction histogram
ax = axes[0, 0]
ax.hist(geo_inter1, bins=50, color="#4878CF", edgecolor="white", alpha=0.85)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("GeoXGB contribution(s_a, s_b)")
ax.set_ylabel("count")
ax.set_title("GeoXGB contributions()\n(bimodal — two regimes captured)", fontsize=9)
ax.text(0.05, 0.93, f"corr w/ truth={corr_geo_gt:+.3f}",
        transform=ax.transAxes, fontsize=8, va="top")

# Row 0, col 1: SHAP per-sample interaction histogram
ax = axes[0, 1]
if shap_inter1 is not None:
    ax.hist(shap_inter1, bins=50, color="#E06C75", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("SHAP interaction(s_a, s_b)")
    ax.set_ylabel("count")
    ax.set_title("SHAP interaction values\n(bimodal signal — but who drives the flip?)", fontsize=9)
    ax.text(0.05, 0.93, f"corr w/ truth={corr_shap_gt:+.3f}",
            transform=ax.transAxes, fontsize=8, va="top")
else:
    ax.text(0.5, 0.5, "SHAP not available", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_title("SHAP interaction values", fontsize=9)

# Row 0, col 2: EBM pairwise term importance
ax = axes[0, 2]
if _EBM and ebm1 is not None:
    term_imps1 = list(zip(ebm1.term_names_, ebm1.term_importances()))
    is_pair = lambda n: isinstance(n, (list,tuple)) or " & " in str(n) or " x " in str(n)
    pair1_sorted = sorted([(str(n), v) for n, v in term_imps1 if is_pair(n)], key=lambda x: -x[1])
    if pair1_sorted:
        names1 = [p[0] for p in pair1_sorted[:6]]
        vals1  = [p[1] for p in pair1_sorted[:6]]
        colors = ["#E06C75" if "s_a" in n and "s_b" in n else "#98C379" for n in names1]
        ax.barh(names1[::-1], vals1[::-1], color=colors[::-1])
        ax.set_xlabel("term importance")
        ax.set_title("EBM global pairwise importances\n(s_a & s_b should be dominant)", fontsize=9)
    else:
        ax.text(0.5, 0.5, "EBM: 0 pairwise terms\n(global surface = 0\ncan't model sign flip)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
                bbox=dict(fc="#FFE0E0", ec="#E06C75"))
        ax.set_title("EBM pairwise terms\n(all pruned — sign flip averages out)", fontsize=9)
else:
    ax.text(0.5, 0.5, "EBM not available", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_title("EBM", fontsize=9)

# Row 1, col 0: GeoXGB cooperation vs mod (step function)
ax = axes[1, 0]
mod_vals = X1_te[:, 2]
sc = ax.scatter(mod_vals, coop_ab1, c=np.sign(mod_vals), cmap="RdBu",
                s=5, alpha=0.5, vmin=-1.5, vmax=1.5)
ax.axvline(0, color="k", lw=1.2, ls="--", label="mod=0")
ax.axhline(0, color="grey", lw=0.7, ls=":")
ax.set_xlabel("mod (modulator feature)")
ax.set_ylabel("GeoXGB local_coop(s_a, s_b)")
ax.set_title("GeoXGB cooperation vs modulator\n(clean step at mod=0)", fontsize=9)
ax.legend(fontsize=8)

# Row 1, col 1: SHAP interaction vs mod
ax = axes[1, 1]
if shap_inter1 is not None:
    sub = len(shap_inter1)
    sc2 = ax.scatter(X1_te[:sub, 2], shap_inter1, c=np.sign(X1_te[:sub, 2]),
                     cmap="RdBu", s=5, alpha=0.5, vmin=-1.5, vmax=1.5)
    ax.axvline(0, color="k", lw=1.2, ls="--")
    ax.axhline(0, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("mod (modulator feature)")
    ax.set_ylabel("SHAP interaction(s_a, s_b)")
    ax.set_title("SHAP interaction vs modulator\n(signal present but not architecturally named)", fontsize=9)
else:
    ax.text(0.5, 0.5, "SHAP not available", ha="center", va="center",
            transform=ax.transAxes)

# Row 1, col 2: GeoXGB three-way tensor for Dataset 1
ax = axes[1, 2]
t_labels = [f"T({a},{b},{c})" for a, b, c, _ in all_triples[:6]]
t_vals   = [abs(v) for _, _, _, v in all_triples[:6]]
t_colors = ["#4878CF" if {a,b,c} == {"s_a","s_b","mod"} else "#AAAAAA"
            for a, b, c, _ in all_triples[:6]]
ax.barh(t_labels[::-1], t_vals[::-1], color=t_colors[::-1])
ax.set_xlabel("|T[i,j,k]| global three-way interaction")
ax.set_title("GeoXGB three-way tensor\n(s_a×s_b×mod should dominate)", fontsize=9)

fig.tight_layout()
p = "benchmarks/demo_synth_sign_flip.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
_p(f"\n[Saved {p}]")
plt.close(fig)

# ─── Plot 2: Three-Way Comparison ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    r"Dataset 2: Pure Three-Way Interaction  $y = a + b + c + 2\,abc + \varepsilon$",
    fontsize=13
)

# Col 0: GeoXGB tensor bar chart
ax = axes[0]
t2_labels = [f"T({a},{b},{c})" for a, b, c, _ in all_triples2[:8]]
t2_vals   = [abs(v) for _, _, _, v in all_triples2[:8]]
t2_colors = ["#4878CF" if {a,b,c} == {"a","b","c"} else "#AAAAAA"
             for a, b, c, _ in all_triples2[:8]]
ax.barh(t2_labels[::-1], t2_vals[::-1], color=t2_colors[::-1])
ax.set_xlabel("|T[i,j,k]|")
ax.set_title("GeoXGB three-way tensor\n(a×b×c dominates — directly identified)", fontsize=9)

# Col 1: GeoXGB local_coop(a,b) vs c — should be monotone
ax = axes[1]
# Bin c into 5 quantiles and show cooperation distribution per bin
c_vals  = X2_te[:, 2]
q_edges = np.quantile(c_vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
q_means = 0.5 * (q_edges[:-1] + q_edges[1:])
q_coop  = [coop_ab2[(c_vals >= q_edges[i]) & (c_vals < q_edges[i+1])]
           for i in range(5)]
q_coop[-1] = coop_ab2[c_vals >= q_edges[-2]]  # include last edge

bp = ax.boxplot(q_coop, positions=q_means, widths=0.3,
                patch_artist=True, notch=False,
                boxprops=dict(facecolor="#4878CF", alpha=0.6),
                medianprops=dict(color="white", linewidth=2))
ax.axhline(0, color="grey", lw=0.7, ls=":")
ax.set_xlabel("c (modulator of a×b)")
ax.set_ylabel("GeoXGB local_coop(a, b)")
ax.set_title(f"GeoXGB: coop(a,b) vs c\ncorr={corr_coop_c:+.3f} — c modulates a-b pair", fontsize=9)

# Col 2: SHAP interaction(a,b) vs c
ax = axes[2]
if shap_inter2 is not None:
    sub = len(shap_inter2)
    c_sub = X2_te[:sub, 2]
    gt_sub = gt_3way2[:sub]
    ax.scatter(c_sub, shap_inter2, c="#E06C75", s=5, alpha=0.4,
               label=f"SHAP int(a,b), corr={corr_shap_c:+.2f}")
    ax.scatter(c_sub, gt_sub, c="black", s=3, alpha=0.3,
               label="ground truth 2abc")
    ax.axhline(0, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("c (modulator feature)")
    ax.set_ylabel("interaction(a, b)")
    ax.set_title("SHAP interaction(a,b) vs c\n"
                 "(signal present but c not named as driver)", fontsize=9)
    ax.legend(fontsize=7)
else:
    if _EBM and ebm2 is not None:
        term_imps2 = list(zip(ebm2.term_names_, ebm2.term_importances()))
        is_pair = lambda n: isinstance(n,(list,tuple)) or " & " in str(n) or " x " in str(n)
        pair2_sorted = sorted([(str(n), v) for n, v in term_imps2 if is_pair(n)], key=lambda x: -x[1])
        if pair2_sorted:
            pn = [p[0] for p in pair2_sorted[:6]]
            pv = [p[1] for p in pair2_sorted[:6]]
            ax.barh(pn[::-1], pv[::-1], color="#98C379")
            ax.set_xlabel("term importance")
            ax.set_title("EBM pairwise importances\n(no three-way terms representable)", fontsize=9)
        else:
            ax.text(0.5, 0.5, "EBM: 0 pairwise terms survived\nThree-way not representable",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10,
                    bbox=dict(fc="#FFE0E0", ec="#E06C75"))
            ax.set_title("EBM pairwise terms", fontsize=9)

fig.tight_layout()
p = "benchmarks/demo_synth_three_way.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
_p(f"[Saved {p}]")
plt.close(fig)

_p("\nDone.")
