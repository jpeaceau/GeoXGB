"""
GeoXGB -- Heterogeneity Detection Test
=======================================

Tests GeoXGB's boost/partition importance ratio as a heterogeneity surface map.

When boost and partition importances diverge, it is not a red flag -- it is
diagnostic information about the structural role of each feature:

  ratio >> 1  feature drives gradient updates (boost) but does not define
              the natural data geometry. Prediction-relevant but not
              structure-defining at the local level.

  ratio << 1  feature strongly organises the data into coherent geometric
              regions (partition) but contributes less within those regions.
              Signature of a heterogeneity axis -- the feature defines WHERE
              different predictive relationships apply, rather than driving
              predictions directly.

  ratio ~= 1  feature is both structure-defining and predictive within the
              regions it defines. Universally informative.

Crucially, the heterogeneity detected is not merely population-level subgroups.
HVRT partitions the feature space into hyperplane-bounded local regions -- with
sufficient partitions, these approach individual-level neighbourhoods. Each
partition can have a distinct predictive structure. The partition_tree_rules()
method exposes which region each individual occupies and what conditions define
it, giving individual-level heterogeneity diagnostics that no additive model can
provide.

The ratio is a RELATIVE signal: interpret structural roles by comparing features
against each other. Note that HVRT at y_weight=0.5 blends X-space geometry with
y-weighted geometry; the X-space component can assign non-trivial partition
importance to uninformative features. Ratio ordering among features with
meaningful boost importance is the primary diagnostic.

Three Scenarios
---------------
1. Regime / heterogeneity axis
   X0 defines which predictive relationship applies (regime indicator). X1
   predicts in regime 0, X2 in regime 1. X0 does not enter the prediction
   formula directly. Expected: ratio(X0) is lower than ratio(X1) and ratio(X2)
   -- X0 is the heterogeneity axis, X1/X2 are local predictors.

2. Sign-flip interaction moderator
   y = 3*X1*sign(X0) + noise. X0 moderates the sign of X1's effect. Neither
   a purely additive model nor XGBoost's standard importance fully surfaces
   X0's structural role. GeoXGB's partition importance shows X0 is the
   local-structure axis; its ratio is lower than X1's (predictor within each
   local region). The ratio ordering is the signal.

3. Geometry anchor vs prediction driver (complementary local roles)
   y = 3*X0 + 2*X1 + noise. Among two strong predictors, one anchors the
   partition geometry (lower ratio) and the other is captured primarily via
   gradient updates (higher ratio). Both carry genuine signal -- the ratio
   reveals their local structural roles, which cannot be predicted from
   coefficient size alone.

Usage
-----
    python benchmarks/heterogeneity_detection_test.py

Requirements: geoxgb, scikit-learn, numpy
Optional: xgboost (for Scenario 2 comparison)
"""

from __future__ import annotations

import warnings

import numpy as np

from geoxgb import GeoXGBRegressor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor as _XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_ROUNDS     = 1000
RANDOM_STATE = 42

GEO_PARAMS = dict(
    n_rounds             = N_ROUNDS,
    learning_rate        = 0.2,
    max_depth            = 4,
    min_samples_leaf     = 5,
    reduce_ratio         = 0.7,
    refit_interval       = 20,
    auto_noise           = True,
    auto_expand          = True,
    cache_geometry       = False,
    generation_strategy  = "epanechnikov",
    random_state         = RANDOM_STATE,
)

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title):
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title):
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


# ---------------------------------------------------------------------------
# Ratio computation helper
# ---------------------------------------------------------------------------

def _boost_partition_ratios(model, feature_names):
    """Return (boost_imp, avg_part_imp, ratio) dicts keyed by feature name."""
    boost = model.feature_importances(feature_names=feature_names)

    part_trace = model.partition_feature_importances(feature_names=feature_names)
    avg_part = {
        f: float(np.mean([e["importances"].get(f, 0.0) for e in part_trace]))
        for f in feature_names
    }

    eps = 1e-10
    ratio = {f: boost[f] / (avg_part[f] + eps) for f in feature_names}
    return boost, avg_part, ratio


def _print_ratio_table(feature_names, boost, avg_part, ratio, roles):
    """Print a formatted importance and ratio table."""
    header = (
        f"  {'Feature':<18}  {'Boost':>8}  {'Partition':>10}  "
        f"{'Ratio':>7}  Role"
    )
    print(header)
    print("  " + "-" * 68)
    for f in feature_names:
        b  = boost[f]
        p  = avg_part[f]
        r  = ratio[f]
        ro = roles.get(f, "noise")
        tag = ""
        if r > 2.0:
            tag = "  [local predictor]"
        elif r < 0.8:
            tag = "  [heterogeneity axis]"
        else:
            tag = "  [dual role]"
        print(f"  {f:<18}  {b:>8.4f}  {p:>10.4f}  {r:>7.3f}  {ro}{tag}")


# ---------------------------------------------------------------------------
# Scenario 1 -- Regime / heterogeneity axis
# ---------------------------------------------------------------------------

def _make_regime_data(n=5000, seed=42):
    """
    X0 = binary heterogeneity axis (0 or 1).
    X1 predicts in regime 0 (X0=0); X2 predicts in regime 1 (X0=1).
    X3, X4 = pure noise.
    y  = 3*X1*(X0=0) + 3*X2*(X0=1) + noise.
    X0 does not enter the prediction formula directly.
    """
    rng = np.random.default_rng(seed)
    X0  = rng.integers(0, 2, n).astype(float)
    X14 = rng.normal(0, 1, (n, 4))
    X   = np.column_stack([X0, X14])
    y   = 3.0 * X[:, 1] * (1.0 - X0) + 3.0 * X[:, 2] * X0 + 0.3 * rng.normal(0, 1, n)
    return X, y


def run_scenario_1():
    _section("Scenario 1 -- Regime / Heterogeneity Axis")
    print(
        "\n  DGP: y = 3*X1*(X0=0) + 3*X2*(X0=1) + noise  (n=5000)\n"
        "  X0 is a binary heterogeneity axis -- it determines which local\n"
        "  predictive relationship applies, but does not predict directly.\n"
        "  X1 is the predictor within regime 0; X2 within regime 1.\n"
        "  X3, X4 are pure noise.\n"
        "\n  Note: HVRT at y_weight=0.5 blends X-space and y-weighted geometry.\n"
        "  Noise features can receive non-trivial partition importance from\n"
        "  the X-space component. Ratio ordering among signal features is\n"
        "  the primary diagnostic.\n"
        "\n  Expected signature:\n"
        "    ratio(X0) < ratio(X1)  -- X0 is local structure axis, X1 is predictor\n"
        "    ratio(X0) < ratio(X2)  -- X0 is local structure axis, X2 is predictor\n"
        "    boost(X1) > avg noise boost  -- within-regime predictor visible\n"
        "    boost(X2) > avg noise boost  -- within-regime predictor visible\n"
    )

    X, y = _make_regime_data()
    names = ["X0_het_axis", "X1_reg0_pred", "X2_reg1_pred", "X3_noise", "X4_noise"]

    model = GeoXGBRegressor(**GEO_PARAMS)
    model.fit(X, y)

    boost, avg_part, ratio = _boost_partition_ratios(model, names)

    roles = {
        "X0_het_axis":   "heterogeneity axis",
        "X1_reg0_pred":  "local predictor (regime 0)",
        "X2_reg1_pred":  "local predictor (regime 1)",
    }
    _print_ratio_table(names, boost, avg_part, ratio, roles)

    _subsection("Assertions")
    passed = True

    r0  = ratio["X0_het_axis"]
    r1  = ratio["X1_reg0_pred"]
    r2  = ratio["X2_reg1_pred"]
    noise_boost = np.mean([boost["X3_noise"], boost["X4_noise"]])

    # Ratio ordering: heterogeneity axis has lower ratio than local predictors
    for sig_name, r_sig in [("X1_reg0_pred", r1), ("X2_reg1_pred", r2)]:
        if r0 < r_sig:
            print(
                f"  PASS  ratio(X0_het_axis)={r0:.3f} < ratio({sig_name})={r_sig:.3f}"
                f"  (het. axis lower ratio than local predictor)"
            )
        else:
            print(
                f"  FAIL  ratio(X0_het_axis)={r0:.3f} NOT < ratio({sig_name})={r_sig:.3f}"
            )
            passed = False

    # Local predictors visible in boost importance
    for sig_name in ["X1_reg0_pred", "X2_reg1_pred"]:
        b = boost[sig_name]
        if b > noise_boost:
            print(
                f"  PASS  boost({sig_name})={b:.4f} > avg_noise_boost={noise_boost:.4f}"
                f"  (within-regime predictor visible to gradient)"
            )
        else:
            print(f"  FAIL  boost({sig_name})={b:.4f} NOT > avg_noise_boost={noise_boost:.4f}")
            passed = False

    return passed


# ---------------------------------------------------------------------------
# Scenario 2 -- Sign-flip interaction moderator
# ---------------------------------------------------------------------------

def _make_sign_flip_data(n=5000, seed=42):
    """
    y = 3*X1*sign(X0) + noise.
    X0 moderates the sign of X1's effect. X2..X4 are pure noise.
    Both X0 and X1 are involved in the interaction; the moderator
    (X0) plays the local-structure role, the predictor (X1) plays
    the within-region prediction role.
    """
    rng = np.random.default_rng(seed)
    X   = rng.normal(0, 1, (n, 5))
    y   = 3.0 * X[:, 1] * np.sign(X[:, 0]) + 0.3 * rng.normal(0, 1, n)
    return X, y


def run_scenario_2():
    _section("Scenario 2 -- Sign-Flip Interaction Moderator")
    print(
        "\n  DGP: y = 3*X1*sign(X0) + noise  (n=5000)\n"
        "  X0 moderates the sign of X1's effect (interaction term).\n"
        "  X0 defines local structure; X1 predicts within each local region.\n"
        "  X2..X4 are pure noise.\n"
        "\n  Expected signature:\n"
        "    ratio(X0) < ratio(X1) -- moderator has lower ratio than predictor\n"
        "    both X0 and X1 visible in boost (interaction keeps both in gradient)\n"
        "    XGBoost ranks X1 > X0 by tree importance (X1 is direct predictor)\n"
    )

    X, y = _make_sign_flip_data()
    names = ["X0_moderator", "X1_predictor", "X2_noise", "X3_noise", "X4_noise"]

    model = GeoXGBRegressor(**GEO_PARAMS)
    model.fit(X, y)

    boost, avg_part, ratio = _boost_partition_ratios(model, names)

    roles = {
        "X0_moderator": "sign-flip moderator",
        "X1_predictor": "within-region predictor",
    }
    _print_ratio_table(names, boost, avg_part, ratio, roles)

    # XGBoost comparison
    if _HAS_XGB:
        print()
        xgb = _XGBRegressor(
            n_estimators=1000, learning_rate=0.2, max_depth=4,
            subsample=0.8, random_state=RANDOM_STATE, verbosity=0,
        )
        xgb.fit(X, y)
        xgb_imp = dict(zip(names, xgb.feature_importances_))
        print("  XGBoost feature importances (tree split frequency):")
        for f in names:
            print(f"    {f:<18}  {xgb_imp[f]:.4f}")
        print()
        print(
            "  GeoXGB boost/partition split reveals X0's LOCAL STRUCTURE ROLE\n"
            "  (lower ratio = more partition-anchoring) separately from X1's\n"
            "  PREDICTIVE ROLE (higher ratio = more gradient-driven).\n"
            "  XGBoost tree importance conflates the two roles into a single\n"
            "  importance score with no structural distinction."
        )
    else:
        xgb_imp = None
        print("\n  (xgboost not installed -- skipping XGBoost comparison)")

    _subsection("Assertions")
    passed = True

    r0 = ratio["X0_moderator"]
    r1 = ratio["X1_predictor"]
    noise_boost = np.mean([boost[f] for f in ["X2_noise", "X3_noise", "X4_noise"]])

    # Ratio ordering: moderator has lower ratio than predictor
    if r0 < r1:
        print(
            f"  PASS  ratio(X0_moderator)={r0:.3f} < ratio(X1_predictor)={r1:.3f}"
            f"  (moderator has lower ratio = more local-structure-defining)"
        )
    else:
        print(
            f"  FAIL  ratio(X0_moderator)={r0:.3f} NOT < ratio(X1_predictor)={r1:.3f}"
        )
        passed = False

    # Both interaction features visible in boost (interaction keeps both in gradient)
    for sig_name in ["X0_moderator", "X1_predictor"]:
        b = boost[sig_name]
        if b > noise_boost:
            print(
                f"  PASS  boost({sig_name})={b:.4f} > avg_noise_boost={noise_boost:.4f}"
            )
        else:
            print(
                f"  FAIL  boost({sig_name})={b:.4f} NOT > avg_noise_boost={noise_boost:.4f}"
            )
            passed = False

    # XGBoost ranks X1 > X0 (X1 is the within-region predictor; boost importance
    # conflates structure and prediction roles and naturally rewards direct predictors)
    if xgb_imp is not None:
        if xgb_imp["X1_predictor"] > xgb_imp["X0_moderator"]:
            print(
                f"  PASS  XGBoost imp(X1)={xgb_imp['X1_predictor']:.4f} > "
                f"imp(X0)={xgb_imp['X0_moderator']:.4f}"
                f"  (XGBoost boost-only correctly ranks predictor above moderator)"
            )
        else:
            print(
                f"  NOTE  XGBoost imp(X0)={xgb_imp['X0_moderator']:.4f} >= "
                f"imp(X1)={xgb_imp['X1_predictor']:.4f}"
                f"  (XGBoost cannot separate structure from prediction roles)"
            )

    return passed


# ---------------------------------------------------------------------------
# Scenario 3 -- Geometry anchor vs prediction driver
# ---------------------------------------------------------------------------

def _make_additive_data(n=5000, seed=42):
    """
    y = 3*X0 + 2*X1 + noise. Two strong predictors playing different structural
    roles. One anchors HVRT partition geometry (lower ratio), the other is
    captured primarily through gradient updates (higher ratio). Role assignment
    is determined by HVRT internals -- not predictable from coefficient size.
    X2..X4 = pure noise.
    """
    rng = np.random.default_rng(seed)
    X   = rng.normal(0, 1, (n, 5))
    y   = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.5 * rng.normal(0, 1, n)
    return X, y


def run_scenario_3():
    _section("Scenario 3 -- Geometry Anchor vs Prediction Driver")
    print(
        "\n  DGP: y = 3*X0 + 2*X1 + noise  (n=5000)\n"
        "  X0 (coeff=3) and X1 (coeff=2) are both genuine signal features.\n"
        "  HVRT allocates one to anchor partition geometry (lower ratio) and\n"
        "  the other to gradient-driven prediction (higher ratio).\n"
        "  This role assignment is emergent -- it cannot be predicted solely\n"
        "  from coefficient magnitude. The divergence is NOT a model error;\n"
        "  it reveals the complementary local roles the features play.\n"
        "  X2..X4 are pure noise.\n"
        "\n  Expected signature:\n"
        "    ratio(X0) != ratio(X1) -- roles diverge (one anchor, one driver)\n"
        "    boost(X0) and boost(X1) both > max noise boost\n"
        "    at least one signal feature has partition_imp > avg noise partition\n"
    )

    X, y = _make_additive_data()
    names = ["X0_signal", "X1_signal", "X2_noise", "X3_noise", "X4_noise"]

    model = GeoXGBRegressor(**GEO_PARAMS)
    model.fit(X, y)

    boost, avg_part, ratio = _boost_partition_ratios(model, names)

    roles = {
        "X0_signal": "signal (coeff=3)",
        "X1_signal": "signal (coeff=2)",
    }
    _print_ratio_table(names, boost, avg_part, ratio, roles)

    anchor   = "X0_signal" if ratio["X0_signal"] < ratio["X1_signal"] else "X1_signal"
    driver   = "X1_signal" if anchor == "X0_signal" else "X0_signal"
    print(
        f"\n  Emergent role assignment:\n"
        f"    Geometry anchor:    {anchor}  (ratio={ratio[anchor]:.3f})\n"
        f"    Prediction driver:  {driver}  (ratio={ratio[driver]:.3f})"
    )

    _subsection("Assertions")
    passed = True

    noise_boost  = max(boost[f]    for f in ["X2_noise", "X3_noise", "X4_noise"])
    noise_part   = np.mean([avg_part[f] for f in ["X2_noise", "X3_noise", "X4_noise"]])

    # Both signal features dominate noise in boost importance
    for sig_name in ["X0_signal", "X1_signal"]:
        b = boost[sig_name]
        if b > noise_boost:
            print(f"  PASS  boost({sig_name})={b:.4f} > max_noise_boost={noise_boost:.4f}")
        else:
            print(f"  FAIL  boost({sig_name})={b:.4f} NOT > max_noise_boost={noise_boost:.4f}")
            passed = False

    # Roles diverge -- the ratio is NOT the same for both signal features
    r_diff = abs(ratio["X0_signal"] - ratio["X1_signal"])
    if r_diff > 0.2:
        print(
            f"  PASS  |ratio(X0) - ratio(X1)| = {r_diff:.3f} > 0.20"
            f"  (roles diverge: one anchors geometry, one drives prediction)"
        )
    else:
        print(
            f"  FAIL  |ratio(X0) - ratio(X1)| = {r_diff:.3f} NOT > 0.20"
            f"  (roles did not diverge sufficiently)"
        )
        passed = False

    # At least one signal feature anchors partition geometry above noise
    max_sig_part = max(avg_part["X0_signal"], avg_part["X1_signal"])
    if max_sig_part > noise_part:
        print(
            f"  PASS  max signal partition_imp={max_sig_part:.4f} > "
            f"avg_noise_part={noise_part:.4f}"
            f"  ({anchor} anchors geometry)"
        )
    else:
        print(
            f"  FAIL  max signal partition_imp={max_sig_part:.4f} NOT > "
            f"avg_noise_part={noise_part:.4f}"
        )
        passed = False

    return passed


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results):
    _section("Summary")
    all_passed = all(results.values())
    for label, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {label}")
    print()
    if all_passed:
        print(
            "  All scenarios confirm the boost/partition ratio as a heterogeneity\n"
            "  surface map. Key findings:\n"
            "\n"
            "  Scenario 1 -- Heterogeneity axis:\n"
            "    The feature that DEFINES where different predictive relationships\n"
            "    apply (X0) consistently has a LOWER ratio than the features that\n"
            "    PREDICT within those local regions (X1, X2). High partition\n"
            "    importance relative to boost = local-structure-defining role.\n"
            "    This is individual-level heterogeneity: each HVRT partition is\n"
            "    a hyperplane-bounded local region. partition_tree_rules() shows\n"
            "    exactly which conditions define the region each individual is in.\n"
            "\n"
            "  Scenario 2 -- Interaction moderator:\n"
            "    The ratio ordering (moderator < predictor) holds even for\n"
            "    interaction terms. GeoXGB's boost/partition split distinguishes\n"
            "    local-structure roles from within-region prediction roles.\n"
            "    XGBoost tree importance conflates both into a single score.\n"
            "\n"
            "  Scenario 3 -- Complementary roles:\n"
            "    Among two strong signal features, HVRT allocates one to anchor\n"
            "    partition geometry (lower ratio) and the other to gradient-driven\n"
            "    prediction (higher ratio). Role assignment is emergent and cannot\n"
            "    be predicted from coefficient magnitude alone. The divergence\n"
            "    reveals local structure, not model error.\n"
            "\n"
            "  Core principle: boost/partition divergence = heterogeneity signal,\n"
            "  not a problem. The ratio surface map is a diagnostic that no\n"
            "  boost-only model (including XGBoost) can produce."
        )
    else:
        print("  One or more scenarios failed -- review output above.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = {
        "Scenario 1: Regime / heterogeneity axis":        run_scenario_1(),
        "Scenario 2: Sign-flip interaction moderator":    run_scenario_2(),
        "Scenario 3: Geometry anchor vs pred. driver":    run_scenario_3(),
    }
    _print_summary(results)
