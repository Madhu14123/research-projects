"""
supervised_pattern_deep_analysis.py
====================================
Deep diagnostic of supervised features BEFORE model building.
1. Loads 13 high-consensus features (>=6/7 votes) + student exam scores
2. Analyses: distribution shapes, non-linearity type per feature,
   interaction effects, monotonicity, threshold/plateau detection
3. Outlier detection: students whose emotional pattern contradicts exam score
4. Dimension-level coherence: do features within a dimension agree?
5. Pairwise feature interactions: which pairs amplify/cancel each other?
6. Student-level scatter diagnostics: who are the anomalies and why?

Goal: understand the data deeply so we can choose the RIGHT model architecture.
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import (
    pearsonr, spearmanr, kendalltau, shapiro,
    skew, kurtosis, iqr,
)
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── paths ────────────────────────────────────────────────────────────────
BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

FILES = {
    "D2": os.path.join(RES, "features_D2.csv"),
    "D3": os.path.join(RES, "features_D3.csv"),
    "D4": os.path.join(RES, "features_D4.csv"),
    "D5": os.path.join(RES, "features_D5.csv"),
}
TARGET = "pct"

# ── 13 HIGH-CONSENSUS FEATURES (>=6/7 votes) ────────────────────────────
DIMENSIONS = {
    "D2_Regulation": [
        "c2_cross_stage_regulation_var",
        "c2_anticipatory_regulation",
    ],
    "D3_Utilisation": [
        "c3_total_productive_time_ratio",
        "c3_struggle_benefit_index",
        "c3_effective_time_efficiency",
        "c3_recovery_acceleration",
        "c3_task_completion_persistence",
    ],
    "D4_Self_Awareness": [
        "c4_arousal_trend",
        "c4_block_stage_awareness",
        "c4_arousal_optimality",
        "c4_arousal_midrange_proportion",
    ],
    "D5_Context_Application": [
        "c5_flow_stability",
        "c5_engagement_recovery_speed",
    ],
}

ALL_FEATURES = []
for feats in DIMENSIONS.values():
    ALL_FEATURES.extend(feats)

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("LOADING DATA")
print("=" * 72)

frames = []
for tag, path in FILES.items():
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("c")]
    keep = ["pid"] + feat_cols
    if TARGET in df.columns:
        keep.append(TARGET)
    frames.append(df[[c for c in keep if c in df.columns]])

merged = frames[0]
for f in frames[1:]:
    cols_to_use = [c for c in f.columns if c not in merged.columns or c == "pid"]
    if TARGET in f.columns and TARGET not in cols_to_use:
        pass  # already have it
    merged = merged.merge(f[cols_to_use], on="pid", how="outer")

# resolve duplicate pct
pct_cols = [c for c in merged.columns if c.startswith("pct")]
if len(pct_cols) > 1:
    merged[TARGET] = merged[pct_cols[0]]
    merged.drop(columns=pct_cols[1:], inplace=True)

y_full = merged[TARGET]
valid = y_full.notna()
pids = merged.loc[valid, "pid"].reset_index(drop=True)
y = y_full.loc[valid].reset_index(drop=True).values

available = [f for f in ALL_FEATURES if f in merged.columns]
X_raw = merged.loc[valid, available].reset_index(drop=True)
X_raw = X_raw.apply(lambda c: c.fillna(c.median()))

scaler = StandardScaler()
X_z = pd.DataFrame(scaler.fit_transform(X_raw), columns=available)

n, p = X_z.shape
print(f"  Students: {n}, Features: {p}")
print(f"  Exam range: {y.min():.1f}% -- {y.max():.1f}% (mean={y.mean():.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════
#  PART A: PER-FEATURE DEEP DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART A: PER-FEATURE PATTERN DIAGNOSTICS")
print("=" * 72)

def fit_relationships(x, y_vals):
    """Fit linear, quadratic, log, sigmoid and return R2 for each."""
    from sklearn.metrics import r2_score as r2s
    results = {}

    # 1. Linear
    try:
        z = np.polyfit(x, y_vals, 1)
        pred = np.polyval(z, x)
        results["linear"] = {"R2": max(r2s(y_vals, pred), -1), "coefs": z}
    except:
        results["linear"] = {"R2": -1, "coefs": None}

    # 2. Quadratic
    try:
        z = np.polyfit(x, y_vals, 2)
        pred = np.polyval(z, x)
        results["quadratic"] = {"R2": max(r2s(y_vals, pred), -1), "coefs": z}
    except:
        results["quadratic"] = {"R2": -1, "coefs": None}

    # 3. Cubic
    try:
        z = np.polyfit(x, y_vals, 3)
        pred = np.polyval(z, x)
        results["cubic"] = {"R2": max(r2s(y_vals, pred), -1), "coefs": z}
    except:
        results["cubic"] = {"R2": -1, "coefs": None}

    # 4. Logarithmic (only if all x > 0)
    x_shift = x - x.min() + 0.01
    try:
        z = np.polyfit(np.log(x_shift), y_vals, 1)
        pred = np.polyval(z, np.log(x_shift))
        results["logarithmic"] = {"R2": max(r2s(y_vals, pred), -1), "coefs": z}
    except:
        results["logarithmic"] = {"R2": -1, "coefs": None}

    # 5. Sigmoid / logistic
    try:
        def sigmoid(x, L, k, x0, b):
            return L / (1 + np.exp(-k * (x - x0))) + b
        p0 = [y_vals.max() - y_vals.min(), 1, np.median(x), y_vals.min()]
        popt, _ = curve_fit(sigmoid, x, y_vals, p0=p0, maxfev=2000)
        pred = sigmoid(x, *popt)
        results["sigmoid"] = {"R2": max(r2s(y_vals, pred), -1), "coefs": popt}
    except:
        results["sigmoid"] = {"R2": -1, "coefs": None}

    # 6. Piecewise (split at median)
    try:
        med = np.median(x)
        lo = x <= med
        hi = x > med
        pred = np.zeros_like(y_vals)
        if lo.sum() > 2 and hi.sum() > 2:
            z_lo = np.polyfit(x[lo], y_vals[lo], 1)
            z_hi = np.polyfit(x[hi], y_vals[hi], 1)
            pred[lo] = np.polyval(z_lo, x[lo])
            pred[hi] = np.polyval(z_hi, x[hi])
            results["piecewise"] = {"R2": max(r2s(y_vals, pred), -1),
                                    "coefs": (z_lo, z_hi)}
        else:
            results["piecewise"] = {"R2": -1, "coefs": None}
    except:
        results["piecewise"] = {"R2": -1, "coefs": None}

    return results


feature_diagnostics = []

print(f"\n  {'Feature':<40} {'lin_r':>7} {'quad_R2':>8} {'best_fit':>12} "
      f"{'best_R2':>8} {'monotonic':>9} {'direction':>9}")
print(f"  {'-'*95}")

for feat in available:
    x = X_z[feat].values
    x_raw = X_raw[feat].values

    # correlations
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    tau, tp = kendalltau(x, y)

    # monotonicity: |spearman| vs |pearson|
    # if spearman >> pearson, relationship is monotonic but non-linear
    mono_ratio = abs(sr) / max(abs(pr), 0.001)

    # fit multiple relationship types
    fits = fit_relationships(x, y)
    best_fit = max(fits.keys(), key=lambda k: fits[k]["R2"])
    best_r2 = fits[best_fit]["R2"]

    # non-linearity gain
    nl_gain = fits["quadratic"]["R2"] - fits["linear"]["R2"]

    # distribution shape
    sk = skew(x_raw[~np.isnan(x_raw)])
    ku = kurtosis(x_raw[~np.isnan(x_raw)])
    _, shap_p = shapiro(x_raw[~np.isnan(x_raw)][:50])  # shapiro max ~50

    # direction: positive or negative relationship?
    direction = "+" if pr > 0 else "-"

    # is monotonic?
    mono_flag = "YES" if abs(sr) > abs(pr) * 1.1 and abs(sr) > 0.15 else "no"

    # threshold detection: is there a sudden jump?
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    # compute rolling mean difference
    w = max(n // 5, 3)
    if n > 2 * w:
        lo_mean = np.mean(y_sorted[:w])
        hi_mean = np.mean(y_sorted[-w:])
        mid_lo = np.mean(y_sorted[n//4 - w//2 : n//4 + w//2])
        mid_hi = np.mean(y_sorted[3*n//4 - w//2 : 3*n//4 + w//2])
        # check if jump is concentrated in one region
        total_diff = hi_mean - lo_mean
        lower_half_diff = mid_lo - lo_mean
        upper_half_diff = hi_mean - mid_hi
    else:
        total_diff = lo_mean = hi_mean = 0

    diag = {
        "feature": feat,
        "dimension": feat.split("_")[0],
        "pearson_r": pr, "pearson_p": pp,
        "spearman_r": sr, "spearman_p": sp,
        "kendall_tau": tau,
        "linear_R2": fits["linear"]["R2"],
        "quadratic_R2": fits["quadratic"]["R2"],
        "cubic_R2": fits["cubic"]["R2"],
        "log_R2": fits["logarithmic"]["R2"],
        "sigmoid_R2": fits["sigmoid"]["R2"],
        "piecewise_R2": fits["piecewise"]["R2"],
        "best_fit_type": best_fit,
        "best_fit_R2": best_r2,
        "nl_gain_quad": nl_gain,
        "mono_ratio": mono_ratio,
        "is_monotonic_nonlinear": mono_flag,
        "direction": direction,
        "skewness": sk,
        "kurtosis": ku,
        "shapiro_p": shap_p,
    }
    feature_diagnostics.append(diag)

    print(f"  {feat:<40} {pr:+7.3f} {fits['quadratic']['R2']:8.4f} "
          f"{best_fit:>12} {best_r2:8.4f} {mono_flag:>9} {direction:>9}")

diag_df = pd.DataFrame(feature_diagnostics)

# Summary
print(f"\n  --- SUMMARY ---")
nl_feats = diag_df[diag_df["nl_gain_quad"] > 0.02]
print(f"  Features with non-linear gain > 0.02: {len(nl_feats)}/{len(available)}")
for _, row in nl_feats.iterrows():
    print(f"    {row['feature']}: linear R2={row['linear_R2']:.3f} -> "
          f"quad R2={row['quadratic_R2']:.3f} (gain={row['nl_gain_quad']:+.3f}), "
          f"best={row['best_fit_type']} R2={row['best_fit_R2']:.3f}")

mono_feats = diag_df[diag_df["is_monotonic_nonlinear"] == "YES"]
print(f"\n  Monotonic non-linear features: {len(mono_feats)}")
for _, row in mono_feats.iterrows():
    print(f"    {row['feature']}: |spearman|={abs(row['spearman_r']):.3f} vs "
          f"|pearson|={abs(row['pearson_r']):.3f} (ratio={row['mono_ratio']:.2f})")

# ═══════════════════════════════════════════════════════════════════════════
#  PART B: PAIRWISE INTERACTION EFFECTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART B: PAIRWISE FEATURE INTERACTIONS")
print("=" * 72)

from sklearn.metrics import r2_score

interaction_results = []
print(f"\n  Top interaction pairs (R2 gain from adding interaction term):")
print(f"  {'Feature A':<35} {'Feature B':<35} {'ind_R2':>7} {'inter_R2':>8} {'gain':>7}")
print(f"  {'-'*95}")

for i in range(len(available)):
    for j in range(i + 1, len(available)):
        fa, fb = available[i], available[j]
        xa = X_z[fa].values
        xb = X_z[fb].values

        # individual: just fa + fb
        X_ind = np.column_stack([xa, xb])
        z_ind = np.polyfit(range(n), y, 1)  # dummy
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X_ind, y)
        pred_ind = lr.predict(X_ind)
        r2_ind = r2_score(y, pred_ind)

        # with interaction: fa + fb + fa*fb
        X_inter = np.column_stack([xa, xb, xa * xb])
        lr2 = LinearRegression()
        lr2.fit(X_inter, y)
        pred_inter = lr2.predict(X_inter)
        r2_inter = r2_score(y, pred_inter)

        gain = r2_inter - r2_ind
        interaction_results.append({
            "feat_a": fa, "feat_b": fb,
            "r2_individual": r2_ind, "r2_interaction": r2_inter,
            "interaction_gain": gain,
            "interaction_coef": lr2.coef_[2],
        })

inter_df = pd.DataFrame(interaction_results).sort_values("interaction_gain", ascending=False)
top_inter = inter_df.head(15)
for _, row in top_inter.iterrows():
    print(f"  {row['feat_a']:<35} {row['feat_b']:<35} "
          f"{row['r2_individual']:7.4f} {row['r2_interaction']:8.4f} "
          f"{row['interaction_gain']:+7.4f}")

print(f"\n  Mean interaction gain: {inter_df['interaction_gain'].mean():.4f}")
print(f"  Max interaction gain: {inter_df['interaction_gain'].max():.4f}")
strong_inter = inter_df[inter_df["interaction_gain"] > 0.03]
print(f"  Pairs with gain > 0.03: {len(strong_inter)}")

# ═══════════════════════════════════════════════════════════════════════════
#  PART C: OUTLIER / ANOMALOUS STUDENT DETECTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART C: OUTLIER / ANOMALOUS STUDENT DETECTION")
print("=" * 72)

X_arr = X_z.values

# Method 1: Isolation Forest on feature space
iso = IsolationForest(contamination=0.15, random_state=42)
iso_labels = iso.fit_predict(X_arr)  # -1 = outlier
iso_scores = iso.decision_function(X_arr)

# Method 2: Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=min(10, n-1), contamination=0.15)
lof_labels = lof.fit_predict(X_arr)
lof_scores = lof.negative_outlier_factor_

# Method 3: Residual-based (students whose exam score doesn't match their features)
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge.fit(X_arr, y)
y_pred_ridge = ridge.predict(X_arr)
residuals = y - y_pred_ridge
residual_z = (residuals - np.mean(residuals)) / np.std(residuals)

# Method 4: Mahalanobis-like distance from cluster centers
# simple: compute each student's z-score composite and see if it aligns with exam
composite_z = np.mean(X_z.values * np.sign(diag_df["pearson_r"].values), axis=1)
composite_r, composite_p = pearsonr(composite_z, y)
composite_residual = y - np.polyval(np.polyfit(composite_z, y, 1), composite_z)
composite_resid_z = (composite_residual - composite_residual.mean()) / composite_residual.std()

# Combine outlier signals
outlier_df = pd.DataFrame({
    "pid": pids,
    "exam_pct": y,
    "ridge_predicted": y_pred_ridge,
    "ridge_residual": residuals,
    "residual_z": residual_z,
    "iso_score": iso_scores,
    "iso_outlier": (iso_labels == -1).astype(int),
    "lof_score": lof_scores,
    "lof_outlier": (lof_labels == -1).astype(int),
    "composite_z": composite_z,
    "composite_resid_z": composite_resid_z,
})

# flag severe outliers: those flagged by >= 2 methods OR |residual_z| > 2
outlier_df["n_outlier_flags"] = (
    outlier_df["iso_outlier"] +
    outlier_df["lof_outlier"] +
    (np.abs(outlier_df["residual_z"]) > 2.0).astype(int) +
    (np.abs(outlier_df["composite_resid_z"]) > 2.0).astype(int)
)
outlier_df["is_anomalous"] = (outlier_df["n_outlier_flags"] >= 2).astype(int)

# EI-Exam inversion: high EI pattern but low exam, or vice versa
# Direction: positive composite_z should mean higher exam
outlier_df["ei_exam_mismatch"] = 0
# high EI (top 25% composite) but low exam (bottom 25%)
q75_comp = np.percentile(composite_z, 75)
q25_comp = np.percentile(composite_z, 25)
q75_exam = np.percentile(y, 75)
q25_exam = np.percentile(y, 25)

hi_ei_lo_exam = (composite_z >= q75_comp) & (y <= q25_exam)
lo_ei_hi_exam = (composite_z <= q25_comp) & (y >= q75_exam)
outlier_df.loc[hi_ei_lo_exam, "ei_exam_mismatch"] = 1    # high EI, low exam
outlier_df.loc[lo_ei_hi_exam, "ei_exam_mismatch"] = -1   # low EI, high exam

print(f"\n  Composite EI vs exam: r={composite_r:.4f}, p={composite_p:.4f}")
print(f"\n  Outlier detection summary:")
print(f"    Isolation Forest outliers: {outlier_df['iso_outlier'].sum()}")
print(f"    LOF outliers: {outlier_df['lof_outlier'].sum()}")
print(f"    Large residual (|z|>2): {(np.abs(outlier_df['residual_z'])>2).sum()}")
print(f"    Multi-method anomalous (>=2 flags): {outlier_df['is_anomalous'].sum()}")
print(f"    EI-Exam mismatch (high EI, low exam): {(outlier_df['ei_exam_mismatch']==1).sum()}")
print(f"    EI-Exam mismatch (low EI, high exam): {(outlier_df['ei_exam_mismatch']==-1).sum()}")

# Print detailed anomalous students
anomalous = outlier_df[outlier_df["is_anomalous"] == 1].sort_values("residual_z", key=abs, ascending=False)
print(f"\n  ANOMALOUS STUDENTS (flagged by >=2 methods):")
print(f"  {'PID':<8} {'Exam%':>6} {'Predicted':>9} {'Residual':>9} {'Resid_z':>8} "
      f"{'Comp_z':>7} {'IsoF':>5} {'LOF':>5} {'Mismatch':>8}")
print(f"  {'-'*75}")
for _, row in anomalous.iterrows():
    mismatch_str = {1: "HiEI/LoEx", -1: "LoEI/HiEx", 0: "---"}[row["ei_exam_mismatch"]]
    print(f"  {row['pid']:<8} {row['exam_pct']:6.1f} {row['ridge_predicted']:9.1f} "
          f"{row['ridge_residual']:+9.1f} {row['residual_z']:+8.2f} "
          f"{row['composite_z']:+7.2f} {row['iso_outlier']:5d} {row['lof_outlier']:5d} "
          f"{mismatch_str:>8}")

# EI mismatch detail
mismatch = outlier_df[outlier_df["ei_exam_mismatch"] != 0].sort_values("ei_exam_mismatch")
if len(mismatch) > 0:
    print(f"\n  EI-EXAM MISMATCH STUDENTS (inverse relationship):")
    print(f"  {'PID':<8} {'Exam%':>6} {'Comp_z':>7} {'Type':>12}")
    print(f"  {'-'*40}")
    for _, row in mismatch.iterrows():
        mtype = "HiEI/LoExam" if row["ei_exam_mismatch"] == 1 else "LoEI/HiExam"
        print(f"  {row['pid']:<8} {row['exam_pct']:6.1f} {row['composite_z']:+7.2f} {mtype:>12}")

# ═══════════════════════════════════════════════════════════════════════════
#  PART D: WITHIN-DIMENSION COHERENCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART D: WITHIN-DIMENSION COHERENCE")
print("=" * 72)

for dim_name, feats in DIMENSIONS.items():
    present = [f for f in feats if f in available]
    if len(present) < 2:
        print(f"\n  {dim_name}: only {len(present)} feature(s), skipping coherence")
        continue

    print(f"\n  {dim_name} ({len(present)} features):")

    # pairwise correlations within dimension
    dim_data = X_z[present].values
    print(f"    Pairwise correlations:")
    for i in range(len(present)):
        for j in range(i+1, len(present)):
            r, p = pearsonr(dim_data[:, i], dim_data[:, j])
            print(f"      {present[i]} <-> {present[j]}: r={r:+.3f} (p={p:.4f})")

    # Cronbach's alpha
    if len(present) >= 2:
        k = len(present)
        item_vars = np.var(dim_data, axis=0, ddof=1)
        total_var = np.var(np.sum(dim_data, axis=1), ddof=1)
        alpha = (k / (k - 1)) * (1 - np.sum(item_vars) / total_var)
        print(f"    Cronbach's alpha: {alpha:.3f} "
              f"({'good' if alpha > 0.7 else 'acceptable' if alpha > 0.5 else 'poor'})")

    # dimension composite vs exam
    # sign-correct features based on pearson with exam
    signs = np.array([np.sign(pearsonr(X_z[f].values, y)[0]) for f in present])
    dim_composite = np.mean(dim_data * signs, axis=1)
    r, p = pearsonr(dim_composite, y)
    sr, sp = spearmanr(dim_composite, y)
    print(f"    Dimension composite vs exam: pearson r={r:.4f} (p={p:.4f}), "
          f"spearman rho={sr:.4f} (p={sp:.4f})")

    # PCA on dimension
    from sklearn.decomposition import PCA as PCA_
    pca_dim = PCA_(n_components=min(len(present), len(present)))
    pca_dim.fit(dim_data)
    print(f"    PCA variance explained: {pca_dim.explained_variance_ratio_}")
    print(f"    PC1 explains: {pca_dim.explained_variance_ratio_[0]*100:.1f}% "
          f"({'unidimensional' if pca_dim.explained_variance_ratio_[0] > 0.5 else 'multidimensional'})")

# ═══════════════════════════════════════════════════════════════════════════
#  PART E: EFFECT OF REMOVING OUTLIERS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART E: IMPACT OF REMOVING OUTLIER STUDENTS")
print("=" * 72)

# Try removing anomalous students and see if correlations improve
anom_mask = outlier_df["is_anomalous"].values == 1
n_anom = anom_mask.sum()
clean_mask = ~anom_mask

print(f"\n  Removing {n_anom} anomalous students ({n} -> {clean_mask.sum()})")
print(f"\n  {'Feature':<40} {'r_full':>8} {'r_clean':>8} {'delta':>8}")
print(f"  {'-'*68}")

for feat in available:
    x_full = X_z[feat].values
    r_full, _ = pearsonr(x_full, y)
    r_clean, _ = pearsonr(x_full[clean_mask], y[clean_mask])
    delta = abs(r_clean) - abs(r_full)
    flag = " ***" if delta > 0.05 else ""
    print(f"  {feat:<40} {r_full:+8.4f} {r_clean:+8.4f} {delta:+8.4f}{flag}")

# overall composite
comp_full_r, _ = pearsonr(composite_z, y)
comp_clean_r, _ = pearsonr(composite_z[clean_mask], y[clean_mask])
print(f"\n  Composite EI:  full r={comp_full_r:.4f}, clean r={comp_clean_r:.4f}, "
      f"delta={abs(comp_clean_r)-abs(comp_full_r):+.4f}")

# Also try removing just the EI-Exam mismatch students
mismatch_mask = outlier_df["ei_exam_mismatch"].values != 0
n_mismatch = mismatch_mask.sum()
no_mismatch = ~mismatch_mask

if n_mismatch > 0:
    print(f"\n  Removing {n_mismatch} EI-Exam mismatch students ({n} -> {no_mismatch.sum()})")
    comp_nomis_r, _ = pearsonr(composite_z[no_mismatch], y[no_mismatch])
    print(f"  Composite EI without mismatch: r={comp_nomis_r:.4f} "
          f"(delta={abs(comp_nomis_r)-abs(comp_full_r):+.4f})")

    for feat in available:
        x_f = X_z[feat].values
        r_full, _ = pearsonr(x_f, y)
        r_nomis, _ = pearsonr(x_f[no_mismatch], y[no_mismatch])
        if abs(r_nomis) - abs(r_full) > 0.05:
            print(f"    {feat}: r={r_full:+.4f} -> r={r_nomis:+.4f} "
                  f"(gain={abs(r_nomis)-abs(r_full):+.4f})")

# ═══════════════════════════════════════════════════════════════════════════
#  PART F: OVERALL PATTERN TYPE DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART F: OVERALL PATTERN DIAGNOSIS & RECOMMENDATIONS")
print("=" * 72)

# 1. Non-linearity summary
avg_nl_gain = diag_df["nl_gain_quad"].mean()
max_nl_gain = diag_df["nl_gain_quad"].max()
pct_nl = (diag_df["nl_gain_quad"] > 0.02).mean() * 100

print(f"\n  NON-LINEARITY:")
print(f"    Mean quadratic R2 gain over linear: {avg_nl_gain:.4f}")
print(f"    Max quadratic R2 gain: {max_nl_gain:.4f}")
print(f"    % features with meaningful non-linearity: {pct_nl:.1f}%")
if pct_nl > 50:
    nl_verdict = "STRONG non-linearity -- use non-linear models (GBR, XGB, SVR-poly)"
elif pct_nl > 20:
    nl_verdict = "MODERATE non-linearity -- ensemble of linear + non-linear models"
else:
    nl_verdict = "WEAK non-linearity -- linear models may suffice, try Ridge/ElasticNet"
print(f"    Verdict: {nl_verdict}")

# 2. Interaction effects
mean_inter_gain = inter_df["interaction_gain"].mean()
pct_strong_inter = (inter_df["interaction_gain"] > 0.03).mean() * 100
print(f"\n  INTERACTIONS:")
print(f"    Mean interaction gain: {mean_inter_gain:.4f}")
print(f"    % pairs with strong interaction (>0.03): {pct_strong_inter:.1f}%")
if pct_strong_inter > 30:
    inter_verdict = "STRONG interactions -- include interaction terms or use tree models"
elif pct_strong_inter > 10:
    inter_verdict = "MODERATE interactions -- tree-based models will capture these"
else:
    inter_verdict = "WEAK interactions -- simple additive models may work"
print(f"    Verdict: {inter_verdict}")

# 3. Monotonicity
pct_mono = (diag_df["is_monotonic_nonlinear"] == "YES").mean() * 100
print(f"\n  MONOTONICITY:")
print(f"    % monotonic non-linear: {pct_mono:.1f}%")
if pct_mono > 40:
    mono_verdict = "Many monotonic non-linear -- rank-based models or isotonic regression useful"
else:
    mono_verdict = "Mostly linear monotonic -- standard approaches work"
print(f"    Verdict: {mono_verdict}")

# 4. Outlier impact
print(f"\n  OUTLIER IMPACT:")
print(f"    Anomalous students: {n_anom}/{n} ({n_anom/n*100:.1f}%)")
print(f"    EI-Exam mismatch: {n_mismatch}")
print(f"    Composite r improvement after removing anomalies: "
      f"{abs(comp_clean_r)-abs(comp_full_r):+.4f}")
if abs(comp_clean_r) - abs(comp_full_r) > 0.1:
    out_verdict = "LARGE outlier impact -- consider robust models or outlier removal"
elif abs(comp_clean_r) - abs(comp_full_r) > 0.03:
    out_verdict = "MODERATE outlier impact -- use robust estimators (Huber, RANSAC)"
else:
    out_verdict = "SMALL outlier impact -- outliers not distorting main signal"
print(f"    Verdict: {out_verdict}")

# 5. Dimension coherence summary
print(f"\n  DIMENSION COHERENCE:")
for dim_name, feats in DIMENSIONS.items():
    present = [f for f in feats if f in available]
    if len(present) >= 2:
        dim_data = X_z[present].values
        signs = np.array([np.sign(pearsonr(X_z[f].values, y)[0]) for f in present])
        dim_comp = np.mean(dim_data * signs, axis=1)
        r, _ = pearsonr(dim_comp, y)
        print(f"    {dim_name}: composite r={r:+.4f} ({len(present)} features)")
    else:
        print(f"    {dim_name}: {len(present)} feature(s)")

# 6. Best fit type distribution
print(f"\n  BEST FIT TYPE DISTRIBUTION:")
for fit_type in diag_df["best_fit_type"].value_counts().items():
    print(f"    {fit_type[0]}: {fit_type[1]} features")

# ═══════════════════════════════════════════════════════════════════════════
#  SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════
diag_df.to_csv(os.path.join(RES, "supervised_feature_diagnostics.csv"), index=False)
outlier_df.to_csv(os.path.join(RES, "student_outlier_analysis.csv"), index=False)
inter_df.head(30).to_csv(os.path.join(RES, "top_feature_interactions.csv"), index=False)
print(f"\n  Saved: supervised_feature_diagnostics.csv")
print(f"  Saved: student_outlier_analysis.csv")
print(f"  Saved: top_feature_interactions.csv")

# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Generating diagnostic plots...")

# ── PLOT 1: 13-panel feature scatter with fit curves ─────────────────────
n_feats = len(available)
ncols = 4
nrows = (n_feats + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
fig.suptitle("Per-Feature Pattern: Exam % vs Feature (with best-fit curves)",
             fontsize=16, fontweight="bold", y=1.01)

for idx, feat in enumerate(available):
    ax = axes.flat[idx]
    x = X_z[feat].values
    diag_row = diag_df[diag_df["feature"] == feat].iloc[0]

    # color anomalous students differently
    normal = ~anom_mask
    ax.scatter(x[normal], y[normal], c="steelblue", alpha=0.6, s=40, edgecolors="k",
               linewidth=0.5, label="Normal", zorder=3)
    if anom_mask.sum() > 0:
        ax.scatter(x[anom_mask], y[anom_mask], c="red", alpha=0.8, s=60, marker="X",
                   edgecolors="k", linewidth=0.5, label="Anomalous", zorder=4)

    # label mismatch students
    mismatch_idx = outlier_df["ei_exam_mismatch"].values != 0
    for i in np.where(mismatch_idx)[0]:
        ax.annotate(pids.iloc[i], (x[i], y[i]), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    # fit curves
    x_line = np.linspace(x.min(), x.max(), 100)
    # linear
    z_lin = np.polyfit(x, y, 1)
    ax.plot(x_line, np.polyval(z_lin, x_line), "b--", alpha=0.5, linewidth=1, label="Linear")
    # quadratic
    z_quad = np.polyfit(x, y, 2)
    ax.plot(x_line, np.polyval(z_quad, x_line), "r-", alpha=0.7, linewidth=1.5, label="Quadratic")

    r = diag_row["pearson_r"]
    best = diag_row["best_fit_type"]
    best_r2 = diag_row["best_fit_R2"]
    nl = diag_row["nl_gain_quad"]

    ax.set_title(f"{feat}\nr={r:+.3f}, best={best}(R2={best_r2:.3f}), NL={nl:+.3f}",
                 fontsize=8)
    ax.set_xlabel("Feature (z-score)", fontsize=7)
    ax.set_ylabel("Exam %", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    if idx == 0:
        ax.legend(fontsize=6, loc="upper left")

# remove empty subplots
for idx in range(n_feats, nrows * ncols):
    axes.flat[idx].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(RES, "supervised_pattern_diagnostics.png"), dpi=150,
            bbox_inches="tight")
print("  Saved: supervised_pattern_diagnostics.png")

# ── PLOT 2: Outlier diagnostic ──────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle("Student Outlier / Anomaly Diagnostics", fontsize=14, fontweight="bold")

# 2a: Composite EI vs Exam (with anomalies marked)
ax = axes2[0, 0]
ax.scatter(composite_z[~anom_mask], y[~anom_mask], c="steelblue", alpha=0.6,
           s=50, edgecolors="k", label="Normal")
ax.scatter(composite_z[anom_mask], y[anom_mask], c="red", alpha=0.8, s=80,
           marker="X", edgecolors="k", label="Anomalous")
# annotate mismatch students
for i in np.where(outlier_df["ei_exam_mismatch"].values != 0)[0]:
    ax.annotate(pids.iloc[i], (composite_z[i], y[i]), fontsize=8,
                fontweight="bold", color="darkred",
                xytext=(5, 5), textcoords="offset points")
z_fit = np.polyfit(composite_z, y, 1)
xline = np.linspace(composite_z.min(), composite_z.max(), 50)
ax.plot(xline, np.polyval(z_fit, xline), "r--", lw=2)
ax.set_xlabel("Composite EI (z-score)")
ax.set_ylabel("Exam %")
ax.set_title(f"Composite EI vs Exam\nr={composite_r:.3f} (full), "
             f"r={comp_clean_r:.3f} (clean)")
ax.legend()
ax.grid(True, alpha=0.3)

# 2b: Residual plot
ax = axes2[0, 1]
ax.scatter(y_pred_ridge, residuals, c="steelblue", alpha=0.6, s=50, edgecolors="k")
ax.scatter(y_pred_ridge[anom_mask], residuals[anom_mask], c="red", alpha=0.8,
           s=80, marker="X", edgecolors="k")
ax.axhline(0, color="k", linewidth=1)
ax.axhline(2 * np.std(residuals), color="r", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axhline(-2 * np.std(residuals), color="r", linewidth=0.8, linestyle="--", alpha=0.5)
for i in np.where(anom_mask)[0]:
    ax.annotate(pids.iloc[i], (y_pred_ridge[i], residuals[i]), fontsize=7)
ax.set_xlabel("Predicted Exam %")
ax.set_ylabel("Residual (Actual - Predicted)")
ax.set_title("Ridge Residual Plot")
ax.grid(True, alpha=0.3)

# 2c: PCA 2D with anomalies
ax = axes2[1, 0]
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_arr)
scatter = ax.scatter(X_pca[~anom_mask, 0], X_pca[~anom_mask, 1],
                     c=y[~anom_mask], cmap="RdYlGn", alpha=0.7,
                     s=60, edgecolors="k", vmin=y.min(), vmax=y.max())
ax.scatter(X_pca[anom_mask, 0], X_pca[anom_mask, 1], c="red", alpha=0.8,
           s=100, marker="X", edgecolors="k", linewidth=1.5)
for i in np.where(anom_mask)[0]:
    ax.annotate(pids.iloc[i], (X_pca[i, 0], X_pca[i, 1]), fontsize=7)
plt.colorbar(scatter, ax=ax, label="Exam %")
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA Feature Space (color=exam, X=anomaly)")
ax.grid(True, alpha=0.3)

# 2d: Isolation Forest score distribution
ax = axes2[1, 1]
ax.hist(iso_scores[~anom_mask], bins=15, color="steelblue", alpha=0.6,
        label="Normal", edgecolor="k")
ax.hist(iso_scores[anom_mask], bins=8, color="red", alpha=0.6,
        label="Anomalous", edgecolor="k")
ax.set_xlabel("Isolation Forest Score")
ax.set_ylabel("Count")
ax.set_title("Anomaly Score Distribution")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(RES, "student_outlier_diagnostics.png"), dpi=150,
             bbox_inches="tight")
print("  Saved: student_outlier_diagnostics.png")

# ── PLOT 3: Interaction heatmap ──────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 10))
inter_matrix = np.zeros((len(available), len(available)))
for _, row in inter_df.iterrows():
    i = available.index(row["feat_a"])
    j = available.index(row["feat_b"])
    inter_matrix[i, j] = row["interaction_gain"]
    inter_matrix[j, i] = row["interaction_gain"]

im = ax3.imshow(inter_matrix, cmap="YlOrRd", aspect="auto")
ax3.set_xticks(range(len(available)))
ax3.set_yticks(range(len(available)))
short_names = [f.replace("c2_", "").replace("c3_", "").replace("c4_", "").replace("c5_", "")[:20]
               for f in available]
ax3.set_xticklabels(short_names, rotation=90, fontsize=7)
ax3.set_yticklabels(short_names, fontsize=7)
for i in range(len(available)):
    for j in range(len(available)):
        if inter_matrix[i, j] > 0.02:
            ax3.text(j, i, f"{inter_matrix[i, j]:.2f}", ha="center", va="center",
                     fontsize=6, color="white" if inter_matrix[i, j] > 0.06 else "black")
plt.colorbar(im, ax=ax3, label="Interaction R2 Gain")
ax3.set_title("Feature Interaction Strength (R2 gain from multiplicative term)",
              fontweight="bold")

# draw dimension boxes
dim_boundaries = []
start = 0
for dim_name, feats in DIMENSIONS.items():
    present = [f for f in feats if f in available]
    end = start + len(present)
    dim_boundaries.append((start, end, dim_name))
    rect = plt.Rectangle((start - 0.5, start - 0.5), len(present), len(present),
                          fill=False, edgecolor="blue", linewidth=2)
    ax3.add_patch(rect)
    start = end

plt.tight_layout()
fig3.savefig(os.path.join(RES, "feature_interaction_heatmap.png"), dpi=150,
             bbox_inches="tight")
print("  Saved: feature_interaction_heatmap.png")

# ── PLOT 4: Non-linearity comparison bar chart ──────────────────────────
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle("Non-linearity Analysis", fontsize=14, fontweight="bold")

# 4a: R2 by fit type for each feature
fit_types = ["linear_R2", "quadratic_R2", "cubic_R2", "log_R2", "sigmoid_R2", "piecewise_R2"]
fit_labels = ["Linear", "Quadratic", "Cubic", "Log", "Sigmoid", "Piecewise"]
x_pos = np.arange(len(available))
width = 0.12
for i, (ft, fl) in enumerate(zip(fit_types, fit_labels)):
    vals = diag_df[ft].values.clip(-0.1, None)
    ax4a.bar(x_pos + i * width, vals, width, label=fl, alpha=0.8)
ax4a.set_xticks(x_pos + width * 2.5)
ax4a.set_xticklabels(short_names, rotation=90, fontsize=7)
ax4a.set_ylabel("R2")
ax4a.set_title("R2 by Fit Type per Feature")
ax4a.legend(fontsize=7, ncol=3)
ax4a.grid(True, alpha=0.2)

# 4b: NL gain (quad - linear)
colors_bar = ["#e74c3c" if g > 0.02 else "#3498db" for g in diag_df["nl_gain_quad"]]
ax4b.bar(x_pos, diag_df["nl_gain_quad"].values, color=colors_bar, edgecolor="k",
         linewidth=0.5)
ax4b.axhline(0.02, color="red", linestyle="--", linewidth=1, label="Threshold (0.02)")
ax4b.set_xticks(x_pos)
ax4b.set_xticklabels(short_names, rotation=90, fontsize=7)
ax4b.set_ylabel("Quadratic R2 - Linear R2")
ax4b.set_title("Non-linearity Gain per Feature (red = significant)")
ax4b.legend()
ax4b.grid(True, alpha=0.2)

plt.tight_layout()
fig4.savefig(os.path.join(RES, "nonlinearity_analysis.png"), dpi=150, bbox_inches="tight")
print("  Saved: nonlinearity_analysis.png")

print("\n" + "=" * 72)
print("DONE -- Deep pattern analysis complete.")
print("Review the diagnostics above and the 4 plot files to decide on model architecture.")
print("=" * 72)
