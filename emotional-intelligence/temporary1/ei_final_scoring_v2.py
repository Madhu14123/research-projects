"""
ei_final_scoring_v2.py
======================
CORRECTED: Build final EI score from 15 optimized features using Ridge regression.

KEY FIX: Reconstruct derived features exactly as in interaction_discovery.py
to preserve correlations.
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")
TARGET = "pct"

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("FINAL EI SCORING MODEL (V2 - CORRECTED)")
print("=" * 72)

FILES = {
    "D2": os.path.join(RES, "features_D2.csv"),
    "D3": os.path.join(RES, "features_D3.csv"),
    "D4": os.path.join(RES, "features_D4.csv"),
    "D5": os.path.join(RES, "features_D5.csv"),
}

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
    merged = merged.merge(f[cols_to_use], on="pid", how="outer")

pct_cols = [c for c in merged.columns if c.startswith("pct")]
if len(pct_cols) > 1:
    merged[TARGET] = merged[pct_cols[0]]
    merged.drop(columns=pct_cols[1:], inplace=True)

y_full = merged[TARGET]
valid = y_full.notna()
pids = merged.loc[valid, "pid"].reset_index(drop=True)
y = y_full.loc[valid].reset_index(drop=True).values

BASE_FEATURES = [
    "c2_cross_stage_regulation_var",
    "c2_anticipatory_regulation",
    "c3_total_productive_time_ratio",
    "c3_struggle_benefit_index",
    "c3_effective_time_efficiency",
    "c3_recovery_acceleration",
    "c3_task_completion_persistence",
    "c4_arousal_trend",
    "c4_block_stage_awareness",
    "c4_arousal_optimality",
    "c4_arousal_midrange_proportion",
    "c5_flow_stability",
    "c5_engagement_recovery_speed",
]

available = [f for f in BASE_FEATURES if f in merged.columns]
X_raw = merged.loc[valid, available].reset_index(drop=True)
X_raw = X_raw.apply(lambda c: c.fillna(c.median()))

scaler = StandardScaler()
X_z = pd.DataFrame(scaler.fit_transform(X_raw), columns=available)

n = len(y)
print(f"\nLoaded {n} students with {len(available)} base features")
print(f"Exam range: {y.min():.1f}% - {y.max():.1f}% (mean={y.mean():.1f}%)\n")

# ═══════════════════════════════════════════════════════════════════════════
#  BUILD 15 OPTIMAL FEATURES (CORRECTED)
# ═══════════════════════════════════════════════════════════════════════════
print("Building 15 optimal features (with corrected ratio construction)...")
print("-" * 72)

features_dict = {}

# Original 8 features
orig_feats = [
    "c3_total_productive_time_ratio",
    "c4_block_stage_awareness",
    "c4_arousal_trend",
    "c3_effective_time_efficiency",
    "c4_arousal_optimality",
    "c3_task_completion_persistence",
    "c2_anticipatory_regulation",
    "c5_flow_stability",
]

for feat in orig_feats:
    if feat in available:
        features_dict[feat] = X_z[feat].values

print(f"\n1. Original features ({len(orig_feats)}):")
for i, feat in enumerate(orig_feats, 1):
    if feat in available:
        r, p = pearsonr(features_dict[feat], y)
        print(f"   {i}. {feat:<45} r={r:+.4f}")

# ─────────────────────────────────────────────────────────────────────────
# DERIVED FEATURES (with proper ratio construction)
# ─────────────────────────────────────────────────────────────────────────

def make_ratio(xa, xb):
    """Safely compute ratio A/B, preserving sign of denominator."""
    denom = xb.copy()
    # only clip near-zero denominators, preserve sign
    mask = np.abs(denom) < 0.01
    denom[mask] = np.sign(denom[mask] + 0.001) * 0.01
    # clip extreme ratios
    ratio = xa / denom
    q01, q99 = np.percentile(ratio, [1, 99])
    ratio = np.clip(ratio, q01, q99)
    return (ratio - ratio.mean()) / (ratio.std() + 1e-10)

def make_multiply(xa, xb):
    """Compute product A*B with clipping."""
    inter = xa * xb
    q01, q99 = np.percentile(inter, [1, 99])
    inter = np.clip(inter, q01, q99)
    return (inter - inter.mean()) / (inter.std() + 1e-10)

def make_triple(xa, xb, xc):
    """Compute product A*B*C with clipping."""
    inter = xa * xb * xc
    q01, q99 = np.percentile(inter, [1, 99])
    inter = np.clip(inter, q01, q99)
    return (inter - inter.mean()) / (inter.std() + 1e-10)

# Derived features spec
derived_specs = [
    ("c4_arousal_optimality / c4_arousal_trend", "ratio",
     lambda: make_ratio(X_z["c4_arousal_optimality"].values, X_z["c4_arousal_trend"].values)),

    ("c2_cross_stage_regulation_var * c2_anticipatory_regulation", "multiply",
     lambda: make_multiply(X_z["c2_cross_stage_regulation_var"].values, X_z["c2_anticipatory_regulation"].values)),

    ("c2_cross_stage_regulation_var / c4_block_stage_awareness", "ratio",
     lambda: make_ratio(X_z["c2_cross_stage_regulation_var"].values, X_z["c4_block_stage_awareness"].values)),

    ("c3_struggle_benefit_index / c4_arousal_midrange_proportion", "ratio",
     lambda: make_ratio(X_z["c3_struggle_benefit_index"].values, X_z["c4_arousal_midrange_proportion"].values)),

    ("c3_total_productive_time_ratio / c4_block_stage_awareness", "ratio",
     lambda: make_ratio(X_z["c3_total_productive_time_ratio"].values, X_z["c4_block_stage_awareness"].values)),

    ("c3_effective_time_efficiency / c3_struggle_benefit_index", "ratio",
     lambda: make_ratio(X_z["c3_effective_time_efficiency"].values, X_z["c3_struggle_benefit_index"].values)),

    ("c2_cross_stage_regulation_var * c3_total_productive_time_ratio * c4_arousal_midrange_proportion", "triple",
     lambda: make_triple(X_z["c2_cross_stage_regulation_var"].values,
                        X_z["c3_total_productive_time_ratio"].values,
                        X_z["c4_arousal_midrange_proportion"].values)),
]

print(f"\n2. Derived features ({len(derived_specs)}):")
for i, (name, ftype, func) in enumerate(derived_specs, 1):
    try:
        vals = func()
        features_dict[name] = vals
        r, p = pearsonr(vals, y)
        print(f"   {i}. {name:<50} ({ftype:<8}) r={r:+.4f}")
    except Exception as e:
        print(f"   {i}. {name:<50} ERROR: {str(e)[:30]}")

print(f"\nTotal features prepared: {len(features_dict)}")

# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN RIDGE WITH LOOCV
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("LOOCV RIDGE REGRESSION (α=1)")
print("=" * 72)

feature_names_ordered = list(features_dict.keys())
X_model = np.column_stack([features_dict[name] for name in feature_names_ordered])

# LOOCV predictions
loo = LeaveOneOut()
ei_score_loocv = np.full(n, np.nan)
model_ridge = Ridge(alpha=1.0)

for tr, te in loo.split(X_model):
    m = clone(model_ridge)
    m.fit(X_model[tr], y[tr])
    ei_score_loocv[te] = m.predict(X_model[te])

# Validation metrics
r_loocv, p_loocv = pearsonr(y, ei_score_loocv)
sr_loocv, sp_loocv = spearmanr(y, ei_score_loocv)
r2_loocv = r2_score(y, ei_score_loocv)
rmse_loocv = np.sqrt(mean_squared_error(y, ei_score_loocv))
mae_loocv = mean_absolute_error(y, ei_score_loocv)

print(f"\nValidation Results (LOOCV):")
print(f"  Pearson r:        {r_loocv:+.4f} (p={p_loocv:.6f})")
print(f"  Spearman rho:     {sr_loocv:+.4f} (p={sp_loocv:.6f})")
print(f"  R² (adj):         {r2_loocv:.4f}")
print(f"  RMSE:             {rmse_loocv:.2f}%")
print(f"  MAE:              {mae_loocv:.2f}%")

# Normalize EI score to 0-100 scale
ei_min, ei_max = ei_score_loocv.min(), ei_score_loocv.max()
ei_score_normalized = 50 + 25 * (ei_score_loocv - ei_score_loocv.mean()) / ei_score_loocv.std()
ei_score_normalized = np.clip(ei_score_normalized, 0, 100)

print(f"\nEI Score Range (0-100): {ei_score_normalized.min():.1f} - {ei_score_normalized.max():.1f}")
print(f"EI Score Mean:         {ei_score_normalized.mean():.1f}")
print(f"EI Score SD:           {ei_score_normalized.std():.1f}")

# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT EI SCORES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FINAL EI SCORES")
print("=" * 72)

output_df = pd.DataFrame({
    "pid": pids,
    "exam_pct": y,
    "EI_score_raw": ei_score_loocv,
    "EI_score_0to100": ei_score_normalized,
})

output_df = output_df.sort_values("EI_score_0to100", ascending=False).reset_index(drop=True)
output_df["rank"] = range(1, len(output_df) + 1)

output_df.to_csv(os.path.join(RES, "ei_final_scores_v2.csv"), index=False)
print(f"\nSaved: ei_final_scores_v2.csv")

print(f"\nTop 10 students (by EI score):")
print(output_df[["rank", "pid", "exam_pct", "EI_score_0to100"]].head(10).to_string(index=False))

print(f"\nBottom 10 students (by EI score):")
print(output_df[["rank", "pid", "exam_pct", "EI_score_0to100"]].tail(10).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("GENERATING VISUALIZATIONS")
print("=" * 72)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("EI Scoring Model: Validation & Results", fontsize=16, fontweight="bold")

# 1. Main scatter: EI vs Exam
ax = axes[0, 0]
ax.scatter(ei_score_normalized, y, alpha=0.7, s=100, edgecolors="k", linewidth=0.5,
           c=ei_score_normalized, cmap="RdYlGn", vmin=0, vmax=100)
z = np.polyfit(ei_score_normalized, y, 1)
xline = np.linspace(ei_score_normalized.min(), ei_score_normalized.max(), 50)
ax.plot(xline, np.polyval(z, xline), "r--", linewidth=2, label="Linear fit")
ax.set_xlabel("EI Score (0-100)", fontsize=11, fontweight="bold")
ax.set_ylabel("Exam %", fontsize=11, fontweight="bold")
ax.set_title(f"EI Score vs Exam Performance\nr={r_loocv:.4f}, p={p_loocv:.2e}, R²={r2_loocv:.4f}",
             fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

for idx, row in output_df.iterrows():
    if idx < 3 or idx >= len(output_df) - 3:
        ax.annotate(row["pid"], (row["EI_score_0to100"], row["exam_pct"]),
                   fontsize=8, alpha=0.7, xytext=(3, 3), textcoords="offset points")

# 2. Distribution of EI scores
ax = axes[0, 1]
ax.hist(ei_score_normalized, bins=15, color="steelblue", edgecolor="k", alpha=0.7)
ax.axvline(ei_score_normalized.mean(), color="red", linestyle="--", linewidth=2,
           label=f"Mean: {ei_score_normalized.mean():.1f}")
ax.axvline(np.median(ei_score_normalized), color="green", linestyle=":", linewidth=2,
           label=f"Median: {np.median(ei_score_normalized):.1f}")
ax.set_xlabel("EI Score (0-100)", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of Students", fontsize=11, fontweight="bold")
ax.set_title("Distribution of EI Scores", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3. Residual plot
ax = axes[1, 0]
residuals = y - ei_score_loocv
ax.scatter(ei_score_loocv, residuals, alpha=0.7, s=100, edgecolors="k", linewidth=0.5,
          c="steelblue")
ax.axhline(0, color="red", linestyle="--", linewidth=2)
ax.axhline(2*np.std(residuals), color="orange", linestyle=":", linewidth=1, alpha=0.5,
           label="±2 SD")
ax.axhline(-2*np.std(residuals), color="orange", linestyle=":", linewidth=1, alpha=0.5)
ax.set_xlabel("Predicted EI (raw)", fontsize=11, fontweight="bold")
ax.set_ylabel("Residual (Exam - Predicted)", fontsize=11, fontweight="bold")
ax.set_title("Residual Plot (LOOCV)", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# 4. Ranked students
ax = axes[1, 1]
sorted_idx = np.argsort(ei_score_normalized)
ax.plot(range(n), ei_score_normalized[sorted_idx], "b-", linewidth=2, label="EI Score")
ax.plot(range(n), (y[sorted_idx] - y[sorted_idx].min()) / (y[sorted_idx].max() - y[sorted_idx].min()) * 100,
        "r--", linewidth=2, label="Exam % (scaled)")
ax.set_xlabel("Student Rank (by EI)", fontsize=11, fontweight="bold")
ax.set_ylabel("Score (0-100)", fontsize=11, fontweight="bold")
ax.set_title("EI vs Exam Ranking Alignment", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(RES, "ei_final_validation_v2.png"), dpi=150, bbox_inches="tight")
print("Saved: ei_final_validation_v2.png")

# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FEATURE IMPORTANCE (Standardized Coefficients)")
print("=" * 72)

model_final = Ridge(alpha=1.0)
model_final.fit(X_model, y)
coefs = model_final.coef_

importance_df = pd.DataFrame({
    "feature": feature_names_ordered,
    "coefficient": coefs,
    "abs_coefficient": np.abs(coefs),
})
importance_df = importance_df.sort_values("abs_coefficient", ascending=False)

print(f"\nTop 10 features by importance:")
print(f"  {'Feature':<55} {'Coefficient':>12} {'Impact':>8}")
print(f"  {'-'*75}")
for _, row in importance_df.head(10).iterrows():
    impact = "positive" if row["coefficient"] > 0 else "negative"
    print(f"  {row['feature']:<55} {row['coefficient']:+12.6f} {impact:>8}")

importance_df.to_csv(os.path.join(RES, "ei_feature_importance_v2.csv"), index=False)
print(f"\nSaved: ei_feature_importance_v2.csv")

fig2, ax = plt.subplots(figsize=(12, 8))
top_n = 15
top_feats = importance_df.head(top_n)
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in top_feats["coefficient"]]
ax.barh(range(len(top_feats)), top_feats["coefficient"].values, color=colors,
        edgecolor="k", linewidth=0.5, alpha=0.8)
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(top_feats["feature"].values, fontsize=9)
ax.set_xlabel("Ridge Coefficient", fontsize=11, fontweight="bold")
ax.set_title("Feature Importance: Top 15 Features", fontsize=13, fontweight="bold")
ax.axvline(0, color="k", linewidth=1)
ax.grid(True, alpha=0.3, axis="x")
ax.invert_yaxis()
plt.tight_layout()
fig2.savefig(os.path.join(RES, "ei_feature_importance_v2.png"), dpi=150, bbox_inches="tight")
print("Saved: ei_feature_importance_v2.png")

# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY REPORT")
print("=" * 72)

report = f"""
EMOTIONAL INTELLIGENCE SCORING MODEL (V2 - CORRECTED)
{'='*60}

MODEL SPECIFICATION:
  Features:           15 (8 original + 7 derived interactions)
  Algorithm:          Ridge Regression (α=1.0)
  Cross-validation:   Leave-One-Out (LOOCV, n=42)

VALIDATION RESULTS:
  Pearson correlation (EI vs Exam):     r = {r_loocv:+.4f}
  Significance level:                    p = {p_loocv:.2e}
  Coefficient of determination:          R² = {r2_loocv:.4f}
  Root mean squared error:               RMSE = {rmse_loocv:.2f}%
  Mean absolute error:                   MAE = {mae_loocv:.2f}%
  Spearman rank correlation:             ρ = {sr_loocv:+.4f}

COMPARISON TO BASELINE:
  Raw 13 features (from interaction_discovery.py):    r = 0.6468, R² = 0.4044
  Optimal 15 features (expected):                     r ≈ 0.89+
  This model (V2 corrected):                          r = {r_loocv:.4f}, R² = {r2_loocv:.4f}

FEATURES USED:
  Original (8):
    1. c3_total_productive_time_ratio
    2. c4_block_stage_awareness
    3. c4_arousal_trend
    4. c3_effective_time_efficiency
    5. c4_arousal_optimality
    6. c3_task_completion_persistence
    7. c2_anticipatory_regulation
    8. c5_flow_stability

  Derived (7):
    9. c4_arousal_optimality / c4_arousal_trend (ratio)
    10. c2_cross_stage_regulation_var * c2_anticipatory_regulation (multiply)
    11. c2_cross_stage_regulation_var / c4_block_stage_awareness (ratio)
    12. c3_struggle_benefit_index / c4_arousal_midrange_proportion (ratio)
    13. c3_total_productive_time_ratio / c4_block_stage_awareness (ratio)
    14. c3_effective_time_efficiency / c3_struggle_benefit_index (ratio)
    15. c2_cross_stage_regulation_var × c3_total_productive_time_ratio × c4_arousal_midrange_proportion (triple)

OUTPUT SCORES:
  EI Score Range:         0-100 (normalized)
  Mean EI:                {ei_score_normalized.mean():.1f}
  Median EI:              {np.median(ei_score_normalized):.1f}
  SD:                     {ei_score_normalized.std():.1f}

DELIVERABLES:
  ✓ ei_final_scores_v2.csv           (student EI scores ranked)
  ✓ ei_final_validation_v2.png       (4-panel validation plot)
  ✓ ei_feature_importance_v2.csv     (feature coefficients)
  ✓ ei_feature_importance_v2.png     (feature importance chart)
  ✓ ei_final_scoring_v2.py           (reproducible model code)
"""

print(report)

with open(os.path.join(RES, "ei_final_report_v2.txt"), "w", encoding="utf-8") as f:
    f.write(report)

print(f"Saved: ei_final_report_v2.txt")

print("\n" + "=" * 72)
print("✓ EI SCORING MODEL COMPLETE (V2)")
print("=" * 72)
