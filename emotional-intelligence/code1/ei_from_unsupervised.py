"""
ei_from_unsupervised.py
========================
Builds an Emotional Intelligence (EI) score purely from the unsupervised
Strong-tier features (features that show a common structural pattern across
all students, selected WITHOUT looking at exam scores).

Then validates: does this EI score correlate with exam performance?

EI construction methods:
  PCA-EI       — first principal component
  FA-EI        — factor analysis latent factor
  Weighted-EI  — pattern_score-weighted signed sum (signs from theory)
  Ensemble-EI  — average of the three (z-scored)

Outputs (results/):
  ei_unsupervised_scores.csv      — EI scores per student
  ei_vs_exam_correlation.png      — scatter plots
  ei_unsupervised_summary.txt     — full report
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load Strong-tier features (from unsupervised pattern analysis)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1  Loading Strong-tier unsupervised features")
print("=" * 70)

tiers = pd.read_csv(os.path.join(RES, "pattern_tiers.csv"), index_col=0)
strong = tiers[tiers["tier"] == "Strong"]
strong_feats = strong["feature"].tolist()
pattern_scores = strong.set_index("feature")["pattern_score"]

# Load raw data
frames = []
for i in range(1, 6):
    df = pd.read_csv(os.path.join(RES, f"features_D{i}.csv"))
    feat_cols = [c for c in df.columns if c.startswith("c")]
    keep = ["pid"] + feat_cols
    if "pct" in df.columns and len(frames) == 0:
        keep.append("pct")
    frames.append(df[keep])

merged = frames[0]
for f in frames[1:]:
    merged = merged.merge(f, on="pid", how="outer")

y = merged["pct"].values
pids = merged["pid"].values
n = len(y)

print(f"  Strong-tier features: {len(strong_feats)}")
print(f"  Students: {n}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Theoretical EI sign assignment
#    +1 = higher value means higher emotional intelligence
#    -1 = higher value means lower emotional intelligence
#    Assigned purely from the feature NAME and construct meaning,
#    NOT from correlation with exam scores.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2  Assigning theoretical EI direction (from construct meaning)")
print("=" * 70)

EI_SIGN = {
    # C1 — Emotion Perception
    "c1_instability_MSSD":                      -1,  # emotional instability is low EI
    "c1_emotion_kurtosis":                      -1,  # extreme peaked distributions = rigid = low EI
    "c1_skewness_balance":                      +1,  # balanced emotional spread = high EI
    "c1_AU_network_density":                    +1,  # rich facial expression network = high EI
    "c1_AU_network_clustering":                 +1,  # organised expression patterns = high EI

    # C2 — Emotion Regulation
    "c2_reappraisal_proxy_coactivation":        +1,  # reappraisal = adaptive regulation
    "c2_stress_peak_attenuation":               +1,  # attenuating stress peaks = good regulation
    "c2_stress_recovery_latency":               -1,  # slower recovery = lower EI
    "c2_stress_instability_MSSD":               -1,  # stress instability = poor regulation
    "c2_recovery_completeness":                 +1,  # complete recovery = high EI
    "c2_neg_to_pos_transition_speed":           +1,  # fast negative-to-positive = good regulation
    "c2_emotion_variability_SD":                -1,  # high variability = dysregulation
    "c2_anticipatory_regulation":               +1,  # proactive regulation = high EI
    "c2_SAM_facial_discrepancy_reduction":      +1,  # aligning self-report with expression = awareness
    "c2_cross_stage_regulation_var":            -1,  # inconsistent regulation across stages = low EI

    # C3 — Emotion Utilisation / Resilience
    "c3_bounce_back_index":                     +1,  # bouncing back from difficulty = high EI
    "c3_late_task_engagement_ratio":            +1,  # staying engaged late = persistence = high EI
    "c3_confusion_resolution_rate":             +1,  # resolving confusion = high EI
    "c3_post_block_engagement_quality":         +1,  # good engagement after blocks = resilience
    "c3_stress_inertia_during_blocks":          -1,  # stress sticking around = low EI
    "c3_longest_productive_streak":             +1,  # longer productive streaks = high EI
    "c3_recovery_acceleration":                 +1,  # faster recovery = high EI
    "c3_block_anger_suppression":               +1,  # managing anger during blocks = regulation
    "c3_second_order_block_escape":             +1,  # escaping blocks efficiently = high EI

    # C4 — Emotional Self-Awareness
    "c4_positive_reporting_bias":               -1,  # bias = low self-awareness
    "c4_emotional_range_concordance":           +1,  # concordance = accurate awareness
    "c4_arousal_midrange_proportion":           +1,  # staying in optimal arousal zone = self-regulation
    "c4_sam_response_range":                    +1,  # using full emotional range = differentiation

    # C5 — Emotion in Context / Application
    "c5_mastery_during_challenge":              +1,  # mastery under pressure = high EI
    "c5_engagement_recovery_speed":             +1,  # fast re-engagement = resilience
    "c5_mastery_mode_ratio":                    +1,  # time in mastery mode = high EI
    "c5_pride_after_production":                +1,  # appropriate pride = healthy emotion
    "c5_mastery_during_challenge_binary":       +1,  # showing mastery = high EI
    "c5_transition_emotional_preparation":      +1,  # emotional prep for transitions = awareness
    "c5_emotion_stage_differentiation":         +1,  # context-appropriate emotions = high EI
    "c5_interest_during_ideation":              +1,  # interest during creative phase = engagement
    "c5_positive_creative_vs_focused":          +1,  # positive creative balance = high EI
    "c5_stress_appropriate_modulation":         +1,  # modulating stress appropriately = regulation
}

# Verify all features have signs
missing = [f for f in strong_feats if f not in EI_SIGN]
if missing:
    print(f"  WARNING: no sign for {missing}, defaulting to +1")
    for f in missing:
        EI_SIGN[f] = +1

signs = np.array([EI_SIGN[f] for f in strong_feats])
print(f"  +1 (higher = more EI): {(signs == 1).sum()} features")
print(f"  -1 (higher = less EI): {(signs == -1).sum()} features")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Impute, scale, flip signs
# ══════════════════════════════════════════════════════════════════════════════
X_raw = merged[strong_feats].copy().apply(lambda c: c.fillna(c.median()))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Flip signs so that higher = more emotionally intelligent for ALL features
X_signed = X_scaled * signs[np.newaxis, :]

# ══════════════════════════════════════════════════════════════════════════════
# 4. Construct four EI indices
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3  Constructing EI indices")
print("=" * 70)

def zscore(arr):
    return (arr - arr.mean()) / (arr.std() + 1e-12)

# 4a. PCA-EI — first principal component
pca = PCA(n_components=1, random_state=42)
pca_ei_raw = pca.fit_transform(X_signed).ravel()
# Ensure PCA direction aligns with EI (positive loadings on most features)
if np.mean(pca.components_[0]) < 0:
    pca_ei_raw = -pca_ei_raw
pca_ei = zscore(pca_ei_raw)
var_pca = pca.explained_variance_ratio_[0]
print(f"  PCA-EI  : variance explained = {var_pca:.1%}")

# 4b. FA-EI — factor analysis, one latent factor
fa = FactorAnalysis(n_components=1, random_state=42)
fa_ei_raw = fa.fit_transform(X_signed).ravel()
if np.corrcoef(fa_ei_raw, pca_ei_raw)[0, 1] < 0:
    fa_ei_raw = -fa_ei_raw
fa_ei = zscore(fa_ei_raw)
print(f"  FA-EI   : done")

# 4c. Weighted-EI — pattern_score * sign * z-scored feature, summed
weights = pattern_scores.loc[strong_feats].values
weighted_ei_raw = X_signed @ weights
weighted_ei = zscore(weighted_ei_raw)
print(f"  Weighted-EI : done (weights = unsupervised pattern_score)")

# 4d. Ensemble-EI — average of the three
ensemble_ei = (pca_ei + fa_ei + weighted_ei) / 3.0
print(f"  Ensemble-EI : done")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Validate against exam scores
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4  Validating EI scores against exam performance")
print("=" * 70)

ei_names = ["PCA_EI", "FA_EI", "Weighted_EI", "Ensemble_EI"]
ei_arrays = [pca_ei, fa_ei, weighted_ei, ensemble_ei]

print(f"\n  {'Index':15s}  {'Pearson r':>10s}  {'p':>8s}  {'Spearman rho':>12s}  {'p':>8s}")
print("  " + "-" * 60)

for name, arr in zip(ei_names, ei_arrays):
    r, p_r = pearsonr(arr, y)
    rho, p_s = spearmanr(arr, y)
    sig = "***" if p_r < 0.01 else "** " if p_r < 0.05 else "*  " if p_r < 0.1 else "   "
    print(f"  {name:15s}  {r:+10.4f}  {p_r:8.4f}  {rho:+12.4f}  {p_s:8.4f} {sig}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Save EI scores
# ══════════════════════════════════════════════════════════════════════════════
ei_df = pd.DataFrame({
    "pid": pids,
    "exam_pct": y,
    "PCA_EI": np.round(pca_ei, 4),
    "FA_EI": np.round(fa_ei, 4),
    "Weighted_EI": np.round(weighted_ei, 4),
    "Ensemble_EI": np.round(ensemble_ei, 4),
})
ei_df = ei_df.sort_values("Ensemble_EI", ascending=False).reset_index(drop=True)
ei_df.to_csv(os.path.join(RES, "ei_unsupervised_scores.csv"), index=False)
print(f"\n  Saved ei_unsupervised_scores.csv")

# Show ranked students
print(f"\n  Students ranked by Ensemble-EI:")
print(f"  {'Rank':>4s}  {'PID':>5s}  {'Exam%':>6s}  {'PCA_EI':>8s}  {'FA_EI':>8s}  {'W_EI':>8s}  {'Ens_EI':>8s}")
print("  " + "-" * 55)
for i, row in ei_df.iterrows():
    print(f"  {i+1:4d}  {row['pid']:>5s}  {row['exam_pct']:6.1f}  "
          f"{row['PCA_EI']:+8.3f}  {row['FA_EI']:+8.3f}  "
          f"{row['Weighted_EI']:+8.3f}  {row['Ensemble_EI']:+8.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Scatter plots: EI vs exam score
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5  Generating plots")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ax, name, arr in zip(axes, ei_names, ei_arrays):
    r, p_val = pearsonr(arr, y)
    rho, _ = spearmanr(arr, y)

    ax.scatter(arr, y, c="#3498db", s=40, alpha=0.7, edgecolors="#2c3e50",
               linewidths=0.5, zorder=5)

    # regression line
    z = np.polyfit(arr, y, 1)
    x_line = np.linspace(arr.min(), arr.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r-", linewidth=2, alpha=0.7)

    # label some extremes
    for idx in [np.argmax(arr), np.argmin(arr), np.argmax(y), np.argmin(y)]:
        ax.annotate(pids[idx], (arr[idx], y[idx]), fontsize=6, alpha=0.6,
                    xytext=(4, 4), textcoords="offset points")

    sig_str = "p<0.01" if p_val < 0.01 else f"p={p_val:.3f}"
    ax.set_title(f"{name}\nr = {r:+.3f} ({sig_str}), rho = {rho:+.3f}",
                 fontsize=11)
    ax.set_xlabel(f"{name} (z-scored)", fontsize=9)
    ax.set_ylabel("Exam score (%)", fontsize=9)
    ax.grid(True, alpha=0.2)

plt.suptitle("Emotional Intelligence (from unsupervised features) vs Exam Score",
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(RES, "ei_vs_exam_correlation.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print("  Saved ei_vs_exam_correlation.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Per-dimension sub-EI scores
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6  Per-dimension sub-EI scores")
print("=" * 70)

dim_labels = {
    "c1": "D1: Emotion Perception",
    "c2": "D2: Emotion Regulation",
    "c3": "D3: Emotion Utilisation",
    "c4": "D4: Self-Awareness",
    "c5": "D5: Emotion in Context",
}

sub_ei_cols = []
print(f"\n  {'Dimension':30s}  {'#Feats':>6s}  {'r vs Exam':>10s}  {'p':>8s}  {'rho':>6s}")
print("  " + "-" * 65)

for dim_prefix, dim_name in dim_labels.items():
    dim_feats = [f for f in strong_feats if f.startswith(dim_prefix + "_")]
    if len(dim_feats) < 2:
        continue

    dim_idx = [strong_feats.index(f) for f in dim_feats]
    dim_signed = X_signed[:, dim_idx]
    dim_weights = weights[dim_idx]
    sub_score = zscore(dim_signed @ dim_weights)

    col_name = f"EI_{dim_prefix.upper()}"
    sub_ei_cols.append(col_name)
    ei_df[col_name] = np.round(sub_score, 4)

    r, p_val = pearsonr(sub_score, y)
    rho, _ = spearmanr(sub_score, y)
    sig = "***" if p_val < 0.01 else "** " if p_val < 0.05 else "*  " if p_val < 0.1 else "   "
    print(f"  {dim_name:30s}  {len(dim_feats):6d}  {r:+10.4f}  {p_val:8.4f}  {rho:+6.3f} {sig}")

# save updated
ei_df.to_csv(os.path.join(RES, "ei_unsupervised_scores.csv"), index=False)

# dimension sub-EI scatter
if sub_ei_cols:
    fig, axes = plt.subplots(1, len(sub_ei_cols), figsize=(4 * len(sub_ei_cols), 4))
    if len(sub_ei_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, sub_ei_cols):
        arr = ei_df[col].values
        r, p_val = pearsonr(arr, y)
        ax.scatter(arr, y, c="#2ecc71", s=30, alpha=0.6, edgecolors="#2c3e50",
                   linewidths=0.4)
        z = np.polyfit(arr, y, 1)
        x_line = np.linspace(arr.min(), arr.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "r-", linewidth=1.5, alpha=0.7)
        sig_str = "p<0.01" if p_val < 0.01 else f"p={p_val:.3f}"
        ax.set_title(f"{col}\nr={r:+.3f} ({sig_str})", fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Exam %", fontsize=8)
        ax.grid(True, alpha=0.2)
    plt.suptitle("Sub-EI dimensions vs Exam Score", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(RES, "ei_subdimensions_vs_exam.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved ei_subdimensions_vs_exam.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Summary report
# ══════════════════════════════════════════════════════════════════════════════
with open(os.path.join(RES, "ei_unsupervised_summary.txt"), "w", encoding="utf-8") as fh:
    fh.write("EMOTIONAL INTELLIGENCE FROM UNSUPERVISED FEATURES\n")
    fh.write("=" * 60 + "\n\n")
    fh.write("Approach:\n")
    fh.write("  1. Features selected by unsupervised pattern analysis\n")
    fh.write("     (consistency, clustering, bimodality, entropy, etc.)\n")
    fh.write("  2. NO exam scores used in feature selection\n")
    fh.write("  3. EI direction assigned from theoretical construct meaning\n")
    fh.write("  4. Four EI indices constructed, then validated against exam scores\n\n")
    fh.write(f"Strong-tier features used: {len(strong_feats)}\n")
    fh.write(f"Students: {n}\n\n")

    fh.write("Features and their EI direction:\n")
    for f in strong_feats:
        direction = "higher = more EI" if EI_SIGN[f] == 1 else "higher = less EI"
        fh.write(f"  {EI_SIGN[f]:+d}  {f:50s}  ({direction})\n")

    fh.write(f"\nEI vs Exam Score Correlation:\n")
    fh.write(f"  {'Index':15s}  {'Pearson r':>10s}  {'p':>8s}  {'Spearman rho':>12s}  {'p':>8s}\n")
    fh.write("  " + "-" * 60 + "\n")
    for name, arr in zip(ei_names, ei_arrays):
        r, p_r = pearsonr(arr, y)
        rho, p_s = spearmanr(arr, y)
        fh.write(f"  {name:15s}  {r:+10.4f}  {p_r:8.4f}  {rho:+12.4f}  {p_s:8.4f}\n")

print("\n  Saved ei_unsupervised_summary.txt")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
