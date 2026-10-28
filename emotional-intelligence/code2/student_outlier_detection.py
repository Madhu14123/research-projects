"""
student_outlier_detection.py
============================
Identify outlier students using multiple detection methods:
1. Residual-based (prediction error anomalies)
2. Multivariate distance (Isolation Forest, LOF, Mahalanobis)
3. EI-exam relationship inconsistencies
Outputs flagged students and justification for removal.
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STUDENT OUTLIER DETECTION")
print("=" * 72)

# Load student feature profiles
profiles = pd.read_csv(os.path.join(RES, "student_feature_profiles_full.csv"))
ei_scores = pd.read_csv(os.path.join(RES, "ei_final_scores_v2.csv"))

n = len(profiles)
print(f"\nLoaded {n} students")

# Extract feature columns (z-scored)
z_cols = [c for c in profiles.columns if c.startswith("z_c")]
X_raw = profiles[z_cols].copy()

# Get exam and EI scores
exam_pct = ei_scores["exam_pct"].values
ei_scores_raw = ei_scores["EI_score_raw"].values
pids = profiles["pid"].values

# Standardize features for distance calculations
scaler = StandardScaler()
X_z = scaler.fit_transform(X_raw)

# ═══════════════════════════════════════════════════════════════════════════
#  1. RESIDUAL-BASED OUTLIERS (Prediction Errors)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("METHOD 1: RESIDUAL ANALYSIS")
print("=" * 72)

# Compute residuals (EI score vs actual exam percentage)
residuals = ei_scores_raw - exam_pct
residual_z = np.abs(zscore(residuals))

# Flag: |residual z-score| > 2.5
residual_outliers = residual_z > 2.5
n_residual = np.sum(residual_outliers)

print(f"\n  Students with large prediction errors (|z| > 2.5):")
print(f"  {'PID':<6} {'Exam %':>8} {'EI Score':>10} {'Residual':>10} {'Z-score':>8}")
print(f"  {'-'*45}")
for i in np.where(residual_outliers)[0]:
    print(f"  {pids[i]:<6} {exam_pct[i]:>8.1f} {ei_scores_raw[i]:>10.2f} {residuals[i]:>10.2f} {residual_z[i]:>8.2f}")

# ═══════════════════════════════════════════════════════════════════════════
#  2. MULTIVARIATE OUTLIERS - Isolation Forest
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("METHOD 2: ISOLATION FOREST (Multivariate anomaly)")
print("=" * 72)

iso_forest = IsolationForest(contamination=0.15, random_state=42)
iso_labels = iso_forest.fit_predict(X_z)
iso_scores = iso_forest.score_samples(X_z)
iso_outliers = iso_labels == -1
n_iso = np.sum(iso_outliers)

print(f"\n  Isolation Forest: {n_iso} students flagged (contamination=0.15)")
print(f"  {'PID':<6} {'Exam %':>8} {'EI Score':>10} {'Anomaly Score':>14}")
print(f"  {'-'*45}")
for i in np.where(iso_outliers)[0]:
    print(f"  {pids[i]:<6} {exam_pct[i]:>8.1f} {ei_scores_raw[i]:>10.2f} {iso_scores[i]:>14.4f}")

# ═══════════════════════════════════════════════════════════════════════════
#  3. LOCAL OUTLIER FACTOR
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("METHOD 3: LOCAL OUTLIER FACTOR (Local density anomaly)")
print("=" * 72)

lof = LocalOutlierFactor(n_neighbors=5, contamination=0.15)
lof_labels = lof.fit_predict(X_z)
lof_scores = lof.negative_outlier_factor_
lof_outliers = lof_labels == -1
n_lof = np.sum(lof_outliers)

print(f"\n  LOF: {n_lof} students flagged (n_neighbors=5, contamination=0.15)")
print(f"  {'PID':<6} {'Exam %':>8} {'EI Score':>10} {'LOF Score':>12}")
print(f"  {'-'*45}")
for i in np.where(lof_outliers)[0]:
    print(f"  {pids[i]:<6} {exam_pct[i]:>8.1f} {ei_scores_raw[i]:>10.2f} {lof_scores[i]:>12.4f}")

# ═══════════════════════════════════════════════════════════════════════════
#  4. MAHALANOBIS DISTANCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("METHOD 4: MAHALANOBIS DISTANCE (Statistical outliers)")
print("=" * 72)

# Compute robust covariance (exclude obvious outliers first)
from sklearn.covariance import EllipticEnvelope
ee = EllipticEnvelope(contamination=0.15, random_state=42)
ee_labels = ee.fit_predict(X_z)
ee_outliers = ee_labels == -1
n_ee = np.sum(ee_outliers)

ee_distances = ee.mahalanobis(X_z)

print(f"\n  Elliptic Envelope (robust covariance): {n_ee} students flagged")
print(f"  {'PID':<6} {'Exam %':>8} {'EI Score':>10} {'Mahal. Dist':>12}")
print(f"  {'-'*45}")
for i in np.where(ee_outliers)[0]:
    print(f"  {pids[i]:<6} {exam_pct[i]:>8.1f} {ei_scores_raw[i]:>10.2f} {ee_distances[i]:>12.2f}")

# ═══════════════════════════════════════════════════════════════════════════
#  5. CONSENSUS FLAGGING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("CONSENSUS OUTLIER FLAGGING")
print("=" * 72)

# Count how many methods flagged each student
outlier_votes = np.zeros(n, dtype=int)
outlier_votes[residual_outliers] += 1
outlier_votes[iso_outliers] += 1
outlier_votes[lof_outliers] += 1
outlier_votes[ee_outliers] += 1

# Consensus: flagged by 2+ methods OR extreme residual
consensus_outliers = (outlier_votes >= 2) | residual_outliers
n_consensus = np.sum(consensus_outliers)

print(f"\n  Consensus criteria: flagged by 2+ methods OR extreme residual (|z|>2.5)")
print(f"  {n_consensus} students identified for potential removal")
print(f"\n  {'PID':<6} {'Exam %':>8} {'EI':>8} {'Votes':>6} {'Residual_z':>12} {'Reason':<20}")
print(f"  {'-'*70}")

for i in np.where(consensus_outliers)[0]:
    votes = int(outlier_votes[i])
    resid_z = f"{residual_z[i]:.2f}"
    reason = ""
    if residual_z[i] > 2.5:
        reason = "Extreme residual"
    else:
        reason = f"Flagged by {votes} methods"

    print(f"  {pids[i]:<6} {exam_pct[i]:>8.1f} {ei_scores_raw[i]:>8.1f} {votes:>6} {resid_z:>12} {reason:<20}")

# ═══════════════════════════════════════════════════════════════════════════
#  6. DETAILED OUTLIER PROFILES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("DETAILED OUTLIER ANALYSIS")
print("=" * 72)

outlier_info = []
for i in np.where(consensus_outliers)[0]:
    # Get top features for this student (absolute z-scores)
    feat_z = np.abs(X_z[i])
    top_feat_idx = np.argsort(feat_z)[-5:][::-1]
    top_feats = [(z_cols[j], X_z[i, j]) for j in top_feat_idx]

    # Determine outlier type
    outlier_type = []
    if residual_z[i] > 2.5:
        outlier_type.append("PredictionError")
    if iso_outliers[i]:
        outlier_type.append("IsoForest")
    if lof_outliers[i]:
        outlier_type.append("LOF")
    if ee_outliers[i]:
        outlier_type.append("Mahal")

    outlier_info.append({
        "pid": pids[i],
        "exam_pct": exam_pct[i],
        "ei_score": ei_scores_raw[i],
        "residual": residuals[i],
        "residual_z": residual_z[i],
        "votes": int(outlier_votes[i]),
        "methods": "|".join(outlier_type),
        "top_features": top_feats
    })

for outlier in outlier_info:
    print(f"\n  {outlier['pid'].upper()}")
    print(f"  {'─' * 50}")
    print(f"    Exam: {outlier['exam_pct']:.1f}%  |  EI Score: {outlier['ei_score']:.2f}")
    print(f"    Residual: {outlier['residual']:+.2f} (z={outlier['residual_z']:.2f})")
    print(f"    Detection: {outlier['votes']} methods flagged → {outlier['methods']}")
    print(f"    Top anomalous features:")
    for feat, z_val in outlier['top_features']:
        print(f"      • {feat}: {z_val:+.2f}")

# ═══════════════════════════════════════════════════════════════════════════
#  7. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SAVING RESULTS")
print("=" * 72)

# Save consensus outliers list
outliers_df = pd.DataFrame({
    "pid": pids[consensus_outliers],
    "exam_pct": exam_pct[consensus_outliers],
    "ei_score": ei_scores_raw[consensus_outliers],
    "residual": residuals[consensus_outliers],
    "residual_z": residual_z[consensus_outliers],
    "votes": outlier_votes[consensus_outliers],
    "residual_flag": residual_outliers[consensus_outliers],
    "isolation_flag": iso_outliers[consensus_outliers],
    "lof_flag": lof_outliers[consensus_outliers],
    "mahal_flag": ee_outliers[consensus_outliers]
})
outliers_df = outliers_df.sort_values("votes", ascending=False)
outliers_df.to_csv(os.path.join(RES, "student_outliers_consensus.csv"), index=False)
print(f"\n  Saved: student_outliers_consensus.csv ({len(outliers_df)} students)")

# Save full anomaly scores for all students
all_anomalies = pd.DataFrame({
    "pid": pids,
    "exam_pct": exam_pct,
    "ei_score": ei_scores_raw,
    "residual": residuals,
    "residual_z": residual_z,
    "isolation_score": iso_scores,
    "lof_score": lof_scores,
    "mahal_score": ee_distances,
    "total_votes": outlier_votes,
    "is_outlier": consensus_outliers.astype(int)
})
all_anomalies = all_anomalies.sort_values("total_votes", ascending=False)
all_anomalies.to_csv(os.path.join(RES, "student_anomaly_scores_all.csv"), index=False)
print(f"  Saved: student_anomaly_scores_all.csv (all {n} students)")

# ═══════════════════════════════════════════════════════════════════════════
#  8. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

# Figure 1: EI vs Exam with outliers highlighted
fig1, ax1 = plt.subplots(figsize=(10, 8))

# Plot normal students
normal_idx = ~consensus_outliers
ax1.scatter(exam_pct[normal_idx], ei_scores_raw[normal_idx],
            s=100, c='#3498db', alpha=0.6, label='Normal', edgecolors='k', linewidth=0.5)

# Plot outliers with different marker
ax1.scatter(exam_pct[consensus_outliers], ei_scores_raw[consensus_outliers],
            s=150, c='#e74c3c', alpha=0.8, marker='^', label='Outlier',
            edgecolors='darkred', linewidth=1.5)

# Add student IDs for outliers
for i in np.where(consensus_outliers)[0]:
    ax1.annotate(pids[i], (exam_pct[i], ei_scores_raw[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

ax1.set_xlabel('Exam Score (%)', fontweight='bold', fontsize=11)
ax1.set_ylabel('EI Score', fontweight='bold', fontsize=11)
ax1.set_title('Student Outlier Detection: EI vs Exam Performance', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=10)
plt.tight_layout()
fig1.savefig(os.path.join(RES, "outlier_ei_vs_exam.png"), dpi=150, bbox_inches='tight')
print("  Saved: outlier_ei_vs_exam.png")

# Figure 2: Residual distribution with outliers
fig2, ax2 = plt.subplots(figsize=(11, 7))

colors = ['#e74c3c' if consensus_outliers[i] else '#3498db' for i in range(n)]
ax2.bar(range(n), residuals[np.argsort(residuals)], color=np.array(colors)[np.argsort(residuals)],
        alpha=0.7, edgecolor='k', linewidth=0.5)

# Add threshold lines
ax2.axhline(2.5 * residuals.std(), color='red', linestyle='--', linewidth=2, label='±2.5σ threshold')
ax2.axhline(-2.5 * residuals.std(), color='red', linestyle='--', linewidth=2)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)

ax2.set_xlabel('Student (sorted by residual)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Residual (EI - Exam)', fontweight='bold', fontsize=11)
ax2.set_title('Residual Analysis: Prediction Errors by Student', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(fontsize=10)
plt.tight_layout()
fig2.savefig(os.path.join(RES, "outlier_residuals.png"), dpi=150, bbox_inches='tight')
print("  Saved: outlier_residuals.png")

# Figure 3: Detection method Venn-like view (votes histogram)
fig3, ax3 = plt.subplots(figsize=(10, 6))

vote_counts = np.bincount(outlier_votes[consensus_outliers], minlength=5)
vote_labels = [f"{i} method{'s' if i != 1 else ''}" for i in range(5)]
colors_votes = ['#2ecc71' if i < 2 else '#f39c12' if i < 3 else '#e74c3c' for i in range(5)]

bars = ax3.bar(range(5), vote_counts, color=colors_votes, alpha=0.7, edgecolor='k', linewidth=1.5)
ax3.set_xticks(range(5))
ax3.set_xticklabels(vote_labels)
ax3.set_ylabel('Number of Outliers', fontweight='bold', fontsize=11)
ax3.set_title(f'Consensus Outlier Detection: Method Agreement (n={n_consensus})',
              fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, vote_counts)):
    if count > 0:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(count)), ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
fig3.savefig(os.path.join(RES, "outlier_method_consensus.png"), dpi=150, bbox_inches='tight')
print("  Saved: outlier_method_consensus.png")

# Figure 4: PCA of features with outliers
fig4, ax4 = plt.subplots(figsize=(10, 8))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_z)

ax4.scatter(X_pca[normal_idx, 0], X_pca[normal_idx, 1],
            s=100, c='#3498db', alpha=0.6, label='Normal', edgecolors='k', linewidth=0.5)
ax4.scatter(X_pca[consensus_outliers, 0], X_pca[consensus_outliers, 1],
            s=150, c='#e74c3c', alpha=0.8, marker='^', label='Outlier',
            edgecolors='darkred', linewidth=1.5)

# Annotate outliers
for i in np.where(consensus_outliers)[0]:
    ax4.annotate(pids[i], (X_pca[i, 0], X_pca[i, 1]),
                xytext=(5, 5), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold', fontsize=11)
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold', fontsize=11)
ax4.set_title('Feature Space PCA: Outlier Distribution', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best', fontsize=10)
plt.tight_layout()
fig4.savefig(os.path.join(RES, "outlier_pca_space.png"), dpi=150, bbox_inches='tight')
print("  Saved: outlier_pca_space.png")

# ═══════════════════════════════════════════════════════════════════════════
#  9. SUMMARY & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 72)

print(f"\n  Total students: {n}")
print(f"  Outliers identified: {n_consensus} ({100*n_consensus/n:.1f}%)")
print(f"\n  Detection method breakdown:")
print(f"    • Residual analysis: {n_residual}")
print(f"    • Isolation Forest: {n_iso}")
print(f"    • Local Outlier Factor: {n_lof}")
print(f"    • Mahalanobis distance: {n_ee}")

print(f"\n  RECOMMENDATION:")
if n_consensus > 0:
    print(f"  • Remove {n_consensus} flagged students from EI model retraining")
    print(f"  • These students show anomalous EI-exam relationships or extreme feature profiles")
    print(f"  • Check student_outliers_consensus.csv for removal list")
    print(f"  • Verify reasons in detailed profiles above before final removal")
else:
    print(f"  • No strong outliers detected; current model uses all {n} students")

print("\n" + "=" * 72)
print("✓ OUTLIER DETECTION COMPLETE")
print("=" * 72)
