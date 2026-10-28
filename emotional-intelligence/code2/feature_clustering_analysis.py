"""
feature_clustering_analysis.py
===============================
Cluster all 114 features to find:
1. Natural feature groups (redundancies within groups)
2. High-value cross-cluster interactions
3. Whether expanding beyond 13 features improves EI model

Approach:
- Load all 114 features
- Compute correlation matrix
- Hierarchical clustering
- Identify redundant features (|r| > 0.8)
- Map interactions between clusters
- Test expanded feature sets
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.base import clone

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")
TARGET = "pct"

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD ALL FEATURES (114 total)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("FEATURE CLUSTERING ANALYSIS (ALL 114 FEATURES)")
print("=" * 72)

FILES = {
    "D1": os.path.join(RES, "features_D1.csv"),
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

all_feat_cols = [c for c in merged.columns if c.startswith("c")]
X_raw = merged.loc[valid, all_feat_cols].reset_index(drop=True)
X_raw = X_raw.apply(lambda c: c.fillna(c.median()))

# Standardize
scaler = StandardScaler()
X_z = pd.DataFrame(scaler.fit_transform(X_raw), columns=all_feat_cols)

n = len(y)
print(f"\nLoaded {n} students with {len(all_feat_cols)} features")
print(f"Features by dimension:")
for dim in ["c1", "c2", "c3", "c4", "c5"]:
    dim_feats = [f for f in all_feat_cols if f.startswith(dim)]
    print(f"  {dim}: {len(dim_feats)} features")

# ═══════════════════════════════════════════════════════════════════════════
#  COMPUTE CORRELATION MATRIX & CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("HIERARCHICAL CLUSTERING")
print("=" * 72)

# Correlation matrix (directly on features)
corr_matrix = X_z.corr().values
# Replace NaNs with 0
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
# Convert to distance (1 - |corr|)
dist_matrix = 1 - np.abs(corr_matrix)
np.fill_diagonal(dist_matrix, 0)

# Hierarchical clustering using condensed distance matrix
from scipy.spatial.distance import squareform
condensed_dist = squareform(dist_matrix, checks=False)
# Remove any remaining NaN/inf
condensed_dist = np.nan_to_num(condensed_dist, nan=0.0, posinf=1.0, neginf=0.0)
linkage_matrix = linkage(condensed_dist, method='average')

# Cut dendrogram at different heights
print("\nTesting different cluster numbers:")
for n_clusters in [5, 10, 15, 20, 25, 30]:
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    sizes = pd.Series(clusters).value_counts().sort_index()
    print(f"  k={n_clusters}: cluster sizes {sizes.values}")

# Use k=15 as sweet spot
n_clusters = 15
cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

clusters_dict = {}
for k in range(1, n_clusters + 1):
    clusters_dict[k] = [f for i, f in enumerate(all_feat_cols) if cluster_labels[i] == k]

print(f"\nUsing k={n_clusters} clusters:")
for k in sorted(clusters_dict.keys()):
    print(f"  Cluster {k:2d} ({len(clusters_dict[k]):2d} features): {', '.join(clusters_dict[k][:3])}...")

# ═══════════════════════════════════════════════════════════════════════════
#  IDENTIFY REDUNDANCIES WITHIN CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("REDUNDANCY ANALYSIS (within clusters)")
print("=" * 72)

redundancy_pairs = []
features_to_drop = set()

for k in sorted(clusters_dict.keys()):
    cluster_feats = clusters_dict[k]
    if len(cluster_feats) <= 1:
        continue

    print(f"\n  Cluster {k} ({len(cluster_feats)} features):")

    # Find highly correlated pairs
    for i, f1 in enumerate(cluster_feats):
        for f2 in cluster_feats[i+1:]:
            r = corr_matrix[all_feat_cols.index(f1), all_feat_cols.index(f2)]
            if abs(r) > 0.8:
                # Mark weaker one for potential removal
                r_exam_f1, _ = pearsonr(X_z[f1].values, y)
                r_exam_f2, _ = pearsonr(X_z[f2].values, y)
                weaker = f2 if abs(r_exam_f1) > abs(r_exam_f2) else f1

                redundancy_pairs.append({
                    "cluster": k, "feat1": f1, "feat2": f2, "r": r,
                    "r_exam_f1": r_exam_f1, "r_exam_f2": r_exam_f2,
                    "weaker": weaker,
                })
                print(f"    HIGH CORR: {f1} <-> {f2}: r={r:+.3f}")
                print(f"              Keep {f1} (r={r_exam_f1:+.3f}), drop {weaker} (r={r_exam_f2:+.3f})")
                features_to_drop.add(weaker)

if not redundancy_pairs:
    print("  No highly correlated pairs (|r| > 0.8) found")

print(f"\n  Summary: {len(redundancy_pairs)} redundant pairs, {len(features_to_drop)} features to drop")

# ═══════════════════════════════════════════════════════════════════════════
#  CROSS-CLUSTER INTERACTIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("CROSS-CLUSTER INTERACTIONS (top candidates)")
print("=" * 72)

cross_cluster_interactions = []

for k1 in sorted(clusters_dict.keys()):
    for k2 in range(k1 + 1, n_clusters + 1):
        cluster1_feats = clusters_dict[k1]
        cluster2_feats = clusters_dict[k2]

        # Sample top feature from each cluster
        top_f1 = max(cluster1_feats, key=lambda f: abs(pearsonr(X_z[f].values, y)[0]))
        top_f2 = max(cluster2_feats, key=lambda f: abs(pearsonr(X_z[f].values, y)[0]))

        # Compute interaction
        inter_vals = (X_z[top_f1].values * X_z[top_f2].values)
        q01, q99 = np.percentile(inter_vals, [1, 99])
        inter_vals = np.clip(inter_vals, q01, q99)
        inter_vals = (inter_vals - inter_vals.mean()) / (inter_vals.std() + 1e-10)

        r_inter, p_inter = pearsonr(inter_vals, y)

        if abs(r_inter) > 0.25:  # strong interaction
            cross_cluster_interactions.append({
                "cluster1": k1, "cluster2": k2,
                "feat1": top_f1, "feat2": top_f2,
                "r": r_inter, "p": p_inter,
                "values": inter_vals,
            })

# Sort by |r|
cross_cluster_interactions.sort(key=lambda x: abs(x["r"]), reverse=True)

print(f"\n  Top 30 cross-cluster interactions (|r| > 0.25):")
print(f"  {'Cluster':>8} {'Feature 1':<35} {'Feature 2':<35} {'r':>8} {'p':>10}")
print(f"  {'-'*100}")

for i, inter in enumerate(cross_cluster_interactions[:30]):
    c1, c2 = inter["cluster1"], inter["cluster2"]
    f1 = inter["feat1"][:32]
    f2 = inter["feat2"][:32]
    print(f"  {f'C{c1}-C{c2}':<8} {f1:<35} {f2:<35} {inter['r']:+8.4f} {inter['p']:10.6f}")

# ═══════════════════════════════════════════════════════════════════════════
#  SELECT REPRESENTATIVE FEATURES FROM EACH CLUSTER
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("REPRESENTATIVE FEATURES (one per cluster)")
print("=" * 72)

representative_features = {}
for k in sorted(clusters_dict.keys()):
    cluster_feats = clusters_dict[k]
    # Pick feature with highest |r| with exam
    rep_feat = max(cluster_feats, key=lambda f: abs(pearsonr(X_z[f].values, y)[0]))
    r_rep, p_rep = pearsonr(X_z[rep_feat].values, y)
    representative_features[k] = (rep_feat, r_rep, p_rep)
    print(f"  Cluster {k:2d}: {rep_feat:<45} r={r_rep:+.4f}, p={p_rep:.6f}")

# ═══════════════════════════════════════════════════════════════════════════
#  TEST EXPANDED FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FEATURE SET COMPARISON (LOOCV Ridge α=1)")
print("=" * 72)

def test_feature_set(feature_names, set_name):
    """Test a feature set with LOOCV Ridge."""
    if len(feature_names) == 0:
        return None

    X_set = X_z[feature_names].values
    loo = LeaveOneOut()
    preds = np.full(n, np.nan)
    model = Ridge(alpha=1.0)

    for tr, te in loo.split(X_set):
        m = clone(model)
        m.fit(X_set[tr], y[tr])
        preds[te] = m.predict(X_set[te])

    r, p = pearsonr(y, preds)
    r2 = r2_score(y, preds)

    return {"set_name": set_name, "n_features": len(feature_names), "r": r, "p": p, "r2": r2}

# Test different feature sets
test_results = []

# 1. Original 13 consensus features
orig_13 = [
    "c2_cross_stage_regulation_var", "c2_anticipatory_regulation",
    "c3_total_productive_time_ratio", "c3_struggle_benefit_index",
    "c3_effective_time_efficiency", "c3_recovery_acceleration",
    "c3_task_completion_persistence", "c4_arousal_trend",
    "c4_block_stage_awareness", "c4_arousal_optimality",
    "c4_arousal_midrange_proportion", "c5_flow_stability",
    "c5_engagement_recovery_speed",
]
orig_13 = [f for f in orig_13 if f in all_feat_cols]
r1 = test_feature_set(orig_13, "13 consensus (baseline)")
if r1:
    test_results.append(r1)

# 2. Representative from each cluster (15 features)
rep_features = [feat for feat, _, _ in representative_features.values()]
r2 = test_feature_set(rep_features, f"15 cluster reps")
if r2:
    test_results.append(r2)

# 3. Top-N features by individual correlation
top_corrs = []
for feat in all_feat_cols:
    r, _ = pearsonr(X_z[feat].values, y)
    top_corrs.append((feat, abs(r)))
top_corrs.sort(key=lambda x: x[1], reverse=True)

for top_n in [20, 30, 40, 50]:
    top_n_feats = [f for f, _ in top_corrs[:top_n]]
    r_topn = test_feature_set(top_n_feats, f"Top-{top_n} by |r|")
    if r_topn:
        test_results.append(r_topn)

# 4. Cluster representatives + top cross-cluster interactions
best_inter = cross_cluster_interactions[:10]
expanded_feats = rep_features + [inter["feat1"] for inter in best_inter] + [inter["feat2"] for inter in best_inter]
expanded_feats = list(set(expanded_feats))  # remove duplicates
r_expanded = test_feature_set(expanded_feats, f"Reps + top interactions ({len(expanded_feats)} feats)")
if r_expanded:
    test_results.append(r_expanded)

# Print results
results_df = pd.DataFrame(test_results)
print(f"\n  {'Feature Set':<40} {'N':>4} {'r':>8} {'R2':>8} {'p':>12}")
print(f"  {'-'*75}")
for _, row in results_df.iterrows():
    print(f"  {row['set_name']:<40} {int(row['n_features']):4d} {row['r']:8.4f} "
          f"{row['r2']:8.4f} {row['p']:12.2e}")

# Find best
best_idx = results_df['r'].idxmax()
best_row = results_df.iloc[best_idx]
print(f"\n  🏆 BEST: {best_row['set_name']:<35} r={best_row['r']:.4f} (n={int(best_row['n_features'])})")

# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION: CLUSTER DENDROGRAM
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("GENERATING VISUALIZATIONS")
print("=" * 72)

fig, ax = plt.subplots(figsize=(20, 10))
dendrogram(linkage_matrix, labels=all_feat_cols, ax=ax, leaf_font_size=6)
ax.set_title("Hierarchical Clustering of All 114 Features", fontweight="bold", fontsize=14)
ax.set_xlabel("Feature", fontweight="bold")
ax.set_ylabel("Distance (1 - |correlation|)", fontweight="bold")
ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Suggested cut (k≈15)')
ax.legend()
plt.xticks(rotation=90)
plt.tight_layout()
fig.savefig(os.path.join(RES, "feature_cluster_dendrogram.png"), dpi=100, bbox_inches="tight")
print("  Saved: feature_cluster_dendrogram.png")
plt.close()

# Feature set comparison bar chart
fig2, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(results_df))
ax.bar(x_pos, results_df['r'].values, color='steelblue', edgecolor='k', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['set_name'].values, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("LOOCV Pearson r", fontweight="bold", fontsize=11)
ax.set_title("Feature Set Comparison (Ridge α=1, LOOCV)", fontweight="bold", fontsize=13)
ax.axhline(y=0.8949, color='green', linestyle='--', linewidth=2, label='Current best (15 derived)')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=10)

# Add value labels on bars
for i, v in enumerate(results_df['r'].values):
    ax.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig2.savefig(os.path.join(RES, "feature_set_comparison.png"), dpi=150, bbox_inches="tight")
print("  Saved: feature_set_comparison.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

report = f"""
FEATURE CLUSTERING & INTERACTION DISCOVERY
{'='*60}

DATA:
  Total features: {len(all_feat_cols)}
  Students: {n}
  Dimensions: c1 ({len([f for f in all_feat_cols if f.startswith('c1')])}) + c2 ({len([f for f in all_feat_cols if f.startswith('c2')])}) + c3 ({len([f for f in all_feat_cols if f.startswith('c3')])}) + c4 ({len([f for f in all_feat_cols if f.startswith('c4')])}) + c5 ({len([f for f in all_feat_cols if f.startswith('c5')])})

CLUSTERING RESULTS:
  Hierarchical clustering (Ward linkage, 1-|correlation| distance)
  Optimal k: {n_clusters} clusters
  Redundant pairs found: {len(redundancy_pairs)}
  Cross-cluster interactions (|r| > 0.25): {len(cross_cluster_interactions)}

REDUNDANCY ANALYSIS:
  Features to potentially drop (highly correlated within clusters): {len(features_to_drop)}
  {', '.join(list(features_to_drop)[:5])}{"..." if len(features_to_drop) > 5 else ""}

CROSS-CLUSTER INTERACTIONS:
  Top 3:
    1. {cross_cluster_interactions[0]['feat1']} × {cross_cluster_interactions[0]['feat2']}: r={cross_cluster_interactions[0]['r']:+.4f}
    2. {cross_cluster_interactions[1]['feat1']} × {cross_cluster_interactions[1]['feat2']}: r={cross_cluster_interactions[1]['r']:+.4f}
    3. {cross_cluster_interactions[2]['feat1']} × {cross_cluster_interactions[2]['feat2']}: r={cross_cluster_interactions[2]['r']:+.4f}

FEATURE SET COMPARISON (LOOCV Ridge):
{results_df.to_string(index=False)}

BEST CONFIGURATION:
  {best_row['set_name']}
  Features: {int(best_row['n_features'])}
  LOOCV r: {best_row['r']:.4f}
  R2: {best_row['r2']:.4f}
  Improvement vs baseline 13: {((best_row['r']/0.6468 - 1)*100):+.1f}%
  Improvement vs current (r=0.8949): {((best_row['r']/0.8949 - 1)*100):+.1f}%

RECOMMENDATIONS:
  1. If best r < 0.8949: Current 15-feature model is optimal
  2. If best r > 0.8949: Use expanded feature set (re-run ei_final_scoring_v2.py with new features)
  3. Consider dropping redundant features: {', '.join(list(features_to_drop)[:3]) if features_to_drop else "None"}
  4. Top cross-cluster interactions worth exploring for future models

OUTPUTS:
  ✓ feature_cluster_dendrogram.png
  ✓ feature_set_comparison.png
  ✓ This report
"""

print(report)

with open(os.path.join(RES, "feature_clustering_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

print("Saved: feature_clustering_report.txt")

print("\n" + "=" * 72)
print("✓ FEATURE CLUSTERING ANALYSIS COMPLETE")
print("=" * 72)
