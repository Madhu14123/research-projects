"""
feature_combination_max_representation.py
=========================================
Create combined features from STRONGLY correlated pairs with NO repetition.
Goal: Maximize feature diversity by combining redundant pairs, each feature used ONCE.

Strategy:
1. Find all high-correlation pairs (|r| > 0.80)
2. Group into clusters (graph connected components)
3. For each cluster, create ONE combined feature
4. Keep unique (unclustered) features as-is
5. Train model on: unique features + combined cluster features
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict, deque

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("MAXIMUM REPRESENTATION FEATURE COMBINATION")
print("=" * 72)

# Load student profiles
profiles = pd.read_csv(os.path.join(RES, "student_feature_profiles_full.csv"))
ei_scores = pd.read_csv(os.path.join(RES, "ei_final_scores_v2.csv"))

n = len(profiles)
exam_pct = ei_scores["exam_pct"].values
y = exam_pct

print(f"\nLoaded {n} students")

# Extract z-scored features
z_cols = [c for c in profiles.columns if c.startswith("z_c")]
X_raw = profiles[z_cols].copy()

# Standardize
scaler = StandardScaler()
X_z = pd.DataFrame(scaler.fit_transform(X_raw), columns=z_cols)

print(f"Total features: {len(z_cols)}")

# ═══════════════════════════════════════════════════════════════════════════
#  1. COMPUTE CORRELATION MATRIX & FIND STRONG PAIRS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 1: IDENTIFY STRONGLY CORRELATED PAIRS")
print("=" * 72)

corr_matrix = X_z.corr().abs()

# Find pairs with |r| > 0.80
correlation_threshold = 0.80
strong_pairs = []

for i in range(len(z_cols)):
    for j in range(i + 1, len(z_cols)):
        corr_val = corr_matrix.iloc[i, j]
        if corr_val > correlation_threshold:
            feat_i = z_cols[i]
            feat_j = z_cols[j]

            # Get votes and rank for both features
            votes_i = 0
            votes_j = 0

            strong_pairs.append({
                'feature_1': feat_i,
                'feature_2': feat_j,
                'correlation': corr_val
            })

strong_pairs_df = pd.DataFrame(strong_pairs).sort_values('correlation', ascending=False)

print(f"\nFound {len(strong_pairs_df)} strongly correlated pairs (|r| > {correlation_threshold})")
print(f"  {'Feature 1':<35} {'Feature 2':<35} {'Corr':>7}")
print(f"  {'-'*80}")

for idx, row in strong_pairs_df.head(20).iterrows():
    print(f"  {row['feature_1']:<35} {row['feature_2']:<35} {row['correlation']:>7.4f}")

if len(strong_pairs_df) > 20:
    print(f"  ... ({len(strong_pairs_df) - 20} more pairs)")

# ═══════════════════════════════════════════════════════════════════════════
#  2. BUILD CLUSTERS (CONNECTED COMPONENTS)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 2: BUILD REDUNDANCY CLUSTERS (CONNECTED COMPONENTS)")
print("=" * 72)

# Union-Find to group features
parent = {feat: feat for feat in z_cols}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[py] = px

# Union all strongly correlated pairs
for _, row in strong_pairs_df.iterrows():
    union(row['feature_1'], row['feature_2'])

# Group features by cluster
clusters = defaultdict(list)
for feat in z_cols:
    root = find(feat)
    clusters[root].append(feat)

# Identify clusters (redundancy groups)
single_features = []
cluster_groups = []

for root, members in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
    if len(members) == 1:
        single_features.append(members[0])
    else:
        cluster_groups.append(members)

print(f"\nCluster analysis:")
print(f"  Total clusters: {len(clusters)}")
print(f"  Single features (unique): {len(single_features)}")
print(f"  Feature clusters (redundancy groups): {len(cluster_groups)}")
print(f"\nCluster sizes:")

for i, cluster in enumerate(sorted(cluster_groups, key=len, reverse=True)[:10], 1):
    print(f"  Cluster {i}: {len(cluster)} features")
    for feat in cluster[:3]:
        r_target = pearsonr(X_z[feat], y)[0]
        print(f"    • {feat:<40} r={r_target:+.4f}")
    if len(cluster) > 3:
        print(f"    ... and {len(cluster) - 3} more")

# ═══════════════════════════════════════════════════════════════════════════
#  3. CREATE COMBINED FEATURES FROM CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 3: CREATE COMBINED FEATURES FROM CLUSTERS")
print("=" * 72)

combined_features_dict = {}
combination_log = []

print(f"\nCreating {len(cluster_groups)} combined features:")

for cluster_idx, cluster in enumerate(cluster_groups, 1):
    # Find feature with highest correlation with target
    target_r_vals = [(feat, abs(pearsonr(X_z[feat], y)[0])) for feat in cluster]
    best_feat, _ = max(target_r_vals, key=lambda x: x[1])

    # Create combined feature: average of all in cluster
    combined = X_z[cluster].mean(axis=1)
    combined = (combined - combined.mean()) / (combined.std() + 1e-10)

    # Name it after best feature + _combined
    combined_name = f"comb_{best_feat.replace('z_', '')}"
    combined_features_dict[combined_name] = combined

    r_combined = pearsonr(combined, y)[0]

    print(f"\n  Cluster {cluster_idx} ({len(cluster)} features):")
    print(f"    Best feature: {best_feat} (r={pearsonr(X_z[best_feat], y)[0]:+.4f})")
    print(f"    Combined (avg): {combined_name} (r={r_combined:+.4f})")
    print(f"    Members:")
    for feat in cluster:
        r = pearsonr(X_z[feat], y)[0]
        print(f"      • {feat:<40} r={r:+.4f}")

    combination_log.append({
        'combined_name': combined_name,
        'num_members': len(cluster),
        'members': '|'.join(cluster),
        'best_feature': best_feat,
        'r_combined': r_combined,
        'r_best_member': pearsonr(X_z[best_feat], y)[0]
    })

# ═══════════════════════════════════════════════════════════════════════════
#  4. BUILD FINAL FEATURE SET (NO REPETITION)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 4: BUILD FINAL FEATURE SET (MAXIMUM REPRESENTATION)")
print("=" * 72)

# Final set: unique features + combined features
final_feature_names = single_features + list(combined_features_dict.keys())
final_features_df = pd.concat([
    X_z[single_features],
    pd.DataFrame(combined_features_dict)
], axis=1)

print(f"\nFinal feature set:")
print(f"  Unique (unclustered) features: {len(single_features)}")
print(f"  Combined (clustered) features: {len(combined_features_dict)}")
print(f"  Total features: {len(final_feature_names)}")
print(f"  Original features represented: {len(z_cols)}")
print(f"  Representation ratio: {100 * len(final_feature_names) / len(z_cols):.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
#  5. FORWARD SELECTION ON FINAL SET
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 5: FORWARD SELECTION ON FINAL FEATURE SET (LOOCV Ridge α=1.0)")
print("=" * 72)

loo = LeaveOneOut()
alpha = 1.0

selected_features_list = []
selected_r_list = []
history = []

candidate_names = final_feature_names.copy()

for step in range(min(20, len(candidate_names))):
    best_feature = None
    best_r_pearson = -np.inf
    best_r_spearman = -np.inf

    # Try each remaining feature
    for feat in candidate_names:
        if feat in [s[0] for s in selected_features_list]:
            continue

        # Build feature set
        if len(selected_features_list) == 0:
            X_test = final_features_df[[feat]]
        else:
            selected = [s[0] for s in selected_features_list]
            X_test = final_features_df[selected + [feat]]

        # LOOCV evaluation
        pred = np.zeros(n)
        for train_idx, test_idx in loo.split(X_test):
            X_train, X_test_fold = X_test.iloc[train_idx], X_test.iloc[test_idx]
            y_train, y_test_fold = y[train_idx], y[test_idx]

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, y_train)
            pred[test_idx[0]] = ridge.predict(X_test_fold)[0]

        # Compute correlations
        r_pearson = pearsonr(pred, y)[0]
        r_spearman = spearmanr(pred, y)[0]

        if r_pearson > best_r_pearson:
            best_feature = feat
            best_r_pearson = r_pearson
            best_r_spearman = r_spearman

    if best_feature is None:
        break

    selected_features_list.append((best_feature, best_r_pearson))
    selected_r_list.append(best_r_pearson)

    is_comb = "✓" if best_feature.startswith("comb_") else "•"
    feat_type = "COMBINED" if best_feature.startswith("comb_") else "UNIQUE"

    history.append({
        'step': step + 1,
        'feature': best_feature,
        'r_pearson': best_r_pearson,
        'r_spearman': best_r_spearman,
        'type': feat_type
    })

    print(f"  Step {step+1:2d}: {best_feature:<45} r={best_r_pearson:.4f} {is_comb} {feat_type}")

print(f"\n  Final selection: {len(selected_features_list)} features")
final_r = selected_r_list[-1] if selected_r_list else 0

# ═══════════════════════════════════════════════════════════════════════════
#  6. COMPARE WITH BASELINE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 6: COMPARISON WITH BASELINE")
print("=" * 72)

baseline_r = 0.8949
improvement = final_r - baseline_r
improvement_pct = 100 * improvement / baseline_r

print(f"\n  Baseline Model (current best):")
print(f"    • 15 features (8 base + 7 derived interactions)")
print(f"    • LOOCV r = {baseline_r:.4f}")
print(f"    • R² = {baseline_r**2:.4f}")

print(f"\n  Max Representation Model (with clusters):")
print(f"    • {len(final_feature_names)} features ({len(single_features)} unique + {len(combined_features_dict)} combined)")
print(f"    • Selected: {len(selected_features_list)}")
print(f"    • LOOCV r = {final_r:.4f}")
print(f"    • R² = {final_r**2:.4f}")

print(f"\n  Improvement:")
print(f"    • Δr = {improvement:+.4f} ({improvement_pct:+.2f}%)")

if improvement > 0.005:
    print(f"    ✓ MEANINGFUL IMPROVEMENT")
elif improvement > 0:
    print(f"    ~ Marginal improvement")
else:
    print(f"    ✗ No improvement")

# Count types in selection
combined_count = sum(1 for feat, _ in selected_features_list if feat.startswith("comb_"))
unique_count = len(selected_features_list) - combined_count

print(f"\n  Final model composition:")
print(f"    • Unique features: {unique_count}")
print(f"    • Combined features: {combined_count}")

# ═══════════════════════════════════════════════════════════════════════════
#  7. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 7: SAVE RESULTS")
print("=" * 72)

# Save combination log
comb_log_df = pd.DataFrame(combination_log)
comb_log_df.to_csv(os.path.join(RES, "max_representation_combinations.csv"), index=False)
print(f"\n  Saved: max_representation_combinations.csv ({len(comb_log_df)} clusters)")

# Save strong pairs
strong_pairs_df.to_csv(os.path.join(RES, "max_representation_strong_pairs.csv"), index=False)
print(f"  Saved: max_representation_strong_pairs.csv ({len(strong_pairs_df)} pairs)")

# Save selection history
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(RES, "max_representation_forward_selection.csv"), index=False)
print(f"  Saved: max_representation_forward_selection.csv ({len(history_df)} steps)")

# Save final selection
selection_df = pd.DataFrame({
    'rank': range(1, len(selected_features_list) + 1),
    'feature': [f[0] for f in selected_features_list],
    'r_pearson': [f[1] for f in selected_features_list],
    'type': ['COMBINED' if f[0].startswith('comb_') else 'UNIQUE' for f in selected_features_list]
})
selection_df.to_csv(os.path.join(RES, "max_representation_final_selection.csv"), index=False)
print(f"  Saved: max_representation_final_selection.csv ({len(selection_df)} features)")

# ═══════════════════════════════════════════════════════════════════════════
#  8. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Forward selection progression
steps = range(1, len(selected_r_list) + 1)
colors_fs = ['#e74c3c' if history_df.iloc[i-1]['type'] == 'COMBINED' else '#3498db'
             for i in steps]

ax1.plot(steps, selected_r_list, 'o-', color='#2c3e50', linewidth=2, markersize=8)
ax1.scatter(steps, selected_r_list, c=colors_fs, s=100, alpha=0.7, edgecolors='k', linewidth=1)
ax1.axhline(baseline_r, color='green', linestyle='--', linewidth=2, label=f'Baseline (r={baseline_r:.4f})')
ax1.set_xlabel('Step', fontweight='bold', fontsize=11)
ax1.set_ylabel('LOOCV r', fontweight='bold', fontsize=11)
ax1.set_title('Forward Selection Progress', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.set_ylim([0.7, 0.95])

# Plot 2: Feature set composition
ax2.bar(['Original\nFeatures', 'Final Feature\nSet', 'Selected in\nModel'],
        [len(z_cols), len(final_feature_names), len(selected_features_list)],
        color=['#95a5a6', '#3498db', '#2ecc71'], alpha=0.7, edgecolor='k', linewidth=2)
ax2.set_ylabel('Feature Count', fontweight='bold', fontsize=11)
ax2.set_title('Feature Set Compression', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

for i, count in enumerate([len(z_cols), len(final_feature_names), len(selected_features_list)]):
    ax2.text(i, count + 1, str(count), ha='center', fontweight='bold', fontsize=11)

# Plot 3: Cluster size distribution
cluster_sizes = [len(c) for c in cluster_groups]
ax3.hist(cluster_sizes, bins=range(2, max(cluster_sizes) + 2), color='#e74c3c', alpha=0.7,
         edgecolor='k', linewidth=1.5)
ax3.set_xlabel('Cluster Size (# of features)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Number of Clusters', fontweight='bold', fontsize=11)
ax3.set_title(f'Redundancy Cluster Distribution ({len(cluster_groups)} clusters)', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Performance comparison
models = ['Baseline\n(15 feat)', 'Max Representation\n(combined clusters)']
r_vals = [baseline_r, final_r]
r2_vals = [baseline_r**2, final_r**2]

x_pos = np.arange(len(models))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, r_vals, width, label='Pearson r', color='#3498db',
                alpha=0.7, edgecolor='k', linewidth=1.5)
ax4.set_ylabel('Correlation Coefficient', fontweight='bold', fontsize=11)
ax4.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models, fontsize=10)
ax4.set_ylim([0.8, 0.95])
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend(fontsize=10)

# Add value labels
for bar, r in zip(bars1, r_vals):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{r:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(RES, "max_representation_analysis.png"), dpi=150, bbox_inches='tight')
print("  Saved: max_representation_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════
#  9. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"\n  Feature Consolidation:")
print(f"    • Original features: {len(z_cols)}")
print(f"    • Strong correlation pairs (|r|>0.80): {len(strong_pairs_df)}")
print(f"    • Redundancy clusters created: {len(cluster_groups)}")
print(f"    • Final feature set: {len(final_feature_names)} ({100*len(final_feature_names)/len(z_cols):.1f}% of original)")

print(f"\n  Model Performance:")
print(f"    • Baseline: r = {baseline_r:.4f}")
print(f"    • Max Representation: r = {final_r:.4f}")
print(f"    • Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")

print(f"\n  Selected Features (n={len(selected_features_list)}):")
print(f"    • Unique (single): {unique_count}")
print(f"    • Combined (clusters): {combined_count}")

print("\n" + "=" * 72)
print("✓ MAXIMUM REPRESENTATION ANALYSIS COMPLETE")
print("=" * 72)
