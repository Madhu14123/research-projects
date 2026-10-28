"""
feature_condensation_and_combination.py
======================================
Condense feature set by:
1. Removing highly correlated redundant features
2. Keeping only strongest representative from each correlation cluster
3. Creating combined features from complementary correlated pairs
4. Output: condensed feature list + combined feature definitions
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("FEATURE CONDENSATION & COMBINATION ANALYSIS")
print("=" * 72)

# Load feature report
feature_report = pd.read_csv(os.path.join(RES, "feature_variation_report.csv"))

# Load raw features for correlation computation
features_D1 = pd.read_csv(os.path.join(RES, "features_D1.csv"))
features_D2 = pd.read_csv(os.path.join(RES, "features_D2.csv"))
features_D3 = pd.read_csv(os.path.join(RES, "features_D3.csv"))
features_D4 = pd.read_csv(os.path.join(RES, "features_D4.csv"))
features_D5 = pd.read_csv(os.path.join(RES, "features_D5.csv"))

# Merge all features
merged = features_D1.copy()
for df in [features_D2, features_D3, features_D4, features_D5]:
    cols_to_use = [c for c in df.columns if c not in merged.columns or c == "pid"]
    merged = merged.merge(df[cols_to_use], on="pid", how="outer")

# Extract feature columns (c* prefix)
feat_cols = [c for c in merged.columns if c.startswith("c")]
X_raw = merged[feat_cols].copy()

# Handle missing values
X_raw = X_raw.apply(lambda c: c.fillna(c.median()))

# Standardize for correlation
scaler = StandardScaler()
X_z = pd.DataFrame(scaler.fit_transform(X_raw), columns=feat_cols)

n_features = len(feat_cols)
print(f"\nLoaded {n_features} features from 5 dimensions")
print(f"Students: {len(X_z)}")

# ═══════════════════════════════════════════════════════════════════════════
#  1. COMPUTE CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 1: COMPUTE FEATURE CORRELATION MATRIX")
print("=" * 72)

print(f"\n  Computing correlations between {n_features} features...")
corr_matrix = X_z.corr().abs()

# Find redundant pairs (|r| > 0.80)
redundancy_threshold = 0.80
redundant_pairs = []

for i in range(n_features):
    for j in range(i + 1, n_features):
        corr_val = corr_matrix.iloc[i, j]
        if corr_val > redundancy_threshold:
            feat_i = feat_cols[i]
            feat_j = feat_cols[j]
            redundant_pairs.append({
                "feature_1": feat_i,
                "feature_2": feat_j,
                "correlation": corr_val,
                "rank_1": feature_report[feature_report['feature'] == feat_i]['rank'].values[0],
                "rank_2": feature_report[feature_report['feature'] == feat_j]['rank'].values[0],
                "votes_1": feature_report[feature_report['feature'] == feat_i]['total_votes'].values[0],
                "votes_2": feature_report[feature_report['feature'] == feat_j]['total_votes'].values[0]
            })

redundant_pairs = pd.DataFrame(redundant_pairs).sort_values('correlation', ascending=False)
n_redundant = len(redundant_pairs)

print(f"\n  Found {n_redundant} redundant pairs (|r| > {redundancy_threshold}):")
print(f"  {'Feature 1':<35} {'Feature 2':<35} {'Corr':>7} {'Winner':<20}")
print(f"  {'-'*100}")

for _, row in redundant_pairs.head(20).iterrows():
    winner = row['feature_1'] if row['votes_1'] > row['votes_2'] else row['feature_2']
    print(f"  {row['feature_1']:<35} {row['feature_2']:<35} {row['correlation']:>7.3f} {winner:<20}")

if n_redundant > 20:
    print(f"  ... ({n_redundant - 20} more pairs)")

# ═══════════════════════════════════════════════════════════════════════════
#  2. BUILD REDUNDANCY CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 2: IDENTIFY REDUNDANCY CLUSTERS")
print("=" * 72)

# Use union-find to group redundant features
parent = {feat: feat for feat in feat_cols}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[py] = px

for _, row in redundant_pairs.iterrows():
    union(row['feature_1'], row['feature_2'])

# Group features by cluster
clusters = {}
for feat in feat_cols:
    root = find(feat)
    if root not in clusters:
        clusters[root] = []
    clusters[root].append(feat)

# Get non-singleton clusters (redundancy groups)
redundancy_groups = {root: feats for root, feats in clusters.items() if len(feats) > 1}

print(f"\n  Total clusters: {len(clusters)}")
print(f"  Redundancy groups (size > 1): {len(redundancy_groups)}")
print(f"\n  Redundancy groups (top 10 by size):")
print(f"  {'Group ID':<35} {'Size':>5} {'Members':<60}")
print(f"  {'-'*100}")

for i, (root, group) in enumerate(sorted(redundancy_groups.items(),
                                         key=lambda x: len(x[1]), reverse=True)[:10]):
    print(f"  {root:<35} {len(group):>5} {', '.join(group[:3])}{'...' if len(group) > 3 else '':<50}")

# ═══════════════════════════════════════════════════════════════════════════
#  3. SELECT REPRESENTATIVES FROM EACH CLUSTER
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 3: SELECT FEATURE REPRESENTATIVES")
print("=" * 72)

selected_features = []
removed_features = []

for cluster_root, cluster_members in clusters.items():
    if len(cluster_members) == 1:
        # Keep single features as-is
        selected_features.append(cluster_members[0])
    else:
        # For redundancy groups, keep feature with highest votes
        member_votes = []
        for feat in cluster_members:
            votes = feature_report[feature_report['feature'] == feat]['total_votes'].values[0]
            member_votes.append((feat, votes))

        best_feat = max(member_votes, key=lambda x: x[1])[0]
        selected_features.append(best_feat)

        for feat in cluster_members:
            if feat != best_feat:
                removed_features.append({
                    'feature': feat,
                    'reason': f'Redundant with {best_feat}',
                    'correlation_with_best': corr_matrix.loc[feat, best_feat],
                    'votes': feature_report[feature_report['feature'] == feat]['total_votes'].values[0],
                    'best_votes': feature_report[feature_report['feature'] == best_feat]['total_votes'].values[0]
                })

selected_features.sort()
print(f"\n  Original features: {n_features}")
print(f"  Selected (unique representatives): {len(selected_features)}")
print(f"  Removed (redundant): {len(removed_features)}")
print(f"  Compression ratio: {100 * (1 - len(selected_features) / n_features):.1f}%")

print(f"\n  Top 20 REMOVED features (redundant):")
print(f"  {'Feature':<35} {'Redundant with':<35} {'Corr':>7} {'Votes':>6}")
print(f"  {'-'*85}")

for removed in sorted(removed_features, key=lambda x: x['correlation_with_best'], reverse=True)[:20]:
    print(f"  {removed['feature']:<35} {removed['reason']:<35} {removed['correlation_with_best']:>7.3f} {removed['votes']:>6}")

# ═══════════════════════════════════════════════════════════════════════════
#  4. IDENTIFY STRONG COMPLEMENTARY PAIRS FOR COMBINATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 4: IDENTIFY STRONG COMPLEMENTARY PAIRS FOR COMBINATION")
print("=" * 72)

# Look for pairs with medium-high correlation (0.5-0.8) that are both predictive
combination_threshold_min = 0.50
combination_threshold_max = 0.80

combination_candidates = []

for i in range(len(selected_features)):
    for j in range(i + 1, len(selected_features)):
        feat_i = selected_features[i]
        feat_j = selected_features[j]

        corr_val = corr_matrix.loc[feat_i, feat_j]

        if combination_threshold_min < corr_val < combination_threshold_max:
            votes_i = feature_report[feature_report['feature'] == feat_i]['total_votes'].values[0]
            votes_j = feature_report[feature_report['feature'] == feat_j]['total_votes'].values[0]

            # Only combine if both are reasonably predictive (votes >= 3)
            if votes_i >= 3 and votes_j >= 3:
                combination_candidates.append({
                    'feature_1': feat_i,
                    'feature_2': feat_j,
                    'correlation': corr_val,
                    'votes_1': votes_i,
                    'votes_2': votes_j,
                    'avg_votes': (votes_i + votes_j) / 2,
                    'dimension_1': feature_report[feature_report['feature'] == feat_i]['dimension'].values[0],
                    'dimension_2': feature_report[feature_report['feature'] == feat_j]['dimension'].values[0]
                })

combination_candidates = pd.DataFrame(combination_candidates)
combination_candidates = combination_candidates.sort_values('correlation', ascending=False)

print(f"\n  Found {len(combination_candidates)} strong complementary pairs:")
print(f"  (Correlation 0.50-0.80, both with votes >= 3)")
print(f"  {'Feature 1':<30} {'Feature 2':<30} {'Corr':>7} {'Avg Votes':>10} {'Type':<20}")
print(f"  {'-'*100}")

for _, row in combination_candidates.head(15).iterrows():
    same_dim = "same-dim" if row['dimension_1'] == row['dimension_2'] else "cross-dim"
    print(f"  {row['feature_1']:<30} {row['feature_2']:<30} {row['correlation']:>7.3f} {row['avg_votes']:>10.1f} {same_dim:<20}")

# ═══════════════════════════════════════════════════════════════════════════
#  5. CREATE COMBINED FEATURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 5: CREATE COMBINED FEATURES")
print("=" * 72)

combined_features = []
combination_definitions = []

# Create average combinations for top pairs
for idx, row in combination_candidates.head(10).iterrows():
    feat_1 = row['feature_1']
    feat_2 = row['feature_2']

    # Create combined feature: average
    combined_name = f"comb_avg_{feat_1}_x_{feat_2}"
    combined_values = (X_z[feat_1] + X_z[feat_2]) / 2

    # Test correlation of combined with outcome
    # (use dummy outcome for now - correlation between components)
    test_r = np.corrcoef(combined_values, X_z[feat_1])[0, 1]

    combined_features.append({
        'combined_name': combined_name,
        'feature_1': feat_1,
        'feature_2': feat_2,
        'method': 'average',
        'correlation': row['correlation'],
        'test_r': test_r
    })

    combination_definitions.append({
        'feature': combined_name,
        'dimension': f"{row['dimension_1']}+{row['dimension_2']}",
        'method': 'average',
        'component_1': feat_1,
        'component_2': feat_2,
        'correlation_between_components': row['correlation']
    })

combined_df = pd.DataFrame(combined_features)
print(f"\n  Created {len(combined_features)} combined features from top complementary pairs")
print(f"  {'Combined Name':<50} {'Method':<10} {'Corr':>7}")
print(f"  {'-'*70}")

for _, row in combined_df.iterrows():
    print(f"  {row['combined_name']:<50} {row['method']:<10} {row['correlation']:>7.3f}")

# ═══════════════════════════════════════════════════════════════════════════
#  6. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 6: SAVE RESULTS")
print("=" * 72)

# Save condensed feature list
condensed_report = feature_report[feature_report['feature'].isin(selected_features)].copy()
condensed_report = condensed_report.sort_values('rank')
condensed_report.to_csv(os.path.join(RES, "feature_list_condensed.csv"), index=False)
print(f"\n  Saved: feature_list_condensed.csv ({len(condensed_report)} features)")

# Save removed features list
removed_df = pd.DataFrame(removed_features)
removed_df.to_csv(os.path.join(RES, "feature_list_removed_redundant.csv"), index=False)
print(f"  Saved: feature_list_removed_redundant.csv ({len(removed_df)} features)")

# Save combination candidates and definitions
combination_candidates.to_csv(os.path.join(RES, "feature_combination_candidates.csv"), index=False)
print(f"  Saved: feature_combination_candidates.csv ({len(combination_candidates)} pairs)")

pd.DataFrame(combination_definitions).to_csv(
    os.path.join(RES, "feature_combination_definitions.csv"), index=False
)
print(f"  Saved: feature_combination_definitions.csv ({len(combination_definitions)} definitions)")

# Save redundancy pairs
redundant_pairs.to_csv(os.path.join(RES, "feature_redundancy_pairs.csv"), index=False)
print(f"  Saved: feature_redundancy_pairs.csv ({n_redundant} pairs)")

# ═══════════════════════════════════════════════════════════════════════════
#  7. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

# Figure 1: Redundancy distribution
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Count removed features by dimension
removed_by_dim = {}
for rem in removed_features:
    feat = rem['feature']
    dim = feature_report[feature_report['feature'] == feat]['dimension'].values[0]
    removed_by_dim[dim] = removed_by_dim.get(dim, 0) + 1

dims = sorted(removed_by_dim.keys())
counts = [removed_by_dim[d] for d in dims]
colors = ['#e74c3c' if c > 5 else '#f39c12' if c > 2 else '#2ecc71' for c in counts]

ax1.bar(dims, counts, color=colors, alpha=0.7, edgecolor='k', linewidth=1.5)
ax1.set_ylabel('Removed Features', fontweight='bold')
ax1.set_xlabel('Dimension', fontweight='bold')
ax1.set_title('Redundant Features by Dimension', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Compression ratio by dimension
all_by_dim = {}
for dim in ['c1', 'c2', 'c3', 'c4', 'c5']:
    all_by_dim[dim] = len([f for f in feat_cols if f.startswith(dim)])

dims_all = sorted(all_by_dim.keys())
kept = [all_by_dim[d] - removed_by_dim.get(d, 0) for d in dims_all]
removed = [removed_by_dim.get(d, 0) for d in dims_all]

ax2.bar(dims_all, kept, label='Kept', alpha=0.7, color='#2ecc71', edgecolor='k')
ax2.bar(dims_all, removed, bottom=kept, label='Removed', alpha=0.7, color='#e74c3c', edgecolor='k')
ax2.set_ylabel('Feature Count', fontweight='bold')
ax2.set_xlabel('Dimension', fontweight='bold')
ax2.set_title('Feature Compression by Dimension', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig1.savefig(os.path.join(RES, "feature_condensation_summary.png"), dpi=150, bbox_inches='tight')
print("  Saved: feature_condensation_summary.png")

# Figure 2: Correlation heatmap of selected features (subset)
fig2, ax2 = plt.subplots(figsize=(12, 10))

# Show top 30 selected features
top_selected = condensed_report.head(30)['feature'].values
subset_corr = corr_matrix.loc[top_selected, top_selected]

im = ax2.imshow(subset_corr, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
ax2.set_xticks(range(len(top_selected)))
ax2.set_yticks(range(len(top_selected)))
ax2.set_xticklabels(top_selected, rotation=90, fontsize=8)
ax2.set_yticklabels(top_selected, fontsize=8)
ax2.set_title('Top 30 Selected Features: Correlation Matrix', fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax2, label='|Correlation|')
plt.tight_layout()
fig2.savefig(os.path.join(RES, "feature_condensation_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
print("  Saved: feature_condensation_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
#  8. SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY STATISTICS")
print("=" * 72)

print(f"\n  Feature Set Reduction:")
print(f"    Original features: {n_features}")
print(f"    Condensed features: {len(selected_features)}")
print(f"    Removed redundant: {len(removed_features)}")
print(f"    Compression: {100 * len(removed_features) / n_features:.1f}%")

print(f"\n  Redundancy by Dimension:")
for dim in ['c1', 'c2', 'c3', 'c4', 'c5']:
    orig = len([f for f in feat_cols if f.startswith(dim)])
    rem = sum(1 for r in removed_features if feature_report[
        feature_report['feature'] == r['feature']]['dimension'].values[0] == dim)
    kept = orig - rem
    print(f"    {dim.upper()}: {orig:3d} → {kept:3d} (removed {rem:2d}, {100*rem/orig:.0f}%)")

print(f"\n  Combination Opportunities:")
print(f"    Same-dimension pairs: {sum(1 for _, r in combination_candidates.iterrows() if r['dimension_1'] == r['dimension_2'])}")
print(f"    Cross-dimension pairs: {sum(1 for _, r in combination_candidates.iterrows() if r['dimension_1'] != r['dimension_2'])}")
print(f"    Created combined features: {len(combination_definitions)}")

print("\n" + "=" * 72)
print("✓ FEATURE CONDENSATION COMPLETE")
print("=" * 72)
