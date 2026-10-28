"""
cross_dimensionality_analysis.py
=================================
Analyze relationships BETWEEN dimensions (C1-C5).
Find which cross-dimensional features predict exam best.
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
from sklearn.decomposition import PCA

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")
TARGET = "pct"

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("CROSS-DIMENSIONALITY ANALYSIS (C1-C5 Interactions)")
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

scaler = StandardScaler()
X_z = pd.DataFrame(scaler.fit_transform(X_raw), columns=all_feat_cols)

n = len(y)
print(f"\nLoaded {n} students")

# ═══════════════════════════════════════════════════════════════════════════
#  DIMENSION COMPOSITES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("DIMENSION COMPOSITES")
print("=" * 72)

dimensions = {}
for dim in ["c1", "c2", "c3", "c4", "c5"]:
    dim_feats = [f for f in all_feat_cols if f.startswith(dim)]
    # PCA to get dimension score (first PC)
    pca = PCA(n_components=1)
    dim_score = pca.fit_transform(X_z[dim_feats])[:, 0]
    dimensions[dim] = dim_score
    r, p = pearsonr(dim_score, y)
    print(f"  {dim.upper()} ({len(dim_feats)} features): PCA score r={r:+.4f}, p={p:.6f}, "
          f"var_exp={pca.explained_variance_ratio_[0]:.1%}")

# ═══════════════════════════════════════════════════════════════════════════
#  CROSS-DIMENSIONAL PRODUCTS & RATIOS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("CROSS-DIMENSIONAL INTERACTIONS")
print("=" * 72)

cross_dim_results = []

dim_names = list(dimensions.keys())
for i, d1 in enumerate(dim_names):
    for d2 in dim_names[i+1:]:
        # Product
        prod = dimensions[d1] * dimensions[d2]
        # Normalize
        prod = (prod - prod.mean()) / (prod.std() + 1e-10)
        r_prod, p_prod = pearsonr(prod, y)

        # Ratio
        denom = dimensions[d2].copy()
        denom[np.abs(denom) < 0.01] = np.sign(denom[np.abs(denom) < 0.01] + 0.001) * 0.01
        ratio = dimensions[d1] / denom
        q01, q99 = np.percentile(ratio, [1, 99])
        ratio = np.clip(ratio, q01, q99)
        ratio = (ratio - ratio.mean()) / (ratio.std() + 1e-10)
        r_ratio, p_ratio = pearsonr(ratio, y)

        # Difference
        diff = dimensions[d1] - dimensions[d2]
        r_diff, p_diff = pearsonr(diff, y)

        # Max (bottleneck)
        mx = np.maximum(dimensions[d1], dimensions[d2])
        r_max, p_max = pearsonr(mx, y)

        cross_dim_results.append({
            "pair": f"{d1}-{d2}", "type": "product",
            "r": r_prod, "p": p_prod, "values": prod
        })
        cross_dim_results.append({
            "pair": f"{d1}-{d2}", "type": "ratio",
            "r": r_ratio, "p": p_ratio, "values": ratio
        })
        cross_dim_results.append({
            "pair": f"{d1}-{d2}", "type": "difference",
            "r": r_diff, "p": p_diff, "values": diff
        })
        cross_dim_results.append({
            "pair": f"{d1}-{d2}", "type": "max",
            "r": r_max, "p": p_max, "values": mx
        })

cross_df = pd.DataFrame([{k: v for k, v in row.items() if k != "values"}
                         for row in cross_dim_results])
cross_df['abs_r'] = np.abs(cross_df['r'])
cross_df = cross_df.sort_values('abs_r', ascending=False)

print(f"\n  Top 20 cross-dimensional features:")
print(f"  {'Pair':<8} {'Type':<12} {'r':>8} {'p':>10}")
print(f"  {'-'*45}")
for _, row in cross_df.head(20).iterrows():
    print(f"  {row['pair']:<8} {row['type']:<12} {row['r']:+8.4f} {row['p']:10.6f}")

# ═══════════════════════════════════════════════════════════════════════════
#  DIMENSION CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("DIMENSION CORRELATION MATRIX")
print("=" * 72)

dim_corr_matrix = np.zeros((5, 5))
for i, d1 in enumerate(dim_names):
    for j, d2 in enumerate(dim_names):
        if i == j:
            r = 1.0
        else:
            r, _ = pearsonr(dimensions[d1], dimensions[d2])
        dim_corr_matrix[i, j] = r

print(f"\n  {' '*5} {'c1':>8} {'c2':>8} {'c3':>8} {'c4':>8} {'c5':>8}")
print(f"  {'-'*45}")
for i, d1 in enumerate(dim_names):
    row_str = f"  {d1.upper()}"
    for j in range(5):
        row_str += f" {dim_corr_matrix[i, j]:+8.3f}"
    print(row_str)

# ═══════════════════════════════════════════════════════════════════════════
#  FIND WHICH DIMENSION IS MOST PREDICTIVE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("DIMENSION RANKING (by individual r with exam)")
print("=" * 72)

dim_ranking = []
for dim in dim_names:
    r, p = pearsonr(dimensions[dim], y)
    dim_ranking.append({"dim": dim.upper(), "r": r, "p": p})

dim_ranking.sort(key=lambda x: abs(x['r']), reverse=True)
for _, row in enumerate(dim_ranking):
    print(f"  {row['dim']}: r={row['r']:+.4f}, p={row['p']:.6f}")

# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

# Heatmap: dimension correlation
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(dim_corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels([d.upper() for d in dim_names])
ax.set_yticklabels([d.upper() for d in dim_names])
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{dim_corr_matrix[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(dim_corr_matrix[i, j]) > 0.5 else "black", fontsize=10)
plt.colorbar(im, ax=ax, label="Correlation")
ax.set_title("Dimension-Dimension Correlation Matrix", fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(RES, "cross_dimensionality_heatmap.png"), dpi=150, bbox_inches="tight")
print("  Saved: cross_dimensionality_heatmap.png")

# Bar chart: top cross-dimensional features
fig2, ax = plt.subplots(figsize=(12, 8))
top_n = 20
top_cross = cross_df.head(top_n)
colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in top_cross['r']]
labels = [f"{row['pair']} ({row['type']})" for _, row in top_cross.iterrows()]
ax.barh(range(len(top_cross)), top_cross['r'].values, color=colors, edgecolor="k", alpha=0.7)
ax.set_yticks(range(len(top_cross)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Pearson r with Exam %", fontweight="bold")
ax.set_title("Top 20 Cross-Dimensional Features", fontweight="bold", fontsize=13)
ax.axvline(0, color="k", linewidth=1)
ax.grid(True, alpha=0.3, axis="x")
ax.invert_yaxis()
plt.tight_layout()
fig2.savefig(os.path.join(RES, "cross_dimensionality_features.png"), dpi=150, bbox_inches="tight")
print("  Saved: cross_dimensionality_features.png")

print("\n" + "=" * 72)
print("✓ CROSS-DIMENSIONALITY ANALYSIS COMPLETE")
print("=" * 72)
