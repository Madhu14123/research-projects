"""
consensus_feature_pool.py
=========================
Reads feature_variation_report.csv (produced by feature_variation_and_ranking.py)
and identifies the **most stable, consensus-driven feature pool** using:

  1. Elbow detection    — finds the natural drop-off in composite_rank_score
  2. Vote-threshold     — features with >= 3/7, >= 4/7, etc. selection votes
  3. Multi-signal       — features that rank in the top-N across MULTIPLE
                          independent signals (|r|, MI, eta², RF importance, votes)
  4. Stability bootstrap — re-samples students 500× and checks how often each
                           feature stays selected (bootstrap inclusion probability)
  5. Dimension coverage — ensures the final pool draws from all 5 dimensions

Outputs (results/):
  consensus_pool.csv           — the final selected feature pool with all evidence
  consensus_pool_summary.txt   — human-readable summary
  pool_evidence_plot.png       — visual: votes × correlation × stability
  elbow_plot.png               — composite score curve with detected elbow
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from kneed import KneeLocator   # will fallback if not installed

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")
REPORT_PATH = os.path.join(RES, "feature_variation_report.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load the ranked report
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 1  Loading feature_variation_report.csv")
print("=" * 72)

report = pd.read_csv(REPORT_PATH, index_col=0)
print(f"  {len(report)} features loaded")

# also reload raw data for bootstrap stability
FILES = {f"D{i}": os.path.join(RES, f"features_D{i}.csv") for i in range(1, 6)}
TARGET = "pct"
frames = []
for tag, path in FILES.items():
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("c")]
    keep = ["pid"] + feat_cols
    if TARGET in df.columns and len(frames) == 0:
        keep.append(TARGET)
    frames.append(df[keep])

merged = frames[0]
for f in frames[1:]:
    merged = merged.merge(f, on="pid", how="outer")

pct_cols = [c for c in merged.columns if c.startswith("pct")]
if len(pct_cols) > 1:
    merged[TARGET] = merged[pct_cols[0]]
    merged.drop(columns=pct_cols[1:], inplace=True)

feature_cols = sorted([c for c in merged.columns if c.startswith("c")])
valid = merged[TARGET].notna()
merged_v = merged.loc[valid].reset_index(drop=True)
y = merged_v[TARGET].values
n = len(y)

X_raw = merged_v[feature_cols].copy().apply(lambda c: c.fillna(c.median()))

# ══════════════════════════════════════════════════════════════════════════════
# 2. Elbow detection on composite_rank_score
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 2  Elbow detection on composite rank score")
print("=" * 72)

scores = report["composite_rank_score"].values
x_idx  = np.arange(1, len(scores) + 1)

# try kneed library, fallback to simple gradient method
try:
    from kneed import KneeLocator
    kl = KneeLocator(x_idx, scores, curve="convex", direction="decreasing",
                     S=1.0, interp_method="polynomial")
    elbow_k = kl.knee
    print(f"  Kneed elbow at k = {elbow_k}")
except ImportError:
    elbow_k = None
    print("  kneed not installed, using gradient fallback")

# gradient fallback: find where the second derivative is maximally negative
diffs = np.diff(scores)
d2    = np.diff(diffs)
# the elbow is where the drop accelerates most
if elbow_k is None or elbow_k is None:
    candidates = np.where(d2 < np.percentile(d2, 10))[0]
    if len(candidates) > 0:
        elbow_k = candidates[0] + 2   # +2 for diff offsets
    else:
        elbow_k = 15  # safe default
    print(f"  Gradient-based elbow at k = {elbow_k}")

# also find a generous elbow (composite > 0.3 threshold)
threshold_k = int((report["composite_rank_score"] >= 0.30).sum())
print(f"  Features with composite >= 0.30 : {threshold_k}")

# plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_idx, scores, "b.-", linewidth=1, markersize=3)
ax.axvline(elbow_k, color="red", linestyle="--", alpha=0.7,
           label=f"Elbow k={elbow_k}")
ax.axhline(0.30, color="orange", linestyle=":", alpha=0.5,
           label="threshold=0.30")
ax.set_xlabel("Feature rank")
ax.set_ylabel("Composite rank score")
ax.set_title("Composite score drop-off with elbow detection")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(RES, "elbow_plot.png"), dpi=150)
plt.close()
print("  Saved elbow_plot.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Vote-threshold tiers
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 3  Vote-threshold tiers")
print("=" * 72)

for thresh in [7, 6, 5, 4, 3]:
    cnt = (report["total_votes"] >= thresh).sum()
    print(f"  >= {thresh}/7 votes : {cnt:3d} features")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Multi-signal top-N intersection
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 4  Multi-signal top-N intersection")
print("=" * 72)

# For each signal, rank features and flag top-25%
top_frac = 0.25
top_n = max(5, int(len(report) * top_frac))

signal_cols = {
    "abs_pearson":  "Pearson |r|",
    "abs_spearman": "Spearman |rho|",
    "mutual_info":  "Mutual Info",
    "eta_squared":  "Eta-squared",
    "score_RF":     "RF importance",
    "score_GB":     "GB importance",
    "score_PLS_VIP":"PLS VIP",
}

report["n_top_signals"] = 0
for col, label in signal_cols.items():
    if col not in report.columns:
        continue
    ranked = report[col].fillna(0).rank(ascending=False)
    in_top = (ranked <= top_n).astype(int)
    report[f"top25_{col}"] = in_top
    report["n_top_signals"] += in_top

print(f"  Checking top-{top_n} across {len(signal_cols)} signals:")
for threshold in [7, 6, 5, 4, 3]:
    cnt = (report["n_top_signals"] >= threshold).sum()
    if cnt > 0:
        print(f"    in top-25% of >= {threshold} signals : {cnt} features")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Bootstrap stability (500 resamples)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 5  Bootstrap stability analysis (500 resamples)")
print("=" * 72)

scaler = StandardScaler()
X_sc = pd.DataFrame(scaler.fit_transform(X_raw), columns=feature_cols)

B = 500
inclusion_count = np.zeros(len(feature_cols))
rng = np.random.default_rng(42)

for b in range(B):
    idx = rng.choice(n, size=n, replace=True)
    X_b = X_sc.iloc[idx].values
    y_b = y[idx]

    # use RF as the selector (fast, captures non-linearity)
    rf_b = RandomForestRegressor(n_estimators=100, max_depth=4,
                                  random_state=b, n_jobs=-1)
    rf_b.fit(X_b, y_b)
    imp = rf_b.feature_importances_

    # select features with importance > mean importance
    threshold = imp.mean()
    inclusion_count += (imp > threshold).astype(int)

    if (b + 1) % 100 == 0:
        print(f"  [{b+1}/{B}] done")

stability = inclusion_count / B
report["bootstrap_stability"] = report["feature"].map(
    dict(zip(feature_cols, stability))
)

print(f"  Features with stability > 0.50 : "
      f"{(report['bootstrap_stability'] > 0.50).sum()}")
print(f"  Features with stability > 0.70 : "
      f"{(report['bootstrap_stability'] > 0.70).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Final consensus pool assembly
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 6  Assembling final consensus pool")
print("=" * 72)

# A feature enters the pool if it satisfies ANY 2 of these 3 gates:
#   Gate A: total_votes >= 3  (method consensus)
#   Gate B: composite_rank_score in top-elbow_k  (elbow)
#   Gate C: bootstrap_stability > 0.50  (stable across resamples)

top_elbow_feats = set(report.head(elbow_k)["feature"].values)

gate_a = report["total_votes"] >= 3
gate_b = report["feature"].isin(top_elbow_feats)
gate_c = report["bootstrap_stability"] > 0.50

report["gate_A_votes"] = gate_a.astype(int)
report["gate_B_elbow"] = gate_b.astype(int)
report["gate_C_bootstrap"] = gate_c.astype(int)
report["gates_passed"] = report["gate_A_votes"] + report["gate_B_elbow"] + report["gate_C_bootstrap"]

pool = report[report["gates_passed"] >= 2].copy()
pool = pool.sort_values("composite_rank_score", ascending=False).reset_index(drop=True)
pool.index = pool.index + 1
pool.index.name = "pool_rank"

print(f"\n  CONSENSUS POOL SIZE : {len(pool)} features")
print(f"  Dimension coverage:")
for dim in ["c1", "c2", "c3", "c4", "c5"]:
    dim_count = pool["feature"].str.startswith(dim + "_").sum()
    print(f"    {dim.upper()} : {dim_count} features")

print(f"\n  Pool features:")
for i, row in pool.iterrows():
    print(f"    {i:3d}. {row['feature']:45s}  "
          f"votes={int(row['total_votes'])}  "
          f"composite={row['composite_rank_score']:.3f}  "
          f"stability={row['bootstrap_stability']:.2f}  "
          f"gates={int(row['gates_passed'])}/3")

# save
pool.to_csv(os.path.join(RES, "consensus_pool.csv"))
print(f"\n  Saved consensus_pool.csv")

# Also save a compact summary
with open(os.path.join(RES, "consensus_pool_summary.txt"), "w") as fh:
    fh.write("CONSENSUS FEATURE POOL\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Total features evaluated : {len(report)}\n")
    fh.write(f"Pool size                : {len(pool)}\n")
    fh.write(f"Elbow-k                  : {elbow_k}\n\n")

    fh.write("Selection gates (need >= 2/3 to enter pool):\n")
    fh.write(f"  Gate A  >= 3/7 method votes       : {gate_a.sum()} features pass\n")
    fh.write(f"  Gate B  top-{elbow_k} by composite : {gate_b.sum()} features pass\n")
    fh.write(f"  Gate C  bootstrap stability > 0.50: {gate_c.sum()} features pass\n\n")

    fh.write("Dimension coverage:\n")
    for dim in ["c1", "c2", "c3", "c4", "c5"]:
        dim_count = pool["feature"].str.startswith(dim + "_").sum()
        fh.write(f"  {dim.upper()} : {dim_count}\n")

    fh.write(f"\nRanked pool:\n")
    fh.write(f"{'Rank':>4s}  {'Feature':45s}  {'Votes':>5s}  "
             f"{'Composite':>9s}  {'Stability':>9s}  {'Gates':>5s}\n")
    fh.write("-" * 85 + "\n")
    for i, row in pool.iterrows():
        fh.write(f"{i:4d}  {row['feature']:45s}  "
                 f"{int(row['total_votes']):5d}  "
                 f"{row['composite_rank_score']:9.4f}  "
                 f"{row['bootstrap_stability']:9.3f}  "
                 f"{int(row['gates_passed']):5d}/3\n")

print("  Saved consensus_pool_summary.txt")

# save updated full report too (with bootstrap + gates columns)
report.to_csv(os.path.join(RES, "feature_variation_report.csv"))
print("  Updated feature_variation_report.csv with stability + gate columns")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Evidence plot: votes × |r| × stability bubble chart
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 7  Generating pool evidence plot")
print("=" * 72)

fig, ax = plt.subplots(figsize=(12, 7))

# all features (grey, small)
not_pool = report[report["gates_passed"] < 2]
ax.scatter(not_pool["total_votes"],
           not_pool["abs_pearson"].fillna(0),
           s=not_pool["bootstrap_stability"].fillna(0) * 80 + 5,
           c="#bdc3c7", alpha=0.4, edgecolors="none", label="Not in pool")

# pool features (coloured by dimension)
dim_colors = {"c1": "#e74c3c", "c2": "#3498db", "c3": "#2ecc71",
              "c4": "#9b59b6", "c5": "#f39c12"}
for dim, color in dim_colors.items():
    sub = pool[pool["feature"].str.startswith(dim + "_")]
    if len(sub) == 0:
        continue
    ax.scatter(sub["total_votes"],
               sub["abs_pearson"].fillna(0),
               s=sub["bootstrap_stability"].fillna(0) * 200 + 20,
               c=color, alpha=0.8, edgecolors="k", linewidths=0.5,
               label=f"{dim.upper()} (pool)")
    for _, row in sub.iterrows():
        short = row["feature"].split("_", 1)[1][:30]
        ax.annotate(short,
                    (row["total_votes"], row.get("abs_pearson", 0) or 0),
                    fontsize=5.5, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")

ax.set_xlabel("Selection method votes (out of 7)", fontsize=10)
ax.set_ylabel("|Pearson r| with exam score", fontsize=10)
ax.set_title("Feature pool evidence\n(bubble size = bootstrap stability)",
             fontsize=11)
ax.axhline(0.3, color="grey", linestyle=":", alpha=0.3)
ax.axvline(3, color="grey", linestyle=":", alpha=0.3)
ax.legend(fontsize=7, loc="upper left")
plt.tight_layout()
fig.savefig(os.path.join(RES, "pool_evidence_plot.png"), dpi=150)
plt.close()
print("  Saved pool_evidence_plot.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Correlation heatmap of pool features
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 8  Pool feature inter-correlation heatmap")
print("=" * 72)

pool_feats = pool["feature"].tolist()
if len(pool_feats) >= 3:
    corr_matrix = X_raw[pool_feats].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(pool_feats) * 0.5),
                                    max(6, len(pool_feats) * 0.4)))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1,
                   aspect="auto")
    ax.set_xticks(range(len(pool_feats)))
    ax.set_xticklabels([f.split("_", 1)[1][:25] for f in pool_feats],
                       fontsize=6, rotation=90)
    ax.set_yticks(range(len(pool_feats)))
    ax.set_yticklabels([f.split("_", 1)[1][:25] for f in pool_feats],
                       fontsize=6)
    ax.set_title("Inter-correlation of consensus pool features")
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(RES, "pool_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved pool_correlation_heatmap.png")

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
