"""
feature_pattern_analysis.py
============================
Purely unsupervised — NO exam scores used at all.

For every feature, asks: "Does this feature show a structured, common pattern
across students, or is it just noise?"

Signals computed per feature:
  1. Distribution shape     — CV, skewness, kurtosis, range, IQR
  2. Tightness              — fraction of students within 1 SD of mean
  3. Bimodality coefficient — >0.555 suggests two natural student groups
  4. Entropy                — low entropy = students cluster at similar values
  5. Dip test (Hartigan)    — significant = multimodal (clear subgroups)
  6. Cluster structure      — silhouette score with k=2,3 (KMeans on 1D)
  7. Consistency index      — 1 - (MAD / range), high = students agree
  8. Redundancy             — mean |r| with all other features; high = part of
                              a shared latent factor across dimensions
  9. Cross-dimension links  — how many features from OTHER dimensions it
                              correlates with (|r| > 0.3)

A "pattern_score" composites all these into a single rank.

Outputs (results/):
  feature_pattern_report.csv        — one row per feature, all metrics + rank
  pattern_tiers.csv                 — features grouped into Strong / Moderate / Weak
  pattern_distribution_plots/       — per-dimension histograms showing each feature
  pattern_summary.txt               — human-readable summary
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import (
    skew, kurtosis, shapiro, entropy as sp_entropy
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── paths ────────────────────────────────────────────────────────────────────
BASE     = "D:/A_emotions in exam"
RES      = os.path.join(BASE, "results")
PLOT_DIR = os.path.join(RES, "pattern_distribution_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

FILES = {f"D{i}": os.path.join(RES, f"features_D{i}.csv") for i in range(1, 6)}

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & merge (drop all outcome columns — purely unsupervised)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 1  Loading data (no exam scores)")
print("=" * 72)

frames = []
for tag, path in FILES.items():
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("c")]
    frames.append(df[["pid"] + feat_cols])

merged = frames[0]
for f in frames[1:]:
    merged = merged.merge(f, on="pid", how="outer")

feature_cols = sorted([c for c in merged.columns if c.startswith("c")])
pids = merged["pid"].values
n = len(merged)

print(f"  Students : {n}")
print(f"  Features : {len(feature_cols)}")

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def bimodality_coefficient(x):
    """BC = (skew^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)(n-3))).
    BC > 0.555 suggests bimodality."""
    n = len(x)
    if n < 4:
        return np.nan
    s = skew(x)
    k = kurtosis(x, fisher=False)  # excess=False → Pearson kurtosis
    correction = 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    bc = (s ** 2 + 1) / (k + correction)
    return bc


def discretized_entropy(x, n_bins=10):
    """Entropy of the histogram — low = students pile at few values."""
    counts, _ = np.histogram(x, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return sp_entropy(probs, base=2)


def hartigan_dip_statistic(x):
    """Simple dip test approximation. Returns dip statistic.
    Uses sorted data and the greatest deviation from a unimodal CDF."""
    x_sorted = np.sort(x)
    n = len(x_sorted)
    if n < 5:
        return 0.0

    # empirical CDF
    ecdf = np.arange(1, n + 1) / n

    # best fitting uniform (unimodal) CDF
    lo, hi = x_sorted[0], x_sorted[-1]
    if hi == lo:
        return 0.0
    uniform_cdf = (x_sorted - lo) / (hi - lo)

    # dip = max deviation between ECDF and closest unimodal CDF
    dip = np.max(np.abs(ecdf - uniform_cdf))
    return dip


def cluster_silhouette(x, k=2):
    """Silhouette score for k-means on 1D data. High = clear clusters."""
    if len(np.unique(x)) < k:
        return -1.0
    X2d = x.reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X2d)
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(X2d, labels)


def consistency_index(x):
    """1 - (MAD / range). High = students are tightly grouped."""
    rng = np.ptp(x)
    if rng < 1e-12:
        return 1.0
    mad = np.mean(np.abs(x - np.median(x)))
    return 1.0 - (mad / rng)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Compute per-feature pattern metrics
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 2  Computing pattern metrics for each feature")
print("=" * 72)

rows = []
for i, col in enumerate(feature_cols):
    vals = merged[col].values.astype(float)
    non_nan = vals[~np.isnan(vals)]
    n_valid = len(non_nan)
    pct_miss = 100.0 * (n - n_valid) / n

    if n_valid < 5:
        rows.append({"feature": col, "dimension": col.split("_")[0],
                      "n_valid": n_valid, "pct_missing": round(pct_miss, 1),
                      "pattern_score": 0.0})
        continue

    mn   = np.mean(non_nan)
    sd   = np.std(non_nan, ddof=1)
    cv   = sd / (abs(mn) + 1e-12)
    med  = np.median(non_nan)
    q1, q3 = np.percentile(non_nan, [25, 75])
    iqr  = q3 - q1
    rng  = np.ptp(non_nan)

    # tightness: fraction within 1 SD
    within_1sd = np.mean(np.abs(non_nan - mn) <= sd)

    # shape
    sk  = skew(non_nan)
    ku  = kurtosis(non_nan)  # excess kurtosis

    # normality test
    if n_valid >= 8:
        _, shap_p = shapiro(non_nan)
    else:
        shap_p = np.nan

    # bimodality
    bc = bimodality_coefficient(non_nan)

    # entropy
    ent = discretized_entropy(non_nan, n_bins=min(10, n_valid // 3))

    # dip test
    dip = hartigan_dip_statistic(non_nan)

    # cluster silhouette (k=2 and k=3)
    sil2 = cluster_silhouette(non_nan, k=2)
    sil3 = cluster_silhouette(non_nan, k=3)

    # consistency
    cons = consistency_index(non_nan)

    rows.append({
        "feature": col,
        "dimension": col.split("_")[0],
        "n_valid": n_valid,
        "pct_missing": round(pct_miss, 1),
        "mean": round(mn, 6),
        "std": round(sd, 6),
        "CV": round(cv, 4),
        "median": round(med, 6),
        "IQR": round(iqr, 6),
        "range": round(rng, 6),
        "skewness": round(sk, 4),
        "kurtosis_excess": round(ku, 4),
        "shapiro_p": round(shap_p, 6) if not np.isnan(shap_p) else np.nan,
        "within_1SD_frac": round(within_1sd, 4),
        "bimodality_coeff": round(bc, 4) if not np.isnan(bc) else np.nan,
        "entropy": round(ent, 4),
        "dip_statistic": round(dip, 4),
        "silhouette_k2": round(sil2, 4),
        "silhouette_k3": round(sil3, 4),
        "consistency": round(cons, 4),
    })

    if (i + 1) % 25 == 0:
        print(f"  [{i+1}/{len(feature_cols)}] features processed")

pattern_df = pd.DataFrame(rows)
print(f"  Computed metrics for {len(pattern_df)} features")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Redundancy + cross-dimension links
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 3  Redundancy & cross-dimension correlation structure")
print("=" * 72)

# impute for correlation matrix
X_imp = merged[feature_cols].apply(lambda c: c.fillna(c.median()))
corr_full = X_imp.corr().abs()

# per-feature: mean |r| with all others
mean_abs_corr = corr_full.mean(axis=1) - 1.0 / len(feature_cols)  # subtract self
pattern_df["redundancy"] = pattern_df["feature"].map(
    mean_abs_corr.to_dict()
).round(4)

# cross-dimension links: count features from OTHER dimensions with |r| > 0.3
cross_dim_links = {}
for col in feature_cols:
    dim = col.split("_")[0]
    others = [c for c in feature_cols if not c.startswith(dim + "_")]
    if len(others) == 0:
        cross_dim_links[col] = 0
        continue
    r_vals = corr_full.loc[col, others]
    cross_dim_links[col] = int((r_vals > 0.3).sum())

pattern_df["cross_dim_links"] = pattern_df["feature"].map(cross_dim_links)

print(f"  Mean redundancy   : {pattern_df['redundancy'].mean():.3f}")
print(f"  Mean cross-dim links : {pattern_df['cross_dim_links'].mean():.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Composite pattern_score
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 4  Computing composite pattern score")
print("=" * 72)

# Normalise each signal to [0, 1], then average.
# Signals of structured pattern:
#   - HIGH consistency          → students agree on this feature
#   - HIGH within_1SD_frac      → tight distribution
#   - HIGH silhouette_k2        → clear 2-group structure
#   - HIGH bimodality_coeff     → bimodal = natural subgroups
#   - HIGH dip_statistic        → multimodal
#   - LOW  entropy              → concentrated (invert)
#   - LOW  shapiro_p            → non-normal = structured (invert)
#   - HIGH redundancy           → part of a latent factor
#   - HIGH cross_dim_links      → pattern spans dimensions
#   - LOW  pct_missing          → reliably measurable (invert)

def norm01(s):
    """Min-max normalise to [0, 1], handling NaN."""
    s = s.fillna(0)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)

signals = {}

signals["consistency"]    = norm01(pattern_df["consistency"])
signals["tightness"]      = norm01(pattern_df["within_1SD_frac"])
signals["cluster_k2"]     = norm01(pattern_df["silhouette_k2"].clip(lower=0))
signals["cluster_k3"]     = norm01(pattern_df["silhouette_k3"].clip(lower=0))
signals["bimodality"]     = norm01(pattern_df["bimodality_coeff"].fillna(0))
signals["dip"]            = norm01(pattern_df["dip_statistic"])
signals["low_entropy"]    = 1.0 - norm01(pattern_df["entropy"])
signals["non_normal"]     = 1.0 - norm01(pattern_df["shapiro_p"].fillna(1))
signals["redundancy"]     = norm01(pattern_df["redundancy"])
signals["cross_dim"]      = norm01(pattern_df["cross_dim_links"])
signals["completeness"]   = 1.0 - norm01(pattern_df["pct_missing"])

for name, vals in signals.items():
    pattern_df[f"sig_{name}"] = vals.round(4)

sig_cols = [c for c in pattern_df.columns if c.startswith("sig_")]
pattern_df["pattern_score"] = pattern_df[sig_cols].mean(axis=1).round(4)

# rank
pattern_df = pattern_df.sort_values("pattern_score", ascending=False).reset_index(drop=True)
pattern_df.index = pattern_df.index + 1
pattern_df.index.name = "rank"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Tier assignment
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 5  Assigning pattern tiers")
print("=" * 72)

# top 33% = Strong, middle 33% = Moderate, bottom 33% = Weak
q66 = pattern_df["pattern_score"].quantile(0.67)
q33 = pattern_df["pattern_score"].quantile(0.33)

def tier(score):
    if score >= q66:
        return "Strong"
    elif score >= q33:
        return "Moderate"
    else:
        return "Weak"

pattern_df["tier"] = pattern_df["pattern_score"].apply(tier)

for t in ["Strong", "Moderate", "Weak"]:
    cnt = (pattern_df["tier"] == t).sum()
    print(f"  {t:10s} : {cnt} features")

print(f"\n  Top-20 features with strongest cross-student pattern:")
for i, row in pattern_df.head(20).iterrows():
    print(f"    {i:3d}. {row['feature']:45s}  "
          f"score={row['pattern_score']:.3f}  "
          f"tier={row['tier']:8s}  "
          f"consist={row['consistency']:.2f}  "
          f"sil_k2={row['silhouette_k2']:.2f}  "
          f"BC={row.get('bimodality_coeff', 0):.3f}  "
          f"cross_dim={int(row['cross_dim_links'])}")

# save
out_path = os.path.join(RES, "feature_pattern_report.csv")
pattern_df.to_csv(out_path)
print(f"\n  Saved {out_path}")

# tiers file
tiers_df = pattern_df[["feature", "dimension", "pattern_score", "tier",
                        "consistency", "silhouette_k2", "bimodality_coeff",
                        "redundancy", "cross_dim_links", "entropy"]].copy()
tiers_df.to_csv(os.path.join(RES, "pattern_tiers.csv"))
print(f"  Saved pattern_tiers.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Summary text
# ══════════════════════════════════════════════════════════════════════════════
with open(os.path.join(RES, "pattern_summary.txt"), "w") as fh:
    fh.write("UNSUPERVISED FEATURE PATTERN ANALYSIS\n")
    fh.write("(No exam scores used — purely cross-student structure)\n")
    fh.write("=" * 65 + "\n\n")
    fh.write(f"Total features : {len(pattern_df)}\n")
    fh.write(f"Students       : {n}\n\n")

    fh.write("Tier thresholds:\n")
    fh.write(f"  Strong   : pattern_score >= {q66:.3f}  ({(pattern_df['tier']=='Strong').sum()} features)\n")
    fh.write(f"  Moderate : pattern_score >= {q33:.3f}  ({(pattern_df['tier']=='Moderate').sum()} features)\n")
    fh.write(f"  Weak     : pattern_score <  {q33:.3f}  ({(pattern_df['tier']=='Weak').sum()} features)\n\n")

    fh.write("Dimension breakdown (Strong tier):\n")
    strong = pattern_df[pattern_df["tier"] == "Strong"]
    for dim in ["c1", "c2", "c3", "c4", "c5"]:
        cnt = strong["feature"].str.startswith(dim + "_").sum()
        fh.write(f"  {dim.upper()} : {cnt}\n")

    fh.write(f"\nFull ranking:\n")
    fh.write(f"{'Rank':>4s}  {'Feature':45s}  {'Score':>6s}  {'Tier':>8s}  "
             f"{'Consist':>7s}  {'Sil_k2':>6s}  {'BC':>6s}  "
             f"{'Redund':>6s}  {'XDim':>4s}\n")
    fh.write("-" * 140 + "\n")
    for i, row in pattern_df.iterrows():
        fh.write(f"{i:4d}  {row['feature']:45s}  "
                 f"{row['pattern_score']:6.3f}  {row['tier']:>8s}  "
                 f"{row['consistency']:7.3f}  "
                 f"{row['silhouette_k2']:6.3f}  "
                 f"{row.get('bimodality_coeff', 0):6.3f}  "
                 f"{row['redundancy']:6.3f}  "
                 f"{int(row['cross_dim_links']):4d}\n")

print("  Saved pattern_summary.txt")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Plots
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 7  Generating plots")
print("=" * 72)

# 7a. Overall pattern score bar chart (top 40)
fig, ax = plt.subplots(figsize=(10, max(6, min(40, len(pattern_df)) * 0.3)))
top = pattern_df.head(40)
tier_colors = {"Strong": "#27ae60", "Moderate": "#f39c12", "Weak": "#e74c3c"}
colors = [tier_colors[t] for t in top["tier"]]
ax.barh(range(len(top)), top["pattern_score"], color=colors)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top["feature"], fontsize=6)
ax.set_xlabel("Pattern score (composite)")
ax.set_title("Top features by cross-student pattern strength\n"
             "(green=Strong, orange=Moderate, red=Weak)")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(RES, "pattern_score_ranking.png"), dpi=150)
plt.close()
print("  Saved pattern_score_ranking.png")

# 7b. Per-dimension distribution plots
for dim in ["c1", "c2", "c3", "c4", "c5"]:
    dim_feats = [c for c in feature_cols if c.startswith(dim + "_")]
    if not dim_feats:
        continue

    # sort by pattern score within dimension
    dim_scores = pattern_df[pattern_df["feature"].isin(dim_feats)] \
                    .sort_values("pattern_score", ascending=False)
    dim_feats_sorted = dim_scores["feature"].tolist()

    n_f = len(dim_feats_sorted)
    ncols = 4
    nrows = int(np.ceil(n_f / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.5))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for j, feat in enumerate(dim_feats_sorted):
        ax = axes[j]
        vals = merged[feat].dropna().values

        # histogram
        ax.hist(vals, bins=min(15, max(5, len(vals) // 3)),
                color="#3498db", alpha=0.6, edgecolor="#2c3e50", linewidth=0.5)

        # mark each student as a rug plot
        ax.plot(vals, np.zeros_like(vals) - 0.3, "|",
                color="#e74c3c", markersize=4, alpha=0.5)

        # stats annotation
        row = dim_scores[dim_scores["feature"] == feat].iloc[0]
        short = feat.replace(dim + "_", "")
        tier_tag = row["tier"][0]  # S/M/W
        ax.set_title(f"{short}\n"
                     f"score={row['pattern_score']:.2f} [{tier_tag}]  "
                     f"sil={row['silhouette_k2']:.2f}",
                     fontsize=6, pad=3)
        ax.tick_params(labelsize=5)

    # hide unused axes
    for j in range(n_f, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"{dim.upper()} — Distribution of each feature across {n} students",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"pattern_dist_{dim}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved pattern_dist_{dim}.png")

# 7c. Signal contribution heatmap for top-30
fig, ax = plt.subplots(figsize=(10, 8))
top30 = pattern_df.head(30)
sig_data = top30[sig_cols].values
ax.imshow(sig_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30["feature"], fontsize=6)
ax.set_xticks(range(len(sig_cols)))
ax.set_xticklabels([c.replace("sig_", "") for c in sig_cols],
                   fontsize=7, rotation=45, ha="right")
ax.set_title("Signal contribution heatmap (top 30 features)")
plt.colorbar(ax.images[0], ax=ax, shrink=0.6, label="Normalised signal [0–1]")
plt.tight_layout()
fig.savefig(os.path.join(RES, "pattern_signal_heatmap.png"), dpi=150)
plt.close()
print("  Saved pattern_signal_heatmap.png")

print("\n" + "=" * 72)
print("DONE — No exam scores were used in this analysis")
print("=" * 72)
