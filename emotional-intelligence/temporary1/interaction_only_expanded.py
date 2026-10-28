"""
interaction_only_expanded.py
============================
Same as interaction_only_model.py but uses TOP 25 features (votes >= 4,
missing < 30%) as the base pool instead of 13, generating a much richer
interaction space. NO individual features in final model.

Base pool: top 25 ranked features from consensus voting
Interactions: mult, ratio, quadratic, difference, triple (screened p<0.05)
Model: LOOCV Ridge α=1.0, forward selection, stops at Δr < 0.003
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr, spearmanr

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("INTERACTION-ONLY MODEL (EXPANDED BASE: TOP 25 FEATURES)")
print("=" * 72)

d1 = pd.read_csv(os.path.join(RES, "features_D1.csv"))
d2 = pd.read_csv(os.path.join(RES, "features_D2.csv"))
d3 = pd.read_csv(os.path.join(RES, "features_D3.csv"))
d4 = pd.read_csv(os.path.join(RES, "features_D4.csv"))
d5 = pd.read_csv(os.path.join(RES, "features_D5.csv"))

merged = d1.copy()
for df in [d2, d3, d4, d5]:
    new_cols = [c for c in df.columns if c not in merged.columns or c == "pid"]
    merged = merged.merge(df[new_cols], on="pid", how="outer")

if "pct" in merged.columns:
    exam_pct = merged["pct"].fillna(merged["pct"].median()).values
else:
    ei = pd.read_csv(os.path.join(RES, "ei_final_scores_v2.csv"))
    exam_pct = ei["exam_pct"].values

y   = exam_pct
n   = len(y)
print(f"\nLoaded {n} students")

# ═══════════════════════════════════════════════════════════════════════════
#  SELECT TOP 25 BASE FEATURES (votes>=4, missing<30%)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SELECTING EXPANDED BASE FEATURE POOL (TOP 25)")
print("=" * 72)

report = pd.read_csv(os.path.join(RES, "feature_variation_report.csv"))

# Filter: votes >= 4, missing < 30%, feature exists in merged
eligible = report[
    (report["total_votes"] >= 4) &
    (report["pct_missing"] < 30.0) &
    (report["feature"].isin(merged.columns))
].copy()

eligible = eligible.sort_values("composite_rank_score", ascending=False).head(25)
BASE_FEATURES = eligible["feature"].tolist()

print(f"\n  Eligible features (votes≥4, missing<30%): {len(eligible)}")
print(f"  Selected top 25 for interaction pool\n")
print(f"  {'Rank':>4} {'Feature':<40} {'Votes':>5} {'Missing%':>9} {'|r|':>8} {'Dim'}")
print(f"  {'-'*75}")

for _, row in eligible.iterrows():
    print(f"  {int(row['rank']):>4} {row['feature']:<40} {int(row['total_votes']):>5} "
          f"{row['pct_missing']:>9.1f} {row['abs_pearson']:>8.4f} {row['dimension']}")

# ═══════════════════════════════════════════════════════════════════════════
#  EXTRACT AND STANDARDIZE BASE FEATURES
# ═══════════════════════════════════════════════════════════════════════════
X_base_raw = merged[BASE_FEATURES].copy()
X_base_raw = X_base_raw.apply(lambda c: c.fillna(c.median()))

scaler = StandardScaler()
X_base = pd.DataFrame(scaler.fit_transform(X_base_raw), columns=BASE_FEATURES)

n_base = len(BASE_FEATURES)
print(f"\n  Base pool: {n_base} features from {eligible['dimension'].nunique()} dimensions")
print(f"  Dimension breakdown: {dict(eligible['dimension'].value_counts())}")

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def make_ratio(xa, xb):
    denom = xb.copy()
    near_zero = np.abs(denom) < 0.01
    denom[near_zero] = np.sign(denom[near_zero] + 1e-9) * 0.01
    ratio = xa / denom
    q01, q99 = np.percentile(ratio, [1, 99])
    ratio = np.clip(ratio, q01, q99)
    return (ratio - ratio.mean()) / (ratio.std() + 1e-10)

def standardize(s):
    return (s - s.mean()) / (s.std() + 1e-10)

# ═══════════════════════════════════════════════════════════════════════════
#  GENERATE ALL INTERACTIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("GENERATING INTERACTIONS")
print("=" * 72)

feat_names  = BASE_FEATURES
interactions = {}

# 1. Multiplicative
for f1, f2 in combinations(feat_names, 2):
    interactions[f"mult_{f1}_x_{f2}"] = standardize(X_base[f1] * X_base[f2])
n_mult = len(interactions)

# 2. Ratios (both directions)
before = len(interactions)
for f1, f2 in combinations(feat_names, 2):
    interactions[f"ratio_{f1}_over_{f2}"] = make_ratio(X_base[f1], X_base[f2])
    interactions[f"ratio_{f2}_over_{f1}"] = make_ratio(X_base[f2], X_base[f1])
n_ratio = len(interactions) - before

# 3. Quadratic
before = len(interactions)
for f in feat_names:
    interactions[f"quad_{f}"] = standardize(X_base[f] ** 2)
n_quad = len(interactions) - before

# 4. Differences
before = len(interactions)
for f1, f2 in combinations(feat_names, 2):
    interactions[f"diff_{f1}_minus_{f2}"] = standardize(X_base[f1] - X_base[f2])
n_diff = len(interactions) - before

# 5. Triple products: top-25 pairs (by |r| with target) × each base feature
before = len(interactions)
promising = []
for f1, f2 in combinations(feat_names, 2):
    r_val = abs(pearsonr(standardize(X_base[f1] * X_base[f2]), y)[0])
    if r_val > 0.20:
        promising.append((f1, f2, r_val))
promising.sort(key=lambda x: x[2], reverse=True)

for f1, f2, _ in promising[:25]:
    for f3 in feat_names:
        if f3 not in (f1, f2):
            interactions[f"triple_{f1}_x_{f2}_x_{f3}"] = standardize(
                X_base[f1] * X_base[f2] * X_base[f3])
n_triple = len(interactions) - before

print(f"\n  Multiplicative A×B:    {n_mult:>5}   C({n_base},2) = {n_mult}")
print(f"  Ratios A/B (2-way):    {n_ratio:>5}")
print(f"  Quadratic A²:          {n_quad:>5}")
print(f"  Differences A-B:       {n_diff:>5}")
print(f"  Triples A×B×C:         {n_triple:>5}   (top-25 pairs × {n_base-2} features)")
print(f"  ─────────────────────────────")
print(f"  TOTAL candidates:      {len(interactions):>5}")

# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN FOR SIGNIFICANCE (p < 0.05)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SCREENING INTERACTIONS (p < 0.05)")
print("=" * 72)

significant = {}
for name, vals in interactions.items():
    r, p = pearsonr(vals, y)
    if p < 0.05:
        significant[name] = (vals, r, p)

ranked = sorted(significant.items(), key=lambda x: abs(x[1][1]), reverse=True)
n_sig  = len(ranked)

# Type breakdown
type_counts = {}
for name in significant:
    t = name.split("_")[0]
    type_counts[t] = type_counts.get(t, 0) + 1

print(f"\n  Significant: {n_sig}/{len(interactions)} "
      f"({100*n_sig/len(interactions):.1f}%)")
print(f"\n  By type:")
for t, cnt in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * (cnt // 3)
    print(f"    {t:<8}: {cnt:>4}  {bar}")

print(f"\n  Top 20 significant interactions:")
print(f"  {'Rank':>4} {'r':>8} {'p':>10} {'Type':<8} {'Name':<55}")
print(f"  {'-'*90}")
for i, (name, (vals, r, p)) in enumerate(ranked[:20], 1):
    t = name.split("_")[0]
    print(f"  {i:>4} {r:>+8.4f} {p:>10.5f} {t:<8} {name[:55]}")

# ═══════════════════════════════════════════════════════════════════════════
#  FORWARD SELECTION (LOOCV, Ridge α=1)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FORWARD SELECTION: INTERACTIONS ONLY (LOOCV Ridge α=1.0)")
print("=" * 72)

X_candidates = pd.DataFrame({n: v for n, (v, r, p) in ranked})

loo        = LeaveOneOut()
alpha      = 1.0
max_steps  = 20
min_delta  = 0.003

selected   = []
selected_r = []
history    = []
prev_r     = 0.0

for step in range(max_steps):
    best_feat = None
    best_r    = -np.inf
    best_rho  = -np.inf

    for name in X_candidates.columns:
        if name in [s[0] for s in selected]:
            continue
        cols = [s[0] for s in selected] + [name]
        Xf   = X_candidates[cols]

        pred = np.zeros(n)
        for tr, te in loo.split(Xf):
            clf = Ridge(alpha=alpha)
            clf.fit(Xf.iloc[tr], y[tr])
            pred[te[0]] = clf.predict(Xf.iloc[te])[0]

        r   = pearsonr(pred, y)[0]
        rho = spearmanr(pred, y)[0]
        if r > best_r:
            best_feat, best_r, best_rho = name, r, rho

    if best_feat is None:
        break

    delta_r = best_r - prev_r
    itype   = best_feat.split("_")[0]

    selected.append((best_feat, best_r))
    selected_r.append(best_r)
    prev_r = best_r

    history.append({
        "step": step + 1, "feature": best_feat, "type": itype,
        "r_pearson": best_r, "r_spearman": best_rho, "delta_r": delta_r,
    })

    note = " ← DIMINISHING" if step > 4 and delta_r < min_delta else ""
    print(f"  Step {step+1:2d}: r={best_r:.4f} Δ={delta_r:+.4f} "
          f"[{itype:<6}] {best_feat[:52]}{note}")

    if step >= 5 and delta_r < min_delta:
        print(f"\n  → Early stop: Δr={delta_r:.4f} < {min_delta}")
        break

# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("RESULTS COMPARISON")
print("=" * 72)

baseline_r   = 0.8949
final_r      = selected_r[-1] if selected_r else 0.0
improvement  = final_r - baseline_r

type_in_model = {}
for feat, _ in selected:
    t = feat.split("_")[0]
    type_in_model[t] = type_in_model.get(t, 0) + 1

print(f"\n  {'Model':<35} {'Base Features':>14} {'Selected':>9} {'LOOCV r':>9} {'R²':>8}")
print(f"  {'-'*80}")
print(f"  {'Baseline (8 base + 7 interact)':<35} {'13':>14} {'15':>9} {baseline_r:>9.4f} {baseline_r**2:>8.4f}")
print(f"  {'Interaction-only (13 base)':<35} {'13':>14} {'N/A':>9} {'(see other script)':>9}")
print(f"  {'Interaction-only (25 base) ← THIS':<35} {'25':>14} {len(selected):>9} {final_r:>9.4f} {final_r**2:>8.4f}")
print(f"\n  Improvement over baseline: Δr={improvement:+.4f} "
      f"({100*improvement/baseline_r:+.2f}%)")
verdict = "✓ BETTER" if improvement > 0.005 else "~ MARGINAL" if improvement > 0 else "✗ WORSE"
print(f"  Verdict: {verdict}")

print(f"\n  Selected interaction breakdown:")
for t, cnt in sorted(type_in_model.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * cnt
    print(f"    {t:<8}: {cnt:>3}  {bar}")

# ═══════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════
pd.DataFrame(history).to_csv(
    os.path.join(RES, "interaction_expanded_history.csv"), index=False)

pd.DataFrame({
    "rank":      range(1, len(selected)+1),
    "feature":   [s[0] for s in selected],
    "type":      [s[0].split("_")[0] for s in selected],
    "r_pearson": [s[1] for s in selected],
}).to_csv(os.path.join(RES, "interaction_expanded_features.csv"), index=False)

pd.DataFrame([
    {"feature": n, "r": r, "p": p, "type": n.split("_")[0]}
    for n, (v, r, p) in ranked
]).to_csv(os.path.join(RES, "interaction_expanded_sig_pool.csv"), index=False)

print(f"\n  Saved: interaction_expanded_history.csv")
print(f"  Saved: interaction_expanded_features.csv  ({len(selected)} features)")
print(f"  Saved: interaction_expanded_sig_pool.csv  ({n_sig} candidates)")

# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

type_colors = {"mult": "#3498db", "ratio": "#e74c3c", "quad": "#2ecc71",
               "diff": "#f39c12", "triple": "#9b59b6"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Forward selection curve
ax = axes[0]
steps  = range(1, len(selected_r)+1)
pcols  = [type_colors.get(history[i]["type"], "#95a5a6") for i in range(len(selected_r))]
ax.plot(steps, selected_r, "o-", color="#2c3e50", lw=2, ms=8, zorder=2)
ax.scatter(steps, selected_r, c=pcols, s=130, edgecolors="k", lw=1, zorder=3)
ax.axhline(baseline_r, color="green", ls="--", lw=2)
ax.set_xlabel("Step", fontweight="bold")
ax.set_ylabel("LOOCV Pearson r", fontweight="bold")
ax.set_title("Forward Selection\n(Interactions Only, 25-base)", fontweight="bold")
ax.set_ylim([0.2, 0.95])
ax.grid(True, alpha=0.3)
ax.text(0.98, baseline_r + 0.003, f"Baseline r={baseline_r:.4f}",
        ha="right", va="bottom", color="green", fontsize=8,
        transform=ax.get_xaxis_transform())
from matplotlib.patches import Patch
patches = [Patch(color=c, label=t) for t, c in type_colors.items()]
ax.legend(handles=patches, fontsize=7, loc="lower right")

# 2. Significant pool breakdown
ax2 = axes[1]
sig_types = {}
for name in significant:
    t = name.split("_")[0]
    sig_types[t] = sig_types.get(t, 0) + 1
ax2.bar(sig_types.keys(),
        sig_types.values(),
        color=[type_colors.get(t, "#95a5a6") for t in sig_types],
        alpha=0.7, edgecolor="k", lw=1.5)
ax2.set_xlabel("Interaction Type", fontweight="bold")
ax2.set_ylabel("Count (p<0.05)", fontweight="bold")
ax2.set_title(f"Significant Pool: 13-base vs 25-base\n(25-base: {n_sig} significant)", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")
for i, (t, cnt) in enumerate(sig_types.items()):
    ax2.text(i, cnt + 1, str(cnt), ha="center", fontweight="bold", fontsize=9)

# 3. Model comparison
ax3 = axes[2]
mod_names = ["Baseline\n(13 base)", "Interact-only\n(25 base)"]
r_vals    = [baseline_r, final_r]
bcols     = ["#3498db", "#2ecc71" if final_r >= baseline_r else "#e74c3c"]
bars = ax3.bar(mod_names, r_vals, color=bcols, alpha=0.7, edgecolor="k", lw=1.5, width=0.4)
ax3.set_ylim([0.80, 0.95])
ax3.set_ylabel("LOOCV Pearson r", fontweight="bold")
ax3.set_title("Model Performance", fontweight="bold")
ax3.grid(True, alpha=0.3, axis="y")
for bar, r in zip(bars, r_vals):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.001,
             f"r={r:.4f}", ha="center", fontweight="bold", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(RES, "interaction_expanded_results.png"), dpi=150, bbox_inches="tight")
print("  Saved: interaction_expanded_results.png")

print("\n" + "=" * 72)
print("✓  INTERACTION-ONLY EXPANDED MODEL COMPLETE")
print("=" * 72)
