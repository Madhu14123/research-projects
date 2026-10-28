"""
interaction_only_model.py
=========================
Train model using ONLY derived interaction features (no individual base features).
Goal: Remove individual features, pack more interactions, capture non-linear effects.

Interaction types generated from 13 base supervised features:
  1. Multiplicative pairs   A * B          → C(13,2) = 78
  2. Ratios                 A / B          → 78 pairs × 2 directions = 156
  3. Quadratic              A²             → 13
  4. Difference             A - B          → 78
  5. Triple products        A * B * C      → top pairs × 13 = ~200 (screened)

No individual feature is included in the final model.
Forward selection stops at diminishing returns (Δr < 0.003).
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
#  LOAD RAW FEATURE DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("INTERACTION-ONLY FORWARD SELECTION MODEL")
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

# Get exam scores from dimension files (pct column)
if "pct" in merged.columns:
    exam_pct = merged["pct"].fillna(merged["pct"].median()).values
else:
    ei = pd.read_csv(os.path.join(RES, "ei_final_scores_v2.csv"))
    exam_pct = ei["exam_pct"].values

y = exam_pct
n = len(y)
print(f"\nLoaded {n} students, {len(merged.columns)-1} raw features")

# ═══════════════════════════════════════════════════════════════════════════
#  DEFINE 13 BASE SUPERVISED FEATURES
# ═══════════════════════════════════════════════════════════════════════════
BASE_FEATURES = [
    "c3_total_productive_time_ratio",
    "c4_block_stage_awareness",
    "c4_arousal_optimality",
    "c4_arousal_midrange_proportion",
    "c3_struggle_benefit_index",
    "c4_arousal_trend",
    "c5_flow_stability",
    "c3_task_completion_persistence",
    "c3_stress_inertia_during_blocks",
    "c2_anticipatory_regulation",
    "c3_effective_time_efficiency",
    "c2_cross_stage_regulation_var",
    "c2_SAM_facial_discrepancy_reduction",
]

# Verify features exist and extract
base_present = [f for f in BASE_FEATURES if f in merged.columns]
print(f"Base features found: {len(base_present)}/{len(BASE_FEATURES)}")

X_base_raw = merged[base_present].copy()
X_base_raw = X_base_raw.apply(lambda c: c.fillna(c.median()))

# Standardize base features
scaler = StandardScaler()
X_base = pd.DataFrame(
    scaler.fit_transform(X_base_raw),
    columns=base_present
)

# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: ROBUST RATIO CONSTRUCTION (preserves sign)
# ═══════════════════════════════════════════════════════════════════════════
def make_ratio(xa, xb):
    """Robust ratio preserving denominator sign, clipped at 1st/99th percentile."""
    denom = xb.copy()
    near_zero = np.abs(denom) < 0.01
    denom[near_zero] = np.sign(denom[near_zero] + 1e-9) * 0.01
    ratio = xa / denom
    q01, q99 = np.percentile(ratio, [1, 99])
    ratio = np.clip(ratio, q01, q99)
    return (ratio - ratio.mean()) / (ratio.std() + 1e-10)

def standardize(s):
    """Z-score a series."""
    return (s - s.mean()) / (s.std() + 1e-10)

# ═══════════════════════════════════════════════════════════════════════════
#  GENERATE ALL INTERACTION FEATURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("GENERATING ALL INTERACTION FEATURES")
print("=" * 72)

feat_names = base_present
n_base = len(feat_names)
interactions = {}

# ── 1. MULTIPLICATIVE PAIRS (A * B) ──────────────────────────────────────
for f1, f2 in combinations(feat_names, 2):
    name = f"mult_{f1}_x_{f2}"
    val = standardize(X_base[f1] * X_base[f2])
    interactions[name] = val

n_mult = len(interactions)
print(f"\n  Multiplicative pairs A×B:  {n_mult}")

# ── 2. RATIOS (A / B, both directions) ───────────────────────────────────
before = len(interactions)
for f1, f2 in combinations(feat_names, 2):
    name_ab = f"ratio_{f1}_over_{f2}"
    name_ba = f"ratio_{f2}_over_{f1}"
    interactions[name_ab] = make_ratio(X_base[f1], X_base[f2])
    interactions[name_ba] = make_ratio(X_base[f2], X_base[f1])

n_ratio = len(interactions) - before
print(f"  Ratios A/B (both dirs):    {n_ratio}")

# ── 3. QUADRATIC (A²) ────────────────────────────────────────────────────
before = len(interactions)
for f in feat_names:
    name = f"quad_{f}"
    interactions[name] = standardize(X_base[f] ** 2)

n_quad = len(interactions) - before
print(f"  Quadratic A²:              {n_quad}")

# ── 4. DIFFERENCES (A - B) ───────────────────────────────────────────────
before = len(interactions)
for f1, f2 in combinations(feat_names, 2):
    name = f"diff_{f1}_minus_{f2}"
    interactions[name] = standardize(X_base[f1] - X_base[f2])

n_diff = len(interactions) - before
print(f"  Differences A-B:           {n_diff}")

# ── 5. TRIPLE PRODUCTS (A * B * C) ───────────────────────────────────────
# Generate from each pair × each remaining base feature (screened by pair |r|>0.2)
before = len(interactions)

# First, find multiplicative pairs with |r| > 0.2 with target
promising_pairs = []
for f1, f2 in combinations(feat_names, 2):
    pair_vals = standardize(X_base[f1] * X_base[f2])
    r_pair = abs(pearsonr(pair_vals, y)[0])
    if r_pair > 0.20:
        promising_pairs.append((f1, f2, r_pair))

promising_pairs.sort(key=lambda x: x[2], reverse=True)

# Top-20 pairs × all remaining base features
for f1, f2, _ in promising_pairs[:20]:
    for f3 in feat_names:
        if f3 != f1 and f3 != f2:
            name = f"triple_{f1}_x_{f2}_x_{f3}"
            interactions[name] = standardize(X_base[f1] * X_base[f2] * X_base[f3])

n_triple = len(interactions) - before
print(f"  Triple products A×B×C:     {n_triple}")
print(f"\n  Total interactions generated: {len(interactions)}")

# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN INTERACTIONS FOR SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SCREENING INTERACTIONS (p < 0.05)")
print("=" * 72)

significant = {}
for name, vals in interactions.items():
    r, p = pearsonr(vals, y)
    if p < 0.05:
        significant[name] = (vals, r, p)

n_sig = len(significant)
print(f"\n  Significant interactions: {n_sig}/{len(interactions)}")

# Sort by |r|
ranked = sorted(significant.items(), key=lambda x: abs(x[1][1]), reverse=True)

print(f"\n  Top 30 by |r|:")
print(f"  {'Rank':>4} {'r':>8} {'p':>10} {'Type':<8} {'Name':<60}")
print(f"  {'-'*95}")

type_counts = {"mult": 0, "ratio": 0, "quad": 0, "diff": 0, "triple": 0}

for i, (name, (vals, r, p)) in enumerate(ranked[:30], 1):
    itype = name.split("_")[0]
    type_counts[itype] = type_counts.get(itype, 0) + 1
    print(f"  {i:>4} {r:>+8.4f} {p:>10.5f} {itype:<8} {name:<60}")

print(f"\n  Significant breakdown by type:")
for itype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
    total_type = sum(1 for name in significant if name.startswith(itype))
    print(f"    {itype:<8}: {total_type:>4} significant")

# ═══════════════════════════════════════════════════════════════════════════
#  FORWARD SELECTION ON INTERACTIONS ONLY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FORWARD SELECTION: INTERACTIONS ONLY (LOOCV Ridge α=1.0)")
print("=" * 72)

# Build candidate DataFrame (significant interactions only)
candidates = {name: vals for name, (vals, r, p) in ranked}
X_candidates = pd.DataFrame(candidates)

loo = LeaveOneOut()
alpha = 1.0
max_steps = 20
min_delta_r = 0.003   # Stop if improvement drops below this

selected = []
selected_r = []
history = []
prev_r = 0.0

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
            best_feat = name
            best_r    = r
            best_rho  = rho

    if best_feat is None:
        break

    delta_r = best_r - prev_r
    itype   = best_feat.split("_")[0]
    selected.append((best_feat, best_r))
    selected_r.append(best_r)
    prev_r = best_r

    history.append({
        "step":      step + 1,
        "feature":   best_feat,
        "type":      itype,
        "r_pearson": best_r,
        "r_spearman": best_rho,
        "delta_r":   delta_r,
    })

    flag = " ← DIMINISHING" if delta_r < min_delta_r and step > 0 else ""
    print(f"  Step {step+1:2d}: r={best_r:.4f} Δ={delta_r:+.4f} [{itype:<6}] {best_feat[:55]}{flag}")

    if step >= 5 and delta_r < min_delta_r:
        print(f"\n  → Stopped early: Δr={delta_r:.4f} < threshold ({min_delta_r})")
        break

# ═══════════════════════════════════════════════════════════════════════════
#  COMPARE WITH BASELINE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("RESULTS COMPARISON")
print("=" * 72)

baseline_r  = 0.8949
final_r     = selected_r[-1] if selected_r else 0.0
improvement = final_r - baseline_r

print(f"\n  ┌─────────────────────────────────────────────────────────┐")
print(f"  │  Model                   Features  LOOCV r   R²         │")
print(f"  ├─────────────────────────────────────────────────────────┤")
print(f"  │  Baseline (8 base + 7 interact)  15  {baseline_r:.4f}  {baseline_r**2:.4f}   │")
print(f"  │  Interaction-only              {len(selected):>3}  {final_r:.4f}  {final_r**2:.4f}   │")
print(f"  ├─────────────────────────────────────────────────────────┤")
print(f"  │  Δr = {improvement:+.4f}  ({100*improvement/baseline_r:+.2f}%){'  ✓ BETTER' if improvement>0.005 else '  ~ MARGINAL' if improvement>0 else '  ✗ WORSE':>15}         │")
print(f"  └─────────────────────────────────────────────────────────┘")

# Feature type breakdown in selected model
type_in_model = {}
for feat, _ in selected:
    t = feat.split("_")[0]
    type_in_model[t] = type_in_model.get(t, 0) + 1

print(f"\n  Selected features by interaction type:")
for t, cnt in sorted(type_in_model.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * cnt
    print(f"    {t:<8}: {cnt:>3}  {bar}")

# ═══════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SAVING RESULTS")
print("=" * 72)

hist_df = pd.DataFrame(history)
hist_df.to_csv(os.path.join(RES, "interaction_only_selection_history.csv"), index=False)

sel_df = pd.DataFrame({
    "rank":      range(1, len(selected) + 1),
    "feature":   [s[0] for s in selected],
    "type":      [s[0].split("_")[0] for s in selected],
    "r_pearson": [s[1] for s in selected],
})
sel_df.to_csv(os.path.join(RES, "interaction_only_final_features.csv"), index=False)

sig_df = pd.DataFrame([
    {"feature": name, "r": r, "p": p, "type": name.split("_")[0]}
    for name, (vals, r, p) in ranked
])
sig_df.to_csv(os.path.join(RES, "interaction_only_significant_pool.csv"), index=False)

print(f"\n  Saved: interaction_only_selection_history.csv")
print(f"  Saved: interaction_only_final_features.csv  ({len(selected)} features)")
print(f"  Saved: interaction_only_significant_pool.csv ({n_sig} candidates)")

# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Forward selection curve
ax = axes[0]
steps = range(1, len(selected_r) + 1)
type_colors = {"mult": "#3498db", "ratio": "#e74c3c", "quad": "#2ecc71",
               "diff": "#f39c12", "triple": "#9b59b6"}
point_colors = [type_colors.get(history[i]["type"], "#95a5a6") for i in range(len(selected_r))]

ax.plot(steps, selected_r, "o-", color="#2c3e50", lw=2, ms=8, zorder=2)
ax.scatter(steps, selected_r, c=point_colors, s=120, edgecolors="k", lw=1, zorder=3)
ax.axhline(baseline_r, color="green", ls="--", lw=2, label=f"Baseline r={baseline_r:.4f}")
ax.set_xlabel("Selection Step", fontweight="bold", fontsize=11)
ax.set_ylabel("LOOCV Pearson r", fontweight="bold", fontsize=11)
ax.set_title("Forward Selection (Interactions Only)", fontweight="bold", fontsize=12)
ax.set_ylim([0.0, 0.95])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Legend for types
from matplotlib.patches import Patch
patches = [Patch(color=c, label=t) for t, c in type_colors.items()]
ax.legend(handles=patches + [plt.Line2D([0], [0], color="green", ls="--", label=f"Baseline r={baseline_r:.4f}")],
          fontsize=8, loc="lower right")

# 2. Interaction type distribution in significant pool
ax2 = axes[1]
sig_types = {}
for name in significant:
    t = name.split("_")[0]
    sig_types[t] = sig_types.get(t, 0) + 1
colors_bar = [type_colors.get(t, "#95a5a6") for t in sig_types.keys()]
bars = ax2.bar(sig_types.keys(), sig_types.values(), color=colors_bar, alpha=0.7, edgecolor="k", lw=1.5)
ax2.set_xlabel("Interaction Type", fontweight="bold", fontsize=11)
ax2.set_ylabel("Significant Count (p<0.05)", fontweight="bold", fontsize=11)
ax2.set_title(f"Significant Interactions Pool (n={n_sig})", fontweight="bold", fontsize=12)
ax2.grid(True, alpha=0.3, axis="y")
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(int(bar.get_height())), ha="center", fontweight="bold", fontsize=10)

# 3. Model comparison bar chart
ax3 = axes[2]
models = ["Baseline\n(8 base +\n7 interact)", "Interaction\nOnly"]
r_vals = [baseline_r, final_r]
bar_colors = ["#3498db", "#e74c3c" if final_r < baseline_r else "#2ecc71"]

bars3 = ax3.bar(models, r_vals, color=bar_colors, alpha=0.7, edgecolor="k", lw=1.5, width=0.4)
ax3.set_ylim([0.8, 0.95])
ax3.set_ylabel("LOOCV Pearson r", fontweight="bold", fontsize=11)
ax3.set_title("Model Comparison", fontweight="bold", fontsize=12)
ax3.grid(True, alpha=0.3, axis="y")
for bar, r in zip(bars3, r_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"r={r:.4f}", ha="center", fontweight="bold", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(RES, "interaction_only_results.png"), dpi=150, bbox_inches="tight")
print("  Saved: interaction_only_results.png")

print("\n" + "=" * 72)
print("✓  INTERACTION-ONLY MODEL COMPLETE")
print("=" * 72)
