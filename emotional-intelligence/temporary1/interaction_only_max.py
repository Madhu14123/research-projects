"""
interaction_only_max.py
=======================
Maximum base pool: all features with votes >= 3 AND missing < 40%.
Typically yields 40-55 base features → 3,000-6,000 interaction candidates.
To keep forward selection tractable, top-500 by |r| enter greedy selection.

NO individual features in final model — interactions only.
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io, time
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
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
MIN_VOTES        = 3     # minimum consensus votes to include a base feature
MAX_MISSING_PCT  = 40.0  # exclude features with > this % missing
SIG_THRESHOLD    = 0.05  # p-value cutoff for screening interactions
TOP_K_CANDIDATES = 500   # cap on candidates entering forward selection
MAX_FS_STEPS     = 20    # max forward selection steps
MIN_DELTA_R      = 0.003 # early-stop threshold

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("INTERACTION-ONLY MODEL (MAXIMUM BASE POOL)")
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
    y = merged["pct"].fillna(merged["pct"].median()).values
else:
    ei = pd.read_csv(os.path.join(RES, "ei_final_scores_v2.csv"))
    y  = ei["exam_pct"].values

n = len(y)
print(f"\nLoaded {n} students")

# ═══════════════════════════════════════════════════════════════════════════
#  SELECT BASE FEATURE POOL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"SELECTING BASE POOL (votes >= {MIN_VOTES}, missing < {MAX_MISSING_PCT}%)")
print("=" * 72)

report = pd.read_csv(os.path.join(RES, "feature_variation_report.csv"))

eligible = report[
    (report["total_votes"] >= MIN_VOTES) &
    (report["pct_missing"]  < MAX_MISSING_PCT) &
    (report["feature"].isin(merged.columns))
].sort_values("composite_rank_score", ascending=False).copy()

BASE_FEATURES = eligible["feature"].tolist()
n_base        = len(BASE_FEATURES)

print(f"\n  Base pool: {n_base} features")
print(f"  Dimension breakdown: {dict(eligible['dimension'].value_counts())}")
print(f"\n  {'Rank':>4} {'Feature':<42} {'Votes':>5} {'Miss%':>6} {'|r|':>7} {'Dim'}")
print(f"  {'-'*72}")
for _, row in eligible.iterrows():
    print(f"  {int(row['rank']):>4} {row['feature']:<42} {int(row['total_votes']):>5} "
          f"{row['pct_missing']:>6.1f} {row['abs_pearson']:>7.4f} {row['dimension']}")

# Standardise
X_raw = merged[BASE_FEATURES].apply(lambda c: c.fillna(c.median()))
scaler = StandardScaler()
X_base = pd.DataFrame(scaler.fit_transform(X_raw), columns=BASE_FEATURES)

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def make_ratio(xa, xb):
    d = xb.copy()
    mask = np.abs(d) < 0.01
    d[mask] = np.sign(d[mask] + 1e-9) * 0.01
    r = xa / d
    lo, hi = np.percentile(r, [1, 99])
    r = np.clip(r, lo, hi)
    return (r - r.mean()) / (r.std() + 1e-10)

def std(s):
    return (s - s.mean()) / (s.std() + 1e-10)

# ═══════════════════════════════════════════════════════════════════════════
#  GENERATE INTERACTIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("GENERATING INTERACTIONS")
print("=" * 72)

t0 = time.time()
feats = BASE_FEATURES
interactions = {}

# 1. Multiplicative
for f1, f2 in combinations(feats, 2):
    interactions[f"mult_{f1}_x_{f2}"] = std(X_base[f1] * X_base[f2])
n_mult = len(interactions)

# 2. Ratios (both directions)
before = len(interactions)
for f1, f2 in combinations(feats, 2):
    interactions[f"ratio_{f1}_over_{f2}"] = make_ratio(X_base[f1], X_base[f2])
    interactions[f"ratio_{f2}_over_{f1}"] = make_ratio(X_base[f2], X_base[f1])
n_ratio = len(interactions) - before

# 3. Quadratic
before = len(interactions)
for f in feats:
    interactions[f"quad_{f}"] = std(X_base[f] ** 2)
n_quad = len(interactions) - before

# 4. Differences
before = len(interactions)
for f1, f2 in combinations(feats, 2):
    interactions[f"diff_{f1}_minus_{f2}"] = std(X_base[f1] - X_base[f2])
n_diff = len(interactions) - before

# 5. Triples: top-30 promising pairs × all remaining
before = len(interactions)
promising = sorted(
    [(f1, f2, abs(pearsonr(std(X_base[f1]*X_base[f2]), y)[0]))
     for f1, f2 in combinations(feats, 2)],
    key=lambda x: x[2], reverse=True
)
for f1, f2, _ in promising[:30]:
    for f3 in feats:
        if f3 not in (f1, f2):
            interactions[f"triple_{f1}_x_{f2}_x_{f3}"] = std(
                X_base[f1] * X_base[f2] * X_base[f3])
n_triple = len(interactions) - before

n_total = len(interactions)
print(f"\n  Multiplicative:  {n_mult:>5}   C({n_base},2)")
print(f"  Ratios (2-way):  {n_ratio:>5}")
print(f"  Quadratic:       {n_quad:>5}")
print(f"  Differences:     {n_diff:>5}")
print(f"  Triples:         {n_triple:>5}   (top-30 pairs × {n_base-2} feats)")
print(f"  ──────────────────────────────")
print(f"  TOTAL:           {n_total:>5}   [{time.time()-t0:.1f}s to generate]")

# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN FOR SIGNIFICANCE THEN TAKE TOP-K
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"SCREENING  (p < {SIG_THRESHOLD}, then top-{TOP_K_CANDIDATES} by |r|)")
print("=" * 72)

t1 = time.time()
sig = {}
for name, vals in interactions.items():
    r, p = pearsonr(vals, y)
    if p < SIG_THRESHOLD:
        sig[name] = (vals, r, p)

ranked_all = sorted(sig.items(), key=lambda x: abs(x[1][1]), reverse=True)
ranked     = ranked_all[:TOP_K_CANDIDATES]

type_counts = {}
for name, _ in ranked_all:
    t = name.split("_")[0]
    type_counts[t] = type_counts.get(t, 0) + 1

print(f"\n  Significant (p<{SIG_THRESHOLD}): {len(sig):>5} / {n_total}")
print(f"  Entering forward selection:  {len(ranked):>5}  (top-{TOP_K_CANDIDATES})")
print(f"  [{time.time()-t1:.1f}s to screen]")
print(f"\n  Significant by type:")
for t, cnt in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * (cnt // 5)
    print(f"    {t:<8}: {cnt:>5}  {bar}")

print(f"\n  Top 20 by |r|:")
print(f"  {'#':>3} {'r':>8} {'p':>10} {'Type':<8} {'Name':<55}")
print(f"  {'-'*88}")
for i, (name, (vals, r, p)) in enumerate(ranked[:20], 1):
    print(f"  {i:>3} {r:>+8.4f} {p:>10.5f} {name.split('_')[0]:<8} {name[:55]}")

# ═══════════════════════════════════════════════════════════════════════════
#  FORWARD SELECTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FORWARD SELECTION (LOOCV Ridge α=1.0)")
print("=" * 72)

X_cand = pd.DataFrame({name: vals for name, (vals, r, p) in ranked})
loo    = LeaveOneOut()

selected   = []
selected_r = []
history    = []
prev_r     = 0.0

t2 = time.time()
for step in range(MAX_FS_STEPS):
    best_feat = None
    best_r    = -np.inf
    best_rho  = -np.inf

    for name in X_cand.columns:
        if name in [s[0] for s in selected]:
            continue
        cols = [s[0] for s in selected] + [name]
        Xf   = X_cand[cols]

        pred = np.zeros(n)
        for tr, te in loo.split(Xf):
            clf = Ridge(alpha=1.0)
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
        "step": step+1, "feature": best_feat, "type": itype,
        "r_pearson": best_r, "r_spearman": best_rho, "delta_r": delta_r,
    })

    note = " ← DIMINISHING" if step > 4 and delta_r < MIN_DELTA_R else ""
    print(f"  Step {step+1:2d}: r={best_r:.4f} Δ={delta_r:+.4f} "
          f"[{itype:<6}] {best_feat[:50]}{note}")

    if step >= 5 and delta_r < MIN_DELTA_R:
        print(f"\n  → Early stop: Δr={delta_r:.4f} < {MIN_DELTA_R}")
        break

print(f"\n  Forward selection time: {time.time()-t2:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FINAL RESULTS")
print("=" * 72)

baseline_r  = 0.8949
final_r     = selected_r[-1] if selected_r else 0.0
improvement = final_r - baseline_r

type_in_model = {}
for feat, _ in selected:
    t = feat.split("_")[0]
    type_in_model[t] = type_in_model.get(t, 0) + 1

print(f"\n  {'Model':<38} {'Base':>5} {'Sel':>4} {'r':>8} {'R²':>8}")
print(f"  {'-'*68}")
print(f"  {'Baseline (8 base + 7 interact)':<38} {'13':>5} {'15':>4} "
      f"{baseline_r:>8.4f} {baseline_r**2:>8.4f}")
print(f"  {'Interact-only MAX ← THIS':<38} {n_base:>5} {len(selected):>4} "
      f"{final_r:>8.4f} {final_r**2:>8.4f}")

verdict = "✓ BETTER" if improvement > 0.005 else "~ MARGINAL" if improvement > 0 else "✗ WORSE"
print(f"\n  Δr = {improvement:+.4f} ({100*improvement/baseline_r:+.2f}%)  →  {verdict}")

print(f"\n  Selected features by type:")
for t, cnt in sorted(type_in_model.items(), key=lambda x: x[1], reverse=True):
    print(f"    {t:<8}: {cnt:>3}  {'█'*cnt}")

print(f"\n  All selected features:")
print(f"  {'#':>3} {'r':>8} {'Δr':>8} {'Type':<8} {'Feature'}")
print(f"  {'-'*80}")
for i, (h, (feat, r)) in enumerate(zip(history, selected), 1):
    print(f"  {i:>3} {r:>8.4f} {h['delta_r']:>+8.4f} {h['type']:<8} {feat[:55]}")

# ═══════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════
pd.DataFrame(history).to_csv(
    os.path.join(RES, "interaction_max_history.csv"), index=False)
pd.DataFrame({
    "rank":      range(1, len(selected)+1),
    "feature":   [s[0] for s in selected],
    "type":      [s[0].split("_")[0] for s in selected],
    "r_pearson": [s[1] for s in selected],
}).to_csv(os.path.join(RES, "interaction_max_features.csv"), index=False)
pd.DataFrame([
    {"feature": n, "r": r, "p": p, "type": n.split("_")[0]}
    for n, (v, r, p) in ranked_all
]).to_csv(os.path.join(RES, "interaction_max_sig_pool.csv"), index=False)

print(f"\n  Saved: interaction_max_history.csv")
print(f"  Saved: interaction_max_features.csv  ({len(selected)} features)")
print(f"  Saved: interaction_max_sig_pool.csv   ({len(sig)} interactions)")

# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

type_colors = {"mult":  "#3498db", "ratio": "#e74c3c",
               "quad":  "#2ecc71", "diff":  "#f39c12", "triple": "#9b59b6"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Interaction-Only Model (MAX base: {n_base} features → {n_total:,} candidates)",
             fontweight="bold", fontsize=12)

# 1. Forward selection curve
ax = axes[0]
steps = range(1, len(selected_r)+1)
pcols = [type_colors.get(history[i]["type"], "#95a5a6") for i in range(len(selected_r))]
ax.plot(steps, selected_r, "o-", color="#2c3e50", lw=2, ms=8, zorder=2)
ax.scatter(steps, selected_r, c=pcols, s=130, edgecolors="k", lw=1, zorder=3)
ax.axhline(baseline_r, color="green", ls="--", lw=2)
ax.set_xlabel("Step", fontweight="bold")
ax.set_ylabel("LOOCV Pearson r", fontweight="bold")
ax.set_title("Forward Selection Progress", fontweight="bold")
ax.set_ylim([0.2, 0.97])
ax.grid(True, alpha=0.3)
ax.text(len(steps)*0.98, baseline_r+0.005, f"Baseline\nr={baseline_r:.4f}",
        ha="right", color="green", fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=t) for t, c in type_colors.items()],
          fontsize=7, loc="lower right")

# 2. Three-way pool comparison (13 / 25 / MAX)
ax2 = axes[1]
labels  = [f"13-base\n(~78 mult)", f"25-base\n(~300 mult)", f"{n_base}-base\n(~{n_mult} mult)"]
n_sig_est = [80, 250, len(sig)]
bcols2  = ["#3498db", "#f39c12", "#e74c3c"]
bars2   = ax2.bar(labels, n_sig_est, color=bcols2, alpha=0.7, edgecolor="k", lw=1.5)
ax2.set_ylabel("Significant Interactions (p<0.05)", fontweight="bold")
ax2.set_title("Significant Pool Growth\nwith Larger Base", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")
for bar, cnt in zip(bars2, n_sig_est):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
             str(cnt), ha="center", fontweight="bold", fontsize=10)

# 3. Model comparison
ax3 = axes[2]
mod_labels = ["Baseline\n(13 base)", f"Interact-only\n({n_base}-base MAX)"]
r_vals     = [baseline_r, final_r]
bcols3     = ["#3498db", "#2ecc71" if final_r >= baseline_r else "#e74c3c"]
bars3      = ax3.bar(mod_labels, r_vals, color=bcols3, alpha=0.7, edgecolor="k",
                     lw=1.5, width=0.4)
ax3.set_ylim([0.78, 0.97])
ax3.set_ylabel("LOOCV Pearson r", fontweight="bold")
ax3.set_title("Model Performance", fontweight="bold")
ax3.grid(True, alpha=0.3, axis="y")
for bar, r in zip(bars3, r_vals):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             f"r={r:.4f}", ha="center", fontweight="bold", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(RES, "interaction_max_results.png"), dpi=150, bbox_inches="tight")
print("  Saved: interaction_max_results.png")

print(f"\n  Total runtime: {time.time()-t0:.1f}s")
print("\n" + "=" * 72)
print("✓  INTERACTION-ONLY MAX MODEL COMPLETE")
print("=" * 72)
