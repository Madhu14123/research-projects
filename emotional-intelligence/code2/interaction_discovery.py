"""
interaction_discovery.py
========================
Exhaustive interaction discovery for 13 supervised high-consensus features.

1. PAIRWISE: A*B multiplicative interactions (78 pairs)
2. RATIOS: A/B ratio features (156 directed pairs)
3. QUADRATIC: A^2 self-interactions (13 features)
4. CROSS-DIMENSION: dimension_composite_X * dimension_composite_Y
5. TRIPLE interactions: A*B*C (top combos only, computationally bounded)
6. INCREMENTAL VALUE: does adding each interaction to the base 13 improve LOOCV?
7. FORWARD SELECTION: build minimal optimal feature set from base + interactions
8. FINAL VERDICT: how many features, which ones, and is ensemble needed?
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
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from itertools import combinations

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")
TARGET = "pct"

DIMENSIONS = {
    "D2_Regulation": [
        "c2_cross_stage_regulation_var",
        "c2_anticipatory_regulation",
    ],
    "D3_Utilisation": [
        "c3_total_productive_time_ratio",
        "c3_struggle_benefit_index",
        "c3_effective_time_efficiency",
        "c3_recovery_acceleration",
        "c3_task_completion_persistence",
    ],
    "D4_Self_Awareness": [
        "c4_arousal_trend",
        "c4_block_stage_awareness",
        "c4_arousal_optimality",
        "c4_arousal_midrange_proportion",
    ],
    "D5_Context_Application": [
        "c5_flow_stability",
        "c5_engagement_recovery_speed",
    ],
}

ALL_FEATURES = []
FEAT_TO_DIM = {}
for dim, feats in DIMENSIONS.items():
    ALL_FEATURES.extend(feats)
    for f in feats:
        FEAT_TO_DIM[f] = dim

# ── LOAD DATA ────────────────────────────────────────────────────────────
print("=" * 72)
print("LOADING DATA")
print("=" * 72)

FILES = {
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

available = [f for f in ALL_FEATURES if f in merged.columns]
X_raw = merged.loc[valid, available].reset_index(drop=True)
X_raw = X_raw.apply(lambda c: c.fillna(c.median()))

scaler = StandardScaler()
X_z = pd.DataFrame(scaler.fit_transform(X_raw), columns=available)

n = len(y)
print(f"  Students: {n}, Base features: {len(available)}")


def loocv_r(X, y):
    """LOOCV Pearson r using Ridge regression."""
    loo = LeaveOneOut()
    preds = np.full(len(y), np.nan)
    # auto-tune alpha
    model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
    for tr, te in loo.split(X):
        m = clone(model)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    r, p = pearsonr(y, preds)
    r2 = r2_score(y, preds)
    return r, p, r2, preds


def marginal_r(feat_vals, y):
    """Simple Pearson r of a single feature with target."""
    r, p = pearsonr(feat_vals, y)
    return r, p


# ═══════════════════════════════════════════════════════════════════════════
#  BASELINE: raw 13 features only
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("BASELINE: 13 raw features, Ridge LOOCV")
print("=" * 72)

base_r, base_p, base_r2, base_preds = loocv_r(X_z.values, y)
print(f"  LOOCV:  r={base_r:.4f}, p={base_p:.6f}, R2={base_r2:.4f}")

# individual feature correlations
print(f"\n  Individual feature correlations with exam:")
print(f"  {'Feature':<40} {'r':>8} {'|r|':>8} {'p':>10}")
print(f"  {'-'*70}")
feat_corrs = {}
for feat in available:
    r, p = pearsonr(X_z[feat].values, y)
    feat_corrs[feat] = (r, p)
    print(f"  {feat:<40} {r:+8.4f} {abs(r):8.4f} {p:10.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1: MULTIPLICATIVE INTERACTIONS (A * B)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 1: MULTIPLICATIVE INTERACTIONS (A * B)")
print("=" * 72)

mult_results = []
for i, j in combinations(range(len(available)), 2):
    fa, fb = available[i], available[j]
    xa, xb = X_z[fa].values, X_z[fb].values
    inter = xa * xb

    r, p = pearsonr(inter, y)
    dim_a, dim_b = FEAT_TO_DIM[fa], FEAT_TO_DIM[fb]
    cross_dim = dim_a != dim_b

    mult_results.append({
        "type": "multiply",
        "name": f"{fa} * {fb}",
        "feat_a": fa, "feat_b": fb,
        "dim_a": dim_a, "dim_b": dim_b,
        "cross_dimension": cross_dim,
        "r": r, "abs_r": abs(r), "p": p,
        "values": inter,
    })

mult_df = pd.DataFrame([{k: v for k, v in row.items() if k != "values"}
                         for row in mult_results])
mult_df = mult_df.sort_values("abs_r", ascending=False).reset_index(drop=True)

print(f"\n  Total multiplicative pairs: {len(mult_df)}")
print(f"  Cross-dimension pairs: {mult_df['cross_dimension'].sum()}")
print(f"  Within-dimension pairs: {(~mult_df['cross_dimension']).sum()}")

print(f"\n  Top 20 multiplicative interactions:")
print(f"  {'Name':<65} {'r':>8} {'p':>10} {'Cross':>6}")
print(f"  {'-'*92}")
for _, row in mult_df.head(20).iterrows():
    print(f"  {row['name']:<65} {row['r']:+8.4f} {row['p']:10.6f} "
          f"{'YES' if row['cross_dimension'] else 'no':>6}")

sig_mult = mult_df[mult_df["p"] < 0.05]
print(f"\n  Significant (p<0.05): {len(sig_mult)}/{len(mult_df)}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2: RATIO FEATURES (A / B)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 2: RATIO FEATURES (A / B)")
print("=" * 72)

ratio_results = []
for i in range(len(available)):
    for j in range(len(available)):
        if i == j:
            continue
        fa, fb = available[i], available[j]
        xa, xb = X_z[fa].values, X_z[fb].values

        # avoid division by near-zero: shift denominator
        denom = xb.copy()
        denom[np.abs(denom) < 0.01] = np.sign(denom[np.abs(denom) < 0.01] + 0.001) * 0.01
        ratio = xa / denom

        # clip extreme ratios
        q01, q99 = np.percentile(ratio, [1, 99])
        ratio = np.clip(ratio, q01, q99)

        if np.std(ratio) < 1e-10 or np.any(np.isinf(ratio)) or np.any(np.isnan(ratio)):
            continue

        r, p = pearsonr(ratio, y)
        dim_a, dim_b = FEAT_TO_DIM[fa], FEAT_TO_DIM[fb]

        ratio_results.append({
            "type": "ratio",
            "name": f"{fa} / {fb}",
            "feat_a": fa, "feat_b": fb,
            "dim_a": dim_a, "dim_b": dim_b,
            "cross_dimension": dim_a != dim_b,
            "r": r, "abs_r": abs(r), "p": p,
            "values": ratio,
        })

ratio_df = pd.DataFrame([{k: v for k, v in row.items() if k != "values"}
                          for row in ratio_results])
ratio_df = ratio_df.sort_values("abs_r", ascending=False).reset_index(drop=True)

print(f"\n  Total ratio features: {len(ratio_df)}")
print(f"\n  Top 20 ratio features:")
print(f"  {'Name':<65} {'r':>8} {'p':>10} {'Cross':>6}")
print(f"  {'-'*92}")
for _, row in ratio_df.head(20).iterrows():
    print(f"  {row['name']:<65} {row['r']:+8.4f} {row['p']:10.6f} "
          f"{'YES' if row['cross_dimension'] else 'no':>6}")

sig_ratio = ratio_df[ratio_df["p"] < 0.05]
print(f"\n  Significant (p<0.05): {len(sig_ratio)}/{len(ratio_df)}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3: QUADRATIC / SQUARED FEATURES (A^2)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 3: QUADRATIC SELF-INTERACTIONS (A^2)")
print("=" * 72)

quad_results = []
print(f"\n  {'Feature':<40} {'r(A)':>8} {'r(A^2)':>8} {'p(A^2)':>10} {'type':>8}")
print(f"  {'-'*78}")

for feat in available:
    x = X_z[feat].values
    x2 = x ** 2
    r_lin, _ = pearsonr(x, y)
    r_quad, p_quad = pearsonr(x2, y)

    # is the quadratic capturing something different from linear?
    r_lin_quad, _ = pearsonr(x, x2)  # correlation between x and x^2
    unique_info = abs(r_quad) > 0.1 and abs(r_lin_quad) < 0.7

    quad_type = "U-shape" if r_quad > 0 and abs(r_lin) < 0.15 else \
                "invU" if r_quad < 0 and abs(r_lin) < 0.15 else \
                "accelerating" if np.sign(r_lin) == np.sign(r_quad) else \
                "decelerating"

    quad_results.append({
        "type": "quadratic",
        "name": f"{feat}^2",
        "feat_a": feat, "feat_b": feat,
        "dim_a": FEAT_TO_DIM[feat], "dim_b": FEAT_TO_DIM[feat],
        "cross_dimension": False,
        "r": r_quad, "abs_r": abs(r_quad), "p": p_quad,
        "r_linear": r_lin,
        "unique_info": unique_info,
        "quad_type": quad_type,
    })
    flag = " <-- unique" if unique_info else ""
    print(f"  {feat:<40} {r_lin:+8.4f} {r_quad:+8.4f} {p_quad:10.6f} "
          f"{quad_type:>8}{flag}")

quad_df = pd.DataFrame(quad_results)
sig_quad = quad_df[quad_df["p"] < 0.05]
unique_quad = quad_df[quad_df["unique_info"]]
print(f"\n  Significant (p<0.05): {len(sig_quad)}/{len(quad_df)}")
print(f"  With unique information: {len(unique_quad)}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4: DIMENSION COMPOSITE INTERACTIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 4: DIMENSION COMPOSITE INTERACTIONS")
print("=" * 72)

# build dimension composites (sign-corrected mean)
dim_composites = {}
for dim_name, feats in DIMENSIONS.items():
    present = [f for f in feats if f in available]
    signs = np.array([np.sign(feat_corrs[f][0]) for f in present])
    dim_data = X_z[present].values * signs
    dim_composites[dim_name] = np.mean(dim_data, axis=1)

    r, p = pearsonr(dim_composites[dim_name], y)
    print(f"  {dim_name} composite: r={r:+.4f}, p={p:.6f}")

# cross-dimension interactions
print(f"\n  Cross-dimension composite interactions:")
print(f"  {'Interaction':<45} {'r':>8} {'p':>10}")
print(f"  {'-'*66}")

dim_names = list(dim_composites.keys())
dim_inter_results = []
for i, j in combinations(range(len(dim_names)), 2):
    da, db = dim_names[i], dim_names[j]
    inter = dim_composites[da] * dim_composites[db]
    r, p = pearsonr(inter, y)
    dim_inter_results.append({
        "name": f"{da} * {db}", "r": r, "abs_r": abs(r), "p": p,
        "values": inter,
    })
    print(f"  {da} * {db:<20} {r:+8.4f} {p:10.6f}")

# also try sum, difference, max, min of composites
print(f"\n  Dimension composite combinations:")
for i, j in combinations(range(len(dim_names)), 2):
    da, db = dim_names[i], dim_names[j]
    ca, cb = dim_composites[da], dim_composites[db]

    # difference
    diff = ca - cb
    r_diff, p_diff = pearsonr(diff, y)
    # max
    mx = np.maximum(ca, cb)
    r_mx, p_mx = pearsonr(mx, y)
    # min (bottleneck)
    mn = np.minimum(ca, cb)
    r_mn, p_mn = pearsonr(mn, y)

    if max(abs(r_diff), abs(r_mx), abs(r_mn)) > 0.3:
        print(f"  {da} - {db}: r={r_diff:+.4f}")
        print(f"  max({da}, {db}): r={r_mx:+.4f}")
        print(f"  min({da}, {db}): r={r_mn:+.4f} (bottleneck)")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 5: TRIPLE INTERACTIONS (top combos)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 5: TRIPLE INTERACTIONS (A * B * C)")
print("=" * 72)

# Only test triples where at least 2 features come from different dimensions
triple_results = []
for i, j, k in combinations(range(len(available)), 3):
    fa, fb, fc = available[i], available[j], available[k]
    dims = {FEAT_TO_DIM[fa], FEAT_TO_DIM[fb], FEAT_TO_DIM[fc]}
    if len(dims) < 2:
        continue  # skip all-same-dimension triples

    inter = X_z[fa].values * X_z[fb].values * X_z[fc].values
    r, p = pearsonr(inter, y)
    triple_results.append({
        "type": "triple",
        "name": f"{fa} * {fb} * {fc}",
        "r": r, "abs_r": abs(r), "p": p,
        "n_dims": len(dims),
        "values": inter,
    })

triple_df = pd.DataFrame([{k: v for k, v in row.items() if k != "values"}
                           for row in triple_results])
triple_df = triple_df.sort_values("abs_r", ascending=False).reset_index(drop=True)

print(f"\n  Total cross-dimension triples tested: {len(triple_df)}")
sig_triple = triple_df[triple_df["p"] < 0.05]
print(f"  Significant (p<0.05): {len(sig_triple)}")

print(f"\n  Top 15 triple interactions:")
print(f"  {'Name':<80} {'r':>8} {'p':>10} {'dims':>5}")
print(f"  {'-'*106}")
for _, row in triple_df.head(15).iterrows():
    print(f"  {row['name']:<80} {row['r']:+8.4f} {row['p']:10.6f} {row['n_dims']:5d}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 6: MASTER CANDIDATE POOL -- rank ALL derived features
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 6: MASTER CANDIDATE POOL (all derived features ranked)")
print("=" * 72)

# Combine all candidate derived features
all_candidates = []

# original features
for feat in available:
    r, p = feat_corrs[feat]
    all_candidates.append({
        "name": feat, "type": "original", "r": r, "abs_r": abs(r), "p": p,
        "values": X_z[feat].values,
    })

# multiplicative
for row in mult_results:
    all_candidates.append({
        "name": row["name"], "type": "multiply",
        "r": row["r"], "abs_r": row["abs_r"], "p": row["p"],
        "values": row["values"],
    })

# ratios (top 30 only to keep manageable)
ratio_sorted = sorted(ratio_results, key=lambda x: x["abs_r"], reverse=True)
for row in ratio_sorted[:30]:
    all_candidates.append({
        "name": row["name"], "type": "ratio",
        "r": row["r"], "abs_r": row["abs_r"], "p": row["p"],
        "values": row["values"],
    })

# quadratic (only unique ones)
for feat in available:
    x = X_z[feat].values
    x2 = x ** 2
    r, p = pearsonr(x2, y)
    r_lin_quad, _ = pearsonr(x, x2)
    if abs(r_lin_quad) < 0.7:  # only if adds unique info
        all_candidates.append({
            "name": f"{feat}^2", "type": "quadratic",
            "r": r, "abs_r": abs(r), "p": p,
            "values": x2,
        })

# dimension composite interactions
for row in dim_inter_results:
    all_candidates.append({
        "name": row["name"], "type": "dim_interact",
        "r": row["r"], "abs_r": row["abs_r"], "p": row["p"],
        "values": row["values"],
    })

# triple interactions (top 20)
for row in triple_results[:20]:
    if row["abs_r"] > 0.3:
        all_candidates.append({
            "name": row["name"], "type": "triple",
            "r": row["r"], "abs_r": row["abs_r"], "p": row["p"],
            "values": row["values"],
        })

# sort by |r|
all_candidates.sort(key=lambda x: x["abs_r"], reverse=True)

print(f"\n  Total candidate features: {len(all_candidates)}")
print(f"    Original: {sum(1 for c in all_candidates if c['type']=='original')}")
print(f"    Multiplicative: {sum(1 for c in all_candidates if c['type']=='multiply')}")
print(f"    Ratios: {sum(1 for c in all_candidates if c['type']=='ratio')}")
print(f"    Quadratic: {sum(1 for c in all_candidates if c['type']=='quadratic')}")
print(f"    Dim interactions: {sum(1 for c in all_candidates if c['type']=='dim_interact')}")
print(f"    Triples: {sum(1 for c in all_candidates if c['type']=='triple')}")

print(f"\n  Top 40 candidates by |r| with exam:")
print(f"  {'Rank':>4} {'Name':<70} {'Type':<12} {'r':>8} {'p':>10}")
print(f"  {'-'*108}")
for idx, cand in enumerate(all_candidates[:40]):
    print(f"  {idx+1:4d} {cand['name']:<70} {cand['type']:<12} "
          f"{cand['r']:+8.4f} {cand['p']:10.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 7: FORWARD SELECTION with LOOCV
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 7: FORWARD FEATURE SELECTION (LOOCV Ridge)")
print("=" * 72)
print("  Building optimal feature set incrementally...")
print("  Starting from best single feature, greedily adding the one that")
print("  improves LOOCV r the most, until no improvement.\n")

# use top 60 candidates for tractability
pool = all_candidates[:60]
pool_names = [c["name"] for c in pool]
pool_values = {c["name"]: c["values"] for c in pool}

selected_feats = []
selected_history = []
remaining = set(range(len(pool)))
best_overall_r = -999

for step in range(min(25, len(pool))):  # max 25 features
    best_r_this_step = -999
    best_idx = -1

    for idx in remaining:
        candidate_set = selected_feats + [pool_names[idx]]
        X_trial = np.column_stack([pool_values[name] for name in candidate_set])

        # standardize the trial matrix
        X_trial_std = (X_trial - X_trial.mean(axis=0)) / (X_trial.std(axis=0) + 1e-10)

        r, p, r2, _ = loocv_r(X_trial_std, y)
        if r > best_r_this_step:
            best_r_this_step = r
            best_idx = idx
            best_p = p
            best_r2 = r2

    # check improvement
    improvement = best_r_this_step - best_overall_r
    if improvement < 0.001 and step > 2:
        print(f"  Step {step+1}: best candidate gain = {improvement:+.4f} < 0.001 -- STOPPING")
        break

    selected_feats.append(pool_names[best_idx])
    remaining.discard(best_idx)
    best_overall_r = best_r_this_step

    feat_type = pool[best_idx]["type"]
    marginal_r_val = pool[best_idx]["r"]

    selected_history.append({
        "step": step + 1,
        "added_feature": pool_names[best_idx],
        "type": feat_type,
        "marginal_r": marginal_r_val,
        "loocv_r": best_r_this_step,
        "loocv_p": best_p,
        "loocv_r2": best_r2,
        "improvement": improvement if step > 0 else best_r_this_step,
    })

    print(f"  Step {step+1:2d}: + {pool_names[best_idx]:<60} "
          f"({feat_type:<10}) LOOCV r={best_r_this_step:.4f} "
          f"(+{improvement:.4f})" if step > 0 else
          f"  Step {step+1:2d}: + {pool_names[best_idx]:<60} "
          f"({feat_type:<10}) LOOCV r={best_r_this_step:.4f}")

history_df = pd.DataFrame(selected_history)

# Find the "elbow" -- where adding more features gives diminishing returns
print(f"\n  Forward selection summary:")
print(f"  {'Step':>4} {'Feature':<60} {'Type':<12} {'LOOCV_r':>8} {'Gain':>8}")
print(f"  {'-'*96}")
for _, row in history_df.iterrows():
    imp = row["improvement"]
    flag = " ***" if imp > 0.02 else " *" if imp > 0.005 else ""
    print(f"  {row['step']:4.0f} {row['added_feature']:<60} {row['type']:<12} "
          f"{row['loocv_r']:8.4f} {imp:+8.4f}{flag}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 8: COMPARE MODEL COMPLEXITIES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 8: MODEL COMPLEXITY COMPARISON")
print("=" * 72)

# A) Just raw 13 features
r_raw, p_raw, r2_raw, _ = loocv_r(X_z.values, y)
print(f"\n  A) Raw 13 features:           LOOCV r={r_raw:.4f}, R2={r2_raw:.4f}")

# B) Optimal forward-selected set
opt_names = selected_feats
X_opt = np.column_stack([pool_values[name] for name in opt_names])
X_opt_std = (X_opt - X_opt.mean(axis=0)) / (X_opt.std(axis=0) + 1e-10)
r_opt, p_opt, r2_opt, preds_opt = loocv_r(X_opt_std, y)
print(f"  B) Optimal {len(opt_names)} features:       LOOCV r={r_opt:.4f}, R2={r2_opt:.4f}")

# C) Just dimension composites (4 features)
X_dim = np.column_stack([dim_composites[d] for d in dim_names])
r_dim, p_dim, r2_dim, _ = loocv_r(X_dim, y)
print(f"  C) 4 dimension composites:    LOOCV r={r_dim:.4f}, R2={r2_dim:.4f}")

# D) Dimension composites + their interactions (4 + 6 = 10)
dim_inter_vals = []
for i, j in combinations(range(len(dim_names)), 2):
    dim_inter_vals.append(dim_composites[dim_names[i]] * dim_composites[dim_names[j]])
X_dim_inter = np.column_stack([X_dim] + dim_inter_vals)
r_dimi, p_dimi, r2_dimi, _ = loocv_r(X_dim_inter, y)
print(f"  D) Dim composites + interactions: LOOCV r={r_dimi:.4f}, R2={r2_dimi:.4f}")

# E) Simple composite EI (sign-corrected mean)
signs = np.array([np.sign(feat_corrs[f][0]) for f in available])
composite = np.mean(X_z.values * signs, axis=1).reshape(-1, 1)
r_comp, p_comp = pearsonr(composite.flatten(), y)
print(f"  E) Simple composite (mean):   Pearson r={r_comp:.4f} (no LOOCV needed, 1 feature)")

# F) Top-k cut of forward selection (find elbow)
print(f"\n  Forward selection at different cut points:")
for k in [3, 5, 7, 10, len(opt_names)]:
    k_actual = min(k, len(opt_names))
    feat_k = opt_names[:k_actual]
    X_k = np.column_stack([pool_values[name] for name in feat_k])
    X_k_std = (X_k - X_k.mean(axis=0)) / (X_k.std(axis=0) + 1e-10)
    r_k, p_k, r2_k, _ = loocv_r(X_k_std, y)
    print(f"    Top-{k_actual} features: LOOCV r={r_k:.4f}, R2={r2_k:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 9: DO WE NEED ENSEMBLE? Compare Ridge vs Tree vs Ensemble
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 9: IS ENSEMBLE NEEDED? (Ridge vs GBR vs RF on optimal features)")
print("=" * 72)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb

test_models = {
    "Ridge(a=1)":    Ridge(alpha=1.0),
    "Ridge(a=10)":   Ridge(alpha=10.0),
    "ElasticNet":    ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    "RF":            RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    "GBR":           GradientBoostingRegressor(n_estimators=80, max_depth=2,
                                               learning_rate=0.1, random_state=42),
    "XGBoost":       xgb.XGBRegressor(n_estimators=80, max_depth=2, learning_rate=0.1,
                                      verbosity=0, random_state=42),
    "LightGBM":      lgb.LGBMRegressor(n_estimators=80, max_depth=2, learning_rate=0.1,
                                       verbose=-1, random_state=42),
    "SVR_rbf":       SVR(kernel="rbf", C=1.0),
    "KNN5":          KNeighborsRegressor(n_neighbors=5),
}

print(f"\n  Testing on optimal {len(opt_names)} features:")
print(f"  {'Model':<18} {'LOOCV_r':>8} {'LOOCV_R2':>9} {'p':>10}")
print(f"  {'-'*48}")

model_comparison = []
for mname, model in test_models.items():
    loo = LeaveOneOut()
    preds = np.full(n, np.nan)
    for tr, te in loo.split(X_opt_std):
        m = clone(model)
        m.fit(X_opt_std[tr], y[tr])
        preds[te] = m.predict(X_opt_std[te])
    r, p = pearsonr(y, preds)
    r2 = r2_score(y, preds)
    model_comparison.append({"model": mname, "r": r, "r2": r2, "p": p})
    print(f"  {mname:<18} {r:8.4f} {r2:9.4f} {p:10.6f}")

# also test on raw 13 features
print(f"\n  Testing on raw 13 features (for comparison):")
print(f"  {'Model':<18} {'LOOCV_r':>8} {'LOOCV_R2':>9} {'p':>10}")
print(f"  {'-'*48}")

for mname, model in test_models.items():
    loo = LeaveOneOut()
    preds = np.full(n, np.nan)
    for tr, te in loo.split(X_z.values):
        m = clone(model)
        m.fit(X_z.values[tr], y[tr])
        preds[te] = m.predict(X_z.values[te])
    r, p = pearsonr(y, preds)
    r2 = r2_score(y, preds)
    print(f"  {mname:<18} {r:8.4f} {r2:9.4f} {p:10.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 10: REDUNDANCY CHECK -- are any base features redundant?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 10: FEATURE REDUNDANCY CHECK")
print("=" * 72)

print(f"\n  Pairwise correlation matrix of 13 base features:")
corr_matrix = X_z.corr()
high_corr = []
for i in range(len(available)):
    for j in range(i+1, len(available)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.5:
            high_corr.append((available[i], available[j], r))
            print(f"    {available[i]} <-> {available[j]}: r={r:+.3f} {'!!! HIGH' if abs(r) > 0.7 else ''}")

if not high_corr:
    print(f"    No pairs with |r| > 0.5")

# drop-one analysis: which features can be removed with minimal LOOCV loss?
print(f"\n  Drop-one analysis (LOOCV r when each feature is removed):")
print(f"  {'Dropped Feature':<40} {'LOOCV_r':>8} {'Change':>8}")
print(f"  {'-'*60}")
print(f"  {'(none - baseline)':<40} {base_r:8.4f} {'---':>8}")

drop_results = []
for feat in available:
    remaining_feats = [f for f in available if f != feat]
    X_drop = X_z[remaining_feats].values
    r_drop, _, r2_drop, _ = loocv_r(X_drop, y)
    change = r_drop - base_r
    drop_results.append({"dropped": feat, "loocv_r": r_drop, "change": change})
    flag = " <-- can drop" if change > -0.005 else " <-- HURTS" if change < -0.02 else ""
    print(f"  {feat:<40} {r_drop:8.4f} {change:+8.4f}{flag}")


# ═══════════════════════════════════════════════════════════════════════════
#  SAVE & VISUALIZE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SAVING OUTPUTS")
print("=" * 72)

# Save master candidate rankings
cand_save = pd.DataFrame([{k: v for k, v in c.items() if k != "values"}
                           for c in all_candidates])
cand_save.to_csv(os.path.join(RES, "interaction_candidates_ranked.csv"), index=False)

# Save forward selection history
history_df.to_csv(os.path.join(RES, "forward_selection_history.csv"), index=False)

# Save feature diagnostics
diag_save = pd.DataFrame([{
    "name": c["name"], "type": c["type"],
    "r": c["r"], "abs_r": c["abs_r"], "p": c["p"]
} for c in all_candidates])
diag_save.to_csv(os.path.join(RES, "all_feature_candidates.csv"), index=False)

print(f"  Saved: interaction_candidates_ranked.csv")
print(f"  Saved: forward_selection_history.csv")
print(f"  Saved: all_feature_candidates.csv")

# ── PLOTS ────────────────────────────────────────────────────────────────

# Plot 1: Forward selection curve
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Interaction Discovery & Feature Selection", fontsize=14, fontweight="bold")

ax = axes[0]
steps = history_df["step"].values
rs = history_df["loocv_r"].values
ax.plot(steps, rs, "bo-", linewidth=2, markersize=8)
ax.axhline(base_r, color="red", linestyle="--", linewidth=1.5,
           label=f"Raw 13 baseline (r={base_r:.4f})")
ax.axhline(r_comp, color="green", linestyle=":", linewidth=1.5,
           label=f"Simple composite (r={r_comp:.4f})")
# mark feature types
for _, row in history_df.iterrows():
    color = {"original": "blue", "multiply": "red", "ratio": "orange",
             "quadratic": "purple", "dim_interact": "green", "triple": "brown"}.get(row["type"], "gray")
    ax.scatter(row["step"], row["loocv_r"], c=color, s=100, zorder=5, edgecolors="k")
ax.set_xlabel("Features Added")
ax.set_ylabel("LOOCV Pearson r")
ax.set_title("Forward Selection Curve")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Top candidates by type
ax = axes[1]
type_colors = {"original": "#3498db", "multiply": "#e74c3c", "ratio": "#f39c12",
               "quadratic": "#9b59b6", "dim_interact": "#2ecc71", "triple": "#8B4513"}
top30 = all_candidates[:30]
for i, cand in enumerate(top30):
    ax.barh(i, cand["abs_r"], color=type_colors.get(cand["type"], "gray"),
            edgecolor="k", linewidth=0.5, alpha=0.8)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels([c["name"][:45] for c in top30], fontsize=6)
ax.set_xlabel("|Pearson r| with exam")
ax.set_title("Top 30 Feature Candidates")
ax.invert_yaxis()
# legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
ax.grid(True, alpha=0.2)

# Plot 3: Improvement per step (bar chart)
ax = axes[2]
improvements = history_df["improvement"].values
colors_bar = ["#e74c3c" if imp > 0.01 else "#3498db" for imp in improvements]
ax.bar(steps, improvements, color=colors_bar, edgecolor="k", linewidth=0.5)
ax.axhline(0.005, color="orange", linestyle="--", label="Min threshold (0.005)")
ax.set_xlabel("Step")
ax.set_ylabel("LOOCV r Improvement")
ax.set_title("Marginal Gain per Feature Added")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(RES, "interaction_discovery.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: interaction_discovery.png")

# Plot 4: Correlation heatmap of optimal features
opt_vals = np.column_stack([pool_values[name] for name in opt_names])
opt_corr = np.corrcoef(opt_vals.T)
fig2, ax2 = plt.subplots(figsize=(10, 8))
im = ax2.imshow(opt_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax2.set_xticks(range(len(opt_names)))
ax2.set_yticks(range(len(opt_names)))
short_opt = [n[:35] for n in opt_names]
ax2.set_xticklabels(short_opt, rotation=90, fontsize=7)
ax2.set_yticklabels(short_opt, fontsize=7)
for i in range(len(opt_names)):
    for j in range(len(opt_names)):
        ax2.text(j, i, f"{opt_corr[i,j]:.2f}", ha="center", va="center",
                 fontsize=6, color="white" if abs(opt_corr[i,j]) > 0.5 else "black")
plt.colorbar(im, ax=ax2, label="Correlation")
ax2.set_title("Correlation Among Optimal Features", fontweight="bold")
plt.tight_layout()
fig2.savefig(os.path.join(RES, "optimal_features_correlation.png"), dpi=150,
             bbox_inches="tight")
print(f"  Saved: optimal_features_correlation.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FINAL VERDICT")
print("=" * 72)

print(f"""
  BASELINE COMPARISON:
    Simple composite (1 feature):        r = {r_comp:.4f}
    Raw 13 features (Ridge LOOCV):       r = {base_r:.4f}
    4 dimension composites:              r = {r_dim:.4f}
    Dim composites + interactions:       r = {r_dimi:.4f}
    Forward-selected ({len(opt_names)} features):     r = {r_opt:.4f}

  OPTIMAL FEATURE SET ({len(opt_names)} features):
""")
for i, name in enumerate(opt_names):
    ftype = [c for c in all_candidates if c["name"] == name][0]["type"]
    print(f"    {i+1}. {name} ({ftype})")

n_original = sum(1 for c in all_candidates if c["name"] in opt_names and c["type"] == "original")
n_derived = len(opt_names) - n_original
print(f"""
  Feature composition: {n_original} original + {n_derived} derived (interactions/ratios/quadratic)
  Best model type: Check Part 9 results above

  KEY INSIGHTS:
  1. Interactions {'DO' if n_derived > 0 else 'do NOT'} add meaningful signal beyond raw features
  2. Optimal feature count: {len(opt_names)} (not all 13 + interactions)
  3. The forward selection {'saturated quickly' if len(opt_names) <= 8 else 'kept improving'} -- {'simple model may suffice' if len(opt_names) <= 8 else 'complex model warranted'}
""")

print("=" * 72)
print("DONE. Review interaction_discovery.png and the tables above.")
print("=" * 72)
