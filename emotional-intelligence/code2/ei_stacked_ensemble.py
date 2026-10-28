"""
ei_stacked_ensemble.py
======================
Dimensional Stacked Ensemble for Emotional Intelligence Scoring
---------------------------------------------------------------
1. Groups 13 high-consensus supervised features (>=6/7 votes) into 4 EI
   dimensions: C2-Regulation, C3-Utilisation, C4-Self-Awareness, C5-Context.
2. Trains multiple diverse base models per dimension using LOOCV.
3. Stacks dimension-level predictions into a meta-learner for final EI score.
4. Applies K-means clustering (k=2..6) on the 13-feature space.
5. Validates stacked EI score and clusters against exam performance.
6. Outputs:
   - ei_stacked_scores.csv     (student EI scores + cluster labels)
   - stacked_ensemble_report.txt
   - stacked_ei_vs_exam.png    (scatter + cluster + dimension panels)
   - dimension_model_performance.png
   - cluster_analysis.png
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr, f_oneway
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, LinearRegression,
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb

# force UTF-8 output
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── paths ────────────────────────────────────────────────────────────────────
BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

FILES = {
    "D1": os.path.join(RES, "features_D1.csv"),
    "D2": os.path.join(RES, "features_D2.csv"),
    "D3": os.path.join(RES, "features_D3.csv"),
    "D4": os.path.join(RES, "features_D4.csv"),
    "D5": os.path.join(RES, "features_D5.csv"),
}
TARGET = "pct"

# ── HIGH-CONSENSUS SUPERVISED FEATURES (>=6/7 votes) ────────────────────────
# Grouped by EI dimension
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
for feats in DIMENSIONS.values():
    ALL_FEATURES.extend(feats)

# ── BASE MODEL ZOO ──────────────────────────────────────────────────────────
def get_base_models():
    """Diverse set of base learners for each dimension."""
    return {
        "Ridge":    Ridge(alpha=1.0),
        "Lasso":    Lasso(alpha=0.1, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "BayesRidge": BayesianRidge(),
        "RF":       RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
        "GBR":      GradientBoostingRegressor(n_estimators=80, max_depth=2,
                                              learning_rate=0.1, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=42),
        "SVR_rbf":  SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "SVR_poly": SVR(kernel="poly", degree=2, C=1.0, epsilon=0.1),
        "KNN5":     KNeighborsRegressor(n_neighbors=5),
        "KNN7":     KNeighborsRegressor(n_neighbors=7),
        "XGB":      xgb.XGBRegressor(n_estimators=80, max_depth=2, learning_rate=0.1,
                                     verbosity=0, random_state=42),
        "LGBM":     lgb.LGBMRegressor(n_estimators=80, max_depth=2, learning_rate=0.1,
                                      verbose=-1, random_state=42),
        "MLP":      MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000,
                                 random_state=42, early_stopping=True),
        "Huber":    HuberRegressor(max_iter=500),
        "AdaBoost": AdaBoostRegressor(n_estimators=60, learning_rate=0.1, random_state=42),
    }


def loocv_predict(model, X, y):
    """Leave-one-out cross-validated predictions."""
    loo = LeaveOneOut()
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        model_clone = clone_model(model)
        model_clone.fit(X_tr, y_tr)
        preds[test_idx] = model_clone.predict(X_te)
    return preds


def clone_model(model):
    """Clone a sklearn-compatible model."""
    from sklearn.base import clone
    return clone(model)


def eval_metrics(y_true, y_pred):
    """Compute regression metrics + Pearson/Spearman correlations."""
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    r2   = r2_score(yt, yp)
    pr, pp = pearsonr(yt, yp)
    sr, sp = spearmanr(yt, yp)
    return {"RMSE": rmse, "MAE": mae, "R2": r2,
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp}


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD & MERGE DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1  Loading and merging feature files")
print("=" * 70)

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

# handle duplicated pct columns
pct_cols = [c for c in merged.columns if c.startswith("pct")]
if len(pct_cols) > 1:
    merged[TARGET] = merged[pct_cols[0]]
    merged.drop(columns=pct_cols[1:], inplace=True)

# extract target and high-consensus features only
y_full = merged[TARGET].copy()
valid = y_full.notna()
pids = merged.loc[valid, "pid"].reset_index(drop=True)
y = y_full.loc[valid].reset_index(drop=True).values

# check which features exist
available = [f for f in ALL_FEATURES if f in merged.columns]
missing = [f for f in ALL_FEATURES if f not in merged.columns]
if missing:
    print(f"  WARNING: missing features: {missing}")

X_raw = merged.loc[valid, available].reset_index(drop=True)
X_raw = X_raw.apply(lambda c: c.fillna(c.median()))

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=available)

n = len(y)
print(f"  Students     : {n}")
print(f"  Features     : {len(available)}")
print(f"  Dimensions   : {len(DIMENSIONS)}")
for dim, feats in DIMENSIONS.items():
    present = [f for f in feats if f in available]
    print(f"    {dim}: {len(present)} features")

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: NON-LINEARITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2  Non-linearity analysis (linear vs polynomial)")
print("=" * 70)

X_arr = X_scaled.values

# linear baseline
from sklearn.linear_model import LinearRegression as LR
lin_preds = loocv_predict(LR(), X_arr, y)
lin_r2 = r2_score(y, lin_preds)
lin_r, _ = pearsonr(y, lin_preds)

# polynomial degree 2 (interaction terms)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_arr)
# Ridge to handle high dimensionality
poly_preds = loocv_predict(Ridge(alpha=10.0), X_poly, y)
poly_r2 = r2_score(y, poly_preds)
poly_r, _ = pearsonr(y, poly_preds)

print(f"  Linear LOOCV:     R2={lin_r2:.4f}, r={lin_r:.4f}")
print(f"  Poly-2 LOOCV:     R2={poly_r2:.4f}, r={poly_r:.4f}")
print(f"  R2 gain:          {poly_r2 - lin_r2:+.4f}")
print(f"  --> {'Strong' if (poly_r2 - lin_r2) > 0.05 else 'Moderate' if (poly_r2 - lin_r2) > 0.01 else 'Weak'} non-linearity detected")

# per-feature non-linearity
print("\n  Per-feature non-linearity (quadratic R2 gain):")
nl_gains = {}
for feat in available:
    x_f = X_scaled[feat].values.reshape(-1, 1)
    lin_p = loocv_predict(LR(), x_f, y)
    x_f2 = np.column_stack([x_f, x_f**2])
    quad_p = loocv_predict(Ridge(alpha=1.0), x_f2, y)
    lin_r2_f = max(r2_score(y, lin_p), -1)
    quad_r2_f = max(r2_score(y, quad_p), -1)
    gain = quad_r2_f - lin_r2_f
    nl_gains[feat] = gain
    if gain > 0.02:
        print(f"    {feat}: linear R2={lin_r2_f:.3f} -> quad R2={quad_r2_f:.3f}  (gain={gain:+.3f})")

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: DIMENSION-LEVEL BASE MODELS (LOOCV)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3  Training dimension-level base models (LOOCV)")
print("=" * 70)

dim_predictions = {}   # {dim: {model_name: predictions}}
dim_best_models = {}   # {dim: best_model_name}
dim_results = []       # for reporting

for dim_name, dim_feats in DIMENSIONS.items():
    present = [f for f in dim_feats if f in available]
    if len(present) == 0:
        print(f"\n  {dim_name}: SKIPPED (no features)")
        continue

    X_dim = X_scaled[present].values
    print(f"\n  {dim_name} ({len(present)} features: {', '.join(present)})")
    print(f"  {'Model':<15} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'r':>8} {'p':>10}")
    print(f"  {'-'*60}")

    models = get_base_models()
    dim_predictions[dim_name] = {}
    best_r = -999
    best_name = ""

    for mname, model in models.items():
        try:
            preds = loocv_predict(model, X_dim, y)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                continue
            m = eval_metrics(y, preds)
            dim_predictions[dim_name][mname] = preds
            dim_results.append({
                "dimension": dim_name, "model": mname,
                **m
            })
            flag = ""
            if abs(m["pearson_r"]) > best_r:
                best_r = abs(m["pearson_r"])
                best_name = mname
                flag = " <-- best"
            print(f"  {mname:<15} {m['RMSE']:8.3f} {m['MAE']:8.3f} "
                  f"{m['R2']:8.4f} {m['pearson_r']:8.4f} {m['pearson_p']:10.4f}{flag}")
        except Exception as e:
            pass

    dim_best_models[dim_name] = best_name
    print(f"  --> Best model: {best_name} (|r|={best_r:.4f})")

dim_results_df = pd.DataFrame(dim_results)

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: SELECT TOP-K DIVERSE MODELS PER DIMENSION FOR STACKING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4  Selecting diverse base models for stacking")
print("=" * 70)

def select_diverse_models(dim_preds, y, top_k=5):
    """Select top-k models that are diverse (low inter-correlation) and accurate."""
    if len(dim_preds) == 0:
        return []

    # rank by |pearson_r| with y
    scores = {}
    for mname, preds in dim_preds.items():
        r, _ = pearsonr(y, preds)
        scores[mname] = abs(r)

    ranked = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)

    # greedily select diverse models
    selected = [ranked[0]]
    for candidate in ranked[1:]:
        if len(selected) >= top_k:
            break
        # check correlation with already selected
        cand_preds = dim_preds[candidate]
        max_corr = max(abs(pearsonr(cand_preds, dim_preds[s])[0]) for s in selected)
        # accept if not too correlated with existing selections
        if max_corr < 0.95:
            selected.append(candidate)

    return selected


stacking_models = {}
for dim_name in DIMENSIONS:
    if dim_name not in dim_predictions:
        continue
    selected = select_diverse_models(dim_predictions[dim_name], y, top_k=5)
    stacking_models[dim_name] = selected
    print(f"  {dim_name}: {selected}")

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5: BUILD STACKING META-FEATURES & TRAIN META-LEARNER
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5  Stacking: meta-features from dimension predictions")
print("=" * 70)

# Build meta-feature matrix: for each dimension, use selected models' LOOCV predictions
meta_features = []
meta_names = []
for dim_name, model_names in stacking_models.items():
    for mname in model_names:
        meta_features.append(dim_predictions[dim_name][mname])
        meta_names.append(f"{dim_name}__{mname}")

X_meta = np.column_stack(meta_features)
print(f"  Meta-feature matrix shape: {X_meta.shape}")
print(f"  Meta-features: {meta_names}")

# Also build dimension-aggregated meta-features (mean of selected models per dim)
dim_scores = {}
for dim_name, model_names in stacking_models.items():
    preds_list = [dim_predictions[dim_name][m] for m in model_names]
    dim_score = np.mean(preds_list, axis=0)
    dim_scores[dim_name] = dim_score

X_meta_agg = np.column_stack([dim_scores[d] for d in stacking_models])
meta_agg_names = list(stacking_models.keys())
print(f"  Aggregated meta-features: {X_meta_agg.shape[1]} (one per dimension)")

# Train multiple meta-learners using LOOCV
print("\n  Meta-learner comparison (on individual model predictions):")
meta_models = {
    "Ridge_a1":    Ridge(alpha=1.0),
    "Ridge_a10":   Ridge(alpha=10.0),
    "Lasso_a01":   Lasso(alpha=0.1, max_iter=5000),
    "ElasticNet":  ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    "BayesRidge":  BayesianRidge(),
    "RF_meta":     RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    "GBR_meta":    GradientBoostingRegressor(n_estimators=60, max_depth=2,
                                             learning_rate=0.05, random_state=42),
    "XGB_meta":    xgb.XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05,
                                    verbosity=0, random_state=42),
    "LGBM_meta":   lgb.LGBMRegressor(n_estimators=60, max_depth=2, learning_rate=0.05,
                                     verbose=-1, random_state=42),
    "SVR_meta":    SVR(kernel="rbf", C=1.0),
    "KNN3_meta":   KNeighborsRegressor(n_neighbors=3),
    "KNN5_meta":   KNeighborsRegressor(n_neighbors=5),
}

print(f"  {'Meta-Learner':<15} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'r':>8} {'p':>10}")
print(f"  {'-'*60}")

meta_results = {}
best_meta_r = -999
best_meta_name = ""

for mname, model in meta_models.items():
    try:
        preds = loocv_predict(model, X_meta, y)
        if np.any(np.isnan(preds)):
            continue
        m = eval_metrics(y, preds)
        meta_results[mname] = {"preds": preds, **m}
        flag = ""
        if abs(m["pearson_r"]) > best_meta_r:
            best_meta_r = abs(m["pearson_r"])
            best_meta_name = mname
            flag = " <-- best"
        print(f"  {mname:<15} {m['RMSE']:8.3f} {m['MAE']:8.3f} "
              f"{m['R2']:8.4f} {m['pearson_r']:8.4f} {m['pearson_p']:10.6f}{flag}")
    except Exception as e:
        pass

print(f"\n  Best meta-learner: {best_meta_name} (|r|={best_meta_r:.4f})")

# Also try aggregated meta-features (one per dimension)
print("\n  Meta-learner comparison (on aggregated dimension scores):")
print(f"  {'Meta-Learner':<15} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'r':>8} {'p':>10}")
print(f"  {'-'*60}")

meta_agg_results = {}
best_agg_r = -999
best_agg_name = ""

for mname, model in meta_models.items():
    try:
        preds = loocv_predict(model, X_meta_agg, y)
        if np.any(np.isnan(preds)):
            continue
        m = eval_metrics(y, preds)
        meta_agg_results[mname] = {"preds": preds, **m}
        flag = ""
        if abs(m["pearson_r"]) > best_agg_r:
            best_agg_r = abs(m["pearson_r"])
            best_agg_name = mname
            flag = " <-- best"
        print(f"  {mname:<15} {m['RMSE']:8.3f} {m['MAE']:8.3f} "
              f"{m['R2']:8.4f} {m['pearson_r']:8.4f} {m['pearson_p']:10.6f}{flag}")
    except Exception as e:
        pass

print(f"\n  Best aggregated meta-learner: {best_agg_name} (|r|={best_agg_r:.4f})")

# Choose best overall stacking approach
if best_meta_r >= best_agg_r:
    stacked_ei = meta_results[best_meta_name]["preds"]
    stacking_type = f"individual ({best_meta_name})"
    stacking_metrics = meta_results[best_meta_name]
else:
    stacked_ei = meta_agg_results[best_agg_name]["preds"]
    stacking_type = f"aggregated ({best_agg_name})"
    stacking_metrics = meta_agg_results[best_agg_name]

# Normalize stacked EI to z-scores
stacked_ei_z = (stacked_ei - np.mean(stacked_ei)) / np.std(stacked_ei)

print(f"\n  Final stacking approach: {stacking_type}")
print(f"  Stacked EI vs exam: r={stacking_metrics['pearson_r']:.4f}, "
      f"p={stacking_metrics['pearson_p']:.6f}")

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 6: DIMENSION SUB-SCORES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6  Dimension sub-scores (best model per dimension)")
print("=" * 70)

dim_sub_scores = {}
for dim_name in DIMENSIONS:
    if dim_name not in dim_best_models:
        continue
    best_m = dim_best_models[dim_name]
    preds = dim_predictions[dim_name][best_m]
    # z-score the dimension prediction
    dim_sub_scores[dim_name] = (preds - np.mean(preds)) / np.std(preds)
    r, p = pearsonr(y, preds)
    print(f"  {dim_name} ({best_m}): r={r:.4f}, p={p:.4f}")

# Also build a simple ensemble EI (average of dimension z-scores)
ensemble_dims = np.column_stack([dim_sub_scores[d] for d in dim_sub_scores])
simple_ensemble_ei = np.mean(ensemble_dims, axis=1)
simple_r, simple_p = pearsonr(y, simple_ensemble_ei)
print(f"\n  Simple ensemble (mean of dim z-scores): r={simple_r:.4f}, p={simple_p:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 7: K-MEANS CLUSTERING ON 13-FEATURE SPACE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7  K-means clustering on supervised feature space")
print("=" * 70)

cluster_results = []
best_sil = -1
best_k = 2

for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X_scaled[available].values)

    sil = silhouette_score(X_scaled[available].values, labels)
    # ANOVA F-test: do clusters differ in exam scores?
    groups = [y[labels == c] for c in range(k)]
    groups_nonzero = [g for g in groups if len(g) > 1]
    if len(groups_nonzero) >= 2:
        f_stat, f_p = f_oneway(*groups_nonzero)
    else:
        f_stat, f_p = 0, 1.0

    cluster_means = [np.mean(g) for g in groups]
    cluster_sizes = [len(g) for g in groups]

    cluster_results.append({
        "k": k, "silhouette": sil, "F_stat": f_stat, "F_p": f_p,
        "cluster_means": cluster_means, "cluster_sizes": cluster_sizes,
        "labels": labels,
    })

    print(f"  k={k}: silhouette={sil:.4f}, F={f_stat:.2f}, p={f_p:.6f}")
    for c_idx in range(k):
        print(f"    Cluster {c_idx}: n={cluster_sizes[c_idx]}, "
              f"exam_mean={cluster_means[c_idx]:.1f}%")

    if sil > best_sil:
        best_sil = sil
        best_k = k

# Also pick cluster solution with best ANOVA p-value
best_anova_idx = np.argmin([cr["F_p"] for cr in cluster_results])
best_anova_k = cluster_results[best_anova_idx]["k"]

print(f"\n  Best silhouette: k={best_k} (sil={best_sil:.4f})")
print(f"  Best ANOVA separation: k={best_anova_k} "
      f"(F={cluster_results[best_anova_idx]['F_stat']:.2f}, "
      f"p={cluster_results[best_anova_idx]['F_p']:.6f})")

# Use the ANOVA-best k for final clustering (most exam-relevant)
final_k_idx = best_anova_idx
final_labels = cluster_results[final_k_idx]["labels"]
final_k = cluster_results[final_k_idx]["k"]

# Re-order clusters by exam mean (ascending) for interpretability
cluster_order = np.argsort([np.mean(y[final_labels == c]) for c in range(final_k)])
label_map = {old: new for new, old in enumerate(cluster_order)}
final_labels_ordered = np.array([label_map[l] for l in final_labels])

print(f"\n  Final clustering: k={final_k} (ordered by exam mean)")
for c in range(final_k):
    mask = final_labels_ordered == c
    c_y = y[mask]
    c_ei = stacked_ei_z[mask]
    print(f"    Cluster {c}: n={np.sum(mask)}, "
          f"exam_mean={np.mean(c_y):.1f}% (SD={np.std(c_y):.1f}), "
          f"EI_mean={np.mean(c_ei):.3f}")

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 8: CLUSTER PROFILING (which dimensions drive each cluster)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8  Cluster profiling by dimension")
print("=" * 70)

profile_data = []
for c in range(final_k):
    mask = final_labels_ordered == c
    row = {"cluster": c, "n": int(np.sum(mask)),
           "exam_mean": np.mean(y[mask]), "EI_mean": np.mean(stacked_ei_z[mask])}
    for dim_name in dim_sub_scores:
        row[dim_name] = np.mean(dim_sub_scores[dim_name][mask])
    profile_data.append(row)

profile_df = pd.DataFrame(profile_data)
print(profile_df.to_string(index=False))

# Feature importance per dimension (correlation with exam)
print("\n  Feature importance (|r| with exam) per dimension:")
for dim_name, feats in DIMENSIONS.items():
    present = [f for f in feats if f in available]
    print(f"\n  {dim_name}:")
    for feat in present:
        r, p = pearsonr(y, X_scaled[feat].values)
        print(f"    {feat}: r={r:.4f}, p={p:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 9: SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 9  Saving outputs")
print("=" * 70)

# Build output dataframe
out = pd.DataFrame({
    "pid": pids,
    "exam_pct": y,
    "Stacked_EI": stacked_ei,
    "Stacked_EI_z": stacked_ei_z,
    "Simple_Ensemble_EI": simple_ensemble_ei,
})

for dim_name in dim_sub_scores:
    out[f"EI_{dim_name}"] = dim_sub_scores[dim_name]

out[f"cluster_k{final_k}"] = final_labels_ordered

# Sort by Stacked EI descending
out = out.sort_values("Stacked_EI_z", ascending=False).reset_index(drop=True)
out.to_csv(os.path.join(RES, "ei_stacked_scores.csv"), index=False)
print(f"  Saved: ei_stacked_scores.csv ({len(out)} rows)")

# Dimension model results
dim_results_df.to_csv(os.path.join(RES, "dimension_model_results.csv"), index=False)
print(f"  Saved: dimension_model_results.csv")

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 10: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 10  Generating visualizations")
print("=" * 70)

# ── PLOT 1: Stacked EI vs Exam (main scatter + clusters) ────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Stacked Ensemble EI: Validation & Clustering", fontsize=16, fontweight="bold")

# 1a: Stacked EI vs exam scatter
ax = axes[0, 0]
ax.scatter(stacked_ei_z, y, c="steelblue", alpha=0.7, edgecolors="k", s=60)
z = np.polyfit(stacked_ei_z, y, 1)
xline = np.linspace(stacked_ei_z.min(), stacked_ei_z.max(), 50)
ax.plot(xline, np.polyval(z, xline), "r--", lw=2)
ax.set_xlabel("Stacked EI (z-score)")
ax.set_ylabel("Exam %")
ax.set_title(f"Stacked EI vs Exam\nr={stacking_metrics['pearson_r']:.3f}, "
             f"p={stacking_metrics['pearson_p']:.4f}")
ax.grid(True, alpha=0.3)

# 1b: Clusters colored by exam
ax = axes[0, 1]
colors = plt.cm.Set1(np.linspace(0, 1, final_k))
for c in range(final_k):
    mask = final_labels_ordered == c
    ax.scatter(stacked_ei_z[mask], y[mask], c=[colors[c]], alpha=0.7,
               edgecolors="k", s=60, label=f"C{c} (n={np.sum(mask)}, "
               f"mean={np.mean(y[mask]):.0f}%)")
ax.legend(fontsize=8)
ax.set_xlabel("Stacked EI (z-score)")
ax.set_ylabel("Exam %")
ax.set_title(f"K-means Clusters (k={final_k})")
ax.grid(True, alpha=0.3)

# 1c: PCA 2D projection of clusters
ax = axes[0, 2]
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled[available].values)
for c in range(final_k):
    mask = final_labels_ordered == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[c]], alpha=0.7,
               edgecolors="k", s=60, label=f"C{c}")
ax.legend(fontsize=8)
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA Projection of Clusters")
ax.grid(True, alpha=0.3)

# 1d-f: Dimension sub-scores vs exam
dim_names_sorted = sorted(dim_sub_scores.keys())
for idx, dim_name in enumerate(dim_names_sorted):
    if idx >= 3:
        break
    ax = axes[1, idx]
    dscore = dim_sub_scores[dim_name]
    r, p = pearsonr(y, dscore)
    for c in range(final_k):
        mask = final_labels_ordered == c
        ax.scatter(dscore[mask], y[mask], c=[colors[c]], alpha=0.7,
                   edgecolors="k", s=50, label=f"C{c}")
    z_fit = np.polyfit(dscore, y, 1)
    xline = np.linspace(dscore.min(), dscore.max(), 50)
    ax.plot(xline, np.polyval(z_fit, xline), "r--", lw=2)
    ax.set_xlabel(f"{dim_name} (z-score)")
    ax.set_ylabel("Exam %")
    ax.set_title(f"{dim_name}\nr={r:.3f}, p={p:.4f}")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

# Handle 4th dimension if exists
if len(dim_names_sorted) > 3:
    # Add as text annotation
    dim_name = dim_names_sorted[3]
    dscore = dim_sub_scores[dim_name]
    r, p = pearsonr(y, dscore)
    axes[1, 2].text(0.02, 0.02, f"+ {dim_name}: r={r:.3f}, p={p:.4f}",
                    transform=axes[1, 2].transAxes, fontsize=8,
                    verticalalignment="bottom", style="italic")

plt.tight_layout()
fig.savefig(os.path.join(RES, "stacked_ei_vs_exam.png"), dpi=150, bbox_inches="tight")
print("  Saved: stacked_ei_vs_exam.png")

# ── PLOT 2: Dimension model performance heatmap ─────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 8))
pivot = dim_results_df.pivot_table(index="model", columns="dimension", values="pearson_r")
# sort by mean |r|
pivot_abs = pivot.abs()
row_order = pivot_abs.mean(axis=1).sort_values(ascending=False).index
pivot = pivot.loc[row_order]

im = ax2.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=0.8)
ax2.set_xticks(range(len(pivot.columns)))
ax2.set_xticklabels(pivot.columns, rotation=45, ha="right")
ax2.set_yticks(range(len(pivot.index)))
ax2.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(v) > 0.5 else "black")
plt.colorbar(im, ax=ax2, label="Pearson r")
ax2.set_title("Dimension x Model Performance (Pearson r with exam)", fontweight="bold")
plt.tight_layout()
fig2.savefig(os.path.join(RES, "dimension_model_performance.png"), dpi=150, bbox_inches="tight")
print("  Saved: dimension_model_performance.png")

# ── PLOT 3: Cluster analysis detail ─────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle(f"Cluster Analysis (k={final_k})", fontsize=14, fontweight="bold")

# 3a: Box plot of exam scores by cluster
ax = axes3[0]
cluster_data = [y[final_labels_ordered == c] for c in range(final_k)]
bp = ax.boxplot(cluster_data, labels=[f"C{c}" for c in range(final_k)],
                patch_artist=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel("Exam %")
ax.set_title(f"Exam Scores by Cluster\nANOVA F={cluster_results[final_k_idx]['F_stat']:.2f}, "
             f"p={cluster_results[final_k_idx]['F_p']:.4f}")
ax.grid(True, alpha=0.3)

# 3b: Radar/bar chart of cluster dimension profiles
ax = axes3[1]
dim_cols = [d for d in dim_sub_scores.keys()]
x_pos = np.arange(len(dim_cols))
width = 0.8 / final_k
for c in range(final_k):
    mask = final_labels_ordered == c
    means = [np.mean(dim_sub_scores[d][mask]) for d in dim_cols]
    ax.bar(x_pos + c * width, means, width, color=colors[c], alpha=0.7,
           label=f"C{c} (n={np.sum(mask)})", edgecolor="k", linewidth=0.5)
ax.set_xticks(x_pos + width * (final_k - 1) / 2)
ax.set_xticklabels([d.replace("_", "\n") for d in dim_cols], fontsize=8)
ax.set_ylabel("Mean dimension z-score")
ax.set_title("Cluster Dimension Profiles")
ax.legend(fontsize=7)
ax.axhline(0, color="k", linewidth=0.5)
ax.grid(True, alpha=0.3)

# 3c: Silhouette scores across k
ax = axes3[2]
ks = [cr["k"] for cr in cluster_results]
sils = [cr["silhouette"] for cr in cluster_results]
f_ps = [cr["F_p"] for cr in cluster_results]
ax.plot(ks, sils, "bo-", linewidth=2, markersize=8, label="Silhouette")
ax.set_xlabel("k")
ax.set_ylabel("Silhouette Score", color="blue")
ax.tick_params(axis="y", labelcolor="blue")
ax2_twin = ax.twinx()
ax2_twin.plot(ks, [-np.log10(max(p, 1e-20)) for p in f_ps], "rs--",
              linewidth=2, markersize=8, label="-log10(ANOVA p)")
ax2_twin.set_ylabel("-log10(ANOVA p)", color="red")
ax2_twin.tick_params(axis="y", labelcolor="red")
ax.set_title("Clustering Quality vs k")
ax.grid(True, alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.tight_layout()
fig3.savefig(os.path.join(RES, "cluster_analysis.png"), dpi=150, bbox_inches="tight")
print("  Saved: cluster_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 11: SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 11  Summary report")
print("=" * 70)

report = []
report.append("STACKED ENSEMBLE EI SCORING - SUMMARY REPORT")
report.append("=" * 60)
report.append(f"\nStudents: {n}")
report.append(f"High-consensus features (>=6/7 votes): {len(available)}")
report.append(f"Dimensions: {len(DIMENSIONS)}")
for d, f in DIMENSIONS.items():
    present = [x for x in f if x in available]
    report.append(f"  {d}: {present}")

report.append(f"\n--- NON-LINEARITY ---")
report.append(f"Linear LOOCV R2: {lin_r2:.4f}")
report.append(f"Poly-2 interaction LOOCV R2: {poly_r2:.4f}")
report.append(f"R2 gain: {poly_r2 - lin_r2:+.4f}")

report.append(f"\n--- DIMENSION-LEVEL BEST MODELS ---")
for dim_name, best_m in dim_best_models.items():
    preds = dim_predictions[dim_name][best_m]
    r, p = pearsonr(y, preds)
    report.append(f"  {dim_name}: {best_m} (r={r:.4f}, p={p:.4f})")

report.append(f"\n--- STACKING ---")
report.append(f"Stacking type: {stacking_type}")
report.append(f"Meta-features: {X_meta.shape[1]}")
report.append(f"Stacked EI vs exam: r={stacking_metrics['pearson_r']:.4f}, "
              f"p={stacking_metrics['pearson_p']:.6f}")
report.append(f"Simple ensemble EI vs exam: r={simple_r:.4f}, p={simple_p:.4f}")

report.append(f"\n--- CLUSTERING (k={final_k}) ---")
report.append(f"Silhouette: {cluster_results[final_k_idx]['silhouette']:.4f}")
report.append(f"ANOVA F={cluster_results[final_k_idx]['F_stat']:.2f}, "
              f"p={cluster_results[final_k_idx]['F_p']:.6f}")
for c in range(final_k):
    mask = final_labels_ordered == c
    report.append(f"  Cluster {c}: n={np.sum(mask)}, "
                  f"exam_mean={np.mean(y[mask]):.1f}%, "
                  f"EI_mean={np.mean(stacked_ei_z[mask]):.3f}")

report.append(f"\n--- CLUSTER DIMENSION PROFILES ---")
report.append(profile_df.to_string(index=False))

report.append(f"\n--- TOP 10 STUDENTS BY STACKED EI ---")
top10 = out.head(10)[["pid", "exam_pct", "Stacked_EI_z", f"cluster_k{final_k}"]].to_string(index=False)
report.append(top10)

report.append(f"\n--- BOTTOM 10 STUDENTS BY STACKED EI ---")
bot10 = out.tail(10)[["pid", "exam_pct", "Stacked_EI_z", f"cluster_k{final_k}"]].to_string(index=False)
report.append(bot10)

report_txt = "\n".join(report)
with open(os.path.join(RES, "stacked_ensemble_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_txt)
print(f"  Saved: stacked_ensemble_report.txt")

print("\n" + report_txt)

print("\n" + "=" * 70)
print("DONE. All outputs saved to results/")
print("=" * 70)
