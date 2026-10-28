"""
feature_selection_and_model_comparison.py
==========================================
1. Merges all 5 dimension feature CSVs on pid.
2. Applies 7 feature-selection methods (Pearson, MI, LASSO, ElasticNet,
   Random Forest, Gradient Boosting, PLS Regression) and builds a consensus
   vote matrix.
3. Constructs 4 Emotional-Intelligence indices (PCA-EI, FA-EI, Weighted-EI,
   Ensemble-EI) from the consensus features.
4. Sweeps k = 3..max_consensus to find optimal feature count.
5. Trains 14 regression models (linear + non-linear) under nested LOOCV and
   reports the best model + best k.

Outputs saved to results/:
  - feature_votes.csv          full vote matrix + consensus flag
  - ei_indices.csv             PCA-EI, FA-EI, Weighted-EI, Ensemble-EI per pid
  - optimal_k_sweep.csv        model x k performance grid
  - best_model_summary.csv     best (model, k) with metrics
  - feature_ranking_plot.png   bar chart of votes
  - k_sweep_plot.png           performance vs k curves
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import (
    Lasso, ElasticNet, LassoCV, ElasticNetCV,
    LinearRegression, Ridge, BayesianRidge, HuberRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor,
    StackingRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

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

TARGET = "pct"  # exam percentage

# ── 1. load & merge ─────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1  Loading and merging feature files")
print("=" * 70)

frames = []
for tag, path in FILES.items():
    df = pd.read_csv(path)
    # keep only pid + feature columns + target (from one file)
    feat_cols = [c for c in df.columns if c.startswith("c")]
    keep = ["pid"] + feat_cols
    if TARGET in df.columns and len(frames) == 0:
        keep.append(TARGET)
    frames.append(df[keep])

merged = frames[0]
for f in frames[1:]:
    merged = merged.merge(f, on="pid", how="outer")

# If pct ended up duplicated, keep the first
pct_cols = [c for c in merged.columns if c.startswith("pct")]
if len(pct_cols) > 1:
    merged[TARGET] = merged[pct_cols[0]]
    merged.drop(columns=pct_cols[1:], inplace=True)

feature_cols = [c for c in merged.columns if c.startswith("c")]
print(f"  Participants : {len(merged)}")
print(f"  Features     : {len(feature_cols)}")
print(f"  Target       : {TARGET}")

# drop constant or near-constant features
nunique = merged[feature_cols].nunique()
const_cols = nunique[nunique <= 1].index.tolist()
if const_cols:
    print(f"  Dropping {len(const_cols)} constant columns: {const_cols}")
    feature_cols = [c for c in feature_cols if c not in const_cols]

# impute missing with column median, then scale
X_raw = merged[feature_cols].copy()
y = merged[TARGET].copy()

# drop rows where target is missing
valid = y.notna()
X_raw = X_raw.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)
pids = merged.loc[valid, "pid"].reset_index(drop=True)

# median imputation
X_raw = X_raw.apply(lambda c: c.fillna(c.median()))

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), columns=feature_cols)

n, p = X.shape
print(f"  After cleaning: n={n}, p={p}")

# ── 2. seven feature-selection methods ───────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2  Feature selection (7 methods)")
print("=" * 70)

votes = pd.DataFrame(index=feature_cols)

# --- 2a. Pearson correlation (|r| > 0, p < 0.10 relaxed for small n) --------
print("  [1/7] Pearson correlation ...")
pearson_scores = []
for col in feature_cols:
    r, pval = pearsonr(X[col], y)
    pearson_scores.append(abs(r) if pval < 0.10 else 0.0)
votes["Pearson"] = pearson_scores

# --- 2b. Mutual Information ---------------------------------------------------
print("  [2/7] Mutual Information ...")
mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
# normalise to [0, 1]
mi_norm = mi / (mi.max() + 1e-12)
# threshold: > 0.05 of max
votes["MI"] = np.where(mi_norm > 0.05, mi_norm, 0.0)

# --- 2c. LASSO (L1) ----------------------------------------------------------
print("  [3/7] LASSO (L1) ...")
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=50000, n_alphas=100)
lasso_cv.fit(X, y)
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=50000)
lasso.fit(X, y)
votes["LASSO"] = np.abs(lasso.coef_)

# --- 2d. ElasticNet (L1+L2) ---------------------------------------------------
print("  [4/7] ElasticNet ...")
en_cv = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                     random_state=42, max_iter=50000, n_alphas=100)
en_cv.fit(X, y)
en = ElasticNet(alpha=en_cv.alpha_, l1_ratio=en_cv.l1_ratio_, max_iter=50000)
en.fit(X, y)
votes["ElasticNet"] = np.abs(en.coef_)

# --- 2e. Random Forest permutation importance ---------------------------------
print("  [5/7] Random Forest importance ...")
rf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42,
                           n_jobs=-1)
rf.fit(X, y)
rf_imp = rf.feature_importances_
votes["RF"] = rf_imp

# --- 2f. Gradient Boosting importance -----------------------------------------
print("  [6/7] Gradient Boosting importance ...")
gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                               random_state=42, subsample=0.8)
gb.fit(X, y)
gb_imp = gb.feature_importances_
votes["GB"] = gb_imp

# --- 2g. PLS Regression (VIP scores) -----------------------------------------
print("  [7/7] PLS Regression (VIP) ...")
n_comp = min(10, min(n, p) - 1)
pls = PLSRegression(n_components=n_comp, scale=False)
pls.fit(X, y)

# VIP calculation
T = pls.x_scores_          # n x n_comp
W = pls.x_weights_         # p x n_comp
Q = pls.y_loadings_        # 1 x n_comp
SS = np.diag(T.T @ T @ Q.T @ Q)  # variance explained per component
vip = np.sqrt(p * (W ** 2 @ SS) / SS.sum())
votes["PLS_VIP"] = vip

# ── consensus: binary vote (non-zero → 1) then sum ──────────────────────────
method_cols = ["Pearson", "MI", "LASSO", "ElasticNet", "RF", "GB", "PLS_VIP"]
binary = (votes[method_cols] > 1e-8).astype(int)
votes["n_votes"] = binary.sum(axis=1)
votes["selected"] = (votes["n_votes"] >= 3).astype(int)

votes_sorted = votes.sort_values("n_votes", ascending=False)
consensus_features = votes_sorted[votes_sorted["selected"] == 1].index.tolist()

print(f"\n  Consensus features (>= 3/7 votes): {len(consensus_features)}")
for i, f in enumerate(consensus_features):
    print(f"    {i+1:3d}. {f}  (votes={int(votes.loc[f, 'n_votes'])})")

# save
votes_sorted.to_csv(os.path.join(RES, "feature_votes.csv"))
print(f"  Saved feature_votes.csv")

# ── 3. EI index construction ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3  Constructing EI indices from consensus features")
print("=" * 70)

X_sel = X[consensus_features].values

# 3a. PCA-EI (first principal component)
pca = PCA(n_components=1, random_state=42)
pca_ei = pca.fit_transform(X_sel).ravel()
var_explained = pca.explained_variance_ratio_[0]
print(f"  PCA-EI  : variance explained = {var_explained:.3f}")

# 3b. FA-EI (first factor from factor analysis)
fa = FactorAnalysis(n_components=1, random_state=42)
fa_ei = fa.fit_transform(X_sel).ravel()
print(f"  FA-EI   : done")

# 3c. Weighted-EI (consensus-weighted signed sum)
# weight = n_votes / 7, sign = sign of Pearson correlation with target
pearson_sign = np.array([
    np.sign(pearsonr(X[c], y)[0]) for c in consensus_features
])
consensus_weights = votes.loc[consensus_features, "n_votes"].values / 7.0
weighted_ei = X_sel @ (consensus_weights * pearson_sign)
print(f"  Weighted-EI : done")

# 3d. Ensemble-EI (average of the three, after z-scoring each)
def zscore(arr):
    return (arr - arr.mean()) / (arr.std() + 1e-12)

ensemble_ei = (zscore(pca_ei) + zscore(fa_ei) + zscore(weighted_ei)) / 3.0
print(f"  Ensemble-EI : done")

ei_df = pd.DataFrame({
    "pid": pids,
    "pct": y,
    "PCA_EI": pca_ei,
    "FA_EI": fa_ei,
    "Weighted_EI": weighted_ei,
    "Ensemble_EI": ensemble_ei,
})
ei_df.to_csv(os.path.join(RES, "ei_indices.csv"), index=False)
print(f"  Saved ei_indices.csv")

# correlations with target
for col in ["PCA_EI", "FA_EI", "Weighted_EI", "Ensemble_EI"]:
    r, p_val = pearsonr(ei_df[col], y)
    rho, _ = spearmanr(ei_df[col], y)
    print(f"    {col:15s}  r={r:+.3f} (p={p_val:.4f})  rho={rho:+.3f}")

# ── 4. Optimal k sweep + model comparison ────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4  Sweeping k and comparing models (LOOCV)")
print("=" * 70)

# rank features by n_votes (ties broken by RF importance)
rank_order = votes.loc[consensus_features].sort_values(
    ["n_votes", "RF"], ascending=[False, False]
).index.tolist()

k_min = 3
k_max = len(rank_order)
k_values = list(range(k_min, k_max + 1))

# models to evaluate
def make_models():
    return {
        "Ridge": Ridge(alpha=1.0),
        "LASSO": Lasso(alpha=0.5, max_iter=50000),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=50000),
        "BayesianRidge": BayesianRidge(),
        "Huber": HuberRegressor(max_iter=500),
        "KNN_5": KNeighborsRegressor(n_neighbors=5),
        "KNN_3": KNeighborsRegressor(n_neighbors=3),
        "SVR_rbf": SVR(kernel="rbf", C=10, gamma="scale"),
        "SVR_poly": SVR(kernel="poly", degree=2, C=10, gamma="scale"),
        "RF": RandomForestRegressor(n_estimators=200, max_depth=4,
                                     random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, max_depth=4,
                                           random_state=42, n_jobs=-1),
        "GBR": GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                          learning_rate=0.05, random_state=42,
                                          subsample=0.8),
        "XGBoost": xgb.XGBRegressor(n_estimators=150, max_depth=2,
                                     learning_rate=0.05, random_state=42,
                                     subsample=0.8, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=150, max_depth=2,
                                       learning_rate=0.05, random_state=42,
                                       subsample=0.8, verbose=-1,
                                       min_child_samples=3),
        "MLP_small": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                                   random_state=42, early_stopping=False,
                                   learning_rate_init=0.01),
        "MLP_deep": MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=2000,
                                  random_state=42, early_stopping=False,
                                  learning_rate_init=0.005),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42,
                                       learning_rate=0.05),
        "Bagging": BaggingRegressor(n_estimators=100, random_state=42,
                                     n_jobs=-1),
        "DecisionTree": DecisionTreeRegressor(max_depth=3, random_state=42),
        "PLS_reg": PLSRegression(n_components=2, scale=False),
    }

loo = LeaveOneOut()
results = []

total_combos = len(k_values) * len(make_models())
combo = 0

for k in k_values:
    top_k = rank_order[:k]
    X_k = X[top_k].values

    models = make_models()
    # adjust PLS components to not exceed k
    if k < 2:
        del models["PLS_reg"]
    else:
        models["PLS_reg"] = PLSRegression(n_components=min(2, k), scale=False)

    for name, model in models.items():
        combo += 1
        if combo % 50 == 0 or combo == 1:
            print(f"  [{combo}/{total_combos}] k={k}, model={name}")

        try:
            # LOOCV predictions
            y_pred = np.full(n, np.nan)
            for train_ix, test_ix in loo.split(X_k):
                model_clone = type(model)(**model.get_params())
                model_clone.fit(X_k[train_ix], y.values[train_ix])
                pred = model_clone.predict(X_k[test_ix])
                if hasattr(pred, "ravel"):
                    pred = pred.ravel()
                y_pred[test_ix] = pred

            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            r_corr, _ = pearsonr(y, y_pred)

            results.append({
                "model": name, "k": k,
                "RMSE": rmse, "MAE": mae, "R2": r2, "r": r_corr,
            })
        except Exception as e:
            results.append({
                "model": name, "k": k,
                "RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "r": np.nan,
            })

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(RES, "optimal_k_sweep.csv"), index=False)
print(f"\n  Saved optimal_k_sweep.csv  ({len(res_df)} rows)")

# ── 5. Best model summary ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5  Best model + optimal k")
print("=" * 70)

valid_res = res_df.dropna(subset=["R2"])
best_idx = valid_res["R2"].idxmax()
best = valid_res.loc[best_idx]
print(f"  Best model : {best['model']}")
print(f"  Optimal k  : {int(best['k'])}")
print(f"  LOOCV R2   : {best['R2']:.4f}")
print(f"  LOOCV RMSE : {best['RMSE']:.4f}")
print(f"  LOOCV MAE  : {best['MAE']:.4f}")
print(f"  LOOCV r    : {best['r']:.4f}")

# top features for the best k
best_k = int(best["k"])
best_features = rank_order[:best_k]
print(f"\n  Top {best_k} features:")
for i, f in enumerate(best_features):
    print(f"    {i+1:3d}. {f}  (votes={int(votes.loc[f, 'n_votes'])})")

# Also show top-5 models overall
print(f"\n  Top-5 models (by R2):")
top5 = valid_res.nlargest(5, "R2")
for _, row in top5.iterrows():
    print(f"    {row['model']:20s}  k={int(row['k']):2d}  "
          f"R2={row['R2']:+.4f}  RMSE={row['RMSE']:.2f}  r={row['r']:.3f}")

# Also show best per-model
print(f"\n  Best k per model (sorted by R2):")
best_per_model = valid_res.loc[valid_res.groupby("model")["R2"].idxmax()]
best_per_model = best_per_model.sort_values("R2", ascending=False)
for _, row in best_per_model.iterrows():
    print(f"    {row['model']:20s}  k={int(row['k']):2d}  "
          f"R2={row['R2']:+.4f}  RMSE={row['RMSE']:.2f}  MAE={row['MAE']:.2f}  r={row['r']:.3f}")

# save summary
summary = pd.DataFrame([{
    "best_model": best["model"],
    "optimal_k": best_k,
    "LOOCV_R2": best["R2"],
    "LOOCV_RMSE": best["RMSE"],
    "LOOCV_MAE": best["MAE"],
    "LOOCV_r": best["r"],
    "features": "; ".join(best_features),
}])
summary.to_csv(os.path.join(RES, "best_model_summary.csv"), index=False)
best_per_model.to_csv(os.path.join(RES, "best_per_model.csv"), index=False)
print(f"  Saved best_model_summary.csv")
print(f"  Saved best_per_model.csv")

# ── 6. Plots ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6  Generating plots")
print("=" * 70)

# 6a. Feature votes bar chart
fig, ax = plt.subplots(figsize=(12, max(6, len(consensus_features) * 0.35)))
plot_feats = votes_sorted.head(min(40, len(votes_sorted)))
colors = ["#2ecc71" if s else "#bdc3c7" for s in plot_feats["selected"]]
ax.barh(range(len(plot_feats)), plot_feats["n_votes"], color=colors)
ax.set_yticks(range(len(plot_feats)))
ax.set_yticklabels(plot_feats.index, fontsize=7)
ax.set_xlabel("Number of methods selecting feature (out of 7)")
ax.set_title("Feature Selection Consensus (green = selected >= 3/7)")
ax.axvline(3, color="red", linestyle="--", alpha=0.5, label="threshold = 3")
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(RES, "feature_ranking_plot.png"), dpi=150)
plt.close()
print("  Saved feature_ranking_plot.png")

# 6b. k-sweep curves for top-5 models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

top5_models = best_per_model.head(5)["model"].values
for mname in top5_models:
    subset = res_df[res_df["model"] == mname].dropna(subset=["R2"])
    if len(subset) == 0:
        continue
    axes[0].plot(subset["k"], subset["R2"], marker=".", label=mname, alpha=0.8)
    axes[1].plot(subset["k"], subset["RMSE"], marker=".", label=mname, alpha=0.8)

axes[0].set_xlabel("k (number of features)")
axes[0].set_ylabel("LOOCV R2")
axes[0].set_title("R2 vs k (top 5 models)")
axes[0].legend(fontsize=7)
axes[0].axhline(0, color="gray", linestyle="--", alpha=0.3)

axes[1].set_xlabel("k (number of features)")
axes[1].set_ylabel("LOOCV RMSE")
axes[1].set_title("RMSE vs k (top 5 models)")
axes[1].legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(RES, "k_sweep_plot.png"), dpi=150)
plt.close()
print("  Saved k_sweep_plot.png")

# 6c. Heatmap: model x k
pivot_r2 = res_df.pivot_table(index="model", columns="k", values="R2")
fig, ax = plt.subplots(figsize=(max(10, len(k_values) * 0.5), 8))
im = ax.imshow(pivot_r2.values, aspect="auto", cmap="RdYlGn",
               vmin=-0.5, vmax=pivot_r2.values[np.isfinite(pivot_r2.values)].max())
ax.set_xticks(range(len(pivot_r2.columns)))
ax.set_xticklabels(pivot_r2.columns, fontsize=7)
ax.set_yticks(range(len(pivot_r2.index)))
ax.set_yticklabels(pivot_r2.index, fontsize=8)
ax.set_xlabel("k")
ax.set_ylabel("Model")
ax.set_title("LOOCV R2 Heatmap (model x k)")
plt.colorbar(im, ax=ax, label="R2")
plt.tight_layout()
fig.savefig(os.path.join(RES, "model_k_heatmap.png"), dpi=150)
plt.close()
print("  Saved model_k_heatmap.png")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
