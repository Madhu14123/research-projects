"""
EI Final Pipeline — Multi-Architecture Validation + Custom DL + EI Score
=========================================================================
Phases:
  A: Load 11 BCa-validated features from prior run
  B: 8-model LOOCV comparison (Mean, Ridge, ElasticNet, BayesianRidge,
     RF, XGBoost, Custom DEIN, Weighted Ensemble)
  C: SHAP convergence across models → select final 4 features
  D: LOOCV composite EI score, partial correlations, Wilcoxon tests

Reads:
  results_pdf_features/all_features_116.csv  (from prior feature extraction)
  Exam Scores.xlsx                           (criterion variable)

Outputs → results_final_pipeline/
"""

import re, argparse, warnings, logging, traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import (pearsonr, spearmanr, linregress,
                          wilcoxon, norm as sp_norm)
from sklearn.linear_model import (RidgeCV, ElasticNetCV, BayesianRidge)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_DATA_DIR = r"D:\A_emotions in exam\data"
PREV_RESULTS     = "results_pdf_features"
OUT_DIR          = "results_final_pipeline"

# The 11 BCa-validated features (Gate 1 survivors)
VALIDATED_FEATURES = {
    "D2": [
        "c2_stress_peak_attenuation",        # r=+0.569, stability=92.6%
        "c2_stress_recovery_latency",         # r=-0.348, CI excl 0
        "c2_emotion_variability_SD",          # r=-0.316, CI excl 0
        "c2_reappraisal_proxy_coactivation",  # r=+0.240, CI excl 0
    ],
    "D3": [
        "c3_productive_absorbing_probability",  # r=+0.451, stability=73%
        "c3_stress_inertia_during_blocks",      # r=-0.428, stability=65%
        "c3_forward_stage_momentum",            # r=-0.367, CI excl 0
        "c3_bounce_back_index",                 # r=-0.225, CI excl 0
    ],
    "D4": [
        "c4_arousal_trend",                 # r=+0.376, stability=66%
        "c4_arousal_optimality",            # r=+0.363, CI excl 0
        "c4_arousal_midrange_proportion",   # r=+0.335, CI excl 0
    ],
}

ALL_VALIDATED = []
for feats in VALIDATED_FEATURES.values():
    ALL_VALIDATED.extend(feats)

# Direction signs (positive = higher is better for exam score)
FEAT_SIGN = {
    "c2_stress_peak_attenuation":        +1,
    "c2_stress_recovery_latency":        -1,
    "c2_emotion_variability_SD":         -1,
    "c2_reappraisal_proxy_coactivation": +1,
    "c3_productive_absorbing_probability": +1,
    "c3_stress_inertia_during_blocks":   -1,
    "c3_forward_stage_momentum":         -1,
    "c3_bounce_back_index":              -1,
    "c4_arousal_trend":                  +1,
    "c4_arousal_optimality":             +1,
    "c4_arousal_midrange_proportion":    +1,
}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data(data_dir: Path):
    """Load the pre-computed feature matrix and exam scores."""
    feat_path = data_dir / PREV_RESULTS / "all_features_116.csv"
    fm = pd.read_csv(feat_path, index_col=0)
    log.info("Loaded feature matrix: %d × %d", *fm.shape)

    exam_path = data_dir / "Exam Scores.xlsx"
    edf = pd.read_excel(exam_path)
    edf.columns = edf.columns.str.strip()
    scores = {}
    for _, row in edf.iterrows():
        m = re.search(r"P(\d+)", str(row.iloc[0]).upper())
        if not m:
            continue
        try:
            denom = float(row.iloc[2]) if len(row) > 2 else 26.0
            scores[f"P{m.group(1)}"] = float(row.iloc[1]) / denom
        except:
            pass

    # Align participants
    common = sorted(
        set(fm.index) & set(scores.keys()),
        key=lambda p: int(re.search(r"\d+", p).group()),
    )
    log.info("Participants with features + exam: %d", len(common))

    # Extract only the 11 validated features
    available = [f for f in ALL_VALIDATED if f in fm.columns]
    log.info("Validated features available: %d / %d", len(available), len(ALL_VALIDATED))

    X_df = fm.loc[common, available].fillna(0)
    y    = np.array([scores[p] for p in common])

    return X_df, y, common, available


# ═══════════════════════════════════════════════════════════════════════
# PHASE B: MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

def make_models():
    """Return dict of model_name → model_factory_fn."""
    models = {}

    models["Ridge"] = lambda: RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    models["ElasticNet"] = lambda: ElasticNetCV(
        cv=5, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        max_iter=5000, random_state=42)
    models["BayesianRidge"] = lambda: BayesianRidge(
        max_iter=500, tol=1e-6)
    models["RandomForest"] = lambda: RandomForestRegressor(
        n_estimators=200, max_depth=3, min_samples_leaf=5,
        max_features="sqrt", random_state=42)

    if HAS_XGB:
        models["XGBoost"] = lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=1.0, reg_lambda=2.0,
            random_state=42, verbosity=0)

    return models


# ═══════════════════════════════════════════════════════════════════════
# DEIN: DIMENSION-ENCODED EI NETWORK  (Custom DL)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class DimensionBranch(nn.Module):
        """One branch per EI dimension."""
        def __init__(self, n_in, n_hidden, n_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_out),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
        def forward(self, x):
            return self.net(x)

    class DEIN(nn.Module):
        """
        Dimension-Encoded EI Network.

        Architecture:
          D2 branch (4 feats → 8 → 4)
          D3 branch (4 feats → 8 → 4)
          D4 branch (3 feats → 6 → 4)
                    ↓ concatenate (12)
          Fusion: 12 → 8 → ReLU → Dropout(0.4) → 4 → ReLU → 1 → Sigmoid
        """
        def __init__(self, feat_names: list, y_min=0.0, y_max=1.0):
            super().__init__()
            self.feat_names = feat_names
            self.y_min = y_min
            self.y_max = y_max

            # Identify indices for each dimension
            self.d2_idx = [i for i, f in enumerate(feat_names)
                           if f.startswith("c2_")]
            self.d3_idx = [i for i, f in enumerate(feat_names)
                           if f.startswith("c3_")]
            self.d4_idx = [i for i, f in enumerate(feat_names)
                           if f.startswith("c4_")]

            n_d2 = max(len(self.d2_idx), 1)
            n_d3 = max(len(self.d3_idx), 1)
            n_d4 = max(len(self.d4_idx), 1)

            self.branch_d2 = DimensionBranch(n_d2, 8, 4)
            self.branch_d3 = DimensionBranch(n_d3, 8, 4)
            self.branch_d4 = DimensionBranch(n_d4, 6, 4)

            self.fusion = nn.Sequential(
                nn.Linear(12, 8),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            h2 = self.branch_d2(x[:, self.d2_idx])
            h3 = self.branch_d3(x[:, self.d3_idx])
            h4 = self.branch_d4(x[:, self.d4_idx])
            fused = torch.cat([h2, h3, h4], dim=1)
            out = self.fusion(fused)
            # Scale sigmoid output to exam score range
            return out * (self.y_max - self.y_min) + self.y_min

    def train_dein_loocv(X: np.ndarray, y: np.ndarray,
                          feat_names: list, n_epochs=300, lr=0.005,
                          weight_decay=0.01) -> np.ndarray:
        """Train DEIN under LOOCV. Returns predictions array."""
        n = len(y)
        preds = np.zeros(n)
        y_min, y_max = float(y.min()) - 0.05, float(y.max()) + 0.05

        for i in range(n):
            # Split
            X_tr = np.delete(X, i, axis=0)
            y_tr = np.delete(y, i)
            X_te = X[i:i+1]

            # Scale
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Tensors
            Xt = torch.FloatTensor(X_tr_s)
            yt = torch.FloatTensor(y_tr).reshape(-1, 1)
            Xv = torch.FloatTensor(X_te_s)

            # Model
            model = DEIN(feat_names, y_min, y_max)
            optimizer = torch.optim.AdamW(model.parameters(),
                                           lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss()

            # Train
            model.train()
            best_loss = float("inf")
            patience_counter = 0
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                pred = model(Xt)
                loss = criterion(pred, yt)
                loss.backward()
                optimizer.step()

                if loss.item() < best_loss - 1e-5:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= 50:
                    break

            # Predict
            model.eval()
            with torch.no_grad():
                preds[i] = model(Xv).item()

        return preds


# ═══════════════════════════════════════════════════════════════════════
# PHASE B: LOOCV FOR ALL MODELS
# ═══════════════════════════════════════════════════════════════════════

def loocv_all_models(X_df: pd.DataFrame, y: np.ndarray,
                      feat_names: list, out_dir: Path):
    """Run LOOCV for all models. Returns predictions dict + metrics df."""
    n = len(y)
    X = X_df.values
    models_dict = make_models()
    all_preds = {}

    # --- Mean baseline ---
    log.info("Running Mean Baseline ...")
    preds_mean = np.full(n, y.mean())
    # True LOOCV mean baseline: predict mean of training fold
    for i in range(n):
        y_tr = np.delete(y, i)
        preds_mean[i] = y_tr.mean()
    all_preds["MeanBaseline"] = preds_mean

    # --- Sklearn models ---
    for name, factory in models_dict.items():
        log.info("Running %s LOOCV ...", name)
        preds = np.zeros(n)
        for i in range(n):
            X_tr = np.delete(X, i, axis=0)
            y_tr = np.delete(y, i)
            X_te = X[i:i+1]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = factory()
            model.fit(X_tr_s, y_tr)
            preds[i] = model.predict(X_te_s)[0]
        all_preds[name] = preds
        log.info("  %s LOOCV R² = %.4f", name, r2_score(y, preds))

    # --- DEIN (custom DL) ---
    if HAS_TORCH:
        log.info("Running DEIN (Custom DL) LOOCV ...")
        preds_dein = train_dein_loocv(X, y, feat_names)
        all_preds["DEIN"] = preds_dein
        log.info("  DEIN LOOCV R² = %.4f", r2_score(y, preds_dein))
    else:
        log.warning("PyTorch not available — skipping DEIN")

    # --- Weighted Ensemble (poor-man's stacking) ---
    log.info("Computing Weighted Ensemble ...")
    # Weight by max(0, R²) of each model (excluding baseline)
    model_names_for_ens = [k for k in all_preds if k != "MeanBaseline"]
    r2s = {k: max(0, r2_score(y, all_preds[k])) for k in model_names_for_ens}
    total_r2 = sum(r2s.values()) + 1e-9
    weights = {k: r2s[k] / total_r2 for k in model_names_for_ens}

    preds_ens = np.zeros(n)
    for k, w in weights.items():
        preds_ens += w * all_preds[k]
    all_preds["WeightedEnsemble"] = preds_ens
    log.info("  Ensemble weights: %s",
             {k: f"{w:.3f}" for k, w in weights.items()})
    log.info("  Ensemble LOOCV R² = %.4f", r2_score(y, preds_ens))

    # --- Compute metrics ---
    rows = []
    for name, preds in all_preds.items():
        rmse = np.sqrt(mean_squared_error(y, preds))
        mae  = mean_absolute_error(y, preds)
        r2   = r2_score(y, preds)
        r_pe = pearsonr(y, preds)[0]
        rho  = spearmanr(y, preds)[0]
        # Cohen's f² = R²/(1-R²)
        f2 = r2 / (1 - r2 + 1e-9) if r2 > 0 else 0.0
        rows.append({
            "Model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4),
            "R2": round(r2, 4), "Pearson_r": round(r_pe, 4),
            "Spearman_rho": round(rho, 4), "Cohen_f2": round(f2, 4),
            "f2_size": ("large" if f2 >= 0.35 else
                        "medium" if f2 >= 0.15 else
                        "small" if f2 >= 0.02 else "negligible"),
        })

    metrics_df = pd.DataFrame(rows).sort_values("R2", ascending=False)
    metrics_df.to_csv(out_dir / "model_comparison.csv", index=False)

    # Save predictions
    pred_df = pd.DataFrame(all_preds, index=X_df.index)
    pred_df["actual"] = y
    pred_df.to_csv(out_dir / "all_predictions.csv")

    return all_preds, metrics_df, weights


# ═══════════════════════════════════════════════════════════════════════
# WILCOXON SIGNED-RANK TESTS (Bonferroni corrected)
# ═══════════════════════════════════════════════════════════════════════

def wilcoxon_comparison(all_preds: dict, y: np.ndarray,
                         out_dir: Path) -> pd.DataFrame:
    """Pairwise Wilcoxon on absolute residuals, Bonferroni corrected."""
    names = list(all_preds.keys())
    residuals = {k: np.abs(y - all_preds[k]) for k in names}
    n_tests = len(names) * (len(names) - 1) // 2
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            try:
                stat, p = wilcoxon(residuals[a], residuals[b])
                p_corr = min(1.0, p * n_tests)
                rows.append({
                    "Model_A": a, "Model_B": b,
                    "W_stat": round(stat, 2),
                    "p_raw": round(p, 4),
                    "p_bonferroni": round(p_corr, 4),
                    "significant": p_corr < 0.05,
                    "better": a if np.mean(residuals[a]) < np.mean(residuals[b]) else b,
                })
            except Exception as e:
                rows.append({
                    "Model_A": a, "Model_B": b,
                    "W_stat": np.nan, "p_raw": np.nan,
                    "p_bonferroni": np.nan,
                    "significant": False, "better": "—",
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "wilcoxon_comparison.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE C: SHAP CONVERGENCE
# ═══════════════════════════════════════════════════════════════════════

def compute_shap_all(X_df: pd.DataFrame, y: np.ndarray,
                      out_dir: Path) -> pd.DataFrame:
    """Compute SHAP for each model, rank features, find convergent set."""
    if not HAS_SHAP:
        log.warning("SHAP not available — skipping convergence analysis")
        return pd.DataFrame()

    X = X_df.values
    feat_names = list(X_df.columns)
    n = len(y)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s_df = pd.DataFrame(X_s, columns=feat_names)

    shap_ranks = {}
    shap_values_all = {}

    # --- Ridge ---
    log.info("SHAP: Ridge ...")
    m = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    m.fit(X_s, y)
    explainer = shap.LinearExplainer(m, X_s_df)
    sv = explainer.shap_values(X_s_df)
    mean_abs = np.abs(sv).mean(axis=0)
    shap_values_all["Ridge"] = mean_abs
    rank = np.argsort(-mean_abs)
    shap_ranks["Ridge"] = {feat_names[r]: i+1 for i, r in enumerate(rank)}

    # --- ElasticNet ---
    log.info("SHAP: ElasticNet ...")
    m2 = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                       max_iter=5000, random_state=42)
    m2.fit(X_s, y)
    explainer2 = shap.LinearExplainer(m2, X_s_df)
    sv2 = explainer2.shap_values(X_s_df)
    mean_abs2 = np.abs(sv2).mean(axis=0)
    shap_values_all["ElasticNet"] = mean_abs2
    rank2 = np.argsort(-mean_abs2)
    shap_ranks["ElasticNet"] = {feat_names[r]: i+1 for i, r in enumerate(rank2)}

    # --- BayesianRidge ---
    log.info("SHAP: BayesianRidge ...")
    m3 = BayesianRidge(max_iter=500)
    m3.fit(X_s, y)
    explainer3 = shap.LinearExplainer(m3, X_s_df)
    sv3 = explainer3.shap_values(X_s_df)
    mean_abs3 = np.abs(sv3).mean(axis=0)
    shap_values_all["BayesianRidge"] = mean_abs3
    rank3 = np.argsort(-mean_abs3)
    shap_ranks["BayesianRidge"] = {feat_names[r]: i+1 for i, r in enumerate(rank3)}

    # --- RandomForest ---
    log.info("SHAP: RandomForest ...")
    m4 = RandomForestRegressor(n_estimators=200, max_depth=3,
                                min_samples_leaf=5, random_state=42)
    m4.fit(X_s, y)
    explainer4 = shap.TreeExplainer(m4)
    sv4 = explainer4.shap_values(X_s_df)
    mean_abs4 = np.abs(sv4).mean(axis=0)
    shap_values_all["RandomForest"] = mean_abs4
    rank4 = np.argsort(-mean_abs4)
    shap_ranks["RandomForest"] = {feat_names[r]: i+1 for i, r in enumerate(rank4)}

    # --- XGBoost ---
    if HAS_XGB:
        log.info("SHAP: XGBoost ...")
        m5 = xgb.XGBRegressor(n_estimators=100, max_depth=2,
                                learning_rate=0.05, min_child_weight=5,
                                reg_alpha=1.0, reg_lambda=2.0,
                                random_state=42, verbosity=0)
        m5.fit(X_s, y)
        explainer5 = shap.TreeExplainer(m5)
        sv5 = explainer5.shap_values(X_s_df)
        mean_abs5 = np.abs(sv5).mean(axis=0)
        shap_values_all["XGBoost"] = mean_abs5
        rank5 = np.argsort(-mean_abs5)
        shap_ranks["XGBoost"] = {feat_names[r]: i+1
                                  for i, r in enumerate(rank5)}

    # --- Convergence analysis ---
    n_models = len(shap_ranks)
    log.info("SHAP convergence across %d models ...", n_models)
    convergence_rows = []
    for feat in feat_names:
        ranks = [shap_ranks[m].get(feat, len(feat_names))
                 for m in shap_ranks]
        in_top6 = sum(1 for r in ranks if r <= 6)
        convergence_rows.append({
            "Feature": feat,
            "Mean_SHAP_rank": round(np.mean(ranks), 1),
            "Min_rank": min(ranks),
            "Max_rank": max(ranks),
            **{f"Rank_{m}": shap_ranks[m].get(feat, "—")
               for m in shap_ranks},
            "In_top6_count": in_top6,
            "Convergent": in_top6 >= (n_models - 1),  # in top-6 of all but 1
        })

    conv_df = pd.DataFrame(convergence_rows)
    conv_df = conv_df.sort_values("Mean_SHAP_rank")
    conv_df.to_csv(out_dir / "shap_convergence.csv", index=False)

    convergent = conv_df[conv_df["Convergent"]]["Feature"].tolist()
    log.info("Convergent features (top-6 in ≥%d models): %s",
             n_models - 1, convergent)

    # --- SHAP summary plot ---
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6),
                              sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, (mname, vals) in zip(axes, shap_values_all.items()):
        order = np.argsort(vals)[::-1]
        ax.barh([feat_names[i] for i in order], vals[order], color="#e74c3c")
        ax.set_title(mname, fontsize=10)
        ax.set_xlabel("Mean |SHAP|")
    plt.suptitle("SHAP Feature Importance Across Models", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_all_models.png", dpi=150)
    plt.close(fig)

    # --- Convergence heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names_list = list(shap_ranks.keys())
    rank_matrix = np.array([
        [shap_ranks[m].get(f, len(feat_names)) for m in model_names_list]
        for f in feat_names
    ])
    # Sort by mean rank
    sort_idx = np.argsort(rank_matrix.mean(axis=1))
    rank_sorted = rank_matrix[sort_idx]
    feat_sorted = [feat_names[i] for i in sort_idx]

    im = ax.imshow(rank_sorted, cmap="RdYlGn_r", aspect="auto",
                    vmin=1, vmax=len(feat_names))
    ax.set_xticks(range(len(model_names_list)))
    ax.set_xticklabels(model_names_list, rotation=45, ha="right")
    ax.set_yticks(range(len(feat_sorted)))
    ax.set_yticklabels(feat_sorted, fontsize=8)
    for i in range(len(feat_sorted)):
        for j in range(len(model_names_list)):
            ax.text(j, i, str(rank_sorted[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if rank_sorted[i, j] <= 3 else "black")
    plt.colorbar(im, label="SHAP Rank (1=most important)")
    ax.set_title("Feature Rank Convergence Across Models\n"
                 "(Green=low rank=important)")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_convergence_heatmap.png", dpi=150)
    plt.close(fig)

    return conv_df


# ═══════════════════════════════════════════════════════════════════════
# PHASE D: FINAL EI SCORE
# ═══════════════════════════════════════════════════════════════════════

def compute_ei_score(X_df: pd.DataFrame, y: np.ndarray,
                      final_feats: list, pids: list,
                      out_dir: Path) -> tuple:
    """
    LOOCV composite EI score.
    For each held-out participant, weights = Pearson r of each feature
    with exam score in the training fold (n-1).
    """
    n = len(y)
    X = X_df[final_feats].fillna(0).values
    feat_names = final_feats

    ei_scores = np.zeros(n)
    weight_records = []

    for i in range(n):
        X_tr = np.delete(X, i, axis=0)
        y_tr = np.delete(y, i)

        # Compute Pearson r for each feature in training fold
        weights = []
        for j in range(X_tr.shape[1]):
            r, _ = pearsonr(X_tr[:, j], y_tr)
            weights.append(r)
        weights = np.array(weights)

        # Direction-correct: features with known negative direction
        # get their weight sign preserved naturally by Pearson r
        # Normalise to sum of absolute weights = 1
        w_abs = np.abs(weights)
        w_norm = weights / (w_abs.sum() + 1e-9)

        # Standardise features using training fold
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X[i:i+1])

        # Composite = weighted sum of z-scored features
        ei_scores[i] = float(X_te_s @ w_norm)
        weight_records.append(dict(zip(feat_names, w_norm)))

    # Scale to [0, 1]
    ei_min, ei_max = ei_scores.min(), ei_scores.max()
    if ei_max - ei_min > 1e-9:
        ei_scores_scaled = (ei_scores - ei_min) / (ei_max - ei_min)
    else:
        ei_scores_scaled = np.full(n, 0.5)

    # --- Correlations ---
    r_pe, p_pe = pearsonr(ei_scores_scaled, y)
    rho, p_rho = spearmanr(ei_scores_scaled, y)

    # Fisher z-transform 95% CI
    z = 0.5 * np.log((1 + r_pe) / (1 - r_pe + 1e-9))
    se = 1 / np.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    r_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
    r_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)

    log.info("EI Score — Pearson r = %.3f [%.3f, %.3f], p = %.4f",
             r_pe, r_lo, r_hi, p_pe)
    log.info("EI Score — Spearman ρ = %.3f, p = %.4f", rho, p_rho)

    # --- Partial correlation (controlling for strongest predictor) ---
    strongest = final_feats[0]  # c2_stress_peak_attenuation
    x_ctrl = X_df[strongest].fillna(0).values
    # Partial r between EI_score and y, controlling for x_ctrl
    r_ey, _ = pearsonr(ei_scores_scaled, y)
    r_ec, _ = pearsonr(ei_scores_scaled, x_ctrl)
    r_cy, _ = pearsonr(x_ctrl, y)
    partial_r = (r_ey - r_ec * r_cy) / (
        np.sqrt((1 - r_ec**2) * (1 - r_cy**2)) + 1e-9)
    # Approximate p-value for partial r
    t_stat = partial_r * np.sqrt((n - 3) / (1 - partial_r**2 + 1e-9))
    from scipy.stats import t as t_dist
    p_partial = 2 * (1 - t_dist.cdf(abs(t_stat), n - 3))

    log.info("Partial r (controlling %s) = %.3f, p = %.4f",
             strongest, partial_r, p_partial)

    # --- Save EI scores ---
    ei_df = pd.DataFrame({
        "Participant": pids,
        "EI_Score_raw": ei_scores,
        "EI_Score_scaled": ei_scores_scaled,
        "Exam_Score": y,
    })
    ei_df.to_csv(out_dir / "ei_scores.csv", index=False)

    # --- Weight stability ---
    w_df = pd.DataFrame(weight_records, index=pids)
    w_df.to_csv(out_dir / "loocv_weights.csv")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: EI vs exam
    ax1 = axes[0]
    ax1.scatter(ei_scores_scaled, y, c="#2c3e50", s=60, alpha=0.7,
                edgecolors="k", zorder=3)
    z_line = np.polyfit(ei_scores_scaled, y, 1)
    x_line = np.linspace(0, 1, 100)
    ax1.plot(x_line, np.polyval(z_line, x_line), "r--", lw=2)
    ax1.set_xlabel("LOOCV Composite EI Score (scaled 0–1)")
    ax1.set_ylabel("Exam Score (normalised)")
    ax1.set_title(f"Composite EI vs Exam Performance\n"
                  f"r = {r_pe:.3f} [{r_lo:.3f}, {r_hi:.3f}], "
                  f"p = {p_pe:.4f}\n"
                  f"Partial r (ctrl {strongest}) = {partial_r:.3f}")
    # Label participants
    for i, pid in enumerate(pids):
        ax1.annotate(pid, (ei_scores_scaled[i], y[i]),
                     fontsize=5, alpha=0.5)

    # Weight stability
    ax2 = axes[1]
    w_means = w_df.mean()
    w_stds  = w_df.std()
    colors = ["#e74c3c" if abs(m) > s else "#95a5a6"
              for m, s in zip(w_means, w_stds)]
    ax2.barh(w_means.index, w_means.values, xerr=w_stds.values,
             color=colors, capsize=3)
    ax2.axvline(0, color="k", lw=0.5)
    ax2.set_xlabel("Mean LOOCV weight ± SD")
    ax2.set_title("Feature Weight Stability\n(Red = |mean| > SD = stable)")

    plt.tight_layout()
    fig.savefig(out_dir / "ei_score_analysis.png", dpi=150)
    plt.close(fig)

    return ei_scores_scaled, {
        "Pearson_r": r_pe, "r_lo": r_lo, "r_hi": r_hi,
        "p_pearson": p_pe,
        "Spearman_rho": rho, "p_spearman": p_rho,
        "Partial_r": partial_r, "p_partial": p_partial,
        "Controlled_for": strongest,
    }


# ═══════════════════════════════════════════════════════════════════════
# DIMENSION COMPOSITE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def dimension_composites(X_df: pd.DataFrame, y: np.ndarray,
                          pids: list, out_dir: Path):
    """Compute per-dimension composite scores and their criterion validity."""
    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns,
                        index=X_df.index)

    rows = []
    dim_scores = {}
    for dim, feats in VALIDATED_FEATURES.items():
        avail = [f for f in feats if f in X_s.columns]
        if not avail:
            continue
        # Direction-corrected z-scores
        corrected = X_s[avail].copy()
        for f in avail:
            corrected[f] *= FEAT_SIGN.get(f, 1)
        composite = corrected.mean(axis=1).values
        dim_scores[dim] = composite

        r, p = pearsonr(composite, y)
        rho, p_rho = spearmanr(composite, y)
        rows.append({
            "Dimension": dim, "N_features": len(avail),
            "Features": ", ".join(avail),
            "Pearson_r": round(r, 3), "p_pearson": round(p, 4),
            "Spearman_rho": round(rho, 3), "p_spearman": round(p_rho, 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dimension_composites.csv", index=False)
    log.info("Dimension composites:\n%s", df.to_string(index=False))
    return df, dim_scores


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_model_comparison(metrics_df: pd.DataFrame, out_dir: Path):
    """Bar chart comparing all models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    metrics_df_sorted = metrics_df.sort_values("R2", ascending=True)

    for ax, metric, title in zip(axes,
                                  ["R2", "RMSE", "Pearson_r"],
                                  ["R² (higher=better)",
                                   "RMSE (lower=better)",
                                   "Pearson r (higher=better)"]):
        colors = []
        for _, r in metrics_df_sorted.iterrows():
            if r["Model"] == "MeanBaseline":
                colors.append("#95a5a6")
            elif r["Model"] == "DEIN":
                colors.append("#9b59b6")
            elif r["Model"] == "WeightedEnsemble":
                colors.append("#f39c12")
            else:
                colors.append("#e74c3c")

        bars = ax.barh(metrics_df_sorted["Model"],
                       metrics_df_sorted[metric], color=colors)
        for bar, val in zip(bars, metrics_df_sorted[metric]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)
        ax.set_title(title)
        if metric == "R2":
            ax.axvline(0, color="k", ls="--", lw=0.8)

    plt.suptitle("Model Comparison — LOOCV on 11 Validated Features (n=41)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_all_scatters(all_preds: dict, y: np.ndarray, out_dir: Path):
    """Scatter plots for all models."""
    n_models = len(all_preds)
    cols = 4
    rows_n = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5*cols, 5*rows_n))
    axes = axes.flatten() if n_models > 1 else [axes]

    for ax, (name, preds) in zip(axes, all_preds.items()):
        ax.scatter(y, preds, s=40, alpha=0.7, edgecolors="k")
        lims = [min(y.min(), preds.min())-0.05,
                max(y.max(), preds.max())+0.05]
        ax.plot(lims, lims, "r--", lw=1.5)
        r, _ = pearsonr(y, preds)
        r2 = r2_score(y, preds)
        ax.set_title(f"{name}\nr={r:.3f}, R²={r2:.3f}", fontsize=9)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    for ax in axes[n_models:]:
        ax.set_visible(False)

    plt.suptitle("LOOCV Predicted vs Actual — All Models", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "all_model_scatters.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════

def write_final_report(metrics_df, wilcoxon_df, conv_df, ei_stats,
                        final_feats, out_dir):
    """Write comprehensive text report."""
    lines = [
        "="*76,
        "  FINAL EI PIPELINE — MULTI-ARCHITECTURE VALIDATION REPORT",
        "="*76, "",
        "─"*76,
        "  PHASE B: MODEL COMPARISON (LOOCV, 11 validated features)",
        "─"*76,
        "",
        f"  {'Model':<22} {'R²':>7} {'RMSE':>7} {'MAE':>7} "
        f"{'r':>7} {'ρ':>7} {'f²':>7} {'Size':>10}",
        "  " + "─"*70,
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"  {r['Model']:<22} {r['R2']:>7.3f} {r['RMSE']:>7.4f} "
            f"{r['MAE']:>7.4f} {r['Pearson_r']:>7.3f} "
            f"{r['Spearman_rho']:>7.3f} {r['Cohen_f2']:>7.3f} "
            f"{r['f2_size']:>10}")

    # Wilcoxon summary
    if wilcoxon_df is not None and not wilcoxon_df.empty:
        sig_pairs = wilcoxon_df[wilcoxon_df["significant"]]
        lines += ["",
            "─"*76,
            f"  WILCOXON SIGNED-RANK (Bonferroni, {len(wilcoxon_df)} pairs)",
            "─"*76,
            f"  Significant pairs: {len(sig_pairs)} / {len(wilcoxon_df)}",
        ]
        if sig_pairs.empty:
            lines.append("  → No model significantly outperforms another")
            lines.append("    This CONFIRMS: the feature set, not the model,")
            lines.append("    is the limiting factor (as predicted by project plan)")
        else:
            for _, r in sig_pairs.iterrows():
                lines.append(f"  {r['Model_A']} vs {r['Model_B']}: "
                             f"p={r['p_bonferroni']:.4f} → {r['better']} better")

    # SHAP convergence
    if conv_df is not None and not conv_df.empty:
        conv_feats = conv_df[conv_df["Convergent"]]["Feature"].tolist()
        lines += ["",
            "─"*76,
            "  PHASE C: SHAP CONVERGENCE",
            "─"*76,
            f"  Features convergent across models: {conv_feats}",
            "",
            "  Full ranking:",
        ]
        for _, r in conv_df.head(11).iterrows():
            tag = " ★" if r["Convergent"] else ""
            lines.append(f"    {r['Feature']:<42} "
                         f"mean_rank={r['Mean_SHAP_rank']:<5} "
                         f"in_top6={r['In_top6_count']}{tag}")

    # EI score
    lines += ["",
        "─"*76,
        "  PHASE D: FINAL COMPOSITE EI SCORE",
        "─"*76,
        f"  Features used: {final_feats}",
        f"  Pearson r  = {ei_stats['Pearson_r']:+.3f} "
        f"[{ei_stats['r_lo']:+.3f}, {ei_stats['r_hi']:+.3f}], "
        f"p = {ei_stats['p_pearson']:.4f}",
        f"  Spearman ρ = {ei_stats['Spearman_rho']:+.3f}, "
        f"p = {ei_stats['p_spearman']:.4f}",
        f"  Partial r  = {ei_stats['Partial_r']:+.3f} "
        f"(controlling {ei_stats['Controlled_for']}), "
        f"p = {ei_stats['p_partial']:.4f}",
        "",
        "─"*76,
        "  INTERPRETATION",
        "─"*76,
    ]

    if ei_stats["p_pearson"] < 0.05:
        lines.append("  ✓ H₁ SUPPORTED: Significant positive correlation between")
        lines.append("    behavioral EI composite and examination performance")
    else:
        lines.append("  ✗ H₀ NOT REJECTED: Insufficient evidence for correlation")

    if ei_stats["p_partial"] < 0.05:
        lines.append("  ✓ Composite adds explanatory value beyond the strongest")
        lines.append(f"    individual predictor ({ei_stats['Controlled_for']})")
    else:
        lines.append("  → Composite does NOT add significant value beyond")
        lines.append(f"    {ei_stats['Controlled_for']} alone")

    lines += ["",
        "─"*76,
        "  PILOT CAVEATS",
        "─"*76,
        "  • n=41: 80% power only for |r| ≥ 0.42",
        "  • Winner's curse likely inflates effect sizes",
        "  • DEIN result is architectural exploration, not a performance claim",
        "  • All findings exploratory — replicate with n ≥ 150",
        "",
        "─"*76,
        "  OUTPUT FILES",
        "─"*76,
        "  model_comparison.csv         LOOCV metrics for all 8 models",
        "  all_predictions.csv          Per-participant predictions",
        "  wilcoxon_comparison.csv      Pairwise Wilcoxon tests",
        "  shap_convergence.csv         Feature ranks across models",
        "  shap_all_models.png          SHAP bar charts",
        "  shap_convergence_heatmap.png Rank convergence visualization",
        "  ei_scores.csv                Final EI scores per participant",
        "  loocv_weights.csv            Weight stability across folds",
        "  dimension_composites.csv     D2/D3/D4 composite validity",
        "  model_comparison.png         Model R²/RMSE/r comparison",
        "  all_model_scatters.png       Predicted vs actual scatters",
        "  ei_score_analysis.png        EI vs exam + weight stability",
        "  final_report.txt             This report",
        "="*76,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "final_report.txt").write_text(report, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main(data_dir: str):
    dp = Path(data_dir)
    out_dir = dp / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("="*60)
    log.info("  EI FINAL PIPELINE — Multi-Architecture Validation")
    log.info("="*60)

    # ── Phase A: Load data ──
    X_df, y, pids, feat_names = load_data(dp)
    log.info("Features: %s", feat_names)
    log.info("y range: [%.3f, %.3f], mean=%.3f", y.min(), y.max(), y.mean())

    # ── Phase B: LOOCV all models ──
    log.info("\n── PHASE B: Multi-Model LOOCV ──")
    all_preds, metrics_df, ens_weights = loocv_all_models(
        X_df, y, feat_names, out_dir)
    plot_model_comparison(metrics_df, out_dir)
    plot_all_scatters(all_preds, y, out_dir)

    # ── Wilcoxon ──
    log.info("\n── WILCOXON COMPARISON ──")
    wilcoxon_df = wilcoxon_comparison(all_preds, y, out_dir)
    sig = wilcoxon_df[wilcoxon_df["significant"]]
    log.info("Significant pairs: %d / %d", len(sig), len(wilcoxon_df))

    # ── Phase C: SHAP convergence ──
    log.info("\n── PHASE C: SHAP Convergence ──")
    conv_df = compute_shap_all(X_df, y, out_dir)

    # Determine final 4 features for EI score
    if not conv_df.empty:
        convergent = conv_df[conv_df["Convergent"]]["Feature"].tolist()
        if len(convergent) >= 4:
            final_feats = convergent[:4]
        else:
            # Fill from top by mean rank
            top_rank = conv_df.head(4)["Feature"].tolist()
            final_feats = convergent + [f for f in top_rank
                                         if f not in convergent]
            final_feats = final_feats[:4]
    else:
        # Fallback: use the 4 with highest criterion r
        final_feats = [
            "c2_stress_peak_attenuation",
            "c3_productive_absorbing_probability",
            "c3_stress_inertia_during_blocks",
            "c4_arousal_trend",
        ]
    log.info("Final 4 features for EI composite: %s", final_feats)

    # ── Dimension composites ──
    log.info("\n── DIMENSION COMPOSITES ──")
    dim_df, dim_scores = dimension_composites(X_df, y, pids, out_dir)

    # ── Phase D: EI score ──
    log.info("\n── PHASE D: Final EI Score ──")
    ei_scores, ei_stats = compute_ei_score(
        X_df, y, final_feats, pids, out_dir)

    # ── Final report ──
    write_final_report(metrics_df, wilcoxon_df, conv_df, ei_stats,
                        final_feats, out_dir)
    log.info("All outputs → %s", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    main(ap.parse_args().data_dir)
