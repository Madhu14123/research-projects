"""
EI Final Pipeline — Multi-Architecture Validation + Custom DL + EI Score
=========================================================================
UPDATED: Expanded from 11 → top-30 features across all 5 dimensions (D1–D5)
         DEIN architecture updated to handle D1 and D5 branches.

Phases:
  A: Load top-30 features ranked by |Pearson r| with criterion
  B: 8-model LOOCV comparison (Mean, Ridge, ElasticNet, BayesianRidge,
     RF, XGBoost, Custom DEIN, Weighted Ensemble)
  C: SHAP convergence across models → select final features
  D: LOOCV composite EI score, partial correlations, Wilcoxon tests

Reads:
  results_pdf_features/all_features_116.csv  (from prior feature extraction)
  Exam Scores.xlsx                           (criterion variable)

Outputs → results_final_pipeline_top30/
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
OUT_DIR          = "results_final_pipeline_top30"

# ── TOP-30 FEATURES ────────────────────────────────────────────────────
# Ranked by |Pearson r| with exam score from criterion validity phase.
# * = BCa 95% CI excludes 0 (significant by bootstrap)
# Organised by dimension (D1–D5) for DEIN branch routing.
# NOTE: features labelled c1_/c5_ are now included (D1 = Emotion Perception,
#       D5 = Emotion-Cognition Alignment).  If a feature is absent in the
#       feature matrix it is silently skipped (see load_data).
# ──────────────────────────────────────────────────────────────────────

VALIDATED_FEATURES = {
    # D1 — Emotion Perception
    "D1": [
        "c1_emotion_richness",           # r=-0.253
        "c1_sample_entropy_AU",          # r≈-0.24 (stability chart)
        "c1_skewness_balance",           # r≈-0.23 (stability chart)
    ],

    # D2 — Emotion Regulation
    "D2": [
        "c2_stress_peak_attenuation",        # r=+0.569 *, large, stability=92.6%
        "c2_stress_recovery_latency",        # r=-0.348 *
        "c2_emotion_variability_SD",         # r=-0.316 *
        "c2_anticipatory_regulation",        # r=-0.293
        "c2_reappraisal_proxy_coactivation", # r=+0.240 *
        "c2_stress_inertia",                 # r=-0.240
        "c2_SAM_facial_discrepancy_reduction", # r=-0.210
        "c2_valence_arousal_regulation_coupling", # stability chart
        "c2_recovery_completeness",          # stability chart
        "c2_lip_press_index",                # stability chart
    ],

    # D3 — Adaptive Persistence
    "D3": [
        "c3_productive_absorbing_probability",  # r=+0.451 *
        "c3_stress_inertia_during_blocks",      # r=-0.428 *
        "c3_total_productive_time_ratio",       # r=+0.404
        "c3_forward_stage_momentum",            # r=-0.367 *
        "c3_effective_time_efficiency",         # r=+0.227
        "c3_bounce_back_index",                 # r=-0.225 *
        "c3_recovery_acceleration",             # r=+0.203
        "c3_block_exit_diversity",              # r=-0.210
        "c3_late_task_engagement_ratio",        # stability chart
        "c3_second_order_block_escape",         # stability chart
        "c3_block_to_production_rate",          # stability chart
        "c3_post_block_engagement_quality",     # stability chart
    ],

    # D4 — Affective Self-Awareness
    "D4": [
        "c4_arousal_trend",                  # r=+0.376 *
        "c4_arousal_optimality",             # r=+0.363 *
        "c4_arousal_midrange_proportion",    # r=+0.335 *
        "c4_arousal_stability",              # r=+0.274
        "c4_valence_arousal_discrimination", # r=+0.266
        "c4_positive_reporting_bias",        # r=+0.244
        "c4_pride_during_production",        # r=-0.222
        "c4_block_stage_awareness",          # stability chart
    ],

    # D5 — Emotion-Cognition Alignment
    "D5": [
        "c5_mastery_mode_ratio",         # r=+0.214
        "c5_confusion_resolution_rate",  # r=+0.205
        "c5_emotion_stage_differentiation", # stability chart
        "c5_engagement_recovery_speed",  # stability chart
        "c5_arousal_SAM_task_match",     # stability chart
        "c5_pride_after_production",     # stability chart
    ],
}

ALL_VALIDATED = []
for feats in VALIDATED_FEATURES.values():
    ALL_VALIDATED.extend(feats)

# Direction signs (positive = higher is better for exam score)
FEAT_SIGN = {
    # D1
    "c1_emotion_richness":              -1,
    "c1_sample_entropy_AU":             -1,
    "c1_skewness_balance":              -1,

    # D2
    "c2_stress_peak_attenuation":        +1,
    "c2_stress_recovery_latency":        -1,
    "c2_emotion_variability_SD":         -1,
    "c2_anticipatory_regulation":        -1,
    "c2_reappraisal_proxy_coactivation": +1,
    "c2_stress_inertia":                 -1,
    "c2_SAM_facial_discrepancy_reduction": -1,
    "c2_valence_arousal_regulation_coupling": +1,
    "c2_recovery_completeness":          +1,
    "c2_lip_press_index":                +1,

    # D3
    "c3_productive_absorbing_probability": +1,
    "c3_stress_inertia_during_blocks":   -1,
    "c3_total_productive_time_ratio":    +1,
    "c3_forward_stage_momentum":         -1,
    "c3_effective_time_efficiency":      +1,
    "c3_bounce_back_index":              -1,
    "c3_recovery_acceleration":          +1,
    "c3_block_exit_diversity":           -1,
    "c3_late_task_engagement_ratio":     +1,
    "c3_second_order_block_escape":      +1,
    "c3_block_to_production_rate":       +1,
    "c3_post_block_engagement_quality":  +1,

    # D4
    "c4_arousal_trend":                  +1,
    "c4_arousal_optimality":             +1,
    "c4_arousal_midrange_proportion":    +1,
    "c4_arousal_stability":              +1,
    "c4_valence_arousal_discrimination": +1,
    "c4_positive_reporting_bias":        +1,
    "c4_pride_during_production":        -1,
    "c4_block_stage_awareness":          +1,

    # D5
    "c5_mastery_mode_ratio":             +1,
    "c5_confusion_resolution_rate":      +1,
    "c5_emotion_stage_differentiation":  +1,
    "c5_engagement_recovery_speed":      +1,
    "c5_arousal_SAM_task_match":         +1,
    "c5_pride_after_production":         +1,
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

    # ── CHANGED: load top-30 requested features, skip any missing ──────
    available = [f for f in ALL_VALIDATED if f in fm.columns]
    missing   = [f for f in ALL_VALIDATED if f not in fm.columns]
    log.info("Features requested: %d  |  available: %d  |  missing: %d",
             len(ALL_VALIDATED), len(available), len(missing))
    if missing:
        log.warning("Missing features (dropped): %s", missing)

    X_df = fm.loc[common, available].fillna(0)
    y    = np.array([scores[p] for p in common])

    log.info("Final feature matrix: %d participants × %d features",
             X_df.shape[0], X_df.shape[1])
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
        n_estimators=300, max_depth=3, min_samples_leaf=5,
        max_features="sqrt", random_state=42)

    if HAS_XGB:
        models["XGBoost"] = lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=1.0, reg_lambda=2.0,
            random_state=42, verbosity=0)
    return models


# ═══════════════════════════════════════════════════════════════════════
# DEIN: DIMENSION-ENCODED EI NETWORK  (updated for 5 dimensions)
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
        Dimension-Encoded EI Network — updated for 5 branches (D1–D5).

        Architecture:
          D1 branch (n_d1 feats → 8 → 4)
          D2 branch (n_d2 feats → 16 → 8)
          D3 branch (n_d3 feats → 16 → 8)
          D4 branch (n_d4 feats → 12 → 6)
          D5 branch (n_d5 feats → 8 → 4)
                    ↓ concatenate (4+8+8+6+4 = 30)
          Fusion: 30 → 16 → ReLU → Dropout(0.4) → 8 → ReLU → 4 → 1 → Sigmoid
        """
        def __init__(self, feat_names: list, y_min=0.0, y_max=1.0):
            super().__init__()
            self.feat_names = feat_names
            self.y_min = y_min
            self.y_max = y_max

            # Identify indices for each dimension by prefix
            self.d1_idx = [i for i, f in enumerate(feat_names) if f.startswith("c1_")]
            self.d2_idx = [i for i, f in enumerate(feat_names) if f.startswith("c2_")]
            self.d3_idx = [i for i, f in enumerate(feat_names) if f.startswith("c3_")]
            self.d4_idx = [i for i, f in enumerate(feat_names) if f.startswith("c4_")]
            self.d5_idx = [i for i, f in enumerate(feat_names) if f.startswith("c5_")]

            n_d1 = max(len(self.d1_idx), 1)
            n_d2 = max(len(self.d2_idx), 1)
            n_d3 = max(len(self.d3_idx), 1)
            n_d4 = max(len(self.d4_idx), 1)
            n_d5 = max(len(self.d5_idx), 1)

            # Branch output sizes
            self.out_d1 = 4
            self.out_d2 = 8
            self.out_d3 = 8
            self.out_d4 = 6
            self.out_d5 = 4
            fusion_in   = self.out_d1 + self.out_d2 + self.out_d3 + self.out_d4 + self.out_d5

            self.branch_d1 = DimensionBranch(n_d1,  8, self.out_d1)
            self.branch_d2 = DimensionBranch(n_d2, 16, self.out_d2)
            self.branch_d3 = DimensionBranch(n_d3, 16, self.out_d3)
            self.branch_d4 = DimensionBranch(n_d4, 12, self.out_d4)
            self.branch_d5 = DimensionBranch(n_d5,  8, self.out_d5)

            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, 16),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 1),
                nn.Sigmoid(),
            )

        def _safe_slice(self, x, idx):
            """Extract dimension slice; return zeros if dimension absent."""
            if idx:
                return x[:, idx]
            # Fallback: single zero column
            return torch.zeros(x.size(0), 1, device=x.device)

        def forward(self, x):
            h1 = self.branch_d1(self._safe_slice(x, self.d1_idx))
            h2 = self.branch_d2(self._safe_slice(x, self.d2_idx))
            h3 = self.branch_d3(self._safe_slice(x, self.d3_idx))
            h4 = self.branch_d4(self._safe_slice(x, self.d4_idx))
            h5 = self.branch_d5(self._safe_slice(x, self.d5_idx))
            fused = torch.cat([h1, h2, h3, h4, h5], dim=1)
            out = self.fusion(fused)
            return out * (self.y_max - self.y_min) + self.y_min

    def train_dein_loocv(X: np.ndarray, y: np.ndarray,
                          feat_names: list, n_epochs=300, lr=0.003,
                          weight_decay=0.01) -> np.ndarray:
        """Train DEIN under LOOCV. Returns predictions array."""
        n = len(y)
        preds = np.zeros(n)
        y_min, y_max = float(y.min()) - 0.05, float(y.max()) + 0.05

        for i in range(n):
            X_tr = np.delete(X, i, axis=0)
            y_tr = np.delete(y, i)
            X_te = X[i:i+1]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            Xt = torch.FloatTensor(X_tr_s)
            yt = torch.FloatTensor(y_tr).reshape(-1, 1)
            Xv = torch.FloatTensor(X_te_s)

            model = DEIN(feat_names, y_min, y_max)
            optimizer = torch.optim.AdamW(model.parameters(),
                                           lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss()

            model.train()
            best_loss  = float("inf")
            patience_c = 0
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                pred = model(Xt)
                loss = criterion(pred, yt)
                loss.backward()
                optimizer.step()

                if loss.item() < best_loss - 1e-5:
                    best_loss  = loss.item()
                    patience_c = 0
                else:
                    patience_c += 1
                if patience_c >= 50:
                    break

            model.eval()
            with torch.no_grad():
                preds[i] = model(Xv).item()

        return preds


# ═══════════════════════════════════════════════════════════════════════
# PHASE B: LOOCV FOR ALL MODELS
# ═══════════════════════════════════════════════════════════════════════

def loocv_all_models(X_df: pd.DataFrame, y: np.ndarray,
                      feat_names: list, out_dir: Path):
    n = len(y)
    X = X_df.values
    models_dict = make_models()
    all_preds = {}

    # Mean baseline (leave-one-out mean)
    log.info("Running Mean Baseline ...")
    preds_mean = np.zeros(n)
    for i in range(n):
        preds_mean[i] = np.delete(y, i).mean()
    all_preds["MeanBaseline"] = preds_mean

    # Sklearn models
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

    # DEIN
    if HAS_TORCH:
        log.info("Running DEIN (5-branch) LOOCV ...")
        preds_dein = train_dein_loocv(X, y, feat_names)
        all_preds["DEIN"] = preds_dein
        log.info("  DEIN LOOCV R² = %.4f", r2_score(y, preds_dein))
    else:
        log.warning("PyTorch not available — skipping DEIN")

    # Weighted Ensemble
    log.info("Computing Weighted Ensemble ...")
    ens_names = [k for k in all_preds if k != "MeanBaseline"]
    r2s      = {k: max(0, r2_score(y, all_preds[k])) for k in ens_names}
    total_r2 = sum(r2s.values()) + 1e-9
    weights  = {k: r2s[k] / total_r2 for k in ens_names}
    preds_ens = sum(w * all_preds[k] for k, w in weights.items())
    all_preds["WeightedEnsemble"] = preds_ens
    log.info("  Ensemble weights: %s",
             {k: f"{w:.3f}" for k, w in weights.items()})
    log.info("  Ensemble LOOCV R² = %.4f", r2_score(y, preds_ens))

    # Metrics
    rows = []
    for name, preds in all_preds.items():
        rmse = np.sqrt(mean_squared_error(y, preds))
        mae  = mean_absolute_error(y, preds)
        r2   = r2_score(y, preds)
        r_pe = pearsonr(y, preds)[0]
        rho  = spearmanr(y, preds)[0]
        f2   = r2 / (1 - r2 + 1e-9) if r2 > 0 else 0.0
        rows.append({
            "Model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4),
            "R2": round(r2, 4), "Pearson_r": round(r_pe, 4),
            "Spearman_rho": round(rho, 4), "Cohen_f2": round(f2, 4),
            "f2_size": ("large" if f2 >= 0.35 else
                        "medium" if f2 >= 0.15 else
                        "small" if f2 >= 0.02 else "negligible"),
        })

    metrics_df = pd.DataFrame(rows).sort_values("R2", ascending=False)
    metrics_df.to_csv(out_dir / "model_comparison_top30.csv", index=False)

    pred_df = pd.DataFrame(all_preds, index=X_df.index)
    pred_df["actual"] = y
    pred_df.to_csv(out_dir / "all_predictions_top30.csv")

    return all_preds, metrics_df, weights


# ═══════════════════════════════════════════════════════════════════════
# WILCOXON SIGNED-RANK TESTS
# ═══════════════════════════════════════════════════════════════════════

def wilcoxon_comparison(all_preds: dict, y: np.ndarray,
                         out_dir: Path) -> pd.DataFrame:
    names     = list(all_preds.keys())
    residuals = {k: np.abs(y - all_preds[k]) for k in names}
    n_tests   = len(names) * (len(names) - 1) // 2
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            try:
                stat, p = wilcoxon(residuals[a], residuals[b])
                p_corr  = min(1.0, p * n_tests)
                rows.append({
                    "Model_A": a, "Model_B": b,
                    "W_stat": round(stat, 2),
                    "p_raw": round(p, 4),
                    "p_bonferroni": round(p_corr, 4),
                    "significant": p_corr < 0.05,
                    "better": a if np.mean(residuals[a]) < np.mean(residuals[b]) else b,
                })
            except Exception:
                rows.append({
                    "Model_A": a, "Model_B": b,
                    "W_stat": np.nan, "p_raw": np.nan,
                    "p_bonferroni": np.nan,
                    "significant": False, "better": "—",
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "wilcoxon_comparison_top30.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE C: SHAP CONVERGENCE
# ═══════════════════════════════════════════════════════════════════════

def compute_shap_all(X_df: pd.DataFrame, y: np.ndarray,
                      out_dir: Path) -> pd.DataFrame:
    if not HAS_SHAP:
        log.warning("SHAP not available — skipping convergence analysis")
        return pd.DataFrame()

    X         = X_df.values
    feat_names = list(X_df.columns)
    scaler    = StandardScaler()
    X_s       = scaler.fit_transform(X)
    X_s_df    = pd.DataFrame(X_s, columns=feat_names)

    shap_ranks     = {}
    shap_values_all = {}

    def _linear_shap(model_cls, name, **kw):
        m = model_cls(**kw)
        m.fit(X_s, y)
        exp = shap.LinearExplainer(m, X_s_df)
        sv  = exp.shap_values(X_s_df)
        mean_abs = np.abs(sv).mean(axis=0)
        shap_values_all[name] = mean_abs
        rank = np.argsort(-mean_abs)
        shap_ranks[name] = {feat_names[r]: i+1 for i, r in enumerate(rank)}

    log.info("SHAP: Ridge ...")
    _linear_shap(RidgeCV, "Ridge",
                 alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    log.info("SHAP: ElasticNet ...")
    _linear_shap(ElasticNetCV, "ElasticNet",
                 cv=5, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                 max_iter=5000, random_state=42)
    log.info("SHAP: BayesianRidge ...")
    _linear_shap(BayesianRidge, "BayesianRidge", max_iter=500)

    log.info("SHAP: RandomForest ...")
    m4 = RandomForestRegressor(n_estimators=300, max_depth=3,
                                min_samples_leaf=5, random_state=42)
    m4.fit(X_s, y)
    exp4 = shap.TreeExplainer(m4)
    sv4  = exp4.shap_values(X_s_df)
    mean_abs4 = np.abs(sv4).mean(axis=0)
    shap_values_all["RandomForest"] = mean_abs4
    rank4 = np.argsort(-mean_abs4)
    shap_ranks["RandomForest"] = {feat_names[r]: i+1 for i, r in enumerate(rank4)}

    if HAS_XGB:
        log.info("SHAP: XGBoost ...")
        m5 = xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               min_child_weight=5, reg_alpha=1.0, reg_lambda=2.0,
                               random_state=42, verbosity=0)
        m5.fit(X_s, y)
        exp5 = shap.TreeExplainer(m5)
        sv5  = exp5.shap_values(X_s_df)
        mean_abs5 = np.abs(sv5).mean(axis=0)
        shap_values_all["XGBoost"] = mean_abs5
        rank5 = np.argsort(-mean_abs5)
        shap_ranks["XGBoost"] = {feat_names[r]: i+1 for i, r in enumerate(rank5)}

    # ── Convergence: features in top-6 across ≥3 models ──────────────
    n_models  = len(shap_ranks)
    top_k     = min(6, len(feat_names))
    top_feats = {name: set(list(ranks.keys())[:top_k])
                 for name, ranks in
                 {m: dict(sorted(r.items(), key=lambda x: x[1]))
                  for m, r in shap_ranks.items()}.items()}

    conv_rows = []
    for f in feat_names:
        in_top6 = sum(1 for m in top_feats if f in top_feats[m])
        mean_r  = np.mean([shap_ranks[m].get(f, len(feat_names))
                           for m in shap_ranks])
        conv_rows.append({
            "Feature": f,
            "Mean_SHAP_rank": round(mean_r, 2),
            "In_top6_count": in_top6,
            "Convergent": in_top6 >= 3,
        })

    conv_df = pd.DataFrame(conv_rows).sort_values("Mean_SHAP_rank")
    conv_df.to_csv(out_dir / "shap_convergence_top30.csv", index=False)

    # ── SHAP bar chart (top 15) ───────────────────────────────────────
    n_models_plot = len(shap_values_all)
    fig, axes = plt.subplots(1, n_models_plot,
                              figsize=(5 * n_models_plot, 8))
    if n_models_plot == 1:
        axes = [axes]

    top15 = conv_df.head(15)["Feature"].tolist()

    for ax, (mname, vals) in zip(axes, shap_values_all.items()):
        idx = [feat_names.index(f) for f in top15 if f in feat_names]
        ax.barh([feat_names[i] for i in idx],
                [vals[i] for i in idx], color="steelblue")
        ax.set_title(mname, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("mean |SHAP|")

    plt.suptitle("SHAP — Top 15 features (top-30 model)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_all_models_top30.png", dpi=150)
    plt.close(fig)

    log.info("Convergent features (top6, ≥3 models): %s",
             conv_df[conv_df["Convergent"]]["Feature"].tolist())
    return conv_df


# ═══════════════════════════════════════════════════════════════════════
# DIMENSION COMPOSITES
# ═══════════════════════════════════════════════════════════════════════

def dimension_composites(X_df, y, pids, out_dir):
    """Compute one composite per dimension, report Pearson r with exam."""
    rows  = []
    scores_dict = {}
    dims  = {"D1": "c1_", "D2": "c2_", "D3": "c3_",
             "D4": "c4_", "D5": "c5_"}

    for dim, prefix in dims.items():
        cols = [c for c in X_df.columns if c.startswith(prefix)]
        if not cols:
            log.warning("No features for %s — skipping composite", dim)
            continue

        # Sign-correct then z-score average
        X_sub = X_df[cols].copy()
        for c in cols:
            X_sub[c] *= FEAT_SIGN.get(c, 1)
        zscores = (X_sub - X_sub.mean()) / (X_sub.std() + 1e-9)
        composite = zscores.mean(axis=1).values

        r, p = pearsonr(composite, y)
        scores_dict[dim] = composite
        rows.append({"Dimension": dim, "n_features": len(cols),
                     "Pearson_r": round(r, 3), "p": round(p, 4),
                     "features": cols})
        log.info("%s composite: n=%d features, r=%.3f (p=%.4f)", dim, len(cols), r, p)

    dim_df = pd.DataFrame(rows)
    dim_df.to_csv(out_dir / "dimension_composites_top30.csv", index=False)
    return dim_df, scores_dict


# ═══════════════════════════════════════════════════════════════════════
# PHASE D: EI COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════════════

def compute_ei_score(X_df, y, final_feats, pids, out_dir):
    """Compute final composite EI score from selected features."""
    available = [f for f in final_feats if f in X_df.columns]
    X_sub = X_df[available].copy()
    for c in available:
        X_sub[c] *= FEAT_SIGN.get(c, 1)

    zscores   = (X_sub - X_sub.mean()) / (X_sub.std() + 1e-9)
    ei_scores = zscores.mean(axis=1).values
    ei_norm   = (ei_scores - ei_scores.min()) / \
                (ei_scores.max() - ei_scores.min() + 1e-9)

    r_ei, p_r  = pearsonr(ei_norm, y)
    rho, p_rho = spearmanr(ei_norm, y)

    # Bootstrap 95% CI on r
    n_boot = 2000
    boot_r = []
    np.random.seed(42)
    for _ in range(n_boot):
        idx = np.random.choice(len(y), len(y), replace=True)
        boot_r.append(pearsonr(ei_norm[idx], y[idx])[0])
    r_lo, r_hi = np.percentile(boot_r, [2.5, 97.5])

    # Partial r (controlling for strongest single predictor)
    best_feat = available[0]
    ctrl = X_df[best_feat].values
    ctrl_z = (ctrl - ctrl.mean()) / (ctrl.std() + 1e-9)
    resid_ei = ei_norm - np.polyval(np.polyfit(ctrl_z, ei_norm, 1), ctrl_z)
    resid_y  = y        - np.polyval(np.polyfit(ctrl_z, y, 1),      ctrl_z)
    partial_r, p_partial = pearsonr(resid_ei, resid_y)

    ei_stats = {
        "Pearson_r":   round(r_ei, 3),
        "r_lo":        round(r_lo, 3),
        "r_hi":        round(r_hi, 3),
        "p_pearson":   round(p_r, 4),
        "Spearman_rho": round(rho, 3),
        "p_spearman":  round(p_rho, 4),
        "Partial_r":   round(partial_r, 3),
        "p_partial":   round(p_partial, 4),
        "Controlled_for": best_feat,
    }
    log.info("EI composite: r=%.3f [%.3f, %.3f], p=%.4f  |  partial r=%.3f",
             r_ei, r_lo, r_hi, p_r, partial_r)

    pd.DataFrame({
        "participant": pids,
        "ei_score": ei_scores,
        "ei_score_norm": ei_norm,
        "exam_score": y,
    }).to_csv(out_dir / "ei_scores_top30.csv", index=False)

    return ei_scores, ei_stats


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_model_comparison(metrics_df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    metrics_sorted = metrics_df.sort_values("R2", ascending=True)

    colors = {"BayesianRidge": "tomato", "ElasticNet": "tomato",
              "Ridge": "tomato", "WeightedEnsemble": "orange",
              "RandomForest": "tomato", "DEIN": "mediumpurple",
              "XGBoost": "tomato", "MeanBaseline": "silver"}

    def bar(ax, col, label, fmt):
        vals   = metrics_sorted[col]
        models = metrics_sorted["Model"]
        clrs   = [colors.get(m, "steelblue") for m in models]
        bars   = ax.barh(models, vals, color=clrs)
        ax.set_title(label)
        for bar_, v in zip(bars, vals):
            ax.text(v + 0.001, bar_.get_y() + bar_.get_height() / 2,
                    fmt.format(v), va="center", fontsize=8)

    bar(axes[0], "R2",        "R² (higher=better)", "{:.3f}")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)

    metrics_sorted_rmse = metrics_df.sort_values("RMSE", ascending=False)
    vals   = metrics_sorted_rmse["RMSE"]
    models = metrics_sorted_rmse["Model"]
    clrs   = [colors.get(m, "steelblue") for m in models]
    bars   = axes[1].barh(models, vals, color=clrs)
    axes[1].set_title("RMSE (lower=better)")
    for bar_, v in zip(bars, vals):
        axes[1].text(v + 0.0005, bar_.get_y() + bar_.get_height() / 2,
                     f"{v:.3f}", va="center", fontsize=8)

    bar(axes[2], "Pearson_r", "Pearson r (higher=better)", "{:.3f}")

    plt.suptitle("Model Comparison — LOOCV on top-30 features (n=41)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "model_comparison_top30.png", dpi=150)
    plt.close(fig)


def plot_all_scatters(all_preds, y, out_dir):
    names    = list(all_preds.keys())
    n_models = len(names)
    n_cols   = 4
    n_rows   = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        preds = all_preds[name]
        ax.scatter(y, preds, alpha=0.7)
        mn, mx = min(y.min(), preds.min()), max(y.max(), preds.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
        r, _  = pearsonr(y, preds)
        r2    = r2_score(y, preds)
        ax.set_title(f"{name}\nr={r:.3f}, R²={r2:.3f}", fontsize=9)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    for ax in axes[n_models:]:
        ax.set_visible(False)

    plt.suptitle("LOOCV Predicted vs Actual — All Models (top-30 features)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "all_model_scatters_top30.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════

def write_final_report(metrics_df, wilcoxon_df, conv_df, ei_stats,
                        final_feats, n_feats_used, out_dir):
    lines = [
        "=" * 76,
        "  FINAL EI PIPELINE — TOP-30 FEATURE EXPANSION REPORT",
        "=" * 76, "",
        f"  Features used in LOOCV: {n_feats_used}  (requested ≤ 30)",
        "",
        "─" * 76,
        "  PHASE B: MODEL COMPARISON (LOOCV, top-30 features)",
        "─" * 76,
        "",
        f"  {'Model':<22} {'R²':>7} {'RMSE':>7} {'MAE':>7} "
        f"{'r':>7} {'ρ':>7} {'f²':>7} {'Size':>10}",
        "  " + "─" * 70,
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"  {r['Model']:<22} {r['R2']:>7.3f} {r['RMSE']:>7.4f} "
            f"{r['MAE']:>7.4f} {r['Pearson_r']:>7.3f} "
            f"{r['Spearman_rho']:>7.3f} {r['Cohen_f2']:>7.3f} "
            f"{r['f2_size']:>10}")

    if wilcoxon_df is not None and not wilcoxon_df.empty:
        sig_pairs = wilcoxon_df[wilcoxon_df["significant"]]
        lines += ["", "─" * 76,
            f"  WILCOXON (Bonferroni, {len(wilcoxon_df)} pairs)",
            "─" * 76,
            f"  Significant pairs: {len(sig_pairs)} / {len(wilcoxon_df)}"]
        if sig_pairs.empty:
            lines.append("  → No model significantly outperforms another")
        else:
            for _, r in sig_pairs.iterrows():
                lines.append(f"  {r['Model_A']} vs {r['Model_B']}: "
                             f"p={r['p_bonferroni']:.4f} → {r['better']} better")

    if conv_df is not None and not conv_df.empty:
        conv_feats = conv_df[conv_df["Convergent"]]["Feature"].tolist()
        lines += ["", "─" * 76,
            "  PHASE C: SHAP CONVERGENCE", "─" * 76,
            f"  Convergent features (top-6, ≥3 models): {conv_feats}", "",
            "  Top-15 by mean SHAP rank:"]
        for _, r in conv_df.head(15).iterrows():
            tag = " ★" if r["Convergent"] else ""
            lines.append(f"    {r['Feature']:<46} "
                         f"mean_rank={r['Mean_SHAP_rank']:<5} "
                         f"in_top6={r['In_top6_count']}{tag}")

    lines += ["", "─" * 76,
        "  PHASE D: FINAL COMPOSITE EI SCORE", "─" * 76,
        f"  Features used: {final_feats}",
        f"  Pearson r  = {ei_stats['Pearson_r']:+.3f} "
        f"[{ei_stats['r_lo']:+.3f}, {ei_stats['r_hi']:+.3f}], "
        f"p = {ei_stats['p_pearson']:.4f}",
        f"  Spearman ρ = {ei_stats['Spearman_rho']:+.3f}, "
        f"p = {ei_stats['p_spearman']:.4f}",
        f"  Partial r  = {ei_stats['Partial_r']:+.3f} "
        f"(controlling {ei_stats['Controlled_for']}), "
        f"p = {ei_stats['p_partial']:.4f}",
        "", "─" * 76, "  PILOT CAVEATS", "─" * 76,
        "  • n=41: 80% power only for |r| ≥ 0.42",
        "  • Expanding to 30 features increases overfitting risk —",
        "    compare regularised models (Ridge/ElasticNet) carefully.",
        "  • Winner's curse likely inflates effect sizes",
        "  • All findings exploratory — replicate with n ≥ 150",
        "", "=" * 76]

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "final_report_top30.txt").write_text(report, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main(data_dir: str):
    dp      = Path(data_dir)
    out_dir = dp / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  EI PIPELINE — TOP-30 FEATURE EXPANSION")
    log.info("=" * 60)

    # Phase A
    X_df, y, pids, feat_names = load_data(dp)
    log.info("Features loaded: %d", len(feat_names))
    log.info("y range: [%.3f, %.3f], mean=%.3f", y.min(), y.max(), y.mean())

    # Phase B
    log.info("\n── PHASE B: Multi-Model LOOCV ──")
    all_preds, metrics_df, ens_weights = loocv_all_models(
        X_df, y, feat_names, out_dir)
    plot_model_comparison(metrics_df, out_dir)
    plot_all_scatters(all_preds, y, out_dir)

    # Wilcoxon
    log.info("\n── WILCOXON ──")
    wilcoxon_df = wilcoxon_comparison(all_preds, y, out_dir)
    sig = wilcoxon_df[wilcoxon_df["significant"]]
    log.info("Significant pairs: %d / %d", len(sig), len(wilcoxon_df))

    # Phase C
    log.info("\n── PHASE C: SHAP Convergence ──")
    conv_df = compute_shap_all(X_df, y, out_dir)

    # Select final features for EI score
    if not conv_df.empty:
        convergent = conv_df[conv_df["Convergent"]]["Feature"].tolist()
        top_by_rank = conv_df.head(6)["Feature"].tolist()
        final_feats = (convergent + [f for f in top_by_rank
                                      if f not in convergent])[:6]
    else:
        final_feats = [
            "c2_stress_peak_attenuation",
            "c3_productive_absorbing_probability",
            "c3_stress_inertia_during_blocks",
            "c4_arousal_trend",
            "c4_arousal_optimality",
            "c2_stress_recovery_latency",
        ]
    log.info("Final features for EI composite: %s", final_feats)

    # Dimension composites
    log.info("\n── DIMENSION COMPOSITES ──")
    dim_df, dim_scores = dimension_composites(X_df, y, pids, out_dir)

    # Phase D
    log.info("\n── PHASE D: EI Score ──")
    ei_scores, ei_stats = compute_ei_score(
        X_df, y, final_feats, pids, out_dir)

    # Report
    write_final_report(metrics_df, wilcoxon_df, conv_df, ei_stats,
                        final_feats, len(feat_names), out_dir)
    log.info("All outputs → %s", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()
    main(args.data_dir)
