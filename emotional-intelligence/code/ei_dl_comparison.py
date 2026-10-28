"""
EI Deep Learning Comparison
============================
Compares deep learning approaches for EI score regression.
All models evaluated under Leave-One-Out Cross-Validation (LOOCV).
Results saved to results_dl/.

Approaches compared:
  1. MLP_shallow    — 2-layer MLP, strong L2 + Dropout (baseline DL)
  2. MLP_deep       — 4-layer MLP with residual skip connection
  3. CNN1D          — 1D-CNN over the 16 tabular features as a sequence
  4. TabTransformer — Transformer encoder on tabular feature embeddings
  5. LSTM_seq       — LSTM over stage-ordered emotion windows
  6. Autoencoder_MLP— Autoencoder pretraining + fine-tuned MLP head

Why these for n=42:
  MLP: smallest parameter count, easiest to regularise, DL baseline.
  CNN1D: treats feature vector as a 1D "signal" — captures local patterns.
  TabTransformer: attention between features; strong on tabular data (Huang 2020).
  LSTM: for the sequence-level experiment (stage-ordered emotion windows).
  Autoencoder: unsupervised pre-training helps with tiny labelled sets.

Compared against Ridge (sklearn) as the classical benchmark.

Usage:
    python ei_dl_comparison.py
    python ei_dl_comparison.py --data_dir "D:/A_emotions in exam/data"

Requirements:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install pandas numpy scipy scikit-learn shap matplotlib openpyxl
"""

import re
import zipfile
import argparse
import warnings
import logging
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.pipeline import Pipeline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_DATA_DIR = r"D:\A_emotions in exam\data"
RESULTS_DIR      = "results_dl"

FEATURE_MATRIX_CSV = "feature_matrix.csv"   # from v3/v4 run
PREDICTIONS_CSV    = "all_predictions.csv"  # for exam scores

# If you have already run the pipeline, point to the feature matrix directly.
# Otherwise set these to None and the script will look in data_dir/results_v4/
FEATURE_MATRIX_PATH = None
EXAM_SCORES_PATH    = None

# Training hyperparameters (tuned for n=42)
EPOCHS     = 300
LR         = 1e-3
WEIGHT_DECAY = 1e-3     # L2 on all models
DROPOUT    = 0.4
BATCH_SIZE = 16         # small batches for tiny dataset
PATIENCE   = 40         # early-stopping patience

ALPHA      = 0.05       # Wilcoxon significance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", DEVICE)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_feature_matrix(data_dir: Path):
    """
    Load feature matrix and exam scores.
    Tries results_v4/ first, then results_v3_fixed/, then results_v2/.
    """
    for sub in ["results_v4", "results_v3_fixed", "results_v3", "results_v2"]:
        fm_path = data_dir / sub / FEATURE_MATRIX_CSV
        ap_path = data_dir / sub / PREDICTIONS_CSV
        if fm_path.exists() and ap_path.exists():
            log.info("Loading features from: %s", data_dir / sub)
            fm = pd.read_csv(fm_path, index_col=0)
            ap = pd.read_csv(ap_path, index_col=0)
            y  = ap["exam_score"].values
            return fm, y

    raise FileNotFoundError(
        "No feature_matrix.csv found. Run ei_pipeline_v4.py first "
        "to generate the feature matrix, then run this script."
    )


# ============================================================================
# SEQUENCE DATA FOR LSTM
# ============================================================================

def build_stage_sequences(feature_df: pd.DataFrame,
                           y: np.ndarray) -> tuple:
    """
    Build a synthetic sequence representation for LSTM:
    Split the 15 features into 5 groups (one per EI dimension),
    treating each group as one time step. Shape: (n, 5, dim_per_step).

    This is a deliberate design: the LSTM learns temporal patterns
    across the five conceptual stages of EI expression, ordered as:
    Awareness → Regulation → Persistence → Self-Report → Alignment.
    """
    groups = [
        ["c1_entropy", "c1_pos_neg_ratio", "c1_variability"],
        ["c2_stress_recovery", "c2_positive_uplift", "c2_appropriateness"],
        ["c3_block_score", "c3_recovery_speed", "c3_productive_ratio"],
        ["c4_valence_stability", "c4_arousal_stability",
         "c4_mean_valence", "c4_trend"],
        ["c5_smoothness", "c5_engagement"],
    ]
    # Pad each group to the same length (4) for a uniform tensor
    max_len = max(len(g) for g in groups)
    seqs = []
    for _, row in feature_df.iterrows():
        steps = []
        for g in groups:
            vals = [row[f] if f in feature_df.columns else 0.0 for f in g]
            while len(vals) < max_len:
                vals.append(0.0)
            steps.append(vals)
        seqs.append(steps)
    X_seq = np.array(seqs, dtype=np.float32)  # (n, 5, max_len)
    return X_seq, y


# ============================================================================
# PYTORCH MODEL DEFINITIONS
# ============================================================================

class MLP_Shallow(nn.Module):
    """
    2-hidden-layer MLP with BatchNorm and Dropout.
    Reference: Goodfellow et al. (2016) Deep Learning, MIT Press.
    Appropriate for small tabular datasets when heavily regularised.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLP_Deep(nn.Module):
    """
    4-hidden-layer MLP with residual skip connection.
    The skip connection prevents vanishing gradients with deeper nets.
    Reference: He et al. (2016) Deep Residual Learning, CVPR.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, 64)
        self.block1 = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(DROPOUT),
        )
        self.block2 = nn.Sequential(
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(DROPOUT),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        x  = self.input_proj(x)
        x  = x + self.block1(x)     # residual
        x  = self.block2(x)
        return self.head(x).squeeze(-1)


class CNN1D(nn.Module):
    """
    1D-CNN treating the feature vector as a 1D signal.
    Kernel size 3 captures local feature interactions.
    Reference: LeCun et al. (1998); applied to tabular data by
    Baosengul et al. (2022) for psychometric feature extraction.
    Features are unsqueezed to shape (batch, 1, n_features).
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 4, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (batch, 1, features)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.head(x).squeeze(-1)


class TabTransformer(nn.Module):
    """
    Transformer encoder over tabular features.
    Each feature is embedded independently, then Transformer
    learns inter-feature attention.
    Reference: Huang et al. (2020) TabTransformer, arXiv:2012.06678.
    Adapted for regression on small tabular datasets.
    """
    def __init__(self, in_dim: int, n_heads: int = 2,
                 n_layers: int = 2, d_model: int = 32):
        super().__init__()
        self.feature_embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=64, dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model * in_dim, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(-1)                     # (batch, features, 1)
        x = self.feature_embed(x)               # (batch, features, d_model)
        x = self.transformer(x)                 # (batch, features, d_model)
        x = x.view(x.size(0), -1)               # (batch, features*d_model)
        return self.head(x).squeeze(-1)


class LSTM_Seq(nn.Module):
    """
    Bidirectional LSTM over the 5-step EI dimension sequence.
    Each step represents one EI dimension's features.
    Reference: Hochreiter & Schmidhuber (1997) LSTM, Neural Computation.
    Bidirectional captures both forward (awareness→alignment)
    and backward (alignment→awareness) dependencies.
    """
    def __init__(self, input_size: int, hidden_size: int = 32,
                 n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len=5, features_per_step)
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last time step
        return self.head(out).squeeze(-1)


class Autoencoder_MLP(nn.Module):
    """
    Two-stage model: unsupervised autoencoder pre-training
    followed by supervised fine-tuning of the encoder + regression head.
    Reference: Bengio et al. (2013) Representation Learning, IEEE TPAMI.
    Pre-training is particularly useful when n is too small for end-to-end
    supervised training, as it initialises the encoder with structure
    from the full feature distribution.
    """
    def __init__(self, in_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, in_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z).squeeze(-1)

    def reconstruct(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def pretrain_autoencoder(model: Autoencoder_MLP,
                          X_train: np.ndarray,
                          epochs: int = 150) -> None:
    """Unsupervised autoencoder pre-training on training fold."""
    model.train()
    X_t   = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    optim_ae = optim.Adam(
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optim_ae.zero_grad()
        recon = model.reconstruct(X_t)
        loss  = criterion(recon, X_t)
        loss.backward()
        optim_ae.step()


def train_model(model: nn.Module,
                X_train: np.ndarray,
                y_train: np.ndarray,
                is_sequence: bool = False,
                pretrain: bool = False) -> nn.Module:
    """
    Train a PyTorch model with early stopping on training loss.
    Uses AdamW optimiser + cosine annealing LR schedule.
    """
    model = model.to(DEVICE)

    if pretrain and isinstance(model, Autoencoder_MLP):
        pretrain_autoencoder(model, X_train)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(y_train)),
                         shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    best_loss  = float("inf")
    best_state = None
    patience_counter = 0

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_model(model: nn.Module,
                  X_test: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        X_t  = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        pred = model(X_t).cpu().numpy()
    return float(pred[0]) if pred.ndim > 0 else float(pred)


# ============================================================================
# LOOCV FOR ALL MODELS
# ============================================================================

def loocv_dl(X: np.ndarray, X_seq: np.ndarray,
              y: np.ndarray, in_dim: int,
              seq_step_dim: int) -> dict:
    """
    Run LOOCV for all DL approaches + Ridge baseline.
    For each fold: fit scaler on train, transform train+test,
    train model from scratch, predict held-out participant.
    """
    loo     = LeaveOneOut()
    results = {}

    model_specs = {
        "Ridge_baseline": None,   # handled separately via sklearn
        "MLP_shallow":    lambda: MLP_Shallow(in_dim),
        "MLP_deep":       lambda: MLP_Deep(in_dim),
        "CNN1D":          lambda: CNN1D(in_dim),
        "TabTransformer": lambda: TabTransformer(in_dim),
        "LSTM_seq":       lambda: LSTM_Seq(seq_step_dim),
        "Autoenc_MLP":    lambda: Autoencoder_MLP(in_dim),
    }

    for name, model_fn in model_specs.items():
        log.info("LOOCV → %s ...", name)
        preds = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X[train_idx])
            X_te_s = scaler.transform(X[test_idx])

            if name == "Ridge_baseline":
                ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500], cv=5)
                ridge.fit(X_tr_s, y[train_idx])
                preds[test_idx] = ridge.predict(X_te_s)

            elif name == "LSTM_seq":
                # Use sequence data, scale each step independently
                X_seq_tr = X_seq[train_idx]
                X_seq_te = X_seq[test_idx]
                n_tr, n_steps, n_feat = X_seq_tr.shape
                # Flatten → scale → reshape
                flat_tr = X_seq_tr.reshape(n_tr, -1)
                flat_te = X_seq_te.reshape(1,    -1)
                seq_scaler = StandardScaler()
                flat_tr = seq_scaler.fit_transform(flat_tr)
                flat_te = seq_scaler.transform(flat_te)
                X_seq_tr_s = flat_tr.reshape(n_tr, n_steps, n_feat)
                X_seq_te_s = flat_te.reshape(1,    n_steps, n_feat)

                model = model_fn().to(DEVICE)
                model = train_model(model, X_seq_tr_s,
                                    y[train_idx], is_sequence=True)
                preds[test_idx] = predict_model(model, X_seq_te_s)

            else:
                pretrain = (name == "Autoenc_MLP")
                model    = model_fn().to(DEVICE)
                model    = train_model(model, X_tr_s, y[train_idx],
                                       pretrain=pretrain)
                preds[test_idx] = predict_model(model,
                                                X_te_s.astype(np.float32))

        residuals = y - preds
        r2 = float(r2_score(y, preds))
        f2 = r2 / (1 - r2) if r2 < 1.0 else np.inf
        results[name] = {
            "preds":     preds,
            "residuals": residuals,
            "rmse":      float(np.sqrt(mean_squared_error(y, preds))),
            "mae":       float(mean_absolute_error(y, preds)),
            "r2":        r2,
            "cohens_f2": f2,
        }
        log.info("  %-18s RMSE=%.4f  MAE=%.4f  R²=%.4f  f²=%.4f",
                 name, results[name]["rmse"], results[name]["mae"], r2, f2)

    return results


# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================

def statistical_comparison(results: dict) -> pd.DataFrame:
    names  = list(results.keys())
    pairs  = list(itertools.combinations(names, 2))
    n_comp = len(pairs)
    rows   = []
    for m1, m2 in pairs:
        abs1 = np.abs(results[m1]["residuals"])
        abs2 = np.abs(results[m2]["residuals"])
        try:
            stat, p_raw = wilcoxon(abs1, abs2, alternative="two-sided")
        except ValueError:
            stat, p_raw = 0.0, 1.0
        p_bonf  = min(p_raw * n_comp, 1.0)
        better  = m2 if results[m1]["rmse"] > results[m2]["rmse"] else m1
        rows.append({
            "Model A":        m1,
            "Model B":        m2,
            "RMSE (A)":       round(results[m1]["rmse"], 4),
            "RMSE (B)":       round(results[m2]["rmse"], 4),
            "Wilcoxon W":     round(stat, 3),
            "p (raw)":        round(p_raw, 4),
            "p (Bonferroni)": round(p_bonf, 4),
            "Significant?":   "YES *" if p_bonf < ALPHA else "no",
            "Better model":   better,
        })
    return pd.DataFrame(rows)


def correlation_analysis(results: dict, y: np.ndarray) -> pd.DataFrame:
    n = len(y)
    rows = []
    for name, res in results.items():
        pr, pp = pearsonr(res["preds"], y)
        sr, sp = spearmanr(res["preds"], y)
        z  = np.arctanh(np.clip(float(pr), -0.9999, 0.9999))
        se = 1 / np.sqrt(n - 3)
        rows.append({
            "Model":      name,
            "Pearson r":  round(float(pr), 4),
            "95% CI lo":  round(float(np.tanh(z - 1.96*se)), 4),
            "95% CI hi":  round(float(np.tanh(z + 1.96*se)), 4),
            "Pearson p":  round(float(pp), 4),
            "Spearman r": round(float(sr), 4),
            "Spearman p": round(float(sp), 4),
        })
    return pd.DataFrame(rows)


# ============================================================================
# PLOTS
# ============================================================================

def plot_metrics(results: dict, out_dir: Path) -> None:
    names  = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, ylabel, title) in zip(axes, [
        ("rmse", "RMSE", "RMSE (lower is better)"),
        ("mae",  "MAE",  "MAE  (lower is better)"),
        ("r2",   "R²",   "R²   (higher is better)"),
    ]):
        vals = [results[n][key] for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor="black",
                      linewidth=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(max(vals))*0.012,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    plt.suptitle("DL Model Comparison — LOOCV", fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "dl_metrics_comparison.png", dpi=100,
                bbox_inches="tight")
    plt.close(fig)
    log.info("Metrics plot saved.")


def plot_predicted_vs_actual(results: dict, y: np.ndarray,
                              out_dir: Path) -> None:
    names = list(results.keys())
    ncols = min(4, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5*ncols, 3.2*nrows),
                             squeeze=False, dpi=100)
    for idx, name in enumerate(names):
        ax    = axes[idx // ncols][idx % ncols]
        preds = results[name]["preds"]
        ax.scatter(preds, y, alpha=0.65, edgecolors="k",
                   linewidths=0.3, s=22)
        m, b = np.polyfit(preds, y, 1)
        xl   = np.linspace(preds.min(), preds.max(), 80)
        ax.plot(xl, m*xl+b, "r--", linewidth=1.2)
        ax.set_title(
            f"{name}\nR²={results[name]['r2']:.3f}  "
            f"RMSE={results[name]['rmse']:.3f}",
            fontsize=8, fontweight="bold")
        ax.set_xlabel("EI (pred.)", fontsize=7)
        ax.set_ylabel("Exam",       fontsize=7)
        ax.tick_params(labelsize=6)
    for idx in range(len(names), nrows*ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    plt.suptitle("Predicted EI vs Exam Score — LOOCV (DL)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "dl_predicted_vs_actual.png", dpi=100,
                bbox_inches="tight")
    plt.close(fig)
    plt.clf(); plt.cla()
    log.info("Scatter plot saved.")


def plot_residuals(results: dict, out_dir: Path) -> None:
    names  = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot([results[n]["residuals"] for n in names],
                    labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_title("LOOCV Residual Distribution — DL Models",
                 fontweight="bold")
    ax.set_ylabel("Residual  (actual − predicted)")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig.savefig(out_dir / "dl_residuals_boxplot.png", dpi=100)
    plt.close(fig)


def plot_correlation_heatmap(corr_df: pd.DataFrame,
                              out_dir: Path) -> None:
    vals = corr_df.set_index("Model")[["Pearson r", "Spearman r"]]
    fig, ax = plt.subplots(figsize=(5, max(3, len(vals)*0.55)))
    im = ax.imshow(vals.values, cmap="RdYlGn", vmin=-1, vmax=1,
                   aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pearson r", "Spearman r"])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals.index, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    for i in range(len(vals)):
        for j in range(2):
            ax.text(j, i, f"{vals.values[i,j]:.3f}",
                    ha="center", va="center", fontsize=9)
    ax.set_title("Correlation with exam score", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "dl_correlation_heatmap.png", dpi=100)
    plt.close(fig)


# ============================================================================
# REPORT
# ============================================================================

def write_report(results: dict, wilcoxon_df: pd.DataFrame,
                 corr_df: pd.DataFrame, n: int,
                 out_dir: Path) -> None:
    best = min(results, key=lambda k: results[k]["rmse"])
    baseline_rmse = results.get("Ridge_baseline", {}).get("rmse", None)

    L = [
        "=" * 68,
        "   EI DEEP LEARNING COMPARISON — RESULTS REPORT",
        "=" * 68,
        f"   Participants:   {n}",
        f"   Models:         {len(results)}",
        f"   Device:         {DEVICE}",
        "",
        "─" * 68,
        "   1. LOOCV PERFORMANCE  (f²: small=0.02, medium=0.15)",
        "─" * 68,
        f"   {'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'f²':>8}",
        "   " + "-" * 50,
    ]
    for nm, res in sorted(results.items(), key=lambda x: x[1]["rmse"]):
        tag = "  ◄ best" if nm == best else ""
        bl  = "  [baseline]" if nm == "Ridge_baseline" else ""
        f2s = f"{res['cohens_f2']:.4f}" \
              if np.isfinite(res["cohens_f2"]) else "  ∞"
        beats_baseline = ""
        if baseline_rmse and nm != "Ridge_baseline":
            beats_baseline = (
                "  beats Ridge" if res["rmse"] < baseline_rmse
                else "  worse than Ridge"
            )
        L.append(
            f"   {nm:<20} {res['rmse']:>8.4f} "
            f"{res['mae']:>8.4f} {res['r2']:>8.4f} {f2s:>8}"
            f"{tag}{bl}{beats_baseline}"
        )

    L += [
        "",
        "─" * 68,
        f"   2. WILCOXON SIGNED-RANK  (Bonferroni α={ALPHA})",
        "─" * 68,
    ]
    sig_pairs = []
    for _, row in wilcoxon_df.iterrows():
        sig = "★ SIGNIFICANT" if "YES" in str(row["Significant?"]) \
              else "not significant"
        L.append(
            f"   {row['Model A']} vs {row['Model B']}:  "
            f"p(Bonf)={row['p (Bonferroni)']:.4f}  → {sig}"
        )
        if "YES" in str(row["Significant?"]):
            sig_pairs.append((row["Model A"], row["Model B"],
                               row["Better model"]))
            L.append(f"     Better: {row['Better model']}")

    L += [
        "",
        "─" * 68,
        "   3. CORRELATION WITH EXAM SCORES",
        "─" * 68,
        f"   {'Model':<20} {'Pearson r':>10} {'95% CI':>18} "
        f"{'p':>7}  {'Spearman r':>11} {'p':>7}",
        "   " + "-" * 72,
    ]
    for _, row in corr_df.iterrows():
        sp = "*" if row["Pearson p"]  < 0.05 else " "
        ss = "*" if row["Spearman p"] < 0.05 else " "
        ci = f"[{row['95% CI lo']:.3f},{row['95% CI hi']:.3f}]"
        L.append(
            f"   {row['Model']:<20} {row['Pearson r']:>10.4f}{sp} "
            f"{ci:>18} {row['Pearson p']:>6.4f}  "
            f"{row['Spearman r']:>11.4f}{ss}  "
            f"{row['Spearman p']:>6.4f}"
        )
    L.append("   (* p < 0.05)")

    L += [
        "",
        "─" * 68,
        f"   BEST MODEL: {best}  "
        f"RMSE={results[best]['rmse']:.4f}  "
        f"R²={results[best]['r2']:.4f}",
        "=" * 68,
    ]

    report = "\n".join(L)
    print("\n" + report)
    with open(out_dir / "dl_summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Report saved → dl_summary_report.txt")


# ============================================================================
# MAIN
# ============================================================================

def main(data_dir: str) -> None:
    data_path = Path(data_dir)
    out_dir   = data_path / RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load feature matrix ───────────────────────────────────────────────
    log.info("Loading feature matrix ...")
    feature_df, y = load_feature_matrix(data_path)
    log.info("Feature matrix: %s  |  participants: %d",
             feature_df.shape, len(y))

    feature_cols = list(feature_df.columns)
    in_dim       = len(feature_cols)
    X            = feature_df[feature_cols].fillna(0.0).values.astype(np.float32)

    # ── Build sequence data for LSTM ──────────────────────────────────────
    X_seq, _    = build_stage_sequences(feature_df, y)
    seq_step_dim = X_seq.shape[2]

    # ── LOOCV ────────────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("LOOCV — all DL models + Ridge baseline")
    log.info("=" * 50)
    results = loocv_dl(X, X_seq, y, in_dim, seq_step_dim)

    # Save predictions
    pred_df = feature_df.copy()
    pred_df["exam_score"] = y
    for nm, res in results.items():
        pred_df[f"pred_{nm}"]  = res["preds"]
        pred_df[f"resid_{nm}"] = res["residuals"]
    pred_df.to_csv(out_dir / "dl_all_predictions.csv")

    # ── Statistical comparison ───────────────────────────────────────────
    log.info("Wilcoxon comparison ...")
    wilcoxon_df = statistical_comparison(results)
    wilcoxon_df.to_csv(out_dir / "dl_wilcoxon_comparison.csv", index=False)

    # ── Correlation ───────────────────────────────────────────────────────
    log.info("Correlation analysis ...")
    corr_df = correlation_analysis(results, y)
    corr_df.to_csv(out_dir / "dl_correlation_results.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    log.info("Generating plots ...")
    plot_metrics(results, out_dir)
    plot_predicted_vs_actual(results, y, out_dir)
    plot_residuals(results, out_dir)
    plot_correlation_heatmap(corr_df, out_dir)

    # ── Report ────────────────────────────────────────────────────────────
    write_report(results, wilcoxon_df, corr_df, len(y), out_dir)

    log.info("All outputs → %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EI Deep Learning Model Comparison"
    )
    parser.add_argument(
        "--data_dir", default=DEFAULT_DATA_DIR,
        help='Root data directory containing results_v4/ or results_v3/'
    )
    args = parser.parse_args()
    main(args.data_dir)
