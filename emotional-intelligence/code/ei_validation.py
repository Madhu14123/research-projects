# -*- coding: utf-8 -*-
"""
EI Score Validation  --  Unsupervised
======================================
Loads precomputed feature CSVs (D1-D5), selects theoretically grounded
features, computes an EI score via PCA + weighted composite, then runs:

  Layer 1  Internal validity   : Silhouette, Davies-Bouldin, Calinski-Harabasz
  Layer 2  Construct validity  : Spearman ρ with exam pct (proxy label)
  Layer 3  Stability           : Bootstrap CI on EI score (n=1000)
  Layer 4  Robustness          : Leave-One-Out score variance
  Layer 5  Feature diagnostics : PCA loadings, communality, per-branch variance
  Layer 6  Convergent validity : Within-branch Cronbach alpha (internal consistency)
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples,
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = "D:/A_emotions in exam/results"
OUT_DIR     = os.path.join(RESULTS_DIR, "validation")
os.makedirs(OUT_DIR, exist_ok=True)

D1 = os.path.join(RESULTS_DIR, "features_D1.csv")
D2 = os.path.join(RESULTS_DIR, "features_D2.csv")
D3 = os.path.join(RESULTS_DIR, "features_D3.csv")
D4 = os.path.join(RESULTS_DIR, "features_D4.csv")
D5 = os.path.join(RESULTS_DIR, "features_D5.csv")

# ── feature selection  (Tier-1 + top Tier-2, by branch) ────────────────────────
# Each tuple: (csv_column_name, EI_branch, higher_is_better)
SELECTED = [
    # ---- Branch: Perceiving Emotions ----
    ("c1_emotion_differentiation_ICC",    "Perceiving",    True),
    ("c1_neg_emotion_differentiation",    "Perceiving",    True),
    ("c4_facial_valence_accuracy",        "Perceiving",    True),

    # ---- Branch: Using Emotions ----
    ("c5_cognitive_effort_deployment",    "Using",         True),
    ("c5_interest_during_ideation",       "Using",         True),
    ("c3_productive_confusion_index",     "Using",         True),

    # ---- Branch: Understanding Emotions ----
    ("c4_stage_emotion_appropriateness",  "Understanding", True),
    ("c4_valence_arousal_discrimination", "Understanding", True),
    ("c4_temporal_coherence",             "Understanding", True),

    # ---- Branch: Managing Emotions ----
    ("c2_recovery_completeness",          "Managing",      True),
    ("c2_stress_recovery_latency",        "Managing",      False),  # lower = better EI
    ("c3_bounce_back_index",              "Managing",      True),
    ("c2_regulation_strategy_switching",  "Managing",      True),
]

FEATURE_COLS  = [f for f, _, _ in SELECTED]
BRANCHES      = [b for _, b, _ in SELECTED]
HIGHER_BETTER = [h for _, _, h in SELECTED]
BRANCH_ORDER  = ["Perceiving", "Using", "Understanding", "Managing"]
BRANCH_COLORS = {
    "Perceiving":    "#4C72B0",
    "Using":         "#55A868",
    "Understanding": "#C44E52",
    "Managing":      "#8172B2",
}

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
rng         = np.random.default_rng(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1  :  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Merge D1-D5 on pid.
    pct column appears in multiple CSVs — extract once from D1 and drop
    all copies before merging feature columns.
    """
    pct_series = None
    dfs        = []

    for path in [D1, D2, D3, D4, D5]:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df = df.set_index("pid")

        # grab pct from the first file that has it
        if pct_series is None and "pct" in df.columns:
            pct_series = df["pct"].copy()

        # drop pct and score from feature columns
        df = df.drop(columns=[c for c in ("pct", "score") if c in df.columns])
        dfs.append(df)

    merged = pd.concat(dfs, axis=1)

    # deduplicate any remaining column name collisions (keep first)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]

    if pct_series is not None:
        merged["pct"] = pct_series

    return merged


def prepare_features(merged):
    """Select, impute, flip sign, and standardise features."""
    available    = [f for f in FEATURE_COLS if f in merged.columns]
    missing_cols = [f for f in FEATURE_COLS if f not in merged.columns]
    if missing_cols:
        print(f"[WARN] {len(missing_cols)} requested features not found — skipping:")
        for c in missing_cols:
            print(f"       {c}")

    idx                = [FEATURE_COLS.index(f) for f in available]
    branches_used      = [BRANCHES[i]      for i in idx]
    higher_better_used = [HIGHER_BETTER[i] for i in idx]

    X_raw = merged[available].copy()

    # Impute missing with column median (conservative for small n)
    imputer  = SimpleImputer(strategy="median")
    X_imp    = pd.DataFrame(
        imputer.fit_transform(X_raw),
        index=X_raw.index, columns=available,
    )

    # Flip sign for "lower is better" features so all point "higher = better EI"
    for col, hb in zip(available, higher_better_used):
        if not hb:
            X_imp[col] = -X_imp[col]

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imp),
        index=X_imp.index, columns=available,
    )

    pct = merged["pct"].copy() if "pct" in merged.columns else pd.Series(dtype=float)

    return X_scaled, pct, available, branches_used


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2  :  EI SCORE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _orient_pc1(scores, loadings):
    """Flip PC1 so that the majority of loadings are positive (higher = higher EI)."""
    if np.sum(loadings > 0) < np.sum(loadings < 0):
        return -scores
    return scores


def compute_ei_scores(X_scaled, feature_cols, branches_used):
    """
    Two methods returned together:
      A) PCA composite  : PC1 score (data-driven, max variance direction)
      B) Branch-weighted: equal branch weight, mean of standardised features
         within each branch then mean across branches.

    Returns (ei_pca, ei_composite, fitted_pca_object).
    """
    # ── Method A: PCA ──────────────────────────────────────────────────────────
    pca = PCA(random_state=RANDOM_SEED)
    pca.fit(X_scaled)
    pc1 = pca.transform(X_scaled.values)[:, 0]
    pc1 = _orient_pc1(pc1, pca.components_[0])
    ei_pca = pd.Series(pc1, index=X_scaled.index, name="EI_PCA")

    # ── Method B: Branch-weighted composite ────────────────────────────────────
    branch_means = {}
    for branch in BRANCH_ORDER:
        cols = [c for c, b in zip(feature_cols, branches_used) if b == branch]
        if cols:
            branch_means[branch] = X_scaled[cols].mean(axis=1)

    ei_composite = pd.concat(branch_means, axis=1).mean(axis=1)
    ei_composite.name = "EI_Composite"

    return ei_pca, ei_composite, pca


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3  :  VALIDATION LAYERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Layer 1 : Internal cluster validity ───────────────────────────────────────

def validate_internal(X_scaled, ei_pca, ei_composite):
    """
    Cluster students into EI tiers (low / mid / high) using k=3 k-means on
    each EI score, then evaluate partition quality against the full feature
    matrix (Silhouette, Davies-Bouldin, Calinski-Harabasz).

    Also evaluate optimal k using Silhouette across k=2..5.
    """
    Xnp    = X_scaled.values
    results = {}

    for name, scores in [("EI_PCA", ei_pca), ("EI_Composite", ei_composite)]:
        S = scores.values.reshape(-1, 1)

        # Fixed k=3 metrics
        km3    = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=20)
        labels = km3.fit_predict(S)
        sil_full = silhouette_score(Xnp, labels)
        sil_1d   = silhouette_score(S,   labels)
        db        = davies_bouldin_score(Xnp, labels)
        ch        = calinski_harabasz_score(Xnp, labels)

        # Optimal k sweep (k=2..5)
        k_sil = {}
        for k in range(2, 6):
            km_k = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
            lbl  = km_k.fit_predict(S)
            k_sil[k] = round(silhouette_score(Xnp, lbl), 4)

        results[name] = {
            "k=3 Silhouette (full features)": round(sil_full, 4),
            "k=3 Silhouette (1-D score)":     round(sil_1d,   4),
            "k=3 Davies-Bouldin  [lower=better]": round(db, 4),
            "k=3 Calinski-Harabasz [higher=better]": round(ch, 2),
            "Silhouette by k (2-5)": k_sil,
            "best_k": max(k_sil, key=k_sil.get),
        }

    return results


# ── Layer 2 : Construct validity (Spearman ρ with exam pct) ──────────────────

def validate_construct(ei_pca, ei_composite, pct):
    """
    Spearman rank correlation between each EI score and exam percentage.
    EI theory predicts a modest positive correlation (ρ ≈ 0.2–0.5).
    Also reports Pearson r for completeness.
    """
    results   = {}
    pct_clean = pct.dropna()

    if len(pct_clean) == 0:
        print("[WARN] No pct values found — skipping construct validity.")
        return results

    for name, scores in [("EI_PCA", ei_pca), ("EI_Composite", ei_composite)]:
        common       = scores.index.intersection(pct_clean.index)
        s_aligned    = scores.loc[common].dropna()
        pct_aligned  = pct_clean.loc[s_aligned.index]

        rho, p_sp = stats.spearmanr(s_aligned, pct_aligned)
        r,   p_pe = stats.pearsonr(s_aligned,  pct_aligned)

        results[name] = {
            "Spearman rho":       round(float(rho), 4),
            "Spearman p-value":   round(float(p_sp), 4),
            "Pearson r":          round(float(r), 4),
            "Pearson p-value":    round(float(p_pe), 4),
            "n":                  int(len(s_aligned)),
            "interpretation":     _interpret_rho(rho, p_sp),
        }

    return results


def _interpret_rho(rho, p):
    sig = "significant" if p < 0.05 else ("trend p<.10" if p < 0.10 else "n.s.")
    mag = "strong" if abs(rho) > 0.5 else ("moderate" if abs(rho) > 0.3 else "weak")
    return f"{mag}, {sig}"


# ── Layer 3 : Bootstrap stability (CI on EI score) ───────────────────────────

def validate_bootstrap(X_scaled, feature_cols, branches_used):
    """
    Bootstrap resamples students (with replacement) n=1000 times.

    For PCA   : refit PCA on each bootstrap sample, project all students,
                z-normalise. Captures model instability.
    For Composite: also resample students each iteration to capture how the
                cohort composition affects each student's relative position.

    Reports per-student 95% CI width. Narrow CI (< 1.0 SDs) = stable score.
    """
    n          = len(X_scaled)
    boot_pca   = np.zeros((N_BOOTSTRAP, n))
    boot_comp  = np.zeros((N_BOOTSTRAP, n))

    for i in range(N_BOOTSTRAP):
        idx    = rng.integers(0, n, size=n)      # resample student indices
        X_boot = X_scaled.iloc[idx].copy()

        # PCA: refit on bootstrap sample, project all students
        pca_b    = PCA(random_state=RANDOM_SEED)
        pca_b.fit(X_boot)
        pc1_b    = pca_b.transform(X_scaled.values)[:, 0]
        pc1_b    = _orient_pc1(pc1_b, pca_b.components_[0])
        std_b    = pc1_b.std() + 1e-9
        boot_pca[i] = (pc1_b - pc1_b.mean()) / std_b

        # Composite: recompute branch means on bootstrap cohort,
        # then project all students relative to bootstrap mean/std
        branch_vals = []
        for branch in BRANCH_ORDER:
            cols = [c for c, b in zip(feature_cols, branches_used) if b == branch]
            if cols:
                # mean and std estimated from bootstrap cohort
                boot_branch_mean = X_boot[cols].mean(axis=1).mean()
                boot_branch_std  = X_boot[cols].mean(axis=1).std() + 1e-9
                full_branch      = X_scaled[cols].mean(axis=1)
                branch_vals.append(((full_branch - boot_branch_mean) / boot_branch_std).values)
        comp_b       = np.stack(branch_vals).mean(axis=0)
        boot_comp[i] = (comp_b - comp_b.mean()) / (comp_b.std() + 1e-9)

    ci_pca_w  = np.percentile(boot_pca,  97.5, axis=0) - np.percentile(boot_pca,  2.5, axis=0)
    ci_comp_w = np.percentile(boot_comp, 97.5, axis=0) - np.percentile(boot_comp, 2.5, axis=0)

    return {
        "EI_PCA": {
            "mean_95CI_width": round(float(ci_pca_w.mean()), 4),
            "max_95CI_width":  round(float(ci_pca_w.max()),  4),
            "per_student_CI":  dict(zip(X_scaled.index, np.round(ci_pca_w, 4))),
        },
        "EI_Composite": {
            "mean_95CI_width": round(float(ci_comp_w.mean()), 4),
            "max_95CI_width":  round(float(ci_comp_w.max()),  4),
            "per_student_CI":  dict(zip(X_scaled.index, np.round(ci_comp_w, 4))),
        },
        "_boot_pca":  boot_pca,
        "_boot_comp": boot_comp,
    }


# ── Layer 4 : LOO robustness (score sensitivity to each student) ──────────────

def validate_loo(X_scaled, feature_cols, branches_used):
    """
    Leave-one-out: for each student, remove them, refit the scoring model,
    re-project all students, then measure how much the held-out student's
    z-score shifted vs the full-sample model.

    High deviation = that student is a strong anchor for the model.
    Mean deviation < 0.3 SD = robust.

    PCA  : refit PCA on n-1 students, project all n, compare held-out.
    Composite: refit StandardScaler on n-1 students to capture re-centering.
    """
    n    = len(X_scaled)
    pids = list(X_scaled.index)

    # Full-sample reference scores (z-normalised)
    full_pca, full_comp, _ = compute_ei_scores(X_scaled, feature_cols, branches_used)
    fp_z = (full_pca  - full_pca.mean())  / (full_pca.std()  + 1e-9)
    fc_z = (full_comp - full_comp.mean()) / (full_comp.std() + 1e-9)

    loo_dev_pca  = {}
    loo_dev_comp = {}

    for pid in pids:
        mask  = [p != pid for p in pids]
        X_loo = X_scaled.loc[[p for p in pids if p != pid]]

        # ── PCA LOO ────────────────────────────────────────────────────────────
        pca_loo = PCA(random_state=RANDOM_SEED)
        pca_loo.fit(X_loo)
        scores_loo = pca_loo.transform(X_scaled.values)[:, 0]
        scores_loo = _orient_pc1(scores_loo, pca_loo.components_[0])
        scores_loo = (scores_loo - scores_loo.mean()) / (scores_loo.std() + 1e-9)

        pid_idx           = pids.index(pid)
        loo_dev_pca[pid]  = float(abs(scores_loo[pid_idx] - fp_z.iloc[pid_idx]))

        # ── Composite LOO ──────────────────────────────────────────────────────
        # Refit StandardScaler on the n-1 cohort, transform all n students,
        # recompute composite — captures sensitivity to cohort mean/std
        scaler_loo  = StandardScaler()
        scaler_loo.fit(X_loo)
        X_loo_scaled = pd.DataFrame(
            scaler_loo.transform(X_scaled.values),
            index=pids, columns=X_scaled.columns,
        )
        branch_vals = []
        for branch in BRANCH_ORDER:
            cols = [c for c, b in zip(feature_cols, branches_used) if b == branch]
            if cols:
                branch_vals.append(X_loo_scaled[cols].mean(axis=1).values)
        comp_loo      = np.stack(branch_vals).mean(axis=0)
        comp_loo_z    = (comp_loo - comp_loo.mean()) / (comp_loo.std() + 1e-9)
        loo_dev_comp[pid] = float(abs(comp_loo_z[pid_idx] - fc_z.iloc[pid_idx]))

    return {
        "EI_PCA": {
            "mean_LOO_deviation": round(float(np.mean(list(loo_dev_pca.values()))),  4),
            "max_LOO_deviation":  round(float(np.max(list(loo_dev_pca.values()))),   4),
            "most_influential":   max(loo_dev_pca, key=loo_dev_pca.get),
            "per_student":        {k: round(v, 4) for k, v in loo_dev_pca.items()},
        },
        "EI_Composite": {
            "mean_LOO_deviation": round(float(np.mean(list(loo_dev_comp.values()))), 4),
            "max_LOO_deviation":  round(float(np.max(list(loo_dev_comp.values()))),  4),
            "most_influential":   max(loo_dev_comp, key=loo_dev_comp.get),
            "per_student":        {k: round(v, 4) for k, v in loo_dev_comp.items()},
        },
    }


# ── Layer 5 : PCA diagnostics ────────────────────────────────────────────────

def validate_pca_diagnostics(X_scaled, pca, feature_cols, branches_used):
    """
    Explained variance per PC, PC1 loadings per feature,
    communality h² (sum of squared loadings on first 3 PCs),
    and mean communality per EI branch.
    """
    evr      = pca.explained_variance_ratio_
    loadings = pca.components_               # shape (n_pcs, n_features)
    n_pcs    = min(3, loadings.shape[0])

    # h² = proportion of each feature's variance captured by first n_pcs PCs
    communality = np.sum(loadings[:n_pcs] ** 2, axis=0)

    branch_communality = {}
    for branch in BRANCH_ORDER:
        idx = [i for i, b in enumerate(branches_used) if b == branch]
        if idx:
            branch_communality[branch] = round(float(communality[idx].mean()), 4)

    loading_df = pd.DataFrame({
        "feature":        feature_cols,
        "branch":         branches_used,
        "PC1_loading":    np.round(loadings[0], 4),
        "PC2_loading":    np.round(loadings[1], 4) if loadings.shape[0] > 1 else np.nan,
        "communality_h2": np.round(communality, 4),
    }).sort_values("PC1_loading", ascending=False).reset_index(drop=True)

    return {
        "explained_variance_ratio":  [round(float(v), 4) for v in evr[:6]],
        "cumulative_top3_PCs":       round(float(evr[:3].sum()), 4),
        "per_branch_communality":    branch_communality,
        "loading_table":             loading_df,
    }


# ── Layer 6 : Convergent validity (Cronbach alpha per branch) ─────────────────

def validate_convergent(X_scaled, feature_cols, branches_used):
    """
    Cronbach's alpha per EI branch — measures internal consistency of features
    within the same branch. Alpha > 0.6 is acceptable for exploratory work.
    Also computes mean inter-item correlation per branch.
    """
    results = {}
    for branch in BRANCH_ORDER:
        cols = [c for c, b in zip(feature_cols, branches_used) if b == branch]
        if len(cols) < 2:
            results[branch] = {
                "alpha":             None,
                "n_items":           len(cols),
                "mean_inter_item_r": None,
                "note":              "need ≥2 items",
            }
            continue

        mat       = X_scaled[cols].values
        n_items   = mat.shape[1]
        item_var  = mat.var(axis=0, ddof=1).sum()
        total_var = mat.sum(axis=1).var(ddof=1)
        alpha     = (n_items / (n_items - 1)) * (1 - item_var / total_var) if total_var > 0 else np.nan

        # Mean inter-item Pearson r
        corr_matrix   = np.corrcoef(mat.T)
        upper_indices = np.triu_indices(n_items, k=1)
        mean_iic      = float(corr_matrix[upper_indices].mean())

        results[branch] = {
            "alpha":             round(float(alpha), 4) if not np.isnan(alpha) else None,
            "n_items":           n_items,
            "mean_inter_item_r": round(mean_iic, 4),
            "quality":           _alpha_quality(alpha),
        }

    return results


def _alpha_quality(a):
    if a is None or np.isnan(a): return "undefined"
    if a >= 0.80: return "good"
    if a >= 0.70: return "acceptable"
    if a >= 0.60: return "questionable — ok for exploratory"
    return "poor — review branch membership"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4  :  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_ei_overview(ei_pca, ei_composite, pct, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel A: distributions ──────────────────────────────────────────────
    ax = axes[0]
    ax.hist(ei_pca.values,       bins=10, alpha=0.65, label="EI_PCA",       color="#4C72B0")
    ax.hist(ei_composite.values, bins=10, alpha=0.65, label="EI_Composite", color="#DD8452")
    ax.set_title("EI Score Distributions")
    ax.set_xlabel("EI Score")
    ax.set_ylabel("Count")
    ax.legend()

    # ── Panel B: EI_PCA vs exam pct ────────────────────────────────────────
    ax  = axes[1]
    df1 = pd.DataFrame({"x": ei_pca, "y": pct}).dropna()
    if len(df1) >= 3:
        rho, p = stats.spearmanr(df1["x"], df1["y"])
        ax.scatter(df1["x"], df1["y"], color="#4C72B0", alpha=0.75, edgecolors="white", s=60)
        m, b = np.polyfit(df1["x"], df1["y"], 1)
        xs   = np.linspace(df1["x"].min(), df1["x"].max(), 100)
        ax.plot(xs, m * xs + b, "r--", lw=1.5)
        ax.set_title(f"EI_PCA vs Exam %\nSpearman ρ={rho:.3f}, p={p:.3f}")
    else:
        ax.set_title("EI_PCA vs Exam % (insufficient pct data)")
    ax.set_xlabel("EI_PCA")
    ax.set_ylabel("Exam %")

    # ── Panel C: EI_Composite vs exam pct ──────────────────────────────────
    ax  = axes[2]
    df2 = pd.DataFrame({"x": ei_composite, "y": pct}).dropna()
    if len(df2) >= 3:
        rho2, p2 = stats.spearmanr(df2["x"], df2["y"])
        ax.scatter(df2["x"], df2["y"], color="#DD8452", alpha=0.75, edgecolors="white", s=60)
        m2, b2 = np.polyfit(df2["x"], df2["y"], 1)
        xs2    = np.linspace(df2["x"].min(), df2["x"].max(), 100)
        ax.plot(xs2, m2 * xs2 + b2, "r--", lw=1.5)
        ax.set_title(f"EI_Composite vs Exam %\nSpearman ρ={rho2:.3f}, p={p2:.3f}")
    else:
        ax.set_title("EI_Composite vs Exam % (insufficient pct data)")
    ax.set_xlabel("EI_Composite")
    ax.set_ylabel("Exam %")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ei_score_overview.png"), dpi=150)
    plt.close()
    print(f"  Saved: ei_score_overview.png")


def plot_pca_diagnostics(pca, loading_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Scree plot ──────────────────────────────────────────────────────────
    ax  = axes[0]
    evr = pca.explained_variance_ratio_[:10] * 100
    ax.bar(range(1, len(evr) + 1), evr, color="#4C72B0", alpha=0.8)
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr), "o-", color="#DD8452",
            lw=1.5, label="Cumulative %")
    ax.axhline(70, color="gray", linestyle="--", lw=0.8, label="70% threshold")
    ax.set_title("PCA Scree Plot")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.legend(fontsize=8)

    # ── PC1 loadings ────────────────────────────────────────────────────────
    ax     = axes[1]
    colors = [BRANCH_COLORS[b] for b in loading_df["branch"]]
    ax.barh(loading_df["feature"], loading_df["PC1_loading"],
            color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("PC1 Loadings by Feature")
    ax.set_xlabel("PC1 Loading")
    handles = [mpatches.Patch(color=v, label=k) for k, v in BRANCH_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_diagnostics.png"), dpi=150)
    plt.close()
    print(f"  Saved: pca_diagnostics.png")


def plot_bootstrap_ci(boot_results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, key, color in zip(axes,
                               ["EI_PCA", "EI_Composite"],
                               ["#4C72B0", "#DD8452"]):
        ci_w   = list(boot_results[key]["per_student_CI"].values())
        pids   = list(boot_results[key]["per_student_CI"].keys())
        order  = np.argsort(ci_w)
        mean_w = boot_results[key]["mean_95CI_width"]

        ax.barh([pids[i] for i in order], [ci_w[i] for i in order],
                color=color, alpha=0.8, edgecolor="white")
        ax.axvline(mean_w, color="red", linestyle="--", lw=1.5,
                   label=f"Mean = {mean_w:.3f}")
        ax.set_title(f"Bootstrap 95% CI Width — {key}")
        ax.set_xlabel("CI Width (z-score units)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bootstrap_ci.png"), dpi=150)
    plt.close()
    print(f"  Saved: bootstrap_ci.png")


def plot_silhouette(X_scaled, ei_pca, out_dir):
    """Per-student silhouette plot for k=3 clustering on EI_PCA."""
    km     = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=20)
    labels = km.fit_predict(ei_pca.values.reshape(-1, 1))
    sil_v  = silhouette_samples(X_scaled.values, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#4C72B0", "#DD8452", "#55A868"]
    y_lower = 10
    for k in range(3):
        k_vals = np.sort(sil_v[labels == k])
        size   = k_vals.shape[0]
        ax.fill_betweenx(np.arange(y_lower, y_lower + size), 0, k_vals,
                         alpha=0.75, color=colors[k], label=f"Tier {k+1}")
        y_lower += size + 5

    mean_sil = sil_v.mean()
    ax.axvline(mean_sil, color="red", linestyle="--",
               label=f"Mean = {mean_sil:.3f}")
    ax.set_title("Per-Student Silhouette (k=3 EI Tiers)")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Students")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "silhouette_plot.png"), dpi=150)
    plt.close()
    print(f"  Saved: silhouette_plot.png")


def plot_feature_heatmap(X_scaled, feature_cols, ei_pca, out_dir):
    """Feature value heatmap with students sorted by EI_PCA."""
    df_plot = X_scaled[feature_cols].copy()
    df_plot["EI_PCA"] = ei_pca
    df_plot = df_plot.sort_values("EI_PCA")

    fig, ax = plt.subplots(figsize=(max(len(feature_cols) + 2, 14), 10))
    sns.heatmap(df_plot.T, cmap="RdBu_r", center=0, ax=ax,
                xticklabels=df_plot.index,
                yticklabels=True,
                linewidths=0.2, linecolor="lightgrey")
    ax.set_title("Feature Heatmap — Students sorted by EI_PCA (low → high)")
    ax.set_xlabel("Student (sorted)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_heatmap.png"), dpi=150)
    plt.close()
    print(f"  Saved: feature_heatmap.png")


def plot_loo_deviation(loo_results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, color in zip(axes,
                               ["EI_PCA", "EI_Composite"],
                               ["#4C72B0", "#DD8452"]):
        devs   = list(loo_results[key]["per_student"].values())
        pids   = list(loo_results[key]["per_student"].keys())
        order  = np.argsort(devs)[::-1]
        mean_d = loo_results[key]["mean_LOO_deviation"]

        ax.bar([pids[i] for i in order], [devs[i] for i in order],
               color=color, alpha=0.8, edgecolor="white")
        ax.axhline(mean_d, color="red", linestyle="--", lw=1.5,
                   label=f"Mean = {mean_d:.3f}")
        ax.axhline(0.3, color="orange", linestyle=":", lw=1.2,
                   label="Threshold 0.3")
        ax.set_title(f"LOO Score Deviation — {key}")
        ax.set_ylabel("|z-score shift|")
        ax.set_xlabel("Student")
        ax.tick_params(axis="x", rotation=70)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loo_deviation.png"), dpi=150)
    plt.close()
    print(f"  Saved: loo_deviation.png")


def plot_branch_alpha(convergent_results, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    branches = [b for b, r in convergent_results.items() if r["alpha"] is not None]
    alphas   = [convergent_results[b]["alpha"] for b in branches]
    colors   = [BRANCH_COLORS[b] for b in branches]
    ax.bar(branches, alphas, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0.7, color="green",  linestyle="--", lw=1.2, label="Acceptable (0.7)")
    ax.axhline(0.6, color="orange", linestyle=":",  lw=1.2, label="Minimum (0.6)")
    ax.set_title("Cronbach Alpha per EI Branch")
    ax.set_ylabel("Alpha")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "branch_alpha.png"), dpi=150)
    plt.close()
    print(f"  Saved: branch_alpha.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5  :  CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(internal, construct, bootstrap, loo, pca_diag, convergent,
                 ei_pca, ei_composite, pct):
    SEP  = "=" * 72
    SEP2 = "-" * 72

    print(f"\n{SEP}")
    print("  EI SCORE VALIDATION REPORT")
    print(SEP)

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    print("\n▌ LAYER 1 — INTERNAL CLUSTER VALIDITY  (k=3 EI tiers)")
    print("  Silhouette > 0.3 = reasonable  |  DB lower = better  |  CH higher = better")
    for method, m in internal.items():
        print(f"\n  [{method}]")
        for k, v in m.items():
            print(f"    {str(k):50s}: {v}")

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("▌ LAYER 2 — CONSTRUCT VALIDITY  (vs exam %)")
    print("  Expected: Spearman ρ ≈ 0.2–0.5  (EI theory)")
    if construct:
        for method, m in construct.items():
            print(f"\n  [{method}]")
            for k, v in m.items():
                print(f"    {str(k):50s}: {v}")
    else:
        print("  [SKIPPED — no pct data]")

    # ── Layer 3 ──────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"▌ LAYER 3 — BOOTSTRAP STABILITY  (n={N_BOOTSTRAP} resamples)")
    print("  95% CI width < 1.0 SD = stable score")
    for method in ["EI_PCA", "EI_Composite"]:
        print(f"\n  [{method}]")
        print(f"    {'Mean 95% CI width':50s}: {bootstrap[method]['mean_95CI_width']}")
        print(f"    {'Max  95% CI width':50s}: {bootstrap[method]['max_95CI_width']}")
        # flag unstable students (CI > 1.5)
        unstable = {pid: w for pid, w in bootstrap[method]["per_student_CI"].items() if w > 1.5}
        if unstable:
            print(f"    Unstable students (CI > 1.5): {unstable}")

    # ── Layer 4 ──────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("▌ LAYER 4 — LOO ROBUSTNESS")
    print("  Mean deviation < 0.30 SD = robust")
    for method in ["EI_PCA", "EI_Composite"]:
        print(f"\n  [{method}]")
        r = loo[method]
        print(f"    {'Mean LOO deviation':50s}: {r['mean_LOO_deviation']}")
        print(f"    {'Max  LOO deviation':50s}: {r['max_LOO_deviation']}")
        print(f"    {'Most influential student':50s}: {r['most_influential']}  "
              f"(dev={r['per_student'][r['most_influential']]})")

    # ── Layer 5 ──────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("▌ LAYER 5 — PCA DIAGNOSTICS")
    print(f"  Explained variance (PC1–PC6) : {pca_diag['explained_variance_ratio']}")
    print(f"  Cumulative top-3 PCs          : {pca_diag['cumulative_top3_PCs']}")
    print("\n  Per-branch communality h² (proportion of variance captured by 3 PCs):")
    for b, v in pca_diag["per_branch_communality"].items():
        print(f"    {b:20s}: {v}")
    print("\n  PC1 Loadings:")
    print(pca_diag["loading_table"].to_string(index=False))

    # ── Layer 6 ──────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("▌ LAYER 6 — CONVERGENT VALIDITY  (Cronbach α per branch)")
    print("  α > 0.70 = acceptable  |  > 0.60 = ok for exploratory research")
    for branch, r in convergent.items():
        print(f"\n  [{branch}]")
        for k, v in r.items():
            print(f"    {str(k):50s}: {v}")

    # ── Final EI scores ───────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("▌ FINAL EI SCORES  (sorted high → low on EI_PCA)")
    out = pd.DataFrame({
        "EI_PCA":       ei_pca,
        "EI_Composite": ei_composite,
        "Exam_%":       pct,
    }).sort_values("EI_PCA", ascending=False)
    print(out.round(4).to_string())
    print(f"\n{SEP}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  EI VALIDATION PIPELINE")
    print("=" * 72)

    print("\n[1/8] Loading data...")
    merged = load_data()

    print("[2/8] Preparing features...")
    X_scaled, pct, feature_cols, branches_used = prepare_features(merged)
    print(f"      {len(feature_cols)} features × {len(X_scaled)} students")

    print("[3/8] Computing EI scores...")
    ei_pca, ei_composite, pca = compute_ei_scores(X_scaled, feature_cols, branches_used)

    print("[4/8] Layer 1 — Internal validity...")
    internal = validate_internal(X_scaled, ei_pca, ei_composite)

    print("[5/8] Layer 2 — Construct validity...")
    construct = validate_construct(ei_pca, ei_composite, pct)

    print(f"[6/8] Layer 3 — Bootstrap stability (n={N_BOOTSTRAP})...")
    bootstrap = validate_bootstrap(X_scaled, feature_cols, branches_used)

    print("[7/8] Layer 4 — LOO robustness...")
    loo = validate_loo(X_scaled, feature_cols, branches_used)

    print("[7/8] Layer 5 — PCA diagnostics...")
    pca_diag = validate_pca_diagnostics(X_scaled, pca, feature_cols, branches_used)

    print("[7/8] Layer 6 — Convergent validity...")
    convergent = validate_convergent(X_scaled, feature_cols, branches_used)

    print("[8/8] Generating plots...")
    plot_ei_overview(ei_pca, ei_composite, pct, OUT_DIR)
    plot_pca_diagnostics(pca, pca_diag["loading_table"], OUT_DIR)
    plot_bootstrap_ci(bootstrap, OUT_DIR)
    plot_silhouette(X_scaled, ei_pca, OUT_DIR)
    plot_feature_heatmap(X_scaled, feature_cols, ei_pca, OUT_DIR)
    plot_loo_deviation(loo, OUT_DIR)
    plot_branch_alpha(convergent, OUT_DIR)

    print_report(internal, construct, bootstrap, loo, pca_diag, convergent,
                 ei_pca, ei_composite, pct)

    # ── Save scores ───────────────────────────────────────────────────────────
    scores_df = pd.DataFrame({
        "pid":          ei_pca.index,
        "EI_PCA":       ei_pca.values.round(4),
        "EI_Composite": ei_composite.values.round(4),
        "exam_pct":     pct.reindex(ei_pca.index).values,
    })
    out_csv = os.path.join(OUT_DIR, "ei_scores_validated.csv")
    scores_df.to_csv(out_csv, index=False)
    print(f"Scores saved  → {out_csv}")
    print(f"Plots saved   → {OUT_DIR}/")


if __name__ == "__main__":
    main()
