"""
Stage 5 — Confirmatory Correlation with Exam Scores (Three Methods)
====================================================================
FIRST AND ONLY PLACE EXAM SCORES ARE LOADED.

Tests H1: EI composites correlate with exam performance.

Three competing methods:
  1. Stability-weighted composite — each feature weighted by bootstrap stability
     (Highest explainability: direct feature→contribution trace)

  2. Network centrality — features weighted by centrality in correlation network
     (Moderate explainability: network visualization intuitively shows importance)

  3. PCA within dimension — PC1 from each dimension, theory-weighted across
     (Weak explainability: PC1 is mathematical abstraction, loses feature meanings)

Outputs:
  outputs/confirmatory_results.txt
  outputs/ei_comparison.csv
  outputs/figures/ei_comparison_scatter.png
  outputs/figures/ei_method_correlation.png
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, EXAM_FILE, DIM_WEIGHTS

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Statistical helpers ───────────────────────────────────────────────────────

def permutation_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int = 10000, seed: int = 42) -> float:
    """Permutation p-value (two-tailed) for Pearson r."""
    rng = np.random.RandomState(seed)
    obs = abs(pearsonr(x, y)[0])
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        perm_r = abs(pearsonr(x, y_perm)[0])
        if perm_r >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def bca_bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    """BCa bootstrap CI for Pearson r."""
    rng = np.random.RandomState(seed)
    obs = pearsonr(x, y)[0]
    boot_stats = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            boot_stats.append(pearsonr(x[idx], y[idx])[0])
        except:
            pass
    boot_stats = np.array(boot_stats)
    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def load_exam_scores() -> pd.Series:
    """Load exam scores from Excel file."""
    df = pd.read_excel(EXAM_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Exam scores file columns: {list(df.columns)}")

    # Identify ID and score columns
    id_col = score_col = None
    for col in df.columns:
        lc = col.lower()
        if id_col is None and any(k in lc for k in ["participant", "id", "pid", "student"]):
            id_col = col
        if score_col is None and any(k in lc for k in ["score", "mark", "exam", "result", "grade", "total"]):
            score_col = col

    if id_col is None:
        id_col = df.columns[0]
    if score_col is None:
        score_col = df.columns[-1]

    print(f"Using ID: '{id_col}', Score: '{score_col}'")

    scores = df[[id_col, score_col]].dropna()
    import re
    def parse_pid(val):
        m = re.search(r'\d+', str(val))
        return int(m.group()) if m else None

    scores["pid"] = scores[id_col].apply(parse_pid)
    scores = scores.dropna(subset=["pid"])
    scores["pid"] = scores["pid"].astype(int)
    scores = scores.set_index("pid")[score_col].astype(float)
    print(f"Loaded exam scores for {len(scores)} participants")
    print(f"Score range: {scores.min():.1f} – {scores.max():.1f}\n")
    return scores


def compute_stability_weighted(norm_df, selected_features_by_dim, stability_all, dim_weights):
    """
    Method 1: Stability-weighted composite
    Each feature weighted by its bootstrap stability score.
    Highly explainable: every contribution is traceable.
    """
    dim_scores = {}
    feature_weights = {}

    for dim_name, selected in selected_features_by_dim.items():
        if not selected:
            dim_scores[dim_name] = pd.Series(0.5, index=norm_df.index)
            continue

        # Get stability scores for selected features
        stability = stability_all.get(dim_name, {})
        stabilities = np.array([stability.get(f, 1.0) for f in selected])

        # Normalize stability scores to sum to 1 (within-dimension weights)
        if stabilities.sum() > 0:
            within_dim_weights = stabilities / stabilities.sum()
        else:
            within_dim_weights = np.ones(len(selected)) / len(selected)

        # Store for reporting
        for f, w in zip(selected, within_dim_weights):
            feature_weights[f] = {"dim": dim_name, "within_dim_weight": float(w), "stability": float(stability.get(f, 1.0))}

        # Compute weighted average for this dimension
        dim_scores[dim_name] = (norm_df[selected].values * within_dim_weights).sum(axis=1)

    # Across dimensions: theory-weighted
    ei = sum(dim_weights[d] * dim_scores[d] for d in dim_weights)
    return ei, dim_scores, feature_weights


def compute_network_centrality(norm_df, selected_features_by_dim, dim_weights):
    """
    Method 2: Network centrality weighting
    Compute correlation network within each dimension; weight by eigenvector centrality.
    Moderately explainable: centrality visualization is intuitive.
    """
    from scipy.linalg import eigh

    dim_scores = {}
    feature_weights = {}

    for dim_name, selected in selected_features_by_dim.items():
        if not selected or len(selected) < 2:
            # Can't compute centrality for single feature
            dim_scores[dim_name] = norm_df[selected].mean(axis=1) if selected else pd.Series(0.5, index=norm_df.index)
            for f in selected:
                feature_weights[f] = {"dim": dim_name, "centrality_weight": 1.0 / len(selected)}
            continue

        # Compute correlation matrix
        X = norm_df[selected].values
        corr = np.abs(np.corrcoef(X.T))
        np.fill_diagonal(corr, 0)

        # Eigenvector centrality: largest eigenvalue's eigenvector
        try:
            eigenvalues, eigenvectors = eigh(corr)
            centrality = np.abs(eigenvectors[:, -1])  # largest eigenvalue's eigenvector
            centrality = centrality / centrality.sum()  # normalize
        except:
            centrality = np.ones(len(selected)) / len(selected)

        # Store for reporting
        for f, c in zip(selected, centrality):
            feature_weights[f] = {"dim": dim_name, "centrality_weight": float(c)}

        # Compute weighted average
        dim_scores[dim_name] = (norm_df[selected].values * centrality).sum(axis=1)

    # Across dimensions: theory-weighted
    ei = sum(dim_weights[d] * dim_scores[d] for d in dim_weights)
    return ei, dim_scores, feature_weights


def compute_pca_within_dimension(norm_df, selected_features_by_dim, dim_weights):
    """
    Method 3: PCA within each dimension
    Use PC1 from each dimension, weighted across by theory.
    Weak explainability: PC1 is abstract, loses feature meaning.
    """
    dim_scores = {}
    feature_weights = {}

    for dim_name, selected in selected_features_by_dim.items():
        if not selected or len(selected) < 2:
            # Can't do PCA on single feature
            dim_scores[dim_name] = norm_df[selected].mean(axis=1) if selected else pd.Series(0.5, index=norm_df.index)
            for f in selected:
                feature_weights[f] = {"dim": dim_name, "pca_loading": 1.0}
            continue

        # PCA: use PC1
        X = norm_df[selected].values
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X).ravel()

        # Get loadings
        loadings = np.abs(pca.components_[0])
        loadings = loadings / loadings.sum()  # normalize for reporting

        for f, load in zip(selected, loadings):
            feature_weights[f] = {"dim": dim_name, "pca_loading": float(load)}

        # Use PC1 as dimension score
        dim_scores[dim_name] = pc1

    # Across dimensions: theory-weighted
    ei = sum(dim_weights[d] * dim_scores[d] for d in dim_weights)
    return ei, dim_scores, feature_weights


def run():
    print("=" * 70)
    print("STAGE 5 — CONFIRMATORY ANALYSIS (three EI computation methods)")
    print("=" * 70)

    # Load data
    pkl_path = os.path.join(OUTPUT_DIR, "ei_scores.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    norm_df = data["norm_df"]

    sel_path = os.path.join(OUTPUT_DIR, "selected_features.json")
    with open(sel_path) as f:
        sel_data = json.load(f)
    selected_features_by_dim = sel_data["by_dimension"]
    stability_all = sel_data.get("stability", {})

    # Load exam scores
    exam = load_exam_scores()

    # Align indices
    common = sorted(set(norm_df.index) & set(exam.index))
    print(f"Participants with both EI and exam scores: {len(common)}\n")

    norm_df_aligned = norm_df.loc[common]
    exam_aligned = exam.loc[common].values

    # Compute three methods
    print("Computing three EI composites...")
    ei_stable, dims_stable, weights_stable = compute_stability_weighted(
        norm_df_aligned, selected_features_by_dim, stability_all, DIM_WEIGHTS)
    print("  [OK] Stability-weighted")

    ei_central, dims_central, weights_central = compute_network_centrality(
        norm_df_aligned, selected_features_by_dim, DIM_WEIGHTS)
    print("  [OK] Network centrality")

    ei_pca, dims_pca, weights_pca = compute_pca_within_dimension(
        norm_df_aligned, selected_features_by_dim, DIM_WEIGHTS)
    print("  [OK] PCA within dimension\n")

    # Test correlations
    report_lines = [
        "CONFIRMATORY ANALYSIS: THREE EI COMPUTATION METHODS",
        "=" * 60,
        f"N = {len(common)} participants",
        f"Exam score range: {exam_aligned.min():.1f} – {exam_aligned.max():.1f}",
        "",
        "Methods compared:",
        "  1. Stability-weighted: each feature weighted by bootstrap stability",
        "  2. Network centrality: each feature weighted by correlation network centrality",
        "  3. PCA within dimension: PC1 from each dim, theory-weighted across",
        "",
        "Evaluation criteria: |r| >= 0.30 AND p < 0.05 (permutation test)",
        "=" * 60,
    ]

    results = {}
    for label, ei in [("Stability-weighted", ei_stable),
                      ("Network centrality", ei_central),
                      ("PCA within dimension", ei_pca)]:
        r, p_raw = pearsonr(ei, exam_aligned)
        rho, p_spear = spearmanr(ei, exam_aligned)
        perm_p = permutation_pvalue(ei, exam_aligned)
        ci_lo, ci_hi = bca_bootstrap_ci(ei, exam_aligned)

        r2 = r ** 2
        f2 = r2 / (1 - r2) if r2 < 1 else np.nan

        support = abs(r) >= 0.30 and perm_p < 0.05

        report_lines.append(f"\n{label}")
        report_lines.append("-" * 60)
        report_lines.append(f"  Pearson r   = {r:+.4f}  (95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}])")
        report_lines.append(f"  Spearman ρ  = {rho:+.4f}")
        report_lines.append(f"  Permutation p (10k perms) = {perm_p:.4f}")
        report_lines.append(f"  Cohen's f² = {f2:.4f}")
        report_lines.append(f"  H1 {'SUPPORTED [OK]' if support else 'NOT SUPPORTED [FAIL]'} (|r|>=0.30: {abs(r)>=0.30}, p<0.05: {perm_p<0.05})")

        results[label] = {"r": r, "p": perm_p, "ci_lo": ci_lo, "ci_hi": ci_hi, "rho": rho, "f2": f2, "support": support}

    # Summary and recommendation
    report_lines.append("\n" + "=" * 60)
    report_lines.append("SUMMARY & RECOMMENDATION")
    report_lines.append("=" * 60)

    best_r = max(results.items(), key=lambda x: abs(x[1]["r"]))
    report_lines.append(f"\nBest correlation: {best_r[0]} (r = {best_r[1]['r']:+.4f})")

    report_lines.append("\nExplainability ranking (best → worst):")
    report_lines.append("  1. Stability-weighted — Every contribution is traceable.")
    report_lines.append("     Non-technical audiences can understand why a score is what it is.")
    report_lines.append("  2. Network centrality — Centrality is intuitive; network visualization helps.")
    report_lines.append("     Requires some justification but easier than PCA.")
    report_lines.append("  3. PCA within dimension — PC1 is abstract; loses feature meaning.")
    report_lines.append("     Reviewers typically object to this approach.")

    report_lines.append("\nRecommendation:")
    if results["Stability-weighted"]["support"]:
        report_lines.append("  [OK] Use STABILITY-WEIGHTED method (best explainability + significant)")
    elif results["Network centrality"]["support"]:
        report_lines.append("  [WARN] STABILITY-WEIGHTED fails; use NETWORK CENTRALITY (good balance)")
    elif results["PCA within dimension"]["support"]:
        report_lines.append("  [WARN] Both fail; fall back to PCA (last resort, poor explainability)")
    else:
        report_lines.append("  [FAIL] All methods fail to predict exam scores.")
        report_lines.append("     EI may not be predictive of exam performance in this cohort.")
        report_lines.append(f"     Best: {best_r[0]} (r = {best_r[1]['r']:+.4f}, p = {best_r[1]['p']:.4f})")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "confirmatory_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nReport saved → {report_path}")

    # Save results table
    results_df = pd.DataFrame({
        "Method": list(results.keys()),
        "Pearson_r": [results[m]["r"] for m in results.keys()],
        "Perm_p": [results[m]["p"] for m in results.keys()],
        "CI_lower": [results[m]["ci_lo"] for m in results.keys()],
        "CI_upper": [results[m]["ci_hi"] for m in results.keys()],
        "Spearman_rho": [results[m]["rho"] for m in results.keys()],
        "Cohens_f2": [results[m]["f2"] for m in results.keys()],
        "Significant": [results[m]["support"] for m in results.keys()],
    })
    csv_path = os.path.join(OUTPUT_DIR, "ei_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results table → {csv_path}\n")

    # Visualization: scatter plots for all three methods
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    methods = ["Stability-weighted", "Network centrality", "PCA within dimension"]
    eis = [ei_stable, ei_central, ei_pca]

    for ax, method, ei in zip(axes, methods, eis):
        ax.scatter(ei, exam_aligned, alpha=0.6, s=60, edgecolor="black", linewidth=0.5)

        # Regression line
        z = np.polyfit(ei, exam_aligned, 1)
        p = np.poly1d(z)
        ei_sorted = np.sort(ei)
        ax.plot(ei_sorted, p(ei_sorted), "r--", lw=2, alpha=0.7)

        r = results[method]["r"]
        p_val = results[method]["p"]
        ax.set_xlabel("EI Score", fontsize=11)
        ax.set_ylabel("Exam Score", fontsize=11)
        ax.set_title(f"{method}\nr = {r:+.3f}, p = {p_val:.4f}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "ei_comparison_scatter.png")
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Scatter plots → {fig_path}")

    # Visualization: correlation heatmap of the three methods
    fig, ax = plt.subplots(figsize=(6, 5))
    ei_comparison = pd.DataFrame({
        "Stability-weighted": ei_stable,
        "Network centrality": ei_central,
        "PCA within dim": ei_pca,
        "Exam score": exam_aligned
    })
    corr_matrix = ei_comparison.corr()

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.columns)

    # Add text
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    ax.set_title("Correlation Matrix: Three Methods + Exam Scores", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "ei_method_correlation.png")
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Correlation matrix → {fig_path}\n")

    return results


if __name__ == "__main__":
    run()
