"""
Stage 4 — Formative Feature Selection
======================================
REDESIGNED: Unsupervised feature selection for formative constructs.

Goal: Select representative features per dimension (NOT predictive ranking).

For a formative construct, the right criteria are:
  1. Variance - features with near-zero variance carry no information
  2. Distributional quality - skewness/kurtosis within reasonable bounds
  3. Within-dimension representativeness - feature that best captures shared variance
  4. Bootstrap stability - does the feature consistently emerge across resamples?
  5. Simple composition - equal-weighted average of selected features

NO target variable used. NO supervised ranking. NO Ridge stacking.

Outputs:
  outputs/validation_report.txt
  outputs/selected_features.json
  outputs/ei_scores.pkl (with EI_stacked column)
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, CANDIDATE_FEATURES, DIM_WEIGHTS, FEATURE_DIRECTION

CORR_THRESHOLD = 0.70
MIN_STABILITY = 0.05  # Minimum stability threshold to include a feature

# Color palette for dimensions (for visualization)
DIM_COLORS = {
    "D1_EmotionPerception": "#1f77b4",
    "D2_EmotionRegulation": "#ff7f0e",
    "D3_AdaptivePersistence": "#2ca02c",
    "D4_AffectiveSelfAwareness": "#d62728",
    "D5_EmotionCognitionAlign": "#9467bd",
}


def compute_distributional_quality(norm_df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Compute variance, skewness, kurtosis. Flag problematic features."""
    avail = [f for f in features if f in norm_df.columns]
    rows = []
    for feat in avail:
        val = norm_df[feat].values
        var = np.var(val)
        skew = stats.skew(val)
        kurt = stats.kurtosis(val)
        rows.append({
            "feature": feat,
            "variance": var,
            "skewness": skew,
            "kurtosis": kurt,
            "flagged": (var < 1e-6) or (abs(skew) > 2.0) or (abs(kurt) > 3.0),
        })
    return pd.DataFrame(rows)


def correlation_clusters(features: list, norm_df: pd.DataFrame, threshold: float) -> list:
    """Group features with |r| > threshold into clusters."""
    avail = [f for f in features if f in norm_df.columns]
    if len(avail) <= 1:
        return [[f] for f in avail]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(norm_df[avail].values)
    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0)

    assigned = {f: None for f in avail}
    clusters = []
    for i, fi in enumerate(avail):
        if assigned[fi] is not None:
            continue
        cluster = [fi]
        assigned[fi] = len(clusters)
        for j, fj in enumerate(avail):
            if i == j or assigned[fj] is not None:
                continue
            if corr[i, j] > threshold:
                cluster.append(fj)
                assigned[fj] = len(clusters)
        clusters.append(cluster)

    return clusters


def select_cluster_representative(cluster: list, norm_df: pd.DataFrame) -> str:
    """
    Pick feature with highest average |correlation| with cluster siblings.
    This feature best represents the shared variance in the cluster.
    """
    if len(cluster) == 1:
        return cluster[0]
    avail = [f for f in cluster if f in norm_df.columns]
    if not avail:
        return cluster[0]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(norm_df[avail].values)
    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0)

    avg_corr = corr.mean(axis=1)
    best_idx = np.argmax(avg_corr)
    return avail[best_idx]


def bootstrap_stability(norm_df: pd.DataFrame, features: list, n_bootstrap: int = 100) -> dict:
    """
    Resample participants 100x. For each resample, rank features by variance.
    Count how often each feature appears in top-K.
    Stability = fraction of resamples where feature ranked in top-K.
    """
    avail = [f for f in features if f in norm_df.columns]
    stability = {f: 0 for f in avail}
    k = min(3, len(avail))  # Top K to track

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(norm_df), size=len(norm_df), replace=True)
        subset = norm_df.iloc[idx]
        var_ranking = subset[avail].var(axis=0).sort_values(ascending=False).index.tolist()
        for f in var_ranking[:k]:
            stability[f] += 1

    stability = {f: v / n_bootstrap for f, v in stability.items()}
    return stability


def run(norm_df=None, ei_scores=None):
    print("=" * 60)
    print("STAGE 4 — FORMATIVE FEATURE SELECTION")
    print("=" * 60)

    if norm_df is None or ei_scores is None:
        pkl_path = os.path.join(OUTPUT_DIR, "ei_scores.pkl")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        norm_df = data["norm_df"]
        ei_scores = data["scores"]

    report_lines = [
        "EI FEATURE SELECTION REPORT",
        "=" * 60,
        "FORMATIVE APPROACH (UNSUPERVISED)",
        f"Participants: {len(norm_df)}",
        "",
        "Selection criteria (no target variable):",
        "  1. Variance - remove near-zero variance features",
        "  2. Distributional quality - flag extreme skewness/kurtosis",
        "  3. Correlation clustering - group redundant features",
        "  4. Representativeness - pick feature that best captures cluster variance",
        "  5. Bootstrap stability - select features that consistently emerge",
        "",
        "Composition: Equal-weighted average within dimension,",
        "            Theory-weighted average across dimensions",
        "",
    ]

    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    all_selected = {}
    stability_all = {}

    for dim_name, feat_dict in CANDIDATE_FEATURES.items():
        # feat_dict is {feature_name: direction}, extract feature names
        feat_list = list(feat_dict.keys())
        avail = [f for f in feat_list if f in norm_df.columns]
        if not avail:
            print(f"  {dim_name}: No features available in dataset. Skipping.")
            continue

        report_lines.append(f"\n{'='*60}")
        report_lines.append(f"DIMENSION: {dim_name}")
        report_lines.append(f"{'='*60}")
        report_lines.append(f"Starting features: {len(avail)}")

        # Step 1: Quality check
        report_lines.append(f"\nStep 1 — Distributional quality:")
        quality_df = compute_distributional_quality(norm_df, avail)
        problem_feats = quality_df[quality_df["flagged"]]["feature"].tolist()
        quality_feats = quality_df[~quality_df["flagged"]]["feature"].tolist()

        if problem_feats:
            report_lines.append(f"  Flagged (low variance or extreme skewness/kurtosis): {len(problem_feats)}")
        report_lines.append(f"  Retained: {len(quality_feats)}")

        # Step 2: Clustering
        report_lines.append(f"\nStep 2 — Correlation clustering (|r| > {CORR_THRESHOLD}):")
        clusters = correlation_clusters(quality_feats, norm_df, CORR_THRESHOLD)
        report_lines.append(f"  Found {len(clusters)} clusters")

        # Step 3: Representatives
        report_lines.append(f"\nStep 3 — Select cluster representative:")
        representatives = []
        for cluster in clusters:
            rep = select_cluster_representative(cluster, norm_df)
            representatives.append(rep)

        # Step 4: Selection logic (varies by dimension type)
        if dim_name == "D4_AffectiveSelfAwareness":
            # D4 is reflective: use cluster representatives directly
            # (high inter-correlation is expected; select by cluster coverage, not variance)
            report_lines.append(f"\nStep 4 — Reflective dimension (cluster-based selection):")
            report_lines.append(f"  Using all {len(representatives)} cluster representatives")
            report_lines.append(f"  (No stability filter; each cluster represents a facet)")
            selected = representatives
            stability_all[dim_name] = {f: 1.0 for f in selected}  # Mark as selected
            report_lines.append(f"  Selected {len(selected)} features (one per cluster):")
            for f in sorted(selected):
                report_lines.append(f"    {f:45s}: (cluster representative)")
        else:
            # Other dimensions are formative: use bootstrap stability
            report_lines.append(f"\nStep 4 — Bootstrap stability (100 resamples):")
            stability = bootstrap_stability(norm_df, representatives, n_bootstrap=100)
            stability_all[dim_name] = stability

            # Select by stability: top features with stability >= MIN_STABILITY,
            # or top 3 by stability if none meet threshold
            ranked = sorted(stability.items(), key=lambda x: x[1], reverse=True)
            selected = [f for f, stab in ranked if stab >= MIN_STABILITY]
            if not selected:
                # Fallback: select top 3
                selected = [f for f, _ in ranked[:min(3, len(ranked))]]

            report_lines.append(f"  Stability threshold: >= {MIN_STABILITY:.2f}")
            report_lines.append(f"  Selected {len(selected)} features (from {len(ranked)} candidates):")
            for f, stab in ranked:
                marker = " ✓" if f in selected else ""
                report_lines.append(f"    {f:45s}: {stab:.2f}{marker}")

        all_selected[dim_name] = selected

    # Create visualizations
    fig, axes = plt.subplots(len(all_selected), 1, figsize=(12, 4 * len(all_selected)))
    if len(all_selected) == 1:
        axes = [axes]

    for idx, (dim_name, selected) in enumerate(all_selected.items()):
        if not selected:
            continue
        ax = axes[idx]
        stab_dict = stability_all.get(dim_name, {})
        features = list(stab_dict.keys())
        stab_scores = [stab_dict[f] for f in features]
        colors = [DIM_COLORS.get(dim_name, "steelblue") if f in selected else "lightgray"
                  for f in features]

        ax.barh(features, stab_scores, color=colors)
        ax.set_xlabel("Bootstrap Stability Score")
        ax.set_title(f"{dim_name}: Feature Stability (✓ = selected)")
        ax.set_xlim(0, 1)
        ax.axvline(MIN_STABILITY, color="red", linestyle="--", label=f"Min threshold ({MIN_STABILITY})")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    viz_path = os.path.join(figures_dir, "01_feature_stability.png")
    plt.savefig(viz_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Visualization -> {viz_path}")
    report_lines.insert(0, f"Visualization: {viz_path}\n")

    # Composite
    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"FINAL COMPOSITE")
    report_lines.append(f"{'='*60}")

    dim_scores = {}
    for dim_name, selected in all_selected.items():
        if selected:
            dim_scores[dim_name] = norm_df[selected].mean(axis=1)
        else:
            dim_scores[dim_name] = pd.Series(0.5, index=norm_df.index)

    stacked_scores = sum(DIM_WEIGHTS[d] * dim_scores[d] for d in DIM_WEIGHTS)

    # Summary
    report_lines.append(f"\nFINAL SELECTED FEATURES:")
    selected_flat = []
    for dim_name in sorted(all_selected.keys()):
        selected = all_selected[dim_name]
        selected_flat.extend(selected)
        report_lines.append(f"\n  {dim_name}: {len(selected)} features")
        for f in selected:
            report_lines.append(f"    - {f}")

    # Final summary
    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"SUMMARY STATISTICS")
    report_lines.append(f"{'='*60}")
    report_lines.append(f"Total dimensions: {len(all_selected)}")
    report_lines.append(f"Total features selected: {len(selected_flat)}")
    for dim_name, selected in all_selected.items():
        n_cand = len(CANDIDATE_FEATURES.get(dim_name, {}))
        report_lines.append(f"  {dim_name}: {len(selected)}/{n_cand} features")
    report_lines.append("")
    report_lines.append("Composition method:")
    report_lines.append("  - Within dimension: equal-weighted average of selected features")
    report_lines.append("  - Across dimensions: theory-weighted average (from DIM_WEIGHTS)")

    # Save
    report_path = os.path.join(OUTPUT_DIR, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report -> {report_path}")

    sel_path = os.path.join(OUTPUT_DIR, "selected_features.json")
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump({
            "selected_features": selected_flat,
            "by_dimension": all_selected,
            "method": "formative-unsupervised",
            "stability": {d: {k: float(v) for k, v in stability_all[d].items()} for d in stability_all},
        }, f, indent=2)
    print(f"Selected -> {sel_path}")

    # Save EI_stacked
    pkl_path = os.path.join(OUTPUT_DIR, "ei_scores.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    data["scores"]["EI_stacked"] = pd.Series(stacked_scores.values, index=norm_df.index, name="EI_stacked")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"EI_stacked saved")

    print(f"\nFINAL: {len(selected_flat)} features selected")
    return selected_flat, pd.Series({f: 1.0 for f in selected_flat}), all_selected


if __name__ == "__main__":
    run()
