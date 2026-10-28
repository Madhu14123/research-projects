"""
Stage 3 — EI Composite Construction
======================================
Builds the composite EI score from behavioral features alone.
Two complementary approaches:
  A) Theory-weighted composite (pre-specified dimension weights from PDF)
  B) PCA-based composite (factor analysis of behavioral signals, no exam scores)

NO exam scores used here.

Outputs:
  outputs/ei_scores.csv / .pkl  — per-participant EI scores
  outputs/normalized_features.csv
"""

import os, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")

from config import (
    OUTPUT_DIR, DIM_WEIGHTS, CANDIDATE_FEATURES, FEATURE_DIRECTION,
    ALL_CANDIDATES
)


def normalize_to_ei_direction(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip features with direction = -1 so that higher value always means higher EI.
    Then robust-scale each feature to [0, 1] range.
    """
    df = feat_df.copy()
    for feat, direction in FEATURE_DIRECTION.items():
        if feat in df.columns and direction == -1:
            df[feat] = -df[feat]
    return df


def scale_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute NaN with median, then RobustScale each column, then clip to [0,1]
    via min-max normalisation of the robust-scaled output.
    """
    cols = feat_df.columns.tolist()
    imputer = SimpleImputer(strategy="median")
    scaler  = RobustScaler()

    imputed = imputer.fit_transform(feat_df[cols])
    scaled  = scaler.fit_transform(imputed)
    # Min-max normalise column-wise to map to [0, 1]
    col_min = scaled.min(axis=0)
    col_max = scaled.max(axis=0)
    denom   = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    normalised = (scaled - col_min) / denom

    return pd.DataFrame(normalised, index=feat_df.index, columns=cols)


def build_theory_composite(norm_df: pd.DataFrame) -> pd.Series:
    """
    Approach A: Theory-weighted composite.
    Within each dimension: equal-weighted mean of all candidate features.
    Across dimensions: weights from DIM_WEIGHTS (0.15/0.25/0.25/0.15/0.20).
    """
    dim_scores = {}
    for dim_name, feat_dict in CANDIDATE_FEATURES.items():
        dim_feats = [f for f in feat_dict if f in norm_df.columns]
        if dim_feats:
            dim_scores[dim_name] = norm_df[dim_feats].mean(axis=1)
        else:
            dim_scores[dim_name] = pd.Series(0.5, index=norm_df.index)

    composite = sum(DIM_WEIGHTS[d] * dim_scores[d] for d in DIM_WEIGHTS)
    composite.name = "EI_theory"
    return composite, pd.DataFrame(dim_scores)


def build_pca_composite(norm_df: pd.DataFrame) -> pd.Series:
    """
    Approach B: PCA composite.
    Run PCA on all behavioral features, retain components explaining ≥80% variance.
    Composite = projection onto first principal component (orientation: higher = higher EI
    approximated by the component with most positive loadings for productive-stage features).
    """
    pca = PCA(n_components=None, random_state=42)
    components = pca.fit_transform(norm_df.values)

    # Use the first PC weighted by its explained variance share
    # Orient: if majority of D3/D4 features (known stronger EI indicators) load positively, keep as-is
    first_pc  = components[:, 0]
    # Check average loading for D3 features (strongest per literature)
    d3_feats  = list(CANDIDATE_FEATURES["D3_AdaptivePersistence"].keys())
    d3_idx    = [i for i, c in enumerate(norm_df.columns) if c in d3_feats]
    d3_loadings_mean = pca.components_[0, d3_idx].mean() if d3_idx else 0
    if d3_loadings_mean < 0:
        first_pc = -first_pc   # flip to ensure positive orientation

    # Normalise to [0, 1]
    first_pc_norm = (first_pc - first_pc.min()) / (first_pc.max() - first_pc.min() + 1e-10)

    print(f"  PCA: {pca.n_components_} components, "
          f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance")
    print(f"  Cumulative variance (top 5): "
          f"{np.cumsum(pca.explained_variance_ratio_[:5])*100}")

    series = pd.Series(first_pc_norm, index=norm_df.index, name="EI_pca")
    return series, pca


def run(feat_df=None):
    print("=" * 60)
    print("STAGE 3 — EI COMPOSITE CONSTRUCTION")
    print("=" * 60)

    if feat_df is None:
        pkl_path = os.path.join(OUTPUT_DIR, "feature_matrix_raw.pkl")
        with open(pkl_path, "rb") as f:
            feat_df = pickle.load(f)

    # Keep only candidate features
    available = [c for c in ALL_CANDIDATES if c in feat_df.columns]
    print(f"Available candidate features: {len(available)} / {len(ALL_CANDIDATES)}")

    feat_sub = feat_df[available].copy()

    # Step 1: flip directions + scale
    feat_sub_flipped = normalize_to_ei_direction(feat_sub)
    norm_df = scale_features(feat_sub_flipped)
    norm_df.to_csv(os.path.join(OUTPUT_DIR, "normalized_features.csv"))

    # Step 2A: Theory-weighted composite
    ei_theory, dim_score_df = build_theory_composite(norm_df)
    print(f"\n[A] Theory-weighted EI scores:")
    print(f"    Mean: {ei_theory.mean():.4f}  SD: {ei_theory.std():.4f}")
    print(f"    Min:  {ei_theory.min():.4f}  Max: {ei_theory.max():.4f}")

    print(f"\n    Dimension means:")
    for dim in DIM_WEIGHTS:
        print(f"      {dim}: {dim_score_df[dim].mean():.4f} (±{dim_score_df[dim].std():.4f})")

    # Step 2B: PCA composite
    print(f"\n[B] PCA composite:")
    ei_pca, pca_model = build_pca_composite(norm_df)
    print(f"    Mean: {ei_pca.mean():.4f}  SD: {ei_pca.std():.4f}")

    # Step 3: Combine into results table
    results = pd.DataFrame({
        "EI_theory": ei_theory,
        "EI_pca":    ei_pca,
    })
    results = pd.concat([results, dim_score_df], axis=1)

    print(f"\n    Pearson r between theory and PCA composite: "
          f"{results['EI_theory'].corr(results['EI_pca']):.4f}")

    # Save
    csv_path = os.path.join(OUTPUT_DIR, "ei_scores.csv")
    pkl_path = os.path.join(OUTPUT_DIR, "ei_scores.pkl")
    results.to_csv(csv_path)
    with open(pkl_path, "wb") as f:
        pickle.dump({"scores": results, "norm_df": norm_df, "pca_model": pca_model}, f)
    print(f"\nSaved → {csv_path}")

    print(f"\nTop 5 participants by theory EI:")
    print(results["EI_theory"].sort_values(ascending=False).head())
    print(f"\nBottom 5 participants by theory EI:")
    print(results["EI_theory"].sort_values().head())

    return results, norm_df


if __name__ == "__main__":
    run()
