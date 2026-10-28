"""
Stage 6 (Regression) — Feature Importance Ranking
===================================================
Extract feature importance from regression models for consistent feature ranking.

Similar to stage 6 classification version but uses regression models:
  - Linear Regression
  - Ridge
  - Bayesian Ridge
  - Random Forest Regressor
  - XGBoost Regressor

Outputs:
  outputs/regression_feature_importance_ranking.csv
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, DIM_WEIGHTS

def load_data():
    """Load features and exam scores."""
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
    exam_file = os.path.join(OUTPUT_DIR, "..", "data", "Exam Scores.xlsx")
    df_exam = pd.read_excel(exam_file)
    df_exam.columns = [str(c).strip() for c in df_exam.columns]

    import re
    def parse_pid(val):
        m = re.search(r'\d+', str(val))
        return int(m.group()) if m else None

    df_exam["pid"] = df_exam.iloc[:, 0].apply(parse_pid)
    df_exam = df_exam.dropna(subset=["pid"])
    df_exam["pid"] = df_exam["pid"].astype(int)
    exam_scores = df_exam.set_index("pid").iloc[:, -2].astype(float)

    # Align
    common = sorted(set(norm_df.index) & set(exam_scores.index))
    norm_df_aligned = norm_df.loc[common]
    exam_aligned = exam_scores.loc[common]

    return norm_df_aligned, exam_aligned, selected_features_by_dim, stability_all


def prepare_features(norm_df, selected_features_by_dim, stability_all, dim_weights):
    """Prepare 59-dimensional feature matrix (same as regression models)."""
    from sklearn.decomposition import PCA
    from scipy.linalg import eigh

    # Individual features
    individual_features = []
    for dim, selected in selected_features_by_dim.items():
        individual_features.extend(selected)

    df_individual = norm_df[individual_features].copy()

    # Dimension aggregates
    dim_scores = {}
    for dim_name, selected in selected_features_by_dim.items():
        if selected:
            stability = stability_all.get(dim_name, {})
            stabilities = np.array([stability.get(f, 1.0) for f in selected])
            if stabilities.sum() > 0:
                weights = stabilities / stabilities.sum()
            else:
                weights = np.ones(len(selected)) / len(selected)
            dim_scores[dim_name] = (norm_df[selected].values * weights).sum(axis=1)
        else:
            dim_scores[dim_name] = np.full(len(norm_df), 0.5)

    df_dims = pd.DataFrame(dim_scores, index=norm_df.index)

    # EI composites
    ei_stable = {}
    for dim_name, selected in selected_features_by_dim.items():
        if selected:
            stability = stability_all.get(dim_name, {})
            stabilities = np.array([stability.get(f, 1.0) for f in selected])
            if stabilities.sum() > 0:
                weights = stabilities / stabilities.sum()
            else:
                weights = np.ones(len(selected)) / len(selected)
            ei_stable[dim_name] = (norm_df[selected].values * weights).sum(axis=1)
        else:
            ei_stable[dim_name] = np.full(len(norm_df), 0.5)
    ei_stab = sum(dim_weights[d] * ei_stable[d] for d in dim_weights)

    ei_central = {}
    for dim_name, selected in selected_features_by_dim.items():
        if not selected or len(selected) < 2:
            ei_central[dim_name] = norm_df[selected].mean(axis=1) if selected else np.full(len(norm_df), 0.5)
        else:
            X = norm_df[selected].values
            corr = np.abs(np.corrcoef(X.T))
            np.fill_diagonal(corr, 0)
            try:
                eigenvalues, eigenvectors = eigh(corr)
                centrality = np.abs(eigenvectors[:, -1])
                centrality = centrality / centrality.sum()
            except:
                centrality = np.ones(len(selected)) / len(selected)
            ei_central[dim_name] = (norm_df[selected].values * centrality).sum(axis=1)
    ei_cent = sum(dim_weights[d] * ei_central[d] for d in dim_weights)

    ei_pca_dict = {}
    for dim_name, selected in selected_features_by_dim.items():
        if not selected or len(selected) < 2:
            ei_pca_dict[dim_name] = norm_df[selected].mean(axis=1) if selected else np.full(len(norm_df), 0.5)
        else:
            X = norm_df[selected].values
            pca = PCA(n_components=1)
            ei_pca_dict[dim_name] = pca.fit_transform(X).ravel()
    ei_pc = sum(dim_weights[d] * ei_pca_dict[d] for d in dim_weights)

    df_composites = pd.DataFrame({
        "EI_stability_weighted": ei_stab,
        "EI_network_centrality": ei_cent,
        "EI_pca_within_dim": ei_pc
    }, index=norm_df.index)

    # Combine all
    X = pd.concat([df_individual, df_dims, df_composites], axis=1)
    feature_names = list(df_individual.columns) + list(df_dims.columns) + list(df_composites.columns)
    feature_levels = (
        ["individual"] * len(df_individual.columns) +
        ["dimension"] * len(df_dims.columns) +
        ["composite"] * len(df_composites.columns)
    )

    return X, feature_names, feature_levels


def run():
    print("=" * 70)
    print("STAGE 6 (REGRESSION) — FEATURE IMPORTANCE RANKING")
    print("=" * 70)

    # Load data
    norm_df, exam_scores, selected_by_dim, stability_all = load_data()
    print(f"\nLoaded {len(norm_df)} participants")
    print(f"Exam scores: range [{exam_scores.min():.1f}, {exam_scores.max():.1f}], mean={exam_scores.mean():.2f}")

    # Prepare features
    print("\nPreparing features at three levels...")
    X, feature_names, feature_levels = prepare_features(norm_df, selected_by_dim, stability_all, DIM_WEIGHTS)
    print(f"  Individual features: {(np.array(feature_levels) == 'individual').sum()}")
    print(f"  Dimension aggregates: {(np.array(feature_levels) == 'dimension').sum()}")
    print(f"  EI composites: {(np.array(feature_levels) == 'composite').sum()}")
    print(f"  Total: {X.shape[1]} features\n")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    y = exam_scores.values

    # Models for feature importance
    print("=" * 70)
    print("EXTRACTING FEATURE IMPORTANCE FROM REGRESSION MODELS")
    print("=" * 70)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Bayesian Ridge": BayesianRidge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }

    importances = {}

    for model_name, model in models.items():
        print(f"\n{model_name}...")
        model.fit(X_scaled, y)

        # Extract importance
        if model_name in ["Linear Regression", "Ridge", "Bayesian Ridge"]:
            imp = np.abs(model.coef_)
        else:  # Random Forest, XGBoost
            imp = model.feature_importances_

        importances[model_name] = imp / imp.sum()
        print(f"  Importance normalized: sum={imp.sum():.4f}")

    # Consensus ranking
    print("\n" + "=" * 70)
    print("CONSENSUS FEATURE RANKING (across regression models)")
    print("=" * 70)

    importance_df = pd.DataFrame(importances, index=feature_names)
    importance_df["mean_importance"] = importance_df.mean(axis=1)
    importance_df["level"] = feature_levels
    importance_df = importance_df.sort_values("mean_importance", ascending=False)

    print("\nTop 20 features:")
    for idx, (feat, row) in enumerate(importance_df.head(20).iterrows(), 1):
        level = row["level"]
        mean_imp = row["mean_importance"]
        print(f"  {idx:2d}. {feat:45s} [{level:10s}] {mean_imp:.4f}")

    # Save ranking
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    ranking_csv = os.path.join(OUTPUT_DIR, "regression_feature_importance_ranking.csv")
    importance_df.to_csv(ranking_csv)
    print(f"  Feature ranking → regression_feature_importance_ranking.csv")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run()
