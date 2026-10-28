"""
Stage 6 (Regression) — Predictive Models with Multi-Level Features
===================================================================
GOAL: Predict CONTINUOUS exam scores using features at multiple aggregation levels.

This is a REGRESSION version (not classification) for direct comparison with thesis.

Feature levels:
  1. Individual selected features (51) — raw behavioral measures
  2. Dimension-level aggregates (5) — average per dimension
  3. EI composites (3) — stability-weighted, network, PCA

Total: 59 features

Models tested:
  - Linear Regression (baseline)
  - Ridge Regression
  - Bayesian Ridge (same as thesis best model)
  - Random Forest (nonlinear)
  - XGBoost (state-of-art nonlinear)

Metrics: R², Pearson r, RMSE, MAE

Outputs:
  outputs/regression_performance.csv
  outputs/regression_feature_importance.csv
  outputs/figures/regression_model_comparison.png
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, DIM_WEIGHTS

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


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
    """Prepare 59-dimensional feature matrix."""
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
    print("STAGE 6 (REGRESSION) — PREDICTIVE MODELS")
    print("Direct comparison with thesis (r=0.780, R²=0.609)")
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

    # CV setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Bayesian Ridge": BayesianRidge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }

    results = {}
    importances = {}

    print("=" * 70)
    print("TRAINING MODELS (5-fold cross-validation, REGRESSION)")
    print("=" * 70)

    for model_name, model in models.items():
        print(f"\n{model_name}...")

        # Cross-validation predictions
        y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

        # Metrics
        r2 = r2_score(y, y_pred)
        pearson_r, p_val = pearsonr(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        results[model_name] = {
            "r2": r2,
            "pearson_r": pearson_r,
            "rmse": rmse,
            "mae": mae,
            "p_value": p_val,
        }

        print(f"  R²:        {r2:.4f}")
        print(f"  Pearson r: {pearson_r:.4f} (p={p_val:.4f})")
        print(f"  RMSE:      {rmse:.4f}")
        print(f"  MAE:       {mae:.4f}")

        # Feature importance
        if model_name == "Linear Regression":
            model.fit(X_scaled, y)
            imp = np.abs(model.coef_)
        elif model_name == "Ridge":
            model.fit(X_scaled, y)
            imp = np.abs(model.coef_)
        elif model_name == "Bayesian Ridge":
            model.fit(X_scaled, y)
            imp = np.abs(model.coef_)
        elif model_name == "Random Forest":
            model.fit(X_scaled, y)
            imp = model.feature_importances_
        else:  # XGBoost
            model.fit(X_scaled, y)
            imp = model.feature_importances_

        importances[model_name] = imp / imp.sum()

    # Consensus feature ranking
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE RANKING (consensus across models)")
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

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(OUTPUT_DIR, "regression_performance.csv"))
    print(f"  Model performance → regression_performance.csv")

    importance_df.to_csv(os.path.join(OUTPUT_DIR, "regression_feature_importance.csv"))
    print(f"  Feature ranking → regression_feature_importance.csv")

    # Comparison with thesis
    print("\n" + "=" * 70)
    print("COMPARISON WITH THESIS RESULTS")
    print("=" * 70)

    best_model = results_df["pearson_r"].abs().idxmax()
    best_r = results_df.loc[best_model, "pearson_r"]
    best_r2 = results_df.loc[best_model, "r2"]

    print(f"\nCurrent best model: {best_model}")
    print(f"  Pearson r:  {best_r:+.4f}")
    print(f"  R²:         {best_r2:.4f}")
    print(f"\nThesis results:")
    print(f"  Pearson r:  +0.7800 (composite EI)")
    print(f"  R²:         0.6090 (Bayesian Ridge)")
    print(f"\nDifference:")
    print(f"  Δr:         {best_r - 0.7800:+.4f}")
    print(f"  ΔR²:        {best_r2 - 0.6090:+.4f}")

    if abs(best_r) >= 0.78:
        print(f"\n✓ CURRENT METHOD MATCHES OR EXCEEDS THESIS!")
    else:
        print(f"\n✗ Current method underperforms thesis")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # R² comparison
    ax = axes[0, 0]
    values = [results[m]["r2"] for m in models.keys()]
    ax.bar(models.keys(), values, color="steelblue", edgecolor="black")
    ax.axhline(0.6090, color="red", linestyle="--", linewidth=2, label="Thesis (0.609)")
    ax.set_ylabel("R²")
    ax.set_title("R² by Model")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    # Pearson r comparison
    ax = axes[0, 1]
    values = [results[m]["pearson_r"] for m in models.keys()]
    ax.bar(models.keys(), values, color="seagreen", edgecolor="black")
    ax.axhline(0.7800, color="red", linestyle="--", linewidth=2, label="Thesis (0.780)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation by Model")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    # RMSE comparison
    ax = axes[1, 0]
    values = [results[m]["rmse"] for m in models.keys()]
    ax.bar(models.keys(), values, color="coral", edgecolor="black")
    ax.set_ylabel("RMSE")
    ax.set_title("Root Mean Squared Error by Model")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, f"{v:.3f}", ha="center", fontsize=9)

    # Feature level importance
    ax = axes[1, 1]
    level_importance = importance_df.groupby("level")["mean_importance"].mean()
    colors = {"individual": "#1f77b4", "dimension": "#ff7f0e", "composite": "#2ca02c"}
    bar_colors = [colors[level] for level in level_importance.index]
    ax.bar(level_importance.index, level_importance.values, color=bar_colors, edgecolor="black")
    ax.set_ylabel("Mean Importance")
    ax.set_title("Feature Level Importance")
    ax.grid(axis="y", alpha=0.3)
    for i, (idx, v) in enumerate(level_importance.items()):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "regression_model_comparison.png")
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Visualization → figures/regression_model_comparison.png")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run()
