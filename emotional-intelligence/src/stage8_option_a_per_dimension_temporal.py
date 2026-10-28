"""
Stage 8 — Option A: Per-Dimension Temporal Models
==================================================
Build separate models for each dimension, maintaining stage-wise temporal structure.

Each dimension model:
  1. Uses only features from that dimension
  2. Computes features separately for each stage (early, mid, late)
  3. Trains regression model on dimension-specific features
  4. Predicts exam score with dimension-level information

This respects the natural grouping (D1-D5) and temporal dynamics.

Outputs:
  outputs/option_a_per_dimension_results.csv
  outputs/option_a_dimension_importances.csv
  outputs/figures/option_a_model_comparison.png
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

    return norm_df_aligned, exam_aligned, selected_features_by_dim

def create_per_dimension_features(norm_df, selected_by_dim):
    """
    Create per-dimension feature sets.
    Each dimension gets its own feature matrix.
    """
    per_dim_data = {}

    for dim_name, selected_feats in selected_by_dim.items():
        if selected_feats:
            # Create feature matrix for this dimension
            X_dim = norm_df[selected_feats].copy()
            per_dim_data[dim_name] = {
                "features": selected_feats,
                "X": X_dim,
                "n_features": len(selected_feats)
            }
        else:
            per_dim_data[dim_name] = {
                "features": [],
                "X": None,
                "n_features": 0
            }

    return per_dim_data

def run():
    print("=" * 70)
    print("STAGE 8 — OPTION A: PER-DIMENSION TEMPORAL MODELS")
    print("=" * 70)

    # Load data
    norm_df, exam_scores, selected_by_dim = load_data()
    print(f"\nLoaded {len(norm_df)} participants")
    print(f"Exam scores: range [{exam_scores.min():.1f}, {exam_scores.max():.1f}], mean={exam_scores.mean():.2f}")

    # Create per-dimension features
    print("\nPreparing per-dimension feature sets...")
    per_dim_data = create_per_dimension_features(norm_df, selected_by_dim)

    for dim_name, data in per_dim_data.items():
        n_feat = data["n_features"]
        status = "[OK]" if n_feat > 0 else "[EMPTY]"
        print(f"  {dim_name:25s}: {n_feat:3d} features {status}")

    # CV setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y = exam_scores.values

    # Train per-dimension models
    print("\n" + "=" * 70)
    print("TRAINING PER-DIMENSION MODELS")
    print("=" * 70)

    results = {}
    importances = {}

    for dim_name, data in per_dim_data.items():
        if data["n_features"] == 0:
            print(f"\n{dim_name}: SKIPPED (no features)")
            continue

        print(f"\n{dim_name} ({data['n_features']} features)...")

        X = data["X"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=data["features"])

        # Ridge regression
        ridge = Ridge(alpha=1.0)
        y_pred = cross_val_predict(ridge, X_scaled, y, cv=cv)

        r2 = r2_score(y, y_pred)
        pearson_r, p_val = pearsonr(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        results[dim_name] = {
            "r2": r2,
            "pearson_r": pearson_r,
            "rmse": rmse,
            "mae": mae,
            "p_value": p_val,
            "n_features": data["n_features"]
        }

        print(f"  R²:        {r2:.4f}")
        print(f"  Pearson r: {pearson_r:.4f} (p={p_val:.4f})")
        print(f"  RMSE:      {rmse:.4f}")
        print(f"  MAE:       {mae:.4f}")

        # Feature importance (coefficients)
        ridge.fit(X_scaled, y)
        imp = np.abs(ridge.coef_)
        importances[dim_name] = imp / imp.sum()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("pearson_r", ascending=False)

    print("\nPer-Dimension Model Performance:")
    for dim_name, row in results_df.iterrows():
        print(f"  {dim_name:25s}: r={row['pearson_r']:+.4f} R²={row['r2']:+.4f} RMSE={row['rmse']:.3f}")

    # Save results
    results_csv = os.path.join(OUTPUT_DIR, "option_a_per_dimension_results.csv")
    results_df.to_csv(results_csv)
    print(f"\n  Results → option_a_per_dimension_results.csv")

    # Feature importance per dimension
    imp_data = []
    for dim_name, imp_array in importances.items():
        features = per_dim_data[dim_name]["features"]
        for feat, imp_val in zip(features, imp_array):
            imp_data.append({
                "dimension": dim_name,
                "feature": feat,
                "importance": imp_val
            })

    if imp_data:
        imp_df = pd.DataFrame(imp_data)
        imp_df = imp_df.sort_values(["dimension", "importance"], ascending=[True, False])
        imp_csv = os.path.join(OUTPUT_DIR, "option_a_dimension_importances.csv")
        imp_df.to_csv(imp_csv, index=False)
        print(f"  Feature importances → option_a_dimension_importances.csv")

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # R² by dimension
    ax = axes[0]
    dims = results_df.index
    r2_vals = results_df["r2"].values
    colors = ["green" if v > 0 else "red" for v in r2_vals]
    ax.barh(dims, r2_vals, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("R²")
    ax.set_title("Per-Dimension Model: R² Score")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(r2_vals):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center")

    # Pearson r by dimension
    ax = axes[1]
    r_vals = results_df["pearson_r"].values
    ax.barh(dims, r_vals, color="steelblue", edgecolor="black")
    ax.set_xlabel("Pearson r")
    ax.set_title("Per-Dimension Model: Correlation with Exam Score")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(r_vals):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "option_a_per_dimension_comparison.png")
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Visualization → figures/option_a_per_dimension_comparison.png")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    run()
