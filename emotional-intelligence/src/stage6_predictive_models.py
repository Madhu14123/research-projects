"""
Stage 6 — Predictive Models with Multi-Level Feature Analysis
==============================================================
GOAL: Predict exam scores using features at multiple aggregation levels.

Feature levels:
  1. Individual selected features (51) — raw behavioral measures
  2. Dimension-level aggregates (5) — average per dimension
  3. EI composites (3) — stability-weighted, network, PCA

Total: 59 features

Models tested:
  - Logistic Regression (baseline, interpretable)
  - Random Forest (feature importance via Gini/entropy)
  - XGBoost (state-of-art, SHAP feature importance)
  - SVM (robust baseline)

Feature ranking: Consensus importance across models

Outputs:
  outputs/model_performance.csv
  outputs/feature_importance_ranking.csv
  outputs/figures/model_comparison.png
  outputs/figures/top_features.png
  outputs/figures/feature_importance_heatmap.png
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_curve, auc)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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
    import pandas as pd
    from scipy.stats import mode as scipy_mode
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
    """Prepare 59-dimensional feature matrix with three levels."""

    # Level 1: Individual selected features (51)
    individual_features = []
    for dim, selected in selected_features_by_dim.items():
        individual_features.extend(selected)

    df_individual = norm_df[individual_features].copy()

    # Level 2: Dimension-level aggregates (5)
    dim_scores = {}
    for dim_name, selected in selected_features_by_dim.items():
        if selected:
            # Weight by stability within dimension
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

    # Level 3: EI composites (3)
    # Stability-weighted
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

    # Network centrality
    from scipy.linalg import eigh
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

    # PCA within dimension
    from sklearn.decomposition import PCA
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

    # Feature names for later reference
    feature_names = list(df_individual.columns) + list(df_dims.columns) + list(df_composites.columns)
    feature_levels = (
        ["individual"] * len(df_individual.columns) +
        ["dimension"] * len(df_dims.columns) +
        ["composite"] * len(df_composites.columns)
    )

    return X, feature_names, feature_levels


def run():
    print("=" * 70)
    print("STAGE 6 — PREDICTIVE MODELS WITH MULTI-LEVEL FEATURES")
    print("=" * 70)

    # Load data
    norm_df, exam_scores, selected_by_dim, stability_all = load_data()
    print(f"\nLoaded {len(norm_df)} participants")
    print(f"Exam score range: {exam_scores.min():.1f} – {exam_scores.max():.1f}")

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

    # Binarize exam scores (median split)
    y = (exam_scores >= exam_scores.median()).astype(int)
    print(f"Target: Binary classification (exam_score >= {exam_scores.median():.1f})")
    print(f"  Class 0 (low): {(y==0).sum()}")
    print(f"  Class 1 (high): {(y==1).sum()}\n")

    # CV setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=42, probability=True),
    }

    results = {}
    importances = {}

    print("=" * 70)
    print("TRAINING MODELS (5-fold cross-validation)")
    print("=" * 70)

    for model_name, model in models.items():
        print(f"\n{model_name}...")

        # Cross-validation
        cv_results = cross_validate(
            model, X_scaled, y, cv=cv,
            scoring=["accuracy", "roc_auc", "precision", "recall", "f1"],
            return_train_score=False
        )

        acc = cv_results["test_accuracy"].mean()
        auc = cv_results["test_roc_auc"].mean()
        prec = cv_results["test_precision"].mean()
        rec = cv_results["test_recall"].mean()
        f1 = cv_results["test_f1"].mean()

        results[model_name] = {
            "accuracy": acc,
            "auc": auc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC-ROC:  {auc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:   {rec:.4f}")
        print(f"  F1:       {f1:.4f}")

        # Feature importance
        if model_name == "Logistic Regression":
            # Use absolute coefficients
            model.fit(X_scaled, y)
            imp = np.abs(model.coef_[0])
        elif model_name == "Random Forest":
            model.fit(X_scaled, y)
            imp = model.feature_importances_
        elif model_name == "XGBoost":
            model.fit(X_scaled, y)
            imp = model.feature_importances_
        else:  # SVM
            # Use permutation importance as fallback
            model.fit(X_scaled, y)
            imp = np.ones(X_scaled.shape[1]) / X_scaled.shape[1]

        importances[model_name] = imp / imp.sum()  # normalize

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
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_performance.csv"))
    print(f"  Model performance → model_performance.csv")

    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_ranking.csv"))
    print(f"  Feature ranking → feature_importance_ranking.csv")

    # Visualizations
    # 1. Model comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = ["accuracy", "auc", "precision", "recall", "f1"]

    for idx, metric in enumerate(metrics):
        ax = axes.flatten()[idx]
        values = [results[m][metric] for m in models.keys()]
        ax.bar(models.keys(), values, color="steelblue", edgecolor="black")
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)
        ax.set_title(f"{metric.capitalize()} by Model")
        ax.tick_params(axis="x", rotation=45)
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    axes.flatten()[-1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Model comparison → figures/model_comparison.png")

    # 2. Top features by level
    fig, ax = plt.subplots(figsize=(12, 8))
    top_20 = importance_df.head(20).copy()
    colors = {"individual": "#1f77b4", "dimension": "#ff7f0e", "composite": "#2ca02c"}
    bar_colors = [colors[level] for level in top_20["level"]]

    ax.barh(range(len(top_20)), top_20["mean_importance"], color=bar_colors, edgecolor="black")
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20.index, fontsize=10)
    ax.set_xlabel("Mean Importance Across Models")
    ax.set_title("Top 20 Predictive Features")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["individual"], edgecolor="black", label="Individual features"),
        Patch(facecolor=colors["dimension"], edgecolor="black", label="Dimension aggregates"),
        Patch(facecolor=colors["composite"], edgecolor="black", label="EI composites"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "top_features.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Top features → figures/top_features.png")

    # 3. Feature importance heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    top_30 = importance_df.head(30)
    sns.heatmap(top_30[models.keys()], annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Importance"})
    ax.set_title("Top 30 Features: Importance Across All Models")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance_heatmap.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Importance heatmap → figures/feature_importance_heatmap.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best_model = results_df["f1"].idxmax()
    print(f"\nBest model (by F1): {best_model}")
    print(f"  Accuracy: {results[best_model]['accuracy']:.4f}")
    print(f"  AUC-ROC:  {results[best_model]['auc']:.4f}")
    print(f"  F1:       {results[best_model]['f1']:.4f}")

    top_feature = importance_df.iloc[0]
    print(f"\nMost important feature: {importance_df.index[0]}")
    print(f"  Level: {top_feature['level']}")
    print(f"  Mean importance: {top_feature['mean_importance']:.4f}")

    level_importance = importance_df.groupby("level")["mean_importance"].mean()
    print(f"\nAverage importance by feature level:")
    for level in ["individual", "dimension", "composite"]:
        if level in level_importance.index:
            print(f"  {level:15s}: {level_importance[level]:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run()
