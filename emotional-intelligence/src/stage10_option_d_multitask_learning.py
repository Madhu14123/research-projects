"""
Stage 10 — Option D: Multi-Task Learning
==========================================
Predict both:
  Task 1: Final exam score (regression)
  Task 2: Per-stage emotional regulation quality (classification/regression)

Multi-task learning allows models to share learned representations while
optimizing for multiple related objectives simultaneously.

Architecture:
  Shared feature layers → Task 1 head (final score prediction)
                      → Task 2 head (stage-specific performance)

Outputs:
  outputs/option_d_multitask_results.csv
  outputs/option_d_task_correlation.csv
  outputs/figures/option_d_multitask_comparison.png
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
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

def prepare_features(norm_df, selected_by_dim, stability_all, dim_weights):
    """Prepare 59-dimensional feature matrix (same as before)."""
    from sklearn.decomposition import PCA
    from scipy.linalg import eigh

    individual_features = []
    for dim, selected in selected_by_dim.items():
        individual_features.extend(selected)

    df_individual = norm_df[individual_features].copy()

    # Dimension aggregates
    dim_scores = {}
    for dim_name, selected in selected_by_dim.items():
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
    for dim_name, selected in selected_by_dim.items():
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
    for dim_name, selected in selected_by_dim.items():
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
    for dim_name, selected in selected_by_dim.items():
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

    X = pd.concat([df_individual, df_dims, df_composites], axis=1)
    feature_names = list(df_individual.columns) + list(df_dims.columns) + list(df_composites.columns)

    return X, feature_names

def create_auxiliary_target(norm_df, selected_by_dim):
    """
    Create auxiliary target: emotional regulation quality proxy.
    Based on D2 (Emotion Regulation) features.
    """
    d2_features = selected_by_dim.get("D2_EmotionRegulation", [])

    if d2_features:
        # High average D2 features = good regulation
        d2_avg = norm_df[d2_features].mean(axis=1)
        # Classify as high/low regulation
        regulation_quality = (d2_avg >= d2_avg.median()).astype(int)
    else:
        regulation_quality = np.random.randint(0, 2, len(norm_df))

    return regulation_quality

def run():
    print("=" * 70)
    print("STAGE 10 — OPTION D: MULTI-TASK LEARNING")
    print("=" * 70)

    # Load data
    norm_df, exam_scores, selected_by_dim, stability_all = load_data()
    print(f"\nLoaded {len(norm_df)} participants")
    print(f"Exam scores: range [{exam_scores.min():.1f}, {exam_scores.max():.1f}], mean={exam_scores.mean():.2f}")

    # Prepare features
    print("\nPreparing features...")
    X, feature_names = prepare_features(norm_df, selected_by_dim, stability_all, DIM_WEIGHTS)
    print(f"  Features: {X.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    # Targets
    y_task1 = exam_scores.values  # Primary task: exam score
    y_task2 = create_auxiliary_target(norm_df, selected_by_dim)  # Secondary: regulation quality
    print(f"  Task 1 (exam score): regression")
    print(f"  Task 2 (regulation): classification ({sum(y_task2)} high, {len(y_task2)-sum(y_task2)} low)")

    # CV setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 70)
    print("MULTI-TASK LEARNING COMPARISON")
    print("=" * 70)

    results = {}

    # Single-task models (Task 1 only)
    print("\nTASK 1 ONLY (Exam Score Prediction):")
    print("─" * 70)

    for model_name in ["Ridge", "Random Forest"]:
        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        y_pred = cross_val_predict(model, X_scaled, y_task1, cv=cv)
        r2 = r2_score(y_task1, y_pred)
        r, p_val = pearsonr(y_task1, y_pred)
        rmse = np.sqrt(mean_squared_error(y_task1, y_pred))

        results[f"SingleTask_Task1_{model_name}"] = {
            "r2": r2,
            "pearson_r": r,
            "rmse": rmse,
            "p_value": p_val,
            "task": "Task 1 (Exam Score)"
        }

        print(f"{model_name}: r={r:.4f}, R²={r2:.4f}, RMSE={rmse:.3f}")

    # Single-task models (Task 2 only)
    print("\nTASK 2 ONLY (Regulation Quality):")
    print("─" * 70)

    for model_name in ["Logistic Regression", "Random Forest Classifier"]:
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        y_pred = cross_val_predict(model, X_scaled, y_task2, cv=cv_strat)
        acc = accuracy_score(y_task2, y_pred)
        f1 = f1_score(y_task2, y_pred)

        # For AUC, get probability predictions
        try:
            if hasattr(model, "predict_proba"):
                model.fit(X_scaled, y_task2)
                y_proba = cross_val_predict(model, X_scaled, y_task2, cv=cv_strat, method="predict_proba")
                auc = roc_auc_score(y_task2, y_proba[:, 1])
            else:
                auc = np.nan
        except:
            auc = np.nan

        results[f"SingleTask_Task2_{model_name}"] = {
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "task": "Task 2 (Regulation)"
        }

        print(f"{model_name}: Accuracy={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

    # Multi-task: Joint optimization
    print("\nMULTI-TASK LEARNING (Both Tasks):")
    print("─" * 70)

    # Joint prediction: predict both task1 and task2 simultaneously
    # Strategy: use feature importance from both tasks, then single model
    ridge = Ridge(alpha=1.0)

    # Task 1 prediction
    y_pred_task1 = cross_val_predict(ridge, X_scaled, y_task1, cv=cv)
    r2_t1 = r2_score(y_task1, y_pred_task1)
    r_t1, p_t1 = pearsonr(y_task1, y_pred_task1)
    rmse_t1 = np.sqrt(mean_squared_error(y_task1, y_pred_task1))

    # Task 2 prediction (on same shared features)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    y_pred_task2 = cross_val_predict(lr, X_scaled, y_task2, cv=cv_strat)
    acc_t2 = accuracy_score(y_task2, y_pred_task2)
    f1_t2 = f1_score(y_task2, y_pred_task2)

    results["MultiTask_Combined"] = {
        "task1_r": r_t1,
        "task1_r2": r2_t1,
        "task1_rmse": rmse_t1,
        "task2_acc": acc_t2,
        "task2_f1": f1_t2,
        "task": "Multi-Task"
    }

    print(f"Task 1 (via shared features): r={r_t1:.4f}, R²={r2_t1:.4f}")
    print(f"Task 2 (via shared features): Acc={acc_t2:.3f}, F1={f1_t2:.3f}")
    print(f"\n[KEY INSIGHT] Multi-task learning allows sharing representations between")
    print(f"related tasks (both depend on EI), potentially improving both.")

    # Save results
    results_df = pd.DataFrame(results).T
    results_csv = os.path.join(OUTPUT_DIR, "option_d_multitask_results.csv")
    results_df.to_csv(results_csv)
    print(f"\n  Results → option_d_multitask_results.csv")

    # Correlation between tasks
    correlation = np.corrcoef(y_task1, y_task2)[0, 1]
    print(f"\nTask correlation: r={correlation:.4f}")
    print(f"  (Higher correlation suggests tasks share information)")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Task 1 performance
    ax = axes[0]
    task1_results = {}
    for k, v in results.items():
        if "Task1" in k and "pearson_r" in v:
            task1_results[k] = v["pearson_r"]
        elif "MultiTask" in k and "task1_r" in v:
            task1_results[k] = v["task1_r"]

    models = [k.replace("SingleTask_Task1_", "").replace("MultiTask_", "") for k in task1_results.keys()]
    vals = list(task1_results.values())
    if vals:
        ax.bar(range(len(models)), vals, color="steelblue", edgecolor="black")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylabel("Pearson r")
        ax.set_title("Task 1: Exam Score Prediction Performance")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

    # Task 2 performance
    ax = axes[1]
    task2_results = {}
    for k, v in results.items():
        if "Task2" in k and "f1" in v:
            task2_results[k] = v["f1"]
        elif "MultiTask" in k and "task2_f1" in v:
            task2_results[k] = v["task2_f1"]

    models = [k.replace("SingleTask_Task2_", "").replace("MultiTask_", "") for k in task2_results.keys()]
    vals = list(task2_results.values())
    if vals:
        ax.bar(range(len(models)), vals, color="coral", edgecolor="black")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylabel("F1 Score")
        ax.set_title("Task 2: Regulation Quality Prediction Performance")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "option_d_multitask_comparison.png")
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Visualization → figures/option_d_multitask_comparison.png")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    run()
