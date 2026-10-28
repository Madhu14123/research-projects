"""
Stage 6B — Feature Count Optimization
======================================
Test: How many features are actually needed for good prediction accuracy?

Approach:
  1. Rank all 59 features by importance (from Stage 6)
  2. Incrementally add features: train on top 1, top 2, top 3, ... top 59
  3. Use 5-fold cross-validation for each feature count
  4. Track: accuracy, AUC, F1 score
  5. Identify the "elbow point" where adding more features stops helping
  6. Find optimal feature set

Models tested:
  - Logistic Regression (best from Stage 6)
  - Random Forest (for comparison)

Outputs:
  outputs/feature_count_results.csv
  outputs/figures/feature_count_accuracy.png
  outputs/optimal_features.txt
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

    return X, feature_names


def load_feature_ranking():
    """Load feature importance ranking from Stage 6."""
    ranking_path = os.path.join(OUTPUT_DIR, "feature_importance_ranking.csv")
    df_rank = pd.read_csv(ranking_path, index_col=0)
    # Sort by mean importance descending
    df_rank = df_rank.sort_values("mean_importance", ascending=False)
    ranked_features = df_rank.index.tolist()
    return ranked_features


def run():
    print("=" * 70)
    print("STAGE 6B — FEATURE COUNT OPTIMIZATION")
    print("=" * 70)

    # Load data
    norm_df, exam_scores, selected_by_dim, stability_all = load_data()
    print(f"\nLoaded {len(norm_df)} participants")

    # Prepare features
    X, feature_names = prepare_features(norm_df, selected_by_dim, stability_all, DIM_WEIGHTS)
    print(f"Total features: {X.shape[1]}")

    # Load ranking
    ranked_features = load_feature_ranking()
    print(f"Ranked features loaded: {len(ranked_features)}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    # Target
    y = (exam_scores >= exam_scores.median()).astype(int)
    print(f"Target: Binary (N_low={sum(y==0)}, N_high={sum(y==1)})\n")

    # CV setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Test incrementally: 1, 2, 3, ..., 59 features
    print("=" * 70)
    print("TESTING INCREMENTALLY: Feature Count vs Accuracy")
    print("=" * 70)

    results = {
        "num_features": [],
        "LR_accuracy": [],
        "LR_auc": [],
        "LR_f1": [],
        "RF_accuracy": [],
        "RF_auc": [],
        "RF_f1": [],
        "top_features": [],
    }

    feature_counts = list(range(1, len(ranked_features) + 1))

    for num_feat in feature_counts:
        top_feats = ranked_features[:num_feat]
        X_subset = X_scaled[top_feats]

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr_results = cross_validate(
            lr, X_subset, y, cv=cv,
            scoring=["accuracy", "roc_auc", "f1"],
            return_train_score=False
        )

        lr_acc = lr_results["test_accuracy"].mean()
        lr_auc = lr_results["test_roc_auc"].mean()
        lr_f1 = lr_results["test_f1"].mean()

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_results = cross_validate(
            rf, X_subset, y, cv=cv,
            scoring=["accuracy", "roc_auc", "f1"],
            return_train_score=False
        )

        rf_acc = rf_results["test_accuracy"].mean()
        rf_auc = rf_results["test_roc_auc"].mean()
        rf_f1 = rf_results["test_f1"].mean()

        results["num_features"].append(num_feat)
        results["LR_accuracy"].append(lr_acc)
        results["LR_auc"].append(lr_auc)
        results["LR_f1"].append(lr_f1)
        results["RF_accuracy"].append(rf_acc)
        results["RF_auc"].append(rf_auc)
        results["RF_f1"].append(rf_f1)
        results["top_features"].append(", ".join(top_feats))

        if num_feat % 5 == 0 or num_feat <= 3 or num_feat == len(ranked_features):
            print(f"Features: {num_feat:2d} | LR: {lr_acc:.4f} (AUC: {lr_auc:.4f}) | RF: {rf_acc:.4f} (AUC: {rf_auc:.4f})")

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(OUTPUT_DIR, "feature_count_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved → feature_count_results.csv")

    # Find optimal
    lr_acc_array = np.array(results["LR_accuracy"])
    rf_acc_array = np.array(results["RF_accuracy"])

    best_lr_idx = np.argmax(lr_acc_array)
    best_rf_idx = np.argmax(rf_acc_array)

    best_lr_count = results["num_features"][best_lr_idx]
    best_lr_acc = results["LR_accuracy"][best_lr_idx]
    best_rf_count = results["num_features"][best_rf_idx]
    best_rf_acc = results["RF_accuracy"][best_rf_idx]

    # Find elbow (where accuracy plateaus)
    lr_diffs = np.diff(lr_acc_array)
    rf_diffs = np.diff(rf_acc_array)

    # Elbow = point where improvement drops below 1%
    lr_elbow = np.argmax(lr_diffs < 0.01) + 1 if np.any(lr_diffs < 0.01) else best_lr_count
    rf_elbow = np.argmax(rf_diffs < 0.01) + 1 if np.any(rf_diffs < 0.01) else best_rf_count

    # Report
    report_lines = [
        "FEATURE COUNT OPTIMIZATION RESULTS",
        "=" * 70,
        f"\nTotal features tested: {len(ranked_features)}",
        f"\n--- LOGISTIC REGRESSION ---",
        f"Best accuracy: {best_lr_acc:.4f} at {best_lr_count} features",
        f"Features: {results['top_features'][best_lr_idx][:100]}...",
        f"\nElbow point (improvement < 1%): {lr_elbow} features",
        f"Accuracy at elbow: {results['LR_accuracy'][lr_elbow-1]:.4f}",
        f"\n--- RANDOM FOREST ---",
        f"Best accuracy: {best_rf_acc:.4f} at {best_rf_count} features",
        f"Features: {results['top_features'][best_rf_idx][:100]}...",
        f"\nElbow point (improvement < 1%): {rf_elbow} features",
        f"Accuracy at elbow: {results['RF_accuracy'][rf_elbow-1]:.4f}",
        f"\n--- RECOMMENDATION ---",
        f"Optimal feature count: {min(best_lr_count, 15)} features",
        f"  (balances accuracy vs complexity)",
        f"Minimum viable: {min(lr_elbow, rf_elbow)} features",
        f"  (diminishing returns beyond this)",
    ]

    optimal_path = os.path.join(OUTPUT_DIR, "optimal_features.txt")
    with open(optimal_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n" + "\n".join(report_lines))
    print(f"\nOptimal features report → optimal_features.txt")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax = axes[0]
    ax.plot(results["num_features"], results["LR_accuracy"], "o-", label="Logistic Regression", linewidth=2, markersize=4)
    ax.plot(results["num_features"], results["RF_accuracy"], "s-", label="Random Forest", linewidth=2, markersize=4)
    ax.axvline(best_lr_count, color="blue", linestyle="--", alpha=0.5, label=f"LR best: {best_lr_count}")
    ax.axvline(best_rf_count, color="orange", linestyle="--", alpha=0.5, label=f"RF best: {best_rf_count}")
    ax.axvline(15, color="green", linestyle=":", alpha=0.7, linewidth=2, label="Recommended: 15")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Accuracy (5-fold CV)")
    ax.set_title("Accuracy vs Feature Count")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    # AUC
    ax = axes[1]
    ax.plot(results["num_features"], results["LR_auc"], "o-", label="Logistic Regression", linewidth=2, markersize=4)
    ax.plot(results["num_features"], results["RF_auc"], "s-", label="Random Forest", linewidth=2, markersize=4)
    ax.axvline(best_lr_count, color="blue", linestyle="--", alpha=0.5)
    ax.axvline(best_rf_count, color="orange", linestyle="--", alpha=0.5)
    ax.axvline(15, color="green", linestyle=":", alpha=0.7, linewidth=2)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("AUC-ROC (5-fold CV)")
    ax.set_title("AUC vs Feature Count")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "feature_count_accuracy.png")
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Visualization → figures/feature_count_accuracy.png")

    # Print top 15 features for easy reference
    print("\n" + "=" * 70)
    print("TOP 15 FEATURES (Recommended Feature Set)")
    print("=" * 70)
    for i, feat in enumerate(ranked_features[:15], 1):
        print(f"{i:2d}. {feat}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run()
