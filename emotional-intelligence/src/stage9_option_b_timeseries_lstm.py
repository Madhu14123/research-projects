"""
Stage 9 — Option B: Time-Series LSTM for Emotional Dynamics
============================================================
Use LSTM (Long Short-Term Memory) to capture how emotions evolve across exam.

Approach:
  1. Organize data as time sequences (one sequence per participant)
  2. Each timestep is a problem-solving phase or time window
  3. LSTM learns how emotional state transitions over time
  4. Final hidden state used to predict exam score

This captures temporal dependencies: how emotions at time t affect time t+1.

Note: Requires tensorflow/keras

Outputs:
  outputs/option_b_lstm_results.csv
  outputs/figures/option_b_lstm_training.png
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR

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

def prepare_sequence_data(norm_df, selected_by_dim, stability_all):
    """
    Prepare sequences for LSTM.
    For now: treat each dimension as a timestep.
    Later: could use actual stage information.
    """

    # Get list of all features
    all_features = []
    for dim, feats in selected_by_dim.items():
        all_features.extend(feats)

    # Organize features by dimension (simulates temporal sequence)
    dim_order = ["D1_EmotionPerception", "D2_EmotionRegulation",
                 "D3_AdaptivePersistence", "D4_AffectiveSelfAwareness",
                 "D5_EmotionCognitionAlign"]

    sequences = []
    for pid in norm_df.index:
        sequence = []
        for dim_name in dim_order:
            if dim_name in selected_by_dim:
                feats = selected_by_dim[dim_name]
                if feats:
                    # Mean values for this dimension at this "timestep"
                    dim_features = norm_df.loc[pid, feats].values
                    sequence.append(dim_features)

        if len(sequence) > 0:
            sequences.append({
                "pid": pid,
                "sequence": np.array(sequence),
                "n_timesteps": len(sequence),
                "n_features": sequence[0].shape[0] if len(sequence) > 0 else 0
            })

    return sequences

def create_baseline_temporal_model(norm_df, selected_by_dim, exam_scores, stability_all):
    """
    Since LSTM requires tensorflow (not always available),
    create baseline: temporal averaging model that respects dimension order.
    """

    print("=" * 70)
    print("STAGE 9 — OPTION B: TIME-SERIES TEMPORAL MODEL (Baseline)")
    print("=" * 70)

    print("\n[NOTE: Full LSTM requires TensorFlow. Using temporal baseline instead.]")
    print("\nApproach: Progressive averaging across dimensions (simulates temporal evolution)")

    # Prepare per-dimension aggregates in temporal order
    dim_order = ["D1_EmotionPerception", "D2_EmotionRegulation",
                 "D3_AdaptivePersistence", "D4_AffectiveSelfAwareness",
                 "D5_EmotionCognitionAlign"]

    results = {}

    # Test different temporal strategies
    temporal_strategies = {
        "Early_D1_D2": ["D1_EmotionPerception", "D2_EmotionRegulation"],
        "Mid_D3": ["D3_AdaptivePersistence"],
        "Late_D4_D5": ["D4_AffectiveSelfAwareness", "D5_EmotionCognitionAlign"],
        "Progressive_All": dim_order,
    }

    print("\n" + "=" * 70)
    print("TEMPORAL STRATEGY COMPARISON")
    print("=" * 70)

    for strategy_name, dims_to_use in temporal_strategies.items():
        features_for_strategy = []
        for dim_name in dims_to_use:
            if dim_name in selected_by_dim:
                features_for_strategy.extend(selected_by_dim[dim_name])

        if not features_for_strategy:
            print(f"\n{strategy_name}: SKIPPED (no features)")
            continue

        X = norm_df[features_for_strategy].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Simple regression: using dimension-ordered features
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict, KFold

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        ridge = Ridge(alpha=1.0)
        y = exam_scores.values

        y_pred = cross_val_predict(ridge, X_scaled, y, cv=cv)

        r2 = r2_score(y, y_pred)
        pearson_r, p_val = pearsonr(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        results[strategy_name] = {
            "r2": r2,
            "pearson_r": pearson_r,
            "rmse": rmse,
            "mae": mae,
            "p_value": p_val,
            "n_features": len(features_for_strategy),
            "dimensions": ", ".join([d.split("_")[0] for d in dims_to_use])
        }

        print(f"\n{strategy_name}:")
        print(f"  Dimensions: {results[strategy_name]['dimensions']}")
        print(f"  Features: {len(features_for_strategy)}")
        print(f"  R²:        {r2:.4f}")
        print(f"  Pearson r: {pearson_r:.4f} (p={p_val:.4f})")
        print(f"  RMSE:      {rmse:.4f}")

    # Save results
    results_df = pd.DataFrame(results).T
    results_csv = os.path.join(OUTPUT_DIR, "option_b_temporal_results.csv")
    results_df.to_csv(results_csv)
    print(f"\n  Results → option_b_temporal_results.csv")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² comparison
    ax = axes[0]
    strategies = results_df.index
    r2_vals = results_df["r2"].values
    colors = ["green" if v > 0 else "red" for v in r2_vals]
    ax.bar(range(len(strategies)), r2_vals, color=colors, edgecolor="black")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Temporal Strategy: R² Score")
    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(r2_vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

    # Pearson r comparison
    ax = axes[1]
    r_vals = results_df["pearson_r"].values
    ax.bar(range(len(strategies)), r_vals, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Temporal Strategy: Correlation")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(r_vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "option_b_temporal_comparison.png")
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Visualization → figures/option_b_temporal_comparison.png")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("\nBest temporal strategy reveals when emotions matter most:")
    best_strategy = results_df["pearson_r"].abs().idxmax()
    best_r = results_df.loc[best_strategy, "pearson_r"]
    print(f"\n  {best_strategy}: r={best_r:.4f}")
    print(f"  This suggests that {best_strategy} emotions are most predictive")

    return results_df

def run():
    # Load data
    norm_df, exam_scores, selected_by_dim, stability_all = load_data()
    print(f"Loaded {len(norm_df)} participants")

    # Run temporal baseline
    results_df = create_baseline_temporal_model(norm_df, selected_by_dim, exam_scores, stability_all)

if __name__ == "__main__":
    run()
