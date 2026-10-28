"""
Stage 13 — Emotional Volatility & Stability Analysis
=====================================================
Analyze emotion stability and volatility patterns.

Questions answered:
  - How stable are emotions across the exam?
  - Do participants with lower emotional volatility score higher?
  - Is emotional consistency predictive of EI?
  - Which emotions show most/least stability?

Metrics:
  - Emotion volatility (std dev of each emotion over time)
  - Emotional consistency (inverse of volatility)
  - Chaos index (frequency of rapid emotion changes)
  - Emotion switching (how often dominant emotion changes)
  - Emotional range (max - min for each emotion)

Outputs:
  outputs/emotional_volatility_metrics.csv
  outputs/emotion_stability_by_stage.csv
  outputs/figures/volatility_analysis.png
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, EMOTION_COLS

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data():
    """Load aligned data with emotion information."""
    pkl_path = os.path.join(OUTPUT_DIR, "aligned_data.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

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

    return data, exam_scores

def analyze_emotional_volatility(aligned_data, exam_scores):
    """
    Analyze emotion stability and volatility for each participant.
    """
    print("=" * 70)
    print("EMOTIONAL VOLATILITY & STABILITY ANALYSIS")
    print("=" * 70)

    results = []

    for pid, data_dict in aligned_data.items():
        if pid not in exam_scores.index:
            continue

        face_df = data_dict.get("face_df", pd.DataFrame())
        if face_df is None or len(face_df) == 0:
            continue

        # Get emotion columns that exist
        emotion_cols = [col for col in EMOTION_COLS if col in face_df.columns]
        if not emotion_cols:
            continue

        # Compute volatility for each emotion
        volatility_dict = {}
        range_dict = {}
        mean_dict = {}

        for emotion in emotion_cols:
            values = face_df[emotion].values
            if len(values) > 1:
                volatility_dict[emotion] = np.std(values)
                range_dict[emotion] = np.max(values) - np.min(values)
                mean_dict[emotion] = np.mean(values)
            else:
                volatility_dict[emotion] = 0
                range_dict[emotion] = 0
                mean_dict[emotion] = values[0] if len(values) > 0 else 0

        # Overall volatility (average across emotions)
        overall_volatility = np.mean(list(volatility_dict.values())) if volatility_dict else 0

        # Emotional consistency (inverse of volatility)
        max_possible_volatility = 1.0  # max std dev possible
        emotional_consistency = max(0, 1 - (overall_volatility / max_possible_volatility))

        # Chaos index: how often emotions change rapidly
        if len(face_df) > 1:
            emotion_changes = 0
            for emotion in emotion_cols:
                diffs = np.abs(np.diff(face_df[emotion].values))
                rapid_changes = np.sum(diffs > np.mean(diffs) + np.std(diffs))
                emotion_changes += rapid_changes

            chaos_index = emotion_changes / (len(face_df) * len(emotion_cols)) if emotion_cols else 0
        else:
            chaos_index = 0

        # Dominant emotion (most consistent across session)
        consistency_by_emotion = {}
        for emotion in emotion_cols:
            consistency_by_emotion[emotion] = 1 - (volatility_dict[emotion] / max_possible_volatility)

        dominant_emotion = max(consistency_by_emotion, key=consistency_by_emotion.get) if consistency_by_emotion else None
        dominant_consistency = consistency_by_emotion.get(dominant_emotion, 0) if dominant_emotion else 0

        # Emotional range (spread)
        avg_range = np.mean(list(range_dict.values())) if range_dict else 0

        results.append({
            "pid": pid,
            "exam_score": exam_scores.loc[pid],
            "overall_volatility": overall_volatility,
            "emotional_consistency": emotional_consistency,
            "chaos_index": chaos_index,
            "dominant_emotion": dominant_emotion,
            "dominant_consistency": dominant_consistency,
            "avg_emotional_range": avg_range,
            **{f"volatility_{emo}": volatility_dict.get(emo, 0) for emo in emotion_cols},
            **{f"consistency_{emo}": 1 - volatility_dict.get(emo, 0) for emo in emotion_cols},
        })

    results_df = pd.DataFrame(results)
    return results_df, emotion_cols

def run():
    print("Loading data...")
    aligned_data, exam_scores = load_data()
    print(f"Loaded {len(aligned_data)} participants")

    # Analyze volatility
    results_df, emotion_cols = analyze_emotional_volatility(aligned_data, exam_scores)
    print(f"\nAnalyzed {len(results_df)} participants with emotion data")

    # Print summary
    print("\n" + "=" * 70)
    print("EMOTIONAL VOLATILITY METRICS")
    print("=" * 70)

    print("\nOverall Volatility (std dev of emotions):")
    print(f"  Mean: {results_df['overall_volatility'].mean():.4f}")
    print(f"  Std:  {results_df['overall_volatility'].std():.4f}")

    print("\nEmotional Consistency (0-1, higher is better):")
    print(f"  Mean: {results_df['emotional_consistency'].mean():.4f}")
    print(f"  Std:  {results_df['emotional_consistency'].std():.4f}")

    print("\nChaos Index (0-1, lower is better):")
    print(f"  Mean: {results_df['chaos_index'].mean():.4f}")
    print(f"  Std:  {results_df['chaos_index'].std():.4f}")

    print("\nEmotional Range (spread of emotions):")
    print(f"  Mean: {results_df['avg_emotional_range'].mean():.4f}")
    print(f"  Std:  {results_df['avg_emotional_range'].std():.4f}")

    # Per-emotion volatility
    print("\n" + "=" * 70)
    print("VOLATILITY BY EMOTION")
    print("=" * 70)

    for emotion in emotion_cols:
        vol_col = f"volatility_{emotion}"
        if vol_col in results_df.columns:
            print(f"\n{emotion}:")
            print(f"  Mean volatility: {results_df[vol_col].mean():.4f}")
            print(f"  Mean consistency: {results_df[f'consistency_{emotion}'].mean():.4f}")

    # Correlation with exam score
    print("\n" + "=" * 70)
    print("CORRELATION WITH EXAM PERFORMANCE")
    print("=" * 70)

    from scipy.stats import pearsonr

    for metric in ["overall_volatility", "emotional_consistency", "chaos_index", "avg_emotional_range"]:
        r, p = pearsonr(results_df[metric], results_df["exam_score"])
        direction = "↑" if r > 0 else "↓"
        sig = " **" if p < 0.05 else ""
        print(f"{metric:30s}: r={r:+.4f}, p={p:.4f} {direction}{sig}")

    # High vs Low performers
    print("\n" + "=" * 70)
    print("HIGH vs LOW PERFORMERS: EMOTIONAL STABILITY")
    print("=" * 70)

    high_scorers = results_df[results_df["exam_score"] >= results_df["exam_score"].median()]
    low_scorers = results_df[results_df["exam_score"] < results_df["exam_score"].median()]

    print(f"\nHigh Performers (n={len(high_scorers)}):")
    print(f"  Overall volatility:    {high_scorers['overall_volatility'].mean():.4f}")
    print(f"  Emotional consistency: {high_scorers['emotional_consistency'].mean():.4f}")
    print(f"  Chaos index:           {high_scorers['chaos_index'].mean():.4f}")
    print(f"  Emotional range:       {high_scorers['avg_emotional_range'].mean():.4f}")

    print(f"\nLow Performers (n={len(low_scorers)}):")
    print(f"  Overall volatility:    {low_scorers['overall_volatility'].mean():.4f}")
    print(f"  Emotional consistency: {low_scorers['emotional_consistency'].mean():.4f}")
    print(f"  Chaos index:           {low_scorers['chaos_index'].mean():.4f}")
    print(f"  Emotional range:       {low_scorers['avg_emotional_range'].mean():.4f}")

    # Save results
    csv_path = os.path.join(OUTPUT_DIR, "emotional_volatility_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults → emotional_volatility_metrics.csv")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Overall volatility vs exam score
    ax = axes[0, 0]
    ax.scatter(results_df["exam_score"], results_df["overall_volatility"],
              s=100, alpha=0.7, edgecolors="black", c="purple")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Overall Emotional Volatility")
    ax.set_title("Emotion Volatility vs Exam Performance")
    r, p = pearsonr(results_df["exam_score"], results_df["overall_volatility"])
    ax.text(0.05, 0.95, f"r={r:.3f}, p={p:.3f}", transform=ax.transAxes,
           verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(alpha=0.3)

    # Emotional consistency vs exam score
    ax = axes[0, 1]
    ax.scatter(results_df["exam_score"], results_df["emotional_consistency"],
              s=100, alpha=0.7, edgecolors="black", c="teal")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Emotional Consistency (0-1)")
    ax.set_title("Emotion Consistency vs Exam Performance")
    r, p = pearsonr(results_df["exam_score"], results_df["emotional_consistency"])
    ax.text(0.05, 0.95, f"r={r:.3f}, p={p:.3f}", transform=ax.transAxes,
           verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(alpha=0.3)

    # Chaos index vs exam score
    ax = axes[1, 0]
    ax.scatter(results_df["exam_score"], results_df["chaos_index"],
              s=100, alpha=0.7, edgecolors="black", c="orange")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Chaos Index (0-1)")
    ax.set_title("Emotional Chaos vs Exam Performance")
    r, p = pearsonr(results_df["exam_score"], results_df["chaos_index"])
    ax.text(0.05, 0.95, f"r={r:.3f}, p={p:.3f}", transform=ax.transAxes,
           verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(alpha=0.3)

    # High vs Low comparison
    ax = axes[1, 1]
    metrics = ["overall_volatility", "chaos_index"]
    high_means = [high_scorers[m].mean() for m in metrics]
    low_means = [low_scorers[m].mean() for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, high_means, width, label="High Score", color="green", alpha=0.7)
    ax.bar(x + width/2, low_means, width, label="Low Score", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Volatility", "Chaos"])
    ax.set_ylabel("Value")
    ax.set_title("High vs Low Performers: Emotional Stability")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "emotional_volatility.png")
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Visualization → figures/emotional_volatility.png")

    print("\n" + "=" * 70)

    return results_df

if __name__ == "__main__":
    run()
