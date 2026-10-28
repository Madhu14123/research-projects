"""
Stage 11 — Stage Transition Analysis
=====================================
Analyze how participants move through problem-solving stages.

Questions answered:
  - Do higher EI participants show more adaptive transitions?
  - What's the difference between chaotic vs productive stage switching?
  - How do transition patterns correlate with exam performance?

Metrics:
  - Transition frequency (from → to stage)
  - Adaptive score (forward progress in problem-solving stages)
  - Chaotic switching (back-and-forth without progress)
  - Recovery transitions (from block back to productive work)

Outputs:
  outputs/stage_transitions_analysis.csv
  outputs/adaptive_vs_chaotic_patterns.csv
  outputs/figures/transition_patterns.png
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, PRODUCTIVE_STAGES, BLOCK_STAGE

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data():
    """Load aligned data with stage information."""
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

def analyze_stage_transitions(aligned_data, exam_scores):
    """
    Analyze stage transition patterns for each participant.
    """
    print("=" * 70)
    print("STAGE TRANSITION ANALYSIS")
    print("=" * 70)

    # Define stage progression
    productive_sequence = ["Preparation", "Ideation", "Production", "Evaluation", "Refinement"]

    results = []

    for pid, data_dict in aligned_data.items():
        if pid not in exam_scores.index:
            continue

        stage_df = data_dict.get("stage_df")
        if stage_df is None or len(stage_df) == 0:
            continue

        # Get sequence of stages in temporal order
        stages = stage_df["stage"].values
        if len(stages) < 2:
            continue

        # Count transitions
        transitions = []
        for i in range(len(stages) - 1):
            from_stage = stages[i]
            to_stage = stages[i + 1]
            if from_stage != to_stage:
                transitions.append((from_stage, to_stage))

        # Analyze transition patterns
        transition_counter = Counter(transitions)
        n_transitions = len(transitions)

        # Adaptive transitions: moving forward in problem-solving sequence
        adaptive_transitions = 0
        chaotic_transitions = 0

        for (from_s, to_s), count in transition_counter.items():
            if from_s in productive_sequence and to_s in productive_sequence:
                from_idx = productive_sequence.index(from_s)
                to_idx = productive_sequence.index(to_s)
                if to_idx > from_idx:  # Forward progress
                    adaptive_transitions += count
                elif to_idx < from_idx and to_idx >= 2:  # Back to later stages (eval/refine)
                    adaptive_transitions += count
                else:  # Back to early stages
                    chaotic_transitions += count
            elif from_s == BLOCK_STAGE and to_s in productive_sequence:
                adaptive_transitions += count  # Recovery from block
            elif from_s in productive_sequence and to_s == BLOCK_STAGE:
                chaotic_transitions += count  # Going to block
            else:
                chaotic_transitions += count

        # Compute metrics
        if n_transitions > 0:
            adaptive_ratio = adaptive_transitions / n_transitions
            chaotic_ratio = chaotic_transitions / n_transitions
        else:
            adaptive_ratio = 0
            chaotic_ratio = 0

        # Time spent in each stage
        stage_times = defaultdict(float)
        for idx, row in stage_df.iterrows():
            stage = row["stage"]
            start = row["start_sec"]
            end = row["end_sec"]
            if pd.notna(start) and pd.notna(end):
                duration = (end - start) / 60  # minutes
                stage_times[stage] += duration

        # Block time
        block_time = stage_times.get(BLOCK_STAGE, 0)

        # Back-and-forth score (lower is better)
        back_forth = sum(1 for (f, t), c in transition_counter.items()
                        if f in productive_sequence and t in productive_sequence
                        and productive_sequence.index(f) > productive_sequence.index(t))

        results.append({
            "pid": pid,
            "exam_score": exam_scores.loc[pid],
            "n_transitions": n_transitions,
            "adaptive_ratio": adaptive_ratio,
            "chaotic_ratio": chaotic_ratio,
            "adaptive_transitions": adaptive_transitions,
            "chaotic_transitions": chaotic_transitions,
            "block_time_min": block_time,
            "back_forth_transitions": back_forth,
            "total_stage_time": sum(stage_times.values()),
            "stage_diversity": len(set(stages))
        })

    results_df = pd.DataFrame(results)
    return results_df, productive_sequence

def classify_ei_groups(results_df):
    """Classify participants into high vs low EI based on adaptive transitions."""
    # EI proxy: high adaptive ratio + low chaotic ratio + low block time
    results_df["ei_proxy"] = (
        results_df["adaptive_ratio"] * 0.4 +
        (1 - results_df["chaotic_ratio"]) * 0.3 +
        (1 - results_df["block_time_min"] / results_df["block_time_min"].max()) * 0.3
    )

    results_df["ei_group"] = results_df["ei_proxy"].apply(
        lambda x: "High EI" if x >= results_df["ei_proxy"].median() else "Low EI"
    )

    return results_df

def run():
    print("Loading data...")
    aligned_data, exam_scores = load_data()
    print(f"Loaded {len(aligned_data)} participants")

    # Analyze transitions
    results_df, productive_seq = analyze_stage_transitions(aligned_data, exam_scores)
    print(f"\nAnalyzed {len(results_df)} participants with transition data")

    # Classify EI groups
    results_df = classify_ei_groups(results_df)

    # Print summary
    print("\n" + "=" * 70)
    print("TRANSITION PATTERN SUMMARY")
    print("=" * 70)

    print("\nAdaptive Ratio (0-1, higher is better):")
    print(f"  Mean: {results_df['adaptive_ratio'].mean():.3f}")
    print(f"  Std:  {results_df['adaptive_ratio'].std():.3f}")

    print("\nChaotic Ratio (0-1, lower is better):")
    print(f"  Mean: {results_df['chaotic_ratio'].mean():.3f}")
    print(f"  Std:  {results_df['chaotic_ratio'].std():.3f}")

    print("\nBlock Time (minutes):")
    print(f"  Mean: {results_df['block_time_min'].mean():.2f}")
    print(f"  Std:  {results_df['block_time_min'].std():.2f}")

    # Compare high vs low EI groups
    print("\n" + "=" * 70)
    print("HIGH EI vs LOW EI GROUPS")
    print("=" * 70)

    high_ei = results_df[results_df["ei_group"] == "High EI"]
    low_ei = results_df[results_df["ei_group"] == "Low EI"]

    print(f"\nHigh EI (n={len(high_ei)}):")
    print(f"  Adaptive ratio:  {high_ei['adaptive_ratio'].mean():.3f}")
    print(f"  Chaotic ratio:   {high_ei['chaotic_ratio'].mean():.3f}")
    print(f"  Block time:      {high_ei['block_time_min'].mean():.2f} min")
    print(f"  Exam score:      {high_ei['exam_score'].mean():.2f}")
    print(f"  Back-forth:      {high_ei['back_forth_transitions'].mean():.1f}")

    print(f"\nLow EI (n={len(low_ei)}):")
    print(f"  Adaptive ratio:  {low_ei['adaptive_ratio'].mean():.3f}")
    print(f"  Chaotic ratio:   {low_ei['chaotic_ratio'].mean():.3f}")
    print(f"  Block time:      {low_ei['block_time_min'].mean():.2f} min")
    print(f"  Exam score:      {low_ei['exam_score'].mean():.2f}")
    print(f"  Back-forth:      {low_ei['back_forth_transitions'].mean():.1f}")

    # Correlation with exam score
    print("\n" + "=" * 70)
    print("CORRELATION WITH EXAM PERFORMANCE")
    print("=" * 70)

    from scipy.stats import pearsonr

    for col in ["adaptive_ratio", "chaotic_ratio", "block_time_min", "back_forth_transitions"]:
        r, p = pearsonr(results_df[col], results_df["exam_score"])
        print(f"\n{col}:")
        print(f"  r = {r:+.4f}, p = {p:.4f}")

    # Save results
    csv_path = os.path.join(OUTPUT_DIR, "stage_transitions_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults → stage_transitions_analysis.csv")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Adaptive ratio by exam score
    ax = axes[0, 0]
    ax.scatter(results_df["exam_score"], results_df["adaptive_ratio"],
              c=results_df["ei_proxy"], cmap="RdYlGn", s=100, alpha=0.7, edgecolors="black")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Adaptive Transition Ratio")
    ax.set_title("Adaptive Transitions vs Exam Performance")
    ax.grid(alpha=0.3)

    # Block time by exam score
    ax = axes[0, 1]
    ax.scatter(results_df["exam_score"], results_df["block_time_min"],
              c=results_df["ei_proxy"], cmap="RdYlGn", s=100, alpha=0.7, edgecolors="black")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Time in Block (minutes)")
    ax.set_title("Block Time vs Exam Performance")
    ax.grid(alpha=0.3)

    # High vs Low EI comparison
    ax = axes[1, 0]
    metrics = ["adaptive_ratio", "chaotic_ratio", "block_time_min"]
    high_means = [high_ei[m].mean() for m in metrics]
    low_means = [low_ei[m].mean() for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, high_means, width, label="High EI", color="green", alpha=0.7)
    ax.bar(x + width/2, low_means, width, label="Low EI", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Adaptive", "Chaotic", "Block Time"])
    ax.set_ylabel("Value")
    ax.set_title("EI Groups: Behavioral Patterns")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # EI proxy vs exam score
    ax = axes[1, 1]
    ax.scatter(results_df["ei_proxy"], results_df["exam_score"], s=100, alpha=0.7, edgecolors="black")
    from scipy.stats import pearsonr
    r, p = pearsonr(results_df["ei_proxy"], results_df["exam_score"])
    ax.set_xlabel("EI Proxy Score")
    ax.set_ylabel("Exam Score")
    ax.set_title(f"EI Proxy vs Exam Score (r={r:.3f}, p={p:.3f})")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "stage_transitions.png")
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Visualization → figures/stage_transitions.png")

    print("\n" + "=" * 70)

    return results_df

if __name__ == "__main__":
    run()
