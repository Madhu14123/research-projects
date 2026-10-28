"""
Stage 12 — Block & Recovery Analysis
====================================
Analyze time spent in block and recovery patterns.

Questions answered:
  - How much time do participants spend in blocks (hesitation/uncertainty)?
  - How quickly do they recover from blocks?
  - Do participants with better recovery show higher exam scores?
  - Is recovery time predictive of EI?

Metrics:
  - Block frequency (how many times block occurs)
  - Block duration (avg time per block)
  - Total block time (cumulative minutes)
  - Recovery latency (time to leave block and re-engage)
  - Recovery success (successfully return to productive stage)
  - Block-to-productive ratio (inverse: lower is better)

Outputs:
  outputs/block_recovery_metrics.csv
  outputs/recovery_patterns.csv
  outputs/figures/recovery_analysis.png
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from collections import defaultdict
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

def analyze_block_recovery(aligned_data, exam_scores):
    """
    Analyze block patterns and recovery for each participant.
    """
    print("=" * 70)
    print("BLOCK & RECOVERY ANALYSIS")
    print("=" * 70)

    results = []

    for pid, data_dict in aligned_data.items():
        if pid not in exam_scores.index:
            continue

        stage_df = data_dict.get("stage_df")
        if stage_df is None or len(stage_df) == 0:
            continue

        # Get all block occurrences
        block_indices = stage_df[stage_df["stage"] == BLOCK_STAGE].index.tolist()

        if len(block_indices) == 0:
            # No blocks - perfect case
            results.append({
                "pid": pid,
                "exam_score": exam_scores.loc[pid],
                "n_blocks": 0,
                "total_block_time": 0,
                "avg_block_duration": 0,
                "max_block_duration": 0,
                "recovery_latencies": [],
                "avg_recovery_latency": 0,
                "successful_recoveries": 0,
                "recovery_success_rate": 1.0,
                "block_frequency": 0,
                "block_to_total_ratio": 0
            })
            continue

        # Analyze each block occurrence
        block_durations = []
        recovery_latencies = []
        successful_recoveries = 0

        for block_idx in block_indices:
            # Duration of this block
            block_row = stage_df.loc[block_idx]
            start_sec = block_row["start_sec"]
            end_sec = block_row["end_sec"]

            if pd.notna(start_sec) and pd.notna(end_sec):
                block_dur = (end_sec - start_sec) / 60  # minutes
                block_durations.append(block_dur)

                # Find next non-block stage (recovery)
                next_productive_idx = None
                for idx in range(block_idx + 1, len(stage_df)):
                    if stage_df.iloc[idx]["stage"] != BLOCK_STAGE:
                        next_productive_idx = idx
                        break

                if next_productive_idx is not None:
                    next_row = stage_df.iloc[next_productive_idx]
                    next_start = next_row["start_sec"]

                    if pd.notna(next_start):
                        recovery_latency = (next_start - end_sec) / 60  # minutes
                        recovery_latencies.append(max(0, recovery_latency))

                        # Successful recovery: moved to productive stage
                        if next_row["stage"] in PRODUCTIVE_STAGES:
                            successful_recoveries += 1

        # Total block time
        total_block_time = sum(block_durations)
        avg_block_duration = np.mean(block_durations) if block_durations else 0
        max_block_duration = np.max(block_durations) if block_durations else 0

        # Recovery latency
        avg_recovery_latency = np.mean(recovery_latencies) if recovery_latencies else 0
        recovery_success_rate = successful_recoveries / len(block_indices) if block_indices else 1.0

        # Block ratio (total time in block vs total task time)
        total_task_time = (stage_df["end_sec"].max() - stage_df["start_sec"].min()) / 60
        block_to_total_ratio = total_block_time / total_task_time if total_task_time > 0 else 0

        results.append({
            "pid": pid,
            "exam_score": exam_scores.loc[pid],
            "n_blocks": len(block_indices),
            "total_block_time": total_block_time,
            "avg_block_duration": avg_block_duration,
            "max_block_duration": max_block_duration,
            "recovery_latencies": recovery_latencies,
            "avg_recovery_latency": avg_recovery_latency,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": recovery_success_rate,
            "block_frequency": len(block_indices),
            "block_to_total_ratio": block_to_total_ratio
        })

    results_df = pd.DataFrame(results)
    return results_df

def run():
    print("Loading data...")
    aligned_data, exam_scores = load_data()
    print(f"Loaded {len(aligned_data)} participants")

    # Analyze blocks
    results_df = analyze_block_recovery(aligned_data, exam_scores)
    print(f"\nAnalyzed {len(results_df)} participants with block data")

    # Print summary
    print("\n" + "=" * 70)
    print("BLOCK & RECOVERY METRICS")
    print("=" * 70)

    print("\nBlock Frequency (number of block occurrences):")
    print(f"  Mean: {results_df['n_blocks'].mean():.2f}")
    print(f"  Std:  {results_df['n_blocks'].std():.2f}")
    print(f"  Min:  {results_df['n_blocks'].min()}")
    print(f"  Max:  {results_df['n_blocks'].max()}")

    print("\nTotal Block Time (minutes):")
    print(f"  Mean: {results_df['total_block_time'].mean():.2f}")
    print(f"  Std:  {results_df['total_block_time'].std():.2f}")
    print(f"  Min:  {results_df['total_block_time'].min():.2f}")
    print(f"  Max:  {results_df['total_block_time'].max():.2f}")

    print("\nAvg Block Duration (minutes per block):")
    print(f"  Mean: {results_df['avg_block_duration'].mean():.2f}")
    print(f"  Std:  {results_df['avg_block_duration'].std():.2f}")

    print("\nRecovery Latency (time to leave block, minutes):")
    print(f"  Mean: {results_df['avg_recovery_latency'].mean():.2f}")
    print(f"  Std:  {results_df['avg_recovery_latency'].std():.2f}")

    print("\nRecovery Success Rate (% returned to productive stage):")
    print(f"  Mean: {results_df['recovery_success_rate'].mean():.1%}")
    print(f"  Std:  {results_df['recovery_success_rate'].std():.1%}")

    print("\nBlock-to-Total Ratio (% task time in block):")
    print(f"  Mean: {results_df['block_to_total_ratio'].mean():.1%}")
    print(f"  Std:  {results_df['block_to_total_ratio'].std():.1%}")

    # Correlation with exam score
    print("\n" + "=" * 70)
    print("CORRELATION WITH EXAM PERFORMANCE")
    print("=" * 70)

    from scipy.stats import pearsonr

    metrics = ["n_blocks", "total_block_time", "avg_block_duration",
               "avg_recovery_latency", "recovery_success_rate", "block_to_total_ratio"]

    for metric in metrics:
        r, p = pearsonr(results_df[metric], results_df["exam_score"])
        direction = "↑" if r > 0 else "↓"
        sig = " **" if p < 0.05 else ""
        print(f"{metric:30s}: r={r:+.4f}, p={p:.4f} {direction}{sig}")

    # High vs Low performers
    print("\n" + "=" * 70)
    print("HIGH vs LOW PERFORMERS")
    print("=" * 70)

    high_scorers = results_df[results_df["exam_score"] >= results_df["exam_score"].median()]
    low_scorers = results_df[results_df["exam_score"] < results_df["exam_score"].median()]

    print(f"\nHigh Performers (n={len(high_scorers)}):")
    print(f"  Block frequency:      {high_scorers['n_blocks'].mean():.2f}")
    print(f"  Total block time:     {high_scorers['total_block_time'].mean():.2f} min")
    print(f"  Recovery latency:     {high_scorers['avg_recovery_latency'].mean():.2f} min")
    print(f"  Recovery success:     {high_scorers['recovery_success_rate'].mean():.1%}")
    print(f"  Block-to-total:       {high_scorers['block_to_total_ratio'].mean():.1%}")

    print(f"\nLow Performers (n={len(low_scorers)}):")
    print(f"  Block frequency:      {low_scorers['n_blocks'].mean():.2f}")
    print(f"  Total block time:     {low_scorers['total_block_time'].mean():.2f} min")
    print(f"  Recovery latency:     {low_scorers['avg_recovery_latency'].mean():.2f} min")
    print(f"  Recovery success:     {low_scorers['recovery_success_rate'].mean():.1%}")
    print(f"  Block-to-total:       {low_scorers['block_to_total_ratio'].mean():.1%}")

    # Save results
    csv_path = os.path.join(OUTPUT_DIR, "block_recovery_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults → block_recovery_metrics.csv")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total block time vs exam score
    ax = axes[0, 0]
    ax.scatter(results_df["exam_score"], results_df["total_block_time"],
              s=100, alpha=0.7, edgecolors="black", c="coral")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Total Block Time (minutes)")
    ax.set_title("Block Time vs Exam Performance")
    r, p = pearsonr(results_df["exam_score"], results_df["total_block_time"])
    ax.text(0.05, 0.95, f"r={r:.3f}, p={p:.3f}", transform=ax.transAxes,
           verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(alpha=0.3)

    # Recovery latency vs exam score
    ax = axes[0, 1]
    ax.scatter(results_df["exam_score"], results_df["avg_recovery_latency"],
              s=100, alpha=0.7, edgecolors="black", c="lightblue")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Avg Recovery Latency (minutes)")
    ax.set_title("Recovery Speed vs Exam Performance")
    r, p = pearsonr(results_df["exam_score"], results_df["avg_recovery_latency"])
    ax.text(0.05, 0.95, f"r={r:.3f}, p={p:.3f}", transform=ax.transAxes,
           verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(alpha=0.3)

    # Recovery success rate vs exam score
    ax = axes[1, 0]
    ax.scatter(results_df["exam_score"], results_df["recovery_success_rate"],
              s=100, alpha=0.7, edgecolors="black", c="lightgreen")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Recovery Success Rate")
    ax.set_title("Recovery Success vs Exam Performance")
    r, p = pearsonr(results_df["exam_score"], results_df["recovery_success_rate"])
    ax.text(0.05, 0.95, f"r={r:.3f}, p={p:.3f}", transform=ax.transAxes,
           verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(alpha=0.3)

    # High vs Low performers
    ax = axes[1, 1]
    metrics_compare = ["n_blocks", "total_block_time", "avg_recovery_latency"]
    high_vals = [high_scorers[m].mean() for m in metrics_compare]
    low_vals = [low_scorers[m].mean() for m in metrics_compare]

    # Normalize for comparison
    high_vals_norm = [high_vals[i] / (high_vals[i] + low_vals[i] + 1e-6) for i in range(len(high_vals))]
    low_vals_norm = [low_vals[i] / (high_vals[i] + low_vals[i] + 1e-6) for i in range(len(low_vals))]

    x = np.arange(len(metrics_compare))
    width = 0.35
    ax.bar(x - width/2, high_vals_norm, width, label="High Score", color="green", alpha=0.7)
    ax.bar(x + width/2, low_vals_norm, width, label="Low Score", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Block Freq", "Block Time", "Recovery Latency"])
    ax.set_ylabel("Relative Value")
    ax.set_title("High vs Low Performers: Block Patterns")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "block_recovery.png")
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Visualization → figures/block_recovery.png")

    print("\n" + "=" * 70)

    return results_df

if __name__ == "__main__":
    run()
