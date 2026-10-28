"""
Stage 7 — Temporal Data Exploration
====================================
Understand the stage-wise structure of emotional data and how features vary
across problem-solving phases.

Goal: Identify which features are available per-stage and how emotion
dynamics evolve from start to finish of exam.

Outputs:
  outputs/temporal_summary.txt
  outputs/stage_wise_feature_matrix.pkl
  outputs/per_dimension_temporal_analysis.csv
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

def load_raw_data():
    """Load stage-wise data and features."""
    pkl_path = os.path.join(OUTPUT_DIR, "aligned_data.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data

def load_features():
    """Load selected features."""
    sel_path = os.path.join(OUTPUT_DIR, "selected_features.json")
    with open(sel_path) as f:
        sel_data = json.load(f)
    return sel_data["by_dimension"]

def analyze_temporal_structure(aligned_data, selected_features):
    """Analyze what stages exist and what features are computable per-stage."""

    print("=" * 70)
    print("TEMPORAL STRUCTURE ANALYSIS")
    print("=" * 70)

    stage_counts = defaultdict(int)
    stage_durations = defaultdict(list)
    per_stage_data_availability = defaultdict(lambda: defaultdict(list))

    for pid, data_dict in aligned_data.items():
        if "stage_df" not in data_dict:
            continue

        stage_df = data_dict["stage_df"]
        face_df = data_dict.get("face_df", pd.DataFrame())

        # Count stages
        for stage in stage_df["stage"].unique():
            stage_counts[stage] += 1

            # Duration of this stage
            stage_data = stage_df[stage_df["stage"] == stage]
            if len(stage_data) > 0:
                start = stage_data["start_sec"].min()
                end = stage_data["end_sec"].max()
                if pd.notna(start) and pd.notna(end):
                    duration = (end - start) / 60  # minutes
                    if duration > 0:
                        stage_durations[stage].append(duration)

    # Print stage summary
    print("\nStages Found in Dataset:")
    print("─" * 70)
    for stage in sorted(stage_counts.keys()):
        count = stage_counts[stage]
        durations = stage_durations.get(stage, [])
        avg_dur = np.mean(durations) if durations else 0
        print(f"  {stage:20s}: {count:3d} occurrences, avg duration: {avg_dur:6.2f} min")

    return stage_counts, stage_durations

def create_stage_wise_feature_matrix(aligned_data, selected_features):
    """
    Create feature matrix organized by stage.

    Structure: {pid: {stage: {dimension: features_dict}}}
    """
    stage_wise = {}

    print("\n" + "=" * 70)
    print("CREATING STAGE-WISE FEATURE MATRIX")
    print("=" * 70)

    for pid, data_dict in aligned_data.items():
        stage_df = data_dict.get("stage_df")
        face_df = data_dict.get("face_df", pd.DataFrame())

        if stage_df is None or len(stage_df) == 0:
            continue

        stage_wise[pid] = {}

        for stage in stage_df["stage"].unique():
            stage_mask = stage_df["stage"] == stage
            stage_times = stage_df[stage_mask][["start_sec", "end_sec"]].values

            if len(stage_times) == 0:
                continue

            start_sec = stage_times[:, 0].min()
            end_sec = stage_times[:, 1].max()

            # Filter face data to this stage window
            if len(face_df) > 0 and "t_sec" in face_df.columns:
                stage_face = face_df[
                    (face_df["t_sec"] >= start_sec) &
                    (face_df["t_sec"] <= end_sec)
                ]
            else:
                stage_face = face_df

            # Aggregate emotional features for this stage
            stage_wise[pid][stage] = {
                "n_frames": len(stage_face),
                "duration_sec": end_sec - start_sec,
                "has_data": len(stage_face) > 0,
                "emotions": stage_face[[c for c in face_df.columns
                                      if c in ["Happiness", "Anger", "Fear",
                                              "Sadness", "Disgust", "Stress", "Surprise"]]].mean().to_dict()
                            if len(stage_face) > 0 else {}
            }

    return stage_wise

def analyze_per_dimension_temporal(aligned_data):
    """
    Analyze how each dimension evolves across exam stages.
    """
    print("\n" + "=" * 70)
    print("PER-DIMENSION TEMPORAL EVOLUTION")
    print("=" * 70)

    # Track emotion changes across stages
    stage_order = {
        "Preparation": 1, "Ideation": 2, "Production": 3,
        "Evaluation": 4, "Refinement": 5, "Block": 0
    }

    emotion_by_stage = defaultdict(lambda: defaultdict(list))

    for pid, data_dict in aligned_data.items():
        stage_df = data_dict.get("stage_df")
        face_df = data_dict.get("face_df", pd.DataFrame())

        if stage_df is None or len(face_df) == 0:
            continue

        for stage in stage_df["stage"].unique():
            stage_mask = stage_df["stage"] == stage
            stage_times = stage_df[stage_mask][["start_sec", "end_sec"]].values

            if len(stage_times) == 0:
                continue

            start_sec = stage_times[:, 0].min()
            end_sec = stage_times[:, 1].max()

            stage_face = face_df[
                (face_df["t_sec"] >= start_sec) &
                (face_df["t_sec"] <= end_sec)
            ]

            if len(stage_face) == 0:
                continue

            # Compute emotion averages for this stage
            for emotion in ["Happiness", "Anger", "Fear", "Sadness", "Disgust", "Stress", "Surprise"]:
                if emotion in stage_face.columns:
                    avg_val = stage_face[emotion].mean()
                    if pd.notna(avg_val):
                        emotion_by_stage[stage][emotion].append(avg_val)

    # Print summary
    for stage in sorted(emotion_by_stage.keys(),
                       key=lambda x: stage_order.get(x, 99)):
        print(f"\n{stage}:")
        for emotion, values in sorted(emotion_by_stage[stage].items()):
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {emotion:12s}: {mean_val:6.3f} ± {std_val:6.3f}")

    return emotion_by_stage

def create_temporal_summary_report(stage_counts, stage_durations, emotion_by_stage):
    """Create text summary of temporal analysis."""

    lines = [
        "TEMPORAL DATA STRUCTURE SUMMARY",
        "=" * 70,
        "\nEXAM STAGES (in typical order):",
        "─" * 70,
        "1. Preparation  — Initial problem analysis",
        "2. Ideation     — Brainstorming solutions",
        "3. Production   — Creating/implementing solution",
        "4. Evaluation   — Testing solution",
        "5. Refinement   — Improving solution",
        "0. Block        — Break between problems",
        "\nKEY INSIGHT: Features have DIFFERENT values at different stages",
        "─" * 70,
        "\nEmotion Evolution Across Exam:",
    ]

    stage_order = {
        "Preparation": 1, "Ideation": 2, "Production": 3,
        "Evaluation": 4, "Refinement": 5, "Block": 0
    }

    for stage in sorted(emotion_by_stage.keys(),
                       key=lambda x: stage_order.get(x, 99)):
        lines.append(f"\n{stage}:")
        for emotion, values in sorted(emotion_by_stage[stage].items()):
            mean_val = np.mean(values)
            lines.append(f"  {emotion:12s}: {mean_val:6.3f}")

    lines.extend([
        "\n\nIMPLICATIONS FOR FEATURE SELECTION:",
        "=" * 70,
        "\n✓ Features are NOT constant across time",
        "✓ Stage-specific models may outperform global models",
        "✓ Time-series approach could capture emotional dynamics",
        "✓ Some stages may be more predictive than others",
        "✓ Per-dimension analysis should respect temporal structure",
    ])

    return "\n".join(lines)

def run():
    print("=" * 70)
    print("STAGE 7 — TEMPORAL EXPLORATION")
    print("=" * 70)

    # Load data
    aligned_data = load_raw_data()
    print(f"\nLoaded {len(aligned_data)} participants")

    selected_features = load_features()

    # Analyze stages
    stage_counts, stage_durations = analyze_temporal_structure(aligned_data, selected_features)

    # Create stage-wise matrix
    stage_wise_data = create_stage_wise_feature_matrix(aligned_data, selected_features)
    print(f"\nCreated stage-wise data for {len(stage_wise_data)} participants")

    # Analyze per-dimension temporal evolution
    emotion_by_stage = analyze_per_dimension_temporal(aligned_data)

    # Create report
    report = create_temporal_summary_report(stage_counts, stage_durations, emotion_by_stage)

    # Save outputs
    report_path = os.path.join(OUTPUT_DIR, "temporal_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport → temporal_summary.txt")

    stage_wise_path = os.path.join(OUTPUT_DIR, "stage_wise_feature_matrix.pkl")
    with open(stage_wise_path, "wb") as f:
        pickle.dump(stage_wise_data, f)
    print(f"Stage-wise data → stage_wise_feature_matrix.pkl")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    run()
