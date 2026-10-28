"""
Master Pipeline Runner
========================
Executes the 5-stage EI pipeline in sequence.
Stages 1-4 use behavioral data ONLY (no exam scores).
Stage 5 loads exam scores for the confirmatory correlation test.

Usage:
  python run_pipeline.py [--start STAGE] [--stop STAGE]
  e.g.
  python run_pipeline.py              # run all 5 stages
  python run_pipeline.py --start 2   # resume from Stage 2
  python run_pipeline.py --stop 4    # run Stages 1-4 only (no exam scores)
"""

import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from config import OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_stage(stage_num: int, shared: dict):
    t0 = time.time()
    print(f"\n{'#'*60}")
    print(f"# STAGE {stage_num}")
    print(f"{'#'*60}")

    if stage_num == 1:
        from stage1_load_align import run as s1
        shared["aligned"] = s1()

    elif stage_num == 2:
        from stage2_features import run as s2
        shared["feat_df"] = s2(aligned=shared.get("aligned"))

    elif stage_num == 3:
        from stage3_ei_composite import run as s3
        shared["ei_scores"], shared["norm_df"] = s3(feat_df=shared.get("feat_df"))

    elif stage_num == 4:
        from stage4_validation import run as s4
        shared["selected_features"], shared["stability"], shared["dim_results"], _ = s4(
            norm_df=shared.get("norm_df"),
            ei_scores=shared.get("ei_scores", {}).get("EI_theory") if isinstance(shared.get("ei_scores"), dict) else
                      (shared["ei_scores"]["EI_theory"] if shared.get("ei_scores") is not None else None))

    elif stage_num == 5:
        from stage5_confirmatory import run as s5
        shared["confirmatory"] = s5(
            ei_scores=shared.get("ei_scores"),
            norm_df=shared.get("norm_df"),
            selected_features=shared.get("selected_features"))

    elapsed = time.time() - t0
    print(f"\nStage {stage_num} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop",  type=int, default=5)
    args = parser.parse_args()

    print("="*60)
    print("EI COMPUTATIONAL PIPELINE")
    print("Stages 1-4: behavioral data only (no exam scores)")
    print("Stage 5:    confirmatory correlation with exam scores")
    print("="*60)

    shared = {}
    for stage in range(args.start, args.stop + 1):
        run_stage(stage, shared)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"Outputs in: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
