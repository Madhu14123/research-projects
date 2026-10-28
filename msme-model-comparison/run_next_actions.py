from pathlib import Path
import json
import subprocess
import sys

BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "outputs" / "phase2_next_steps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_MAP = {
    "cluster": "train_cluster_model.py",
    "environment": "train_environment_model.py",
    "women": "train_women_model.py",
    "credit": "train_credit_model.py"
}


def run_script(name: str):
    script = BASE_DIR / SCRIPT_MAP[name]
    if not script.exists():
        print(f"Missing script: {script}")
        return False

    print(f"Running {name} pipeline: {script}")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode == 0


if __name__ == "__main__":
    plan_path = OUTPUT_DIR / "phase2_action_plan.json"
    if not plan_path.exists():
        print(f"Action plan not found: {plan_path}")
        sys.exit(1)

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    print("Starting Phase 2 next-action pipeline...")
    for stage in ["cluster", "environment", "women", "credit"]:
        ok = run_script(stage)
        if not ok:
            print(f"Pipeline failed at stage: {stage}")
            sys.exit(1)

    print("All next-action scripts completed successfully.")
