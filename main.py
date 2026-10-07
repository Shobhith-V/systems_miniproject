import subprocess
import sys
from pathlib import Path

DATA_FILES = [
    "data/tep_fault_free_training.parquet",
    "data/tep_faulty_training_runs01-20.parquet",
]

# Order matters: evaluation needs both score files,
# and diagnosis needs the thresholds from evaluation.
STEPS = [
    "run_pca.py",         # -> results/scores_pca.parquet
    "run_ridge.py",       # -> results/scores_ridge.parquet
    "run_evaluation.py",  # -> results/thresholds.csv, results/detection.csv
    "run_diagnosis.py",   # -> results/contributions.csv
]


def main():
    missing = [f for f in DATA_FILES if not Path(f).exists()]
    if missing:
        sys.exit(f"Missing data files: {missing}. Download them into data/ first.")

    Path("results").mkdir(exist_ok=True)

    for step in STEPS:
        print(f"\n=== Running {step} ===")
        # check=True stops the pipeline if a step fails
        subprocess.run([sys.executable, step], check=True)

    print("\nDone. Files in results/:")
    for f in sorted(Path("results").iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
