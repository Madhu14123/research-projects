from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path.cwd()
DATA_FILE = BASE_DIR / "outputs" / "MSME_preprocessed_full.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase2_next_steps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_FILE)
    subset = df[["latitude", "longitude", "investment", "employment", "DISTRICT_NAME", "CATEGOERY_NAME"]].copy()
    subset = subset.dropna(subset=["latitude", "longitude", "investment", "employment"])

    X = subset[["latitude", "longitude", "investment", "employment"]].astype(float).values

    # Use a representative sample for silhouette scoring to avoid memory issues
    sample_size = min(5000, len(X))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X), size=sample_size, replace=False)
    X_sample = X[sample_idx]

    # Try a few cluster counts and save the best silhouette score on the sample
    best = None
    for k in range(3, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        labels_sample = model.labels_[sample_idx]
        score = silhouette_score(X_sample, labels_sample)
        if best is None or score > best["score"]:
            best = {"k": k, "score": float(score), "labels": labels.tolist()}

    subset["cluster"] = best["labels"]
    subset.to_csv(OUTPUT_DIR / "cluster_model_results.csv", index=False)

    with open(OUTPUT_DIR / "cluster_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_k": best["k"],
                "best_silhouette_score": best["score"]
            },
            f,
            indent=2
        )

    print(f"Cluster model complete. Best k={best['k']} | silhouette={best['score']:.4f}")


if __name__ == "__main__":
    main()
