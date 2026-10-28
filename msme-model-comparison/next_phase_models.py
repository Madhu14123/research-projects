from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------
# Configuration
# ---------------------------
BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "outputs"
PHASE2_DIR = OUTPUT_DIR / "phase2"
PHASE2_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = OUTPUT_DIR / "MSME_preprocessed_full.csv"

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

# ---------------------------
# Load prepared data
# ---------------------------
df = pd.read_csv(DATA_FILE)

# Keep only columns needed for next-step modeling
cluster_cols = [
    "latitude", "longitude", "investment", "employment",
    "CATEGOERY_NAME", "DISTRICT_NAME", "industrial_park"
]

risk_cols = [
    "PCBCATEGORY", "DISTRICT_NAME", "CATEGOERY_NAME", "investment",
    "employment", "industrial_park", "promoter_women", "social_status",
    "is_women_led", "is_in_industrial_estate", "risk_score"
]

# ---------------------------
# Helper functions
# ---------------------------
def save_plot(name, fig=None):
    if fig is None:
        fig = plt.gcf()
    path = PHASE2_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def summarize_cluster_results(cluster_df, labels):
    summary = pd.DataFrame({
        "cluster": np.arange(len(np.unique(labels)))
    })
    summary["count"] = pd.Series(labels).value_counts().reindex(summary["cluster"], fill_value=0).values
    for col in ["investment", "employment", "turnover", "risk_score"]:
        if col in cluster_df.columns:
            summary[f"avg_{col}"] = (
                cluster_df.assign(cluster=labels)
                .groupby("cluster")[col]
                .mean()
                .reindex(summary["cluster"], fill_value=np.nan)
                .values
            )
    return summary


# ---------------------------
# 1) Prepare clustering dataset
# ---------------------------
cluster_df = df[cluster_cols].copy()
cluster_df = cluster_df.dropna(subset=["latitude", "longitude", "investment", "employment"])

# For clustering, use numeric features only
numeric_cluster = ["latitude", "longitude", "investment", "employment"]
X_cluster = cluster_df[numeric_cluster]

# Run KMeans baseline (7 clusters per report)
kmeans = KMeans(n_clusters=7, random_state=42, n_init=20)
cluster_df["cluster"] = kmeans.fit_predict(X_cluster)

cluster_summary = summarize_cluster_results(cluster_df, cluster_df["cluster"])
cluster_summary.to_csv(PHASE2_DIR / "cluster_summary.csv", index=False)

# Save cluster assignments
cluster_df[["latitude", "longitude", "investment", "employment", "cluster"]].to_csv(
    PHASE2_DIR / "cluster_assignments.csv", index=False
)

# Simple cluster visualization by district concentration
cluster_dist = (
    cluster_df.groupby(["cluster", "DISTRICT_NAME"])
    .size()
    .reset_index(name="count")
)

fig, ax = plt.subplots(figsize=(12, 6))
cluster_dist_top = cluster_dist.sort_values("count", ascending=False).head(20)
sns.barplot(data=cluster_dist_top, x="DISTRICT_NAME", y="count", hue="cluster", dodge=False, ax=ax)
ax.set_title("Cluster Distribution by District")
ax.set_xlabel("District")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)
save_plot("cluster_by_district", fig)

# ---------------------------
# 2) Prepare environmental risk dataset
# ---------------------------
environment_df = df[risk_cols].copy()

environment_df = environment_df.dropna(subset=["PCBCATEGORY", "DISTRICT_NAME", "CATEGOERY_NAME"])

environment_df["target_high_risk"] = environment_df["PCBCATEGORY"].isin(["ORANGE", "RED"]).astype(int)

# Build modeling features
X_env = environment_df.drop(columns=["PCBCATEGORY", "target_high_risk"])
Y_env = environment_df["target_high_risk"]

# Training split
X_train, X_test, y_train, y_test = train_test_split(
    X_env, Y_env, test_size=0.2, random_state=42, stratify=Y_env
)

# Pipeline for numeric + categorical features
numeric_features = [
    "investment", "employment", "risk_score",
    "is_women_led", "is_in_industrial_estate"
]
categorical_features = [
    "DISTRICT_NAME", "CATEGOERY_NAME", "industrial_park",
    "promoter_women", "social_status"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)

report = classification_report(y_test, preds, target_names=["Low Risk", "High Risk"], output_dict=True)
cm = confusion_matrix(y_test, preds)

# Save metrics
metrics = {
    "accuracy": float(report["accuracy"]),
    "macro_avg_f1": float(report["macro avg"]["f1-score"]),
    "high_risk_precision": float(report["High Risk"]["precision"]),
    "high_risk_recall": float(report["High Risk"]["recall"]),
    "confusion_matrix": cm.tolist()
}

with open(PHASE2_DIR / "environment_risk_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

with open(PHASE2_DIR / "environment_risk_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(
        y_test, preds, target_names=["Low Risk", "High Risk"]
    ))

# Save predictions for review
pred_df = X_test.copy()
pred_df["actual_high_risk"] = y_test.values
pred_df["predicted_high_risk"] = preds
pred_df.to_csv(PHASE2_DIR / "environment_risk_predictions.csv", index=False)

# Visualization: risk score by predicted class
fig, ax = plt.subplots(figsize=(10, 5))
if "risk_score" in pred_df.columns:
    sns.boxplot(data=pred_df, x="predicted_high_risk", y="risk_score", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Low Risk", "High Risk"])
    ax.set_title("Risk Score by Predicted Class")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Risk Score")
    save_plot("risk_score_by_predicted_class", fig)

# ---------------------------
# 3) Save a simple next-step checklist
# ---------------------------
checklist = {
    "next_steps": [
        "Train baseline clustering model using latitude, longitude, investment, and employment.",
        "Evaluate cluster quality and interpret high-density districts.",
        "Use logistic regression to predict high-risk compliance cases.",
        "Validate results on holdout data and refine features.",
        "Prepare model cards and deployment-ready APIs."
    ]
}

with open(PHASE2_DIR / "next_steps_checklist.json", "w", encoding="utf-8") as f:
    json.dump(checklist, f, indent=2)

print(f"Phase 2 modeling setup complete. Outputs saved in: {PHASE2_DIR}")
print(f"- Cluster summary: {PHASE2_DIR / 'cluster_summary.csv'}")
print(f"- Environment metrics: {PHASE2_DIR / 'environment_risk_metrics.json'}")
