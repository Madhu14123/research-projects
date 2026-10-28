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
from sklearn.tree import DecisionTreeClassifier

# ---------------------------
# Configuration
# ---------------------------
BASE_DIR = Path.cwd()
DATA_FILE = BASE_DIR / "outputs" / "MSME_preprocessed_full.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase2_next_steps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(DATA_FILE)

# Standardize text fields for reporting consistency
for col in [
    "DISTRICT_NAME", "mandal_name", "CATEGOERY_NAME", "PCBCATEGORY",
    "industrial_park", "promoter_women", "social_status", "ODOP"
]:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

# ---------------------------
# Helper functions
# ---------------------------
def save_plot(name, fig=None):
    if fig is None:
        fig = plt.gcf()
    path = OUTPUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_json(name, payload):
    path = OUTPUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


# ---------------------------
# 1) Baseline clustering for location + scale typologies
# ---------------------------
cluster_df = df[
    ["latitude", "longitude", "investment", "employment", "CATEGOERY_NAME", "DISTRICT_NAME"]
].copy()
cluster_df = cluster_df.dropna(subset=["latitude", "longitude", "investment", "employment"])

num_cluster = ["latitude", "longitude", "investment", "employment"]
X_cluster = cluster_df[num_cluster]

kmeans = KMeans(n_clusters=7, random_state=42, n_init=20)
cluster_df["cluster"] = kmeans.fit_predict(X_cluster)

cluster_summary = (
    cluster_df.groupby("cluster")
    .agg(
        records=("cluster", "size"),
        avg_investment=("investment", "mean"),
        avg_employment=("employment", "mean"),
        avg_latitude=("latitude", "mean"),
        avg_longitude=("longitude", "mean")
    )
    .reset_index()
)
cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)

cluster_df[["latitude", "longitude", "investment", "employment", "cluster"]].to_csv(
    OUTPUT_DIR / "cluster_assignments.csv", index=False
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(
    data=cluster_df,
    x="longitude",
    y="latitude",
    hue="cluster",
    palette="viridis",
    s=40,
    alpha=0.6,
    ax=ax
)
ax.set_title("K-Means Clusters by Geography")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
save_plot("cluster_geography", fig)

# ---------------------------
# 2) Compliance / environmental-risk baseline model
# ---------------------------
environment_df = df[
    ["PCBCATEGORY", "DISTRICT_NAME", "CATEGOERY_NAME", "investment", "employment",
     "industrial_park", "promoter_women", "social_status", "risk_score"]
].copy()
environment_df = environment_df.dropna(subset=["PCBCATEGORY", "DISTRICT_NAME", "CATEGOERY_NAME"])

environment_df["target_high_risk"] = environment_df["PCBCATEGORY"].isin(["ORANGE", "RED"]).astype(int)

X_env = environment_df.drop(columns=["PCBCATEGORY", "target_high_risk"])
y_env = environment_df["target_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X_env, y_env, test_size=0.2, random_state=42, stratify=y_env
)

numeric_features = ["investment", "employment", "risk_score"]
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

env_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
env_model.fit(X_train, y_train)
pred_env = env_model.predict(X_test)

report = classification_report(
    y_test, pred_env, target_names=["Low Risk", "High Risk"], output_dict=True
)
cm = confusion_matrix(y_test, pred_env)

metrics = {
    "accuracy": float(report["accuracy"]),
    "macro_f1": float(report["macro avg"]["f1-score"]),
    "high_risk_precision": float(report["High Risk"]["precision"]),
    "high_risk_recall": float(report["High Risk"]["recall"]),
    "confusion_matrix": cm.tolist()
}
save_json("environment_model_metrics.json", metrics)

with open(OUTPUT_DIR / "environment_model_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(y_test, pred_env, target_names=["Low Risk", "High Risk"]))

# ---------------------------
# 3) Women-led MSME performance summary (baseline for women-focused model)
# ---------------------------
women_df = df[df["is_women_led"] == 1].copy()
if women_df.empty:
    women_df = df[df["promoter_women"].astype(str).eq("Yes")].copy()

women_summary = (
    women_df.groupby("DISTRICT_NAME")
    .agg(
        records=("UNIT_NAME", "count"),
        avg_investment=("investment", "mean"),
        avg_turnover=("turnover", "mean"),
        avg_risk=("risk_score", "mean")
    )
    .sort_values("records", ascending=False)
    .reset_index()
)
women_summary.to_csv(OUTPUT_DIR / "women_msmes_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=women_summary.head(10), x="DISTRICT_NAME", y="records", ax=ax)
ax.set_title("Top Districts for Women-led MSMEs")
ax.set_xlabel("District")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)
save_plot("women_districts", fig)

# ---------------------------
# 4) Credit-scoring preparation: subset with valid turnover and feature view
# ---------------------------
credit_df = df[df["turnover_missing_flag"] == 0].copy()
credit_df = credit_df.dropna(subset=["investment", "employment", "turnover"])

credit_features = [
    "investment", "employment", "turnover", "risk_score",
    "female_percentage", "ROI_ratio", "revenue_per_employee"
]

credit_summary = (
    credit_df.groupby("CATEGOERY_NAME")
    .agg(
        records=("UNIT_NAME", "count"),
        avg_turnover=("turnover", "mean"),
        avg_investment=("investment", "mean"),
        avg_roi=("ROI_ratio", "mean")
    )
    .sort_values("records", ascending=False)
    .reset_index()
)
credit_summary.to_csv(OUTPUT_DIR / "credit_model_feature_summary.csv", index=False)

# Optional baseline tree model for a simple segmentation view on credit-ready data
if len(credit_df) > 200:
    credit_df["high_turnover"] = (credit_df["turnover"] >= credit_df["turnover"].median()).astype(int)
    X_credit = credit_df[["investment", "employment", "risk_score", "female_percentage"]].fillna(0)
    y_credit = credit_df["high_turnover"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit
    )

    credit_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    credit_tree.fit(Xc_train, yc_train)
    credit_tree_report = classification_report(
        yc_test, credit_tree.predict(Xc_test), output_dict=True
    )

    save_json(
        "credit_tree_metrics.json",
        {
            "accuracy": float(credit_tree_report["accuracy"]),
            "macro_f1": float(credit_tree_report["macro avg"]["f1-score"])
        }
    )

# ---------------------------
# 5) Next-step checklist / action plan
# ---------------------------
plan = {
    "phase_2_priority": [
        "Cluster Analysis (location + product + scale)",
        "Environmental Risk Prediction (high-risk compliance cases)",
        "Women MSME Performance Analysis",
        "Credit Scoring Preparation (turnover-ready subset)"
    ],
    "recommended_inputs": {
        "cluster": ["latitude", "longitude", "investment", "employment"],
        "environment": ["PCBCATEGORY", "DISTRICT_NAME", "investment", "employment", "risk_score"],
        "women": ["is_women_led", "DISTRICT_NAME", "investment", "turnover"],
        "credit": ["investment", "employment", "turnover", "ROI_ratio", "revenue_per_employee"]
    },
    "next_actions": [
        "Tune K-Means cluster count and validate cluster stability.",
        "Use feature engineering to improve environmental risk prediction.",
        "Build a women-specific performance model on the women-led subset.",
        "Experiment with imputation methods for turnover-dependent models."
    ]
}
save_json("phase2_action_plan.json", plan)

print(f"Phase 2 next-step outputs saved to: {OUTPUT_DIR}")
