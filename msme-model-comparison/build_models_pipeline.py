from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path.cwd()
DATA_FILE = BASE_DIR / "outputs" / "MSME_preprocessed_full.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "build_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(name, payload):
    path = OUTPUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def train_logistic_model(X_train, X_test, y_train, y_test, model_name):
    numeric_features = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

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

    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    metrics = {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "per_class": report
    }
    save_json(f"{model_name}_metrics.json", metrics)
    with open(OUTPUT_DIR / f"{model_name}_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, preds))
    return model


def train_tree_model(X_train, X_test, y_train, y_test, model_name):
    numeric_features = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

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
        ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42))
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    metrics = {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }
    save_json(f"{model_name}_metrics.json", metrics)
    with open(OUTPUT_DIR / f"{model_name}_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, preds))
    return model


def main():
    df = pd.read_csv(DATA_FILE)

    # ----- Cluster model -----
    cluster_df = df[[
        "latitude", "longitude", "investment", "employment",
        "DISTRICT_NAME", "CATEGOERY_NAME"
    ]].copy()
    cluster_df = cluster_df.dropna(subset=["latitude", "longitude", "investment", "employment"])
    X_cluster = cluster_df[["latitude", "longitude", "investment", "employment"]].astype(float)

    best = None
    for k in range(3, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_cluster)
        sample_idx = np.random.RandomState(42).choice(len(X_cluster), size=min(3000, len(X_cluster)), replace=False)
        score = silhouette_score(X_cluster.values[sample_idx], labels[sample_idx])
        if best is None or score > best["score"]:
            best = {"k": k, "score": float(score), "labels": labels.tolist()}

    cluster_df["cluster"] = best["labels"]
    cluster_df.to_csv(OUTPUT_DIR / "cluster_results.csv", index=False)
    save_json(
        "cluster_metrics.json",
        {"best_k": best["k"], "best_silhouette_score": best["score"]}
    )

    # ----- Environmental risk model -----
    env_df = df[
        [
            "PCBCATEGORY", "DISTRICT_NAME", "CATEGOERY_NAME", "investment",
            "employment", "industrial_park", "promoter_women", "social_status", "risk_score"
        ]
    ].copy()
    env_df = env_df.dropna(subset=["PCBCATEGORY", "DISTRICT_NAME", "CATEGOERY_NAME"])
    env_df["target_high_risk"] = env_df["PCBCATEGORY"].isin(["ORANGE", "RED"]).astype(int)

    X_env = env_df.drop(columns=["PCBCATEGORY", "target_high_risk"])
    y_env = env_df["target_high_risk"]
    X_train_env, X_test_env, y_train_env, y_test_env = train_test_split(
        X_env, y_env, test_size=0.2, random_state=42, stratify=y_env
    )
    train_logistic_model(
        X_train_env, X_test_env, y_train_env, y_test_env, "environment"
    )

    # ----- Women-led model -----
    women_df = df[df["is_women_led"] == 1].copy()
    if women_df.empty:
        women_df = df[df["promoter_women"].astype(str).eq("Yes")].copy()
    women_df = women_df.dropna(subset=["investment", "employment", "turnover", "DISTRICT_NAME"])
    women_df["high_turnover"] = (women_df["turnover"] >= women_df["turnover"].median()).astype(int)
    X_women = women_df[[
        "investment", "employment", "risk_score", "DISTRICT_NAME",
        "CATEGOERY_NAME", "industrial_park"
    ]].copy()
    y_women = women_df["high_turnover"]
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X_women, y_women, test_size=0.2, random_state=42, stratify=y_women
    )
    train_tree_model(X_train_w, X_test_w, y_train_w, y_test_w, "women")

    # ----- Credit model -----
    credit_df = df[df["turnover_missing_flag"] == 0].copy()
    credit_df = credit_df.dropna(subset=["investment", "employment", "turnover", "DISTRICT_NAME"])
    credit_df["high_turnover"] = (credit_df["turnover"] >= credit_df["turnover"].median()).astype(int)
    X_credit = credit_df[[
        "investment", "employment", "turnover", "risk_score",
        "DISTRICT_NAME", "CATEGOERY_NAME", "industrial_park"
    ]].copy()
    y_credit = credit_df["high_turnover"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit
    )
    train_tree_model(X_train_c, X_test_c, y_train_c, y_test_c, "credit")

    print(f"Build pipeline completed. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
