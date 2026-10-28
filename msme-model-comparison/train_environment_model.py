from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path.cwd()
DATA_FILE = BASE_DIR / "outputs" / "MSME_preprocessed_full.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase2_next_steps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_FILE)
    env_df = df[
        [
            "PCBCATEGORY", "DISTRICT_NAME", "CATEGOERY_NAME", "investment",
            "employment", "industrial_park", "promoter_women", "social_status", "risk_score"
        ]
    ].copy()
    env_df = env_df.dropna(subset=["PCBCATEGORY", "DISTRICT_NAME", "CATEGOERY_NAME"])
    env_df["target_high_risk"] = env_df["PCBCATEGORY"].isin(["ORANGE", "RED"]).astype(int)

    X = env_df.drop(columns=["PCBCATEGORY", "target_high_risk"])
    y = env_df["target_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(
        y_test, preds, target_names=["Low Risk", "High Risk"], output_dict=True
    )
    cm = confusion_matrix(y_test, preds)

    metrics = {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "high_risk_precision": float(report["High Risk"]["precision"]),
        "high_risk_recall": float(report["High Risk"]["recall"]),
        "confusion_matrix": cm.tolist()
    }

    with open(OUTPUT_DIR / "environment_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(OUTPUT_DIR / "environment_model_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, preds, target_names=["Low Risk", "High Risk"]))

    print(
        f"Environment model complete. Accuracy={metrics['accuracy']:.4f} | "
        f"Macro F1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
