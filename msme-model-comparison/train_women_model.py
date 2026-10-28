from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path.cwd()
DATA_FILE = BASE_DIR / "outputs" / "MSME_preprocessed_full.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase2_next_steps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_FILE)

    women_df = df[df["is_women_led"] == 1].copy()
    if women_df.empty:
        women_df = df[df["promoter_women"].astype(str).eq("Yes")].copy()

    women_df = women_df.dropna(subset=["investment", "employment", "turnover", "DISTRICT_NAME"])
    women_df["high_turnover"] = (women_df["turnover"] >= women_df["turnover"].median()).astype(int)

    X = women_df[
        ["investment", "employment", "risk_score", "DISTRICT_NAME", "CATEGOERY_NAME", "industrial_park"]
    ].copy()
    y = women_df["high_turnover"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ["investment", "employment", "risk_score"]
    categorical_features = ["DISTRICT_NAME", "CATEGOERY_NAME", "industrial_park"]

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
    with open(OUTPUT_DIR / "women_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"])
        }, f, indent=2)

    with open(OUTPUT_DIR / "women_model_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, preds))

    print(f"Women model complete. Accuracy={report['accuracy']:.4f} | Macro F1={report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
