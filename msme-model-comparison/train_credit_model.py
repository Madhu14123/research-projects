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
    credit_df = df[df["turnover_missing_flag"] == 0].copy()
    credit_df = credit_df.dropna(subset=["investment", "employment", "turnover", "DISTRICT_NAME"])

    credit_df["high_turnover"] = (credit_df["turnover"] >= credit_df["turnover"].median()).astype(int)

    X = credit_df[
        [
            "investment", "employment", "turnover", "risk_score",
            "DISTRICT_NAME", "CATEGOERY_NAME", "industrial_park"
        ]
    ].copy()
    y = credit_df["high_turnover"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ["investment", "employment", "turnover", "risk_score"]
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
    with open(OUTPUT_DIR / "credit_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"])
        }, f, indent=2)

    with open(OUTPUT_DIR / "credit_model_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, preds))

    print(f"Credit model complete. Accuracy={report['accuracy']:.4f} | Macro F1={report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
