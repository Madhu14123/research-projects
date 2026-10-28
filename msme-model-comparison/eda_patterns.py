from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Configuration
# ---------------------------
BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda_patterns"
EDA_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = OUTPUT_DIR / "MSME_preprocessed_full.csv"
SUMMARY_FILE = EDA_DIR / "eda_summary.txt"

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(DATA_FILE)

# Make sure the key text fields are treated consistently
for col in ["DISTRICT_NAME", "mandal_name", "CATEGOERY_NAME", "PCBCATEGORY", "industrial_park", "promoter_women", "social_status", "ODOP"]:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

numeric_cols = [
    "investment", "employment", "turnover", "no_of_males", "no_of_females",
    "ROI_ratio", "revenue_per_employee", "employees_per_lakh_investment",
    "female_percentage", "risk_score"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------
# Helper functions
# ---------------------------
def save_plot(name):
    path = EDA_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def describe_numeric(series, label):
    return (
        f"{label}:\n"
        f"  count={series.count():,} | mean={series.mean():,.2f} | median={series.median():,.2f} | "
        f"std={series.std():,.2f} | min={series.min():,.2f} | max={series.max():,.2f}\n"
    )

# ---------------------------
# Pattern summary
# ---------------------------
summary_lines = []
summary_lines.append("EDA PATTERN SUMMARY")
summary_lines.append("=" * 80)
summary_lines.append(f"Rows: {len(df):,}")
summary_lines.append(f"Columns: {len(df.columns)}")
summary_lines.append("")

# Top districts
if "DISTRICT_NAME" in df.columns:
    district_counts = df["DISTRICT_NAME"].value_counts().head(10)
    summary_lines.append("TOP 10 DISTRICTS BY COUNT")
    summary_lines.append(district_counts.to_string())
    summary_lines.append("")

# Category distribution
if "CATEGOERY_NAME" in df.columns:
    category_counts = df["CATEGOERY_NAME"].value_counts(dropna=False)
    summary_lines.append("CATEGORY DISTRIBUTION")
    summary_lines.append(category_counts.to_string())
    summary_lines.append("")

# Industrial park distribution
if "industrial_park" in df.columns:
    industrial_counts = df["industrial_park"].value_counts(dropna=False)
    summary_lines.append("INDUSTRIAL PARK DISTRIBUTION")
    summary_lines.append(industrial_counts.to_string())
    summary_lines.append("")

# Women-led / ODOP / social status
for col in ["promoter_women", "ODOP", "social_status"]:
    if col in df.columns:
        summary_lines.append(f"{col.upper()} DISTRIBUTION")
        summary_lines.append(df[col].value_counts(dropna=False).to_string())
        summary_lines.append("")

# Numeric summaries
for col in numeric_cols:
    if col in df.columns:
        summary_lines.append(describe_numeric(df[col].dropna(), col))

# Average turnover by category
if {"CATEGOERY_NAME", "turnover"}.issubset(df.columns):
    avg_turnover_by_category = df.groupby("CATEGOERY_NAME")["turnover"].mean().sort_values(ascending=False)
    summary_lines.append("AVERAGE TURNOVER BY CATEGORY")
    summary_lines.append(avg_turnover_by_category.round(2).to_string())
    summary_lines.append("")

# High-risk units
if {"UNIT_NAME", "risk_score", "turnover", "investment"}.issubset(df.columns):
    high_risk = df.sort_values("risk_score", ascending=False).head(10)
    summary_lines.append("TOP 10 HIGHEST RISK UNITS")
    summary_lines.append(
        high_risk[["UNIT_NAME", "DISTRICT_NAME", "CATEGOERY_NAME", "risk_score", "investment", "turnover"]]
        .round({"risk_score": 3, "investment": 2, "turnover": 2})
        .to_string(index=False)
    )
    summary_lines.append("")

# Write summary report
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

# ---------------------------
# Plot 1: Distribution of key financial metrics
# ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axes, ["investment", "employment", "turnover"]):
    if col in df.columns:
        sns.histplot(df[col].dropna(), bins=40, ax=ax, kde=True)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
save_plot("financial_distributions")

# ---------------------------
# Plot 2: Top districts counts
# ---------------------------
if "DISTRICT_NAME" in df.columns:
    top_districts = df["DISTRICT_NAME"].value_counts().head(12)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_districts.values, y=top_districts.index)
    plt.title("Top Districts by Count")
    plt.xlabel("Number of Units")
    plt.ylabel("District")
    save_plot("top_districts")

# ---------------------------
# Plot 3: Category counts
# ---------------------------
if "CATEGOERY_NAME" in df.columns:
    category_counts = df["CATEGOERY_NAME"].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title("Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    save_plot("category_distribution")

# ---------------------------
# Plot 4: Turnover by category
# ---------------------------
if {"CATEGOERY_NAME", "turnover"}.issubset(df.columns):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="CATEGOERY_NAME", y="turnover")
    plt.title("Turnover by Category")
    plt.xlabel("Category")
    plt.ylabel("Turnover")
    plt.xticks(rotation=45)
    save_plot("turnover_by_category")

# ---------------------------
# Plot 5: Turnover vs investment (log scale)
# ---------------------------
if {"investment", "turnover", "CATEGOERY_NAME"}.issubset(df.columns):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="investment",
        y="turnover",
        hue="CATEGOERY_NAME",
        s=40,
        alpha=0.6
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Turnover vs Investment")
    plt.xlabel("Investment")
    plt.ylabel("Turnover")
    save_plot("turnover_vs_investment")

# ---------------------------
# Plot 6: Correlation heatmap
# ---------------------------
correlation_cols = [col for col in ["investment", "employment", "turnover", "risk_score", "female_percentage", "ROI_ratio", "revenue_per_employee"] if col in df.columns]
if len(correlation_cols) >= 2:
    corr = df[correlation_cols].corr().round(2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap")

# ---------------------------
# Plot 7: Risk score distribution
# ---------------------------
if "risk_score" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df["risk_score"].dropna(), bins=25, kde=True)
    plt.title("Risk Score Distribution")
    plt.xlabel("Risk Score")
    plt.ylabel("Count")
    save_plot("risk_score_distribution")

# ---------------------------
# Plot 8: Female percentage by category
# ---------------------------
if {"CATEGOERY_NAME", "female_percentage"}.issubset(df.columns):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="CATEGOERY_NAME", y="female_percentage")
    plt.title("Female Percentage by Category")
    plt.xlabel("Category")
    plt.ylabel("Female Percentage")
    plt.xticks(rotation=45)
    save_plot("female_percentage_by_category")

print(f"EDA analysis complete. Results saved in: {EDA_DIR}")
print(f"Summary report: {SUMMARY_FILE}")
