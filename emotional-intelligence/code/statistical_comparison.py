import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Load results
results_df = pd.read_csv('results/unsupervised_models_comparison.csv')

print("="*90)
print("STATISTICAL COMPARISON OF UNSUPERVISED CLUSTERING MODELS")
print("="*90)

print(f"\nTotal models evaluated: {len(results_df)}")
print(f"Feature counts tested: {sorted(results_df['k_features'].unique())}")
print(f"Model types: {len(results_df['model'].unique())} different configurations")

# Extract model type from model name
def extract_model_type(model_name):
    if model_name.startswith('KMeans'):
        return 'K-Means'
    elif model_name.startswith('HC'):
        return 'Hierarchical'
    elif model_name.startswith('DBSCAN'):
        return 'DBSCAN'
    elif model_name.startswith('Spectral'):
        return 'Spectral'
    else:
        return 'Other'

results_df['model_type'] = results_df['model'].apply(extract_model_type)

print("\n" + "="*90)
print("DESCRIPTIVE STATISTICS BY MODEL TYPE")
print("="*90)

for metric in ['silhouette', 'davies_bouldin', 'calinski']:
    print(f"\n{metric.upper()}:")
    print("-" * 90)

    for model_type in sorted(results_df['model_type'].unique()):
        subset = results_df[results_df['model_type'] == model_type][metric].dropna()
        print(f"\n{model_type:15} (n={len(subset):2})")
        print(f"  Mean:   {subset.mean():>8.4f}")
        print(f"  Median: {subset.median():>8.4f}")
        print(f"  Std:    {subset.std():>8.4f}")
        print(f"  Min:    {subset.min():>8.4f}")
        print(f"  Max:    {subset.max():>8.4f}")

# Statistical tests
print("\n" + "="*90)
print("STATISTICAL TESTS - COMPARING MODEL TYPES")
print("="*90)

for metric in ['silhouette', 'davies_bouldin', 'calinski']:
    print(f"\n{metric.upper()}:")
    print("-" * 90)

    # Prepare data by model type
    groups = []
    model_types = []
    for model_type in sorted(results_df['model_type'].unique()):
        subset = results_df[results_df['model_type'] == model_type][metric].dropna().values
        if len(subset) > 0:
            groups.append(subset)
            model_types.append(model_type)

    # Kruskal-Wallis test (non-parametric alternative to ANOVA)
    h_stat, p_value = kruskal(*groups)
    print(f"\nKruskal-Wallis Test:")
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (alpha=0.05)")

    # Pairwise Mann-Whitney U tests
    print(f"\nPairwise Comparisons (Mann-Whitney U):")
    print("-" * 90)

    for i in range(len(model_types)):
        for j in range(i+1, len(model_types)):
            u_stat, p_val = mannwhitneyu(groups[i], groups[j])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {model_types[i]:12} vs {model_types[j]:12}: U={u_stat:7.2f}, p={p_val:.6f} {sig}")

# Analysis by K features
print("\n" + "="*90)
print("IMPACT OF FEATURE COUNT (K) ON MODEL PERFORMANCE")
print("="*90)

for metric in ['silhouette', 'davies_bouldin', 'calinski']:
    print(f"\n{metric.upper()}:")
    print("-" * 90)

    k_groups = []
    k_values = []
    for k in sorted(results_df['k_features'].unique()):
        subset = results_df[results_df['k_features'] == k][metric].dropna().values
        if len(subset) > 0:
            k_groups.append(subset)
            k_values.append(k)
            print(f"  K={k:2}: Mean={np.mean(subset):>7.4f}, Median={np.median(subset):>7.4f}, Std={np.std(subset):>7.4f}, n={len(subset):2}")

    # Correlation between K and metric
    all_k = []
    all_metric = []
    for k in sorted(results_df['k_features'].unique()):
        subset_vals = results_df[results_df['k_features'] == k][metric].dropna().values
        all_k.extend([k] * len(subset_vals))
        all_metric.extend(subset_vals)

    if len(all_k) > 2:
        corr, p_val = stats.spearmanr(all_k, all_metric)
        print(f"\nSpearman Correlation (K vs {metric}): r={corr:.4f}, p={p_val:.6f}")

# Best models summary
print("\n" + "="*90)
print("BEST MODELS BY DIFFERENT CRITERIA")
print("="*90)

metrics_info = {
    'silhouette': ('Silhouette Score (higher is better)', True),
    'davies_bouldin': ('Davies-Bouldin Index (lower is better)', False),
    'calinski': ('Calinski-Harabasz Index (higher is better)', True)
}

for metric, (desc, higher_better) in metrics_info.items():
    print(f"\n{desc}")
    print("-" * 90)

    if higher_better:
        top_models = results_df.nlargest(5, metric)[['model', metric, 'k_features', 'model_type']]
    else:
        top_models = results_df.nsmallest(5, metric)[['model', metric, 'k_features', 'model_type']]

    for idx, (_, row) in enumerate(top_models.iterrows(), 1):
        print(f"  {idx}. {row['model']:<40} {metric}={row[metric]:>7.4f} (K={row['k_features']}, {row['model_type']})")

# Feature count analysis
print("\n" + "="*90)
print("FEATURE COUNT ANALYSIS - OPTIMAL K")
print("="*90)

for metric in ['silhouette', 'davies_bouldin', 'calinski']:
    higher_better = metric != 'davies_bouldin'
    if higher_better:
        best_k = results_df.loc[results_df[metric].idxmax(), 'k_features']
        best_val = results_df[metric].max()
    else:
        best_k = results_df.loc[results_df[metric].idxmin(), 'k_features']
        best_val = results_df[metric].min()

    print(f"\n{metric.upper()}:")
    print(f"  Optimal K: {int(best_k)} features")
    print(f"  Best score: {best_val:.4f}")

# Comprehensive ranking
print("\n" + "="*90)
print("COMPREHENSIVE MODEL RANKING (BY MULTIPLE METRICS)")
print("="*90)

# Normalize metrics for ranking
results_df['silhouette_norm'] = (results_df['silhouette'] - results_df['silhouette'].min()) / (results_df['silhouette'].max() - results_df['silhouette'].min())
results_df['davies_bouldin_norm'] = (results_df['davies_bouldin'].max() - results_df['davies_bouldin']) / (results_df['davies_bouldin'].max() - results_df['davies_bouldin'].min())
results_df['calinski_norm'] = (results_df['calinski'] - results_df['calinski'].min()) / (results_df['calinski'].max() - results_df['calinski'].min())

# Handle NaN values
results_df['silhouette_norm'] = results_df['silhouette_norm'].fillna(0)
results_df['davies_bouldin_norm'] = results_df['davies_bouldin_norm'].fillna(0)
results_df['calinski_norm'] = results_df['calinski_norm'].fillna(0)

# Combined score
results_df['combined_score'] = (results_df['silhouette_norm'] + results_df['davies_bouldin_norm'] + results_df['calinski_norm']) / 3

top_10_combined = results_df.nlargest(10, 'combined_score')[['model', 'silhouette', 'davies_bouldin', 'calinski', 'combined_score', 'k_features', 'model_type']]

print("\nTop 10 Models (Ranked by Combined Normalized Score):")
print("-" * 90)
for idx, (_, row) in enumerate(top_10_combined.iterrows(), 1):
    print(f"{idx:2}. {row['model']:<40} Score={row['combined_score']:>5.4f} K={int(row['k_features']):2}")
    print(f"    Silhouette={row['silhouette']:>7.4f}  DB={row['davies_bouldin']:>7.4f}  CH={row['calinski']:>8.2f}  Type={row['model_type']}")

# Save detailed comparison
results_df.to_csv('results/statistical_comparison_detailed.csv', index=False)
top_10_combined.to_csv('results/top_10_models.csv', index=False)

print("\n" + "="*90)
print("FILES SAVED")
print("="*90)
print("  - statistical_comparison_detailed.csv")
print("  - top_10_models.csv")
print("\nAnalysis complete!")
