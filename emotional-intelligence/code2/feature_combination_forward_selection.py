"""
feature_combination_forward_selection.py
========================================
Test if combined features from complementary pairs improve the EI model.

Strategy:
1. Create 5 combined features from top complementary pairs
2. Add to 13 base supervised features (18 total candidates)
3. Run forward selection (LOOCV Ridge α=1.0)
4. Compare with current best model (r=0.8949, 15 features)
5. Identify if combinations add value
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr, spearmanr

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "D:/A_emotions in exam"
RES  = os.path.join(BASE, "results")

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("FEATURE COMBINATION FORWARD SELECTION")
print("=" * 72)

# Load student profiles
profiles = pd.read_csv(os.path.join(RES, "student_feature_profiles_full.csv"))
ei_scores = pd.read_csv(os.path.join(RES, "ei_final_scores_v2.csv"))

n = len(profiles)
exam_pct = ei_scores["exam_pct"].values
y = exam_pct  # Target: actual exam percentage

print(f"\nLoaded {n} students")

# Extract z-scored features
z_cols = [c for c in profiles.columns if c.startswith("z_c")]
X_raw = profiles[z_cols].copy()

# Standardize
scaler = StandardScaler()
X_z = pd.DataFrame(scaler.fit_transform(X_raw), columns=z_cols)

# ═══════════════════════════════════════════════════════════════════════════
#  1. CREATE COMBINED FEATURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 1: CREATE COMBINED FEATURES FROM COMPLEMENTARY PAIRS")
print("=" * 72)

# Top 5 complementary pairs (from feature_combination_candidates.csv)
pairs = [
    {
        'name': 'comb_arousal_opt_mid',
        'f1': 'z_c4_arousal_optimality',
        'f2': 'z_c4_arousal_midrange_proportion',
        'method': 'average',
        'corr': 0.708,
        'votes': 6.5
    },
    {
        'name': 'comb_mastery',
        'f1': 'z_c5_mastery_during_challenge',
        'f2': 'z_c5_mastery_mode_ratio',
        'method': 'average',
        'corr': 0.774,
        'votes': 3.0
    },
    {
        'name': 'comb_inertia',
        'f1': 'z_c1_inertia_composite',
        'f2': 'z_c2_stress_inertia',
        'method': 'average',
        'corr': 0.757,
        'votes': 4.0
    },
    {
        'name': 'comb_regulation_arousal',
        'f1': 'z_c2_reappraisal_proxy_coactivation',
        'f2': 'z_c5_mastery_during_challenge',
        'method': 'product',
        'corr': 0.626,
        'votes': 3.0
    },
    {
        'name': 'comb_confusion_recovery',
        'f1': 'z_c3_productive_confusion_index',
        'f2': 'z_c5_engagement_recovery_speed',
        'method': 'product',
        'corr': 0.52,
        'votes': 5.0
    }
]

combined_features = {}
print(f"\nCreating {len(pairs)} combined features:")
print(f"  {'Name':<30} {'Method':<10} {'F1':<35} {'F2':<35} {'Corr':>7}")
print(f"  {'-'*120}")

for pair in pairs:
    f1 = pair['f1']
    f2 = pair['f2']

    if f1 not in X_z.columns or f2 not in X_z.columns:
        print(f"  WARNING: {f1} or {f2} not in features, skipping {pair['name']}")
        continue

    if pair['method'] == 'average':
        combined = (X_z[f1] + X_z[f2]) / 2
    elif pair['method'] == 'product':
        combined = X_z[f1] * X_z[f2]
    else:
        continue

    # Standardize combined feature
    combined = (combined - combined.mean()) / (combined.std() + 1e-10)
    combined_features[pair['name']] = combined

    # Compute correlation with target
    r_target = pearsonr(combined, y)[0]

    print(f"  {pair['name']:<30} {pair['method']:<10} {f1[-30:]:<35} {f2[-30:]:<35} {r_target:>7.3f}")

print(f"\n  Created {len(combined_features)} combined features successfully")

# ═══════════════════════════════════════════════════════════════════════════
#  2. SELECT 13 BASE SUPERVISED FEATURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 2: DEFINE BASE SUPERVISED FEATURE SET")
print("=" * 72)

# 13 top supervised features from consensus voting
base_features = [
    'z_c3_total_productive_time_ratio',
    'z_c4_block_stage_awareness',
    'z_c4_arousal_optimality',
    'z_c4_arousal_midrange_proportion',
    'z_c3_struggle_benefit_index',
    'z_c4_arousal_trend',
    'z_c5_flow_stability',
    'z_c3_task_completion_persistence',
    'z_c5_confusion_resolution_rate',
    'z_c3_stress_inertia_during_blocks',
    'z_c2_anticipatory_regulation',
    'z_c3_late_task_engagement_ratio',
    'z_c2_SAM_facial_discrepancy_reduction'
]

# Verify all exist
missing = [f for f in base_features if f not in X_z.columns]
if missing:
    print(f"  WARNING: Missing features {missing}")
    base_features = [f for f in base_features if f in X_z.columns]

print(f"\n  Base supervised features: {len(base_features)}")
for i, feat in enumerate(base_features, 1):
    r_target = pearsonr(X_z[feat], y)[0]
    print(f"    {i:2d}. {feat:<40} r={r_target:+.4f}")

# Create combined feature dataframe
X_base = X_z[base_features].copy()
X_combined_df = pd.DataFrame(combined_features)
print(f"\n  Combined features: {len(combined_features)}")
print(f"  Total candidate features: {len(base_features) + len(combined_features)}")

# ═══════════════════════════════════════════════════════════════════════════
#  3. FORWARD SELECTION WITH COMBINED FEATURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 3: FORWARD SELECTION (LOOCV Ridge α=1.0)")
print("=" * 72)

# Combine all candidates
X_candidates = pd.concat([X_base, X_combined_df], axis=1)
candidate_names = list(X_candidates.columns)
n_candidates = len(candidate_names)

print(f"\n  Forward selection on {n_candidates} features:")

# LOOCV setup
loo = LeaveOneOut()
alpha = 1.0

selected_features_list = []
selected_r_list = []
history = []

for step in range(min(15, n_candidates)):  # Max 15 steps like before
    best_feature = None
    best_r = -np.inf
    best_r_pearson = -np.inf
    best_r_spearman = -np.inf

    # Try each remaining feature
    for feat in candidate_names:
        if feat in [s[0] for s in selected_features_list]:
            continue

        # Build feature set: selected + candidate
        if len(selected_features_list) == 0:
            X_test = X_candidates[[feat]]
        else:
            selected = [s[0] for s in selected_features_list]
            X_test = X_candidates[selected + [feat]]

        # LOOCV evaluation
        pred = np.zeros(n)
        for train_idx, test_idx in loo.split(X_test):
            X_train, X_test_fold = X_test.iloc[train_idx], X_test.iloc[test_idx]
            y_train, y_test_fold = y[train_idx], y[test_idx]

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, y_train)
            pred[test_idx[0]] = ridge.predict(X_test_fold)[0]

        # Compute correlations
        r_pearson = pearsonr(pred, y)[0]
        r_spearman = spearmanr(pred, y)[0]

        if r_pearson > best_r_pearson:
            best_feature = feat
            best_r_pearson = r_pearson
            best_r_spearman = r_spearman
            best_r = r_pearson

    if best_feature is None:
        break

    selected_features_list.append((best_feature, best_r))
    selected_r_list.append(best_r)

    is_combined = "✓ COMBINED" if best_feature in combined_features.keys() else ""
    history.append({
        'step': step + 1,
        'feature': best_feature,
        'r_pearson': best_r_pearson,
        'r_spearman': best_r_spearman,
        'is_combined': "YES" if is_combined else "NO"
    })

    print(f"  Step {step+1:2d}: {best_feature:<40} r={best_r:.4f} (ρ={best_r_spearman:.4f}) {is_combined}")

print(f"\n  Final selection: {len(selected_features_list)} features")
final_r = selected_r_list[-1] if selected_r_list else 0
print(f"  Final LOOCV Pearson r: {final_r:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
#  4. ANALYZE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 4: RESULTS ANALYSIS")
print("=" * 72)

# Compare with baseline (current best)
baseline_r = 0.8949
improvement = final_r - baseline_r
improvement_pct = 100 * improvement / baseline_r

print(f"\n  Current best model:")
print(f"    Features: 15 (8 base + 7 derived interactions)")
print(f"    LOOCV r: {baseline_r:.4f}")
print(f"    R²: {baseline_r**2:.4f}")

print(f"\n  Model with combinations:")
print(f"    Features: {len(selected_features_list)}")
print(f"    LOOCV r: {final_r:.4f}")
print(f"    R²: {final_r**2:.4f}")

print(f"\n  Comparison:")
print(f"    Δr = {improvement:+.4f} ({improvement_pct:+.2f}%)")

if improvement > 0.005:
    print(f"    ✓ MEANINGFUL IMPROVEMENT (>0.5% gain)")
elif improvement > 0:
    print(f"    ~ Marginal improvement (<0.5% gain)")
else:
    print(f"    ✗ No improvement or worse")

# Count combined features in selection
combined_in_selection = sum(1 for feat, _ in selected_features_list
                            if feat in combined_features.keys())
print(f"\n  Combined features selected: {combined_in_selection}/{len(selected_features_list)}")

if combined_in_selection > 0:
    print(f"  Combined features in final model:")
    for feat, r in selected_features_list:
        if feat in combined_features.keys():
            idx = list(combined_features.keys()).index(feat)
            pair = pairs[idx]
            print(f"    • {feat} (from {pair['f1'][-25:]} × {pair['f2'][-25:]})")

# ═══════════════════════════════════════════════════════════════════════════
#  5. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 5: SAVE RESULTS")
print("=" * 72)

# Save selection history
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(RES, "combination_forward_selection_history.csv"), index=False)
print(f"\n  Saved: combination_forward_selection_history.csv")

# Save final selection
selection_df = pd.DataFrame({
    'rank': range(1, len(selected_features_list) + 1),
    'feature': [f[0] for f in selected_features_list],
    'r_pearson': [f[1] for f in selected_features_list],
    'is_combined': ["YES" if f[0] in combined_features else "NO" for f in selected_features_list],
    'type': ['combined' if f[0] in combined_features else 'base' for f in selected_features_list]
})
selection_df.to_csv(os.path.join(RES, "combination_final_selection.csv"), index=False)
print(f"  Saved: combination_final_selection.csv ({len(selection_df)} features)")

# ═══════════════════════════════════════════════════════════════════════════
#  6. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: R progression
steps = range(1, len(selected_r_list) + 1)
colors = ['#e74c3c' if history_df.iloc[i-1]['is_combined'] == 'YES' else '#3498db'
          for i in steps]

ax1.plot(steps, selected_r_list, 'o-', color='#2c3e50', linewidth=2, markersize=8, label='Model r')
ax1.scatter(steps, selected_r_list, c=colors, s=100, alpha=0.7, edgecolors='k', linewidth=1)
ax1.axhline(baseline_r, color='green', linestyle='--', linewidth=2, label=f'Baseline (r={baseline_r:.4f})')
ax1.set_xlabel('Forward Selection Step', fontweight='bold', fontsize=11)
ax1.set_ylabel('LOOCV Pearson r', fontweight='bold', fontsize=11)
ax1.set_title('Forward Selection with Combined Features', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_ylim([0.7, 0.95])

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='k', label='Base feature'),
    Patch(facecolor='#e74c3c', edgecolor='k', label='Combined feature')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Plot 2: Feature type composition
base_count = sum(1 for _, t in zip(steps, history_df['is_combined']) if t == 'NO')
combined_count = sum(1 for _, t in zip(steps, history_df['is_combined']) if t == 'YES')

ax2.bar(['Base\nFeatures', 'Combined\nFeatures'], [base_count, combined_count],
        color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='k', linewidth=2)
ax2.set_ylabel('Count in Final Model', fontweight='bold', fontsize=11)
ax2.set_title(f'Feature Composition (n={len(selected_features_list)})', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (label, count) in enumerate(zip(['Base', 'Combined'], [base_count, combined_count])):
    ax2.text(i, count + 0.1, str(count), ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
fig.savefig(os.path.join(RES, "combination_forward_selection_results.png"), dpi=150, bbox_inches='tight')
print("  Saved: combination_forward_selection_results.png")

# ═══════════════════════════════════════════════════════════════════════════
#  7. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"\n  Baseline Model (current):")
print(f"    • 15 features (8 base + 7 derived)")
print(f"    • LOOCV r = 0.8949")
print(f"    • R² = 0.8001")

print(f"\n  Test Model (with combinations):")
print(f"    • {len(selected_features_list)} features ({base_count} base + {combined_count} combined)")
print(f"    • LOOCV r = {final_r:.4f}")
print(f"    • R² = {final_r**2:.4f}")

print(f"\n  Conclusion:")
if improvement > 0.005:
    print(f"    ✓ Combined features IMPROVE model by {improvement_pct:.2f}%")
    print(f"    → Recommend adding top {combined_count} combined features to final model")
elif improvement > 0:
    print(f"    ~ Combined features show marginal improvement ({improvement_pct:.2f}%)")
    print(f"    → Keep current model; combinations don't justify added complexity")
else:
    print(f"    ✗ Combined features DECREASE performance")
    print(f"    → Stick with current baseline model (r=0.8949)")

print("\n" + "=" * 72)
print("✓ COMBINATION FEATURE SELECTION COMPLETE")
print("=" * 72)
