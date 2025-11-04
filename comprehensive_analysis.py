"""
Comprehensive Psycholinguistic Network Analysis
================================================

Phase 1: Exploratory Visualizations
Phase 2: Statistical Testing Framework

This script performs comprehensive analysis of semantic network data
from TD and SLI children for academic presentation.

Author: Data Analysis Pipeline
Date: 2025-11-04
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene, chi2_contingency
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")
sns.set_context("paper", font_scale=1.2)

# Create output directories
OUTPUT_DIR = Path("analysis_outputs")
FIG_DIR = OUTPUT_DIR / "figures"
STAT_DIR = OUTPUT_DIR / "statistics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPREHENSIVE PSYCHOLINGUISTIC NETWORK ANALYSIS")
print("="*80)
print()

# ============================================================================
# DATA LOADING
# ============================================================================

print("Loading data...")
results_dir = Path("network_analysis/results")

# Load all datasets
group_summary = pd.read_csv(results_dir / "group_summary_statistics.csv")
statistical_comparisons = pd.read_csv(results_dir / "statistical_comparisons.csv")
mlu_matched = pd.read_csv(results_dir / "mlu_matched_comparison.csv")
individual_networks = pd.read_csv(results_dir / "individual_networks_threshold_0.7.csv")

# Load metadata
metadata_path = Path("extracted_data/embedding_ready/metadata.csv")
if metadata_path.exists():
    metadata = pd.read_csv(metadata_path)
    print(f"  - Loaded metadata for {len(metadata)} children")
else:
    metadata = None
    print("  - Metadata not found, using individual networks data")

print(f"  - Group summary: {len(group_summary)} rows")
print(f"  - Statistical comparisons: {len(statistical_comparisons)} tests")
print(f"  - MLU-matched pairs: {mlu_matched['n_pairs'].iloc[0] if len(mlu_matched) > 0 else 'N/A'}")
print(f"  - Individual networks: {len(individual_networks)} children")
print()

# ============================================================================
# PHASE 1: EXPLORATORY VISUALIZATIONS
# ============================================================================

print("="*80)
print("PHASE 1: EXPLORATORY VISUALIZATIONS")
print("="*80)
print()

# ----------------------------------------------------------------------------
# 1.1 Network Topology Overview: Violin Plots
# ----------------------------------------------------------------------------

print("Creating Figure 1: Network topology comparison (violin plots)...")

# Key metrics to visualize
key_metrics = [
    'avg_clustering',
    'avg_path_length',
    'density',
    'avg_degree_centrality',
    'transitivity'
]

metric_labels = {
    'avg_clustering': 'Clustering Coefficient',
    'avg_path_length': 'Average Path Length',
    'density': 'Network Density',
    'avg_degree_centrality': 'Degree Centrality',
    'transitivity': 'Transitivity'
}

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Prepare data for plotting
plot_data = []
for _, row in individual_networks.iterrows():
    for metric in key_metrics:
        if metric in row:
            plot_data.append({
                'metric': metric_labels.get(metric, metric),
                'value': row[metric],
                'group': row['development_type'],
                'age': row['age_years']
            })

plot_df = pd.DataFrame(plot_data)

# Create violin plots for each metric
for idx, metric in enumerate(key_metrics):
    metric_data = plot_df[plot_df['metric'] == metric_labels.get(metric, metric)]

    # Violin plot
    sns.violinplot(
        data=metric_data,
        x='group',
        y='value',
        ax=axes[idx],
        inner='box',
        palette={'TD': '#3498db', 'SLI': '#e74c3c'}
    )

    # Add individual points
    sns.stripplot(
        data=metric_data,
        x='group',
        y='value',
        ax=axes[idx],
        alpha=0.3,
        size=2,
        color='black'
    )

    axes[idx].set_title(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(axis='y', alpha=0.3)

    # Add statistical annotation
    td_vals = metric_data[metric_data['group'] == 'TD']['value'].values
    sli_vals = metric_data[metric_data['group'] == 'SLI']['value'].values

    t_stat, p_val = ttest_ind(td_vals, sli_vals)
    cohen_d = (np.mean(td_vals) - np.mean(sli_vals)) / np.sqrt((np.std(td_vals)**2 + np.std(sli_vals)**2) / 2)

    sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    axes[idx].text(0.5, 0.95, f'p={p_val:.4f} {sig_marker}\nd={cohen_d:.2f}',
                   transform=axes[idx].transAxes,
                   ha='center', va='top',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Remove extra subplot
axes[-1].axis('off')

plt.suptitle('Network Topology Comparison: TD vs SLI',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_topology_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig1_topology_comparison.pdf", bbox_inches='tight')
print(f"  ✓ Saved to {FIG_DIR}/fig1_topology_comparison.png")
plt.close()

# ----------------------------------------------------------------------------
# 1.2 Age Progression Plots
# ----------------------------------------------------------------------------

print("Creating Figure 2: Age progression plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, metric in enumerate(key_metrics):
    metric_col = metric + '_mean'

    if metric_col in group_summary.columns:
        # Separate TD and SLI data
        td_data = group_summary[group_summary['development_type'] == 'TD']
        sli_data = group_summary[group_summary['development_type'] == 'SLI']

        # Plot with error bars
        axes[idx].errorbar(
            td_data['age_years'],
            td_data[metric_col],
            yerr=td_data[metric + '_std'],
            marker='o',
            markersize=8,
            linewidth=2,
            capsize=5,
            label='TD',
            color='#3498db'
        )

        axes[idx].errorbar(
            sli_data['age_years'],
            sli_data[metric_col],
            yerr=sli_data[metric + '_std'],
            marker='s',
            markersize=8,
            linewidth=2,
            capsize=5,
            label='SLI',
            color='#e74c3c'
        )

        # Add polynomial fit lines
        for group_data, color, label in [(td_data, '#3498db', 'TD'),
                                          (sli_data, '#e74c3c', 'SLI')]:
            if len(group_data) > 2:
                z = np.polyfit(group_data['age_years'], group_data[metric_col], 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(group_data['age_years'].min(),
                                      group_data['age_years'].max(), 100)
                axes[idx].plot(x_smooth, p(x_smooth), '--',
                             color=color, alpha=0.3, linewidth=1)

        axes[idx].set_title(metric_labels.get(metric, metric),
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Age (years)')
        axes[idx].set_ylabel('Mean Value')
        axes[idx].legend(loc='best')
        axes[idx].grid(alpha=0.3)

# Remove extra subplot
axes[-1].axis('off')

plt.suptitle('Developmental Trajectories: Network Metrics by Age',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_age_progression.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig2_age_progression.pdf", bbox_inches='tight')
print(f"  ✓ Saved to {FIG_DIR}/fig2_age_progression.png")
plt.close()

# ----------------------------------------------------------------------------
# 1.3 Distribution Analysis
# ----------------------------------------------------------------------------

print("Creating Figure 3: Distribution histograms...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, metric in enumerate(key_metrics):
    metric_data = plot_df[plot_df['metric'] == metric_labels.get(metric, metric)]

    # Separate histograms
    td_vals = metric_data[metric_data['group'] == 'TD']['value']
    sli_vals = metric_data[metric_data['group'] == 'SLI']['value']

    axes[idx].hist(td_vals, bins=30, alpha=0.6, label='TD',
                   color='#3498db', density=True, edgecolor='black')
    axes[idx].hist(sli_vals, bins=30, alpha=0.6, label='SLI',
                   color='#e74c3c', density=True, edgecolor='black')

    # Add KDE
    from scipy.stats import gaussian_kde
    if len(td_vals) > 1:
        kde_td = gaussian_kde(td_vals)
        x_range = np.linspace(td_vals.min(), td_vals.max(), 100)
        axes[idx].plot(x_range, kde_td(x_range),
                      color='#3498db', linewidth=2, linestyle='--')

    if len(sli_vals) > 1:
        kde_sli = gaussian_kde(sli_vals)
        x_range = np.linspace(sli_vals.min(), sli_vals.max(), 100)
        axes[idx].plot(x_range, kde_sli(x_range),
                      color='#e74c3c', linewidth=2, linestyle='--')

    axes[idx].set_title(metric_labels.get(metric, metric),
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Density')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

# Remove extra subplot
axes[-1].axis('off')

plt.suptitle('Distribution Analysis: TD vs SLI',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_distributions.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig3_distributions.pdf", bbox_inches='tight')
print(f"  ✓ Saved to {FIG_DIR}/fig3_distributions.png")
plt.close()

# ----------------------------------------------------------------------------
# 1.4 Effect Size Visualization (Forest Plot)
# ----------------------------------------------------------------------------

print("Creating Figure 4: Effect sizes (forest plot)...")

# Extract effect sizes from statistical comparisons
overall_stats = statistical_comparisons[statistical_comparisons['age_group'] == 'Overall']

fig, ax = plt.subplots(figsize=(10, 8))

metrics_for_plot = []
cohens_d = []
ci_lower = []
ci_upper = []

for _, row in overall_stats.iterrows():
    metric = row['metric']
    if metric in key_metrics:
        metrics_for_plot.append(metric_labels.get(metric, metric))
        cohens_d.append(row['cohen_d'])

        # Calculate 95% CI for Cohen's d (approximate)
        n_td = row['n_td']
        n_sli = row['n_sli']
        se = np.sqrt((n_td + n_sli) / (n_td * n_sli) + row['cohen_d']**2 / (2 * (n_td + n_sli)))
        ci_lower.append(row['cohen_d'] - 1.96 * se)
        ci_upper.append(row['cohen_d'] + 1.96 * se)

y_pos = np.arange(len(metrics_for_plot))

# Plot points and error bars
ax.errorbar(cohens_d, y_pos, xerr=[np.array(cohens_d) - np.array(ci_lower),
                                     np.array(ci_upper) - np.array(cohens_d)],
            fmt='o', markersize=10, linewidth=2, capsize=5, color='#2c3e50')

# Add vertical line at 0
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Color code by effect size magnitude
for i, d in enumerate(cohens_d):
    if abs(d) > 0.8:
        color = '#e74c3c'  # Large effect
    elif abs(d) > 0.5:
        color = '#f39c12'  # Medium effect
    else:
        color = '#95a5a6'  # Small effect
    ax.plot(d, i, 'o', markersize=12, color=color)

# Shading for effect size magnitude
ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Small effect')
ax.axvspan(-0.5, -0.2, alpha=0.1, color='orange')
ax.axvspan(0.2, 0.5, alpha=0.1, color='orange', label='Medium effect')
ax.axvspan(-10, -0.5, alpha=0.1, color='red')
ax.axvspan(0.5, 10, alpha=0.1, color='red', label='Large effect')

ax.set_yticks(y_pos)
ax.set_yticklabels(metrics_for_plot)
ax.set_xlabel("Cohen's d (TD - SLI)", fontsize=12, fontweight='bold')
ax.set_ylabel('')
ax.set_title("Effect Sizes: TD vs SLI Network Metrics",
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_effect_sizes.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig4_effect_sizes.pdf", bbox_inches='tight')
print(f"  ✓ Saved to {FIG_DIR}/fig4_effect_sizes.png")
plt.close()

# ----------------------------------------------------------------------------
# 1.5 Correlation Matrix
# ----------------------------------------------------------------------------

print("Creating Figure 5: Correlation matrix...")

# Compute correlation matrix for TD and SLI separately
td_networks = individual_networks[individual_networks['development_type'] == 'TD']
sli_networks = individual_networks[individual_networks['development_type'] == 'SLI']

# Select numeric columns
numeric_cols = ['avg_clustering', 'avg_path_length', 'density',
                'avg_degree_centrality', 'transitivity']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# TD correlation matrix
td_corr = td_networks[numeric_cols].corr()
sns.heatmap(td_corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, ax=ax1,
            xticklabels=[metric_labels.get(c, c) for c in numeric_cols],
            yticklabels=[metric_labels.get(c, c) for c in numeric_cols],
            cbar_kws={'label': 'Correlation'})
ax1.set_title('TD Correlation Matrix', fontsize=14, fontweight='bold')

# SLI correlation matrix
sli_corr = sli_networks[numeric_cols].corr()
sns.heatmap(sli_corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, ax=ax2,
            xticklabels=[metric_labels.get(c, c) for c in numeric_cols],
            yticklabels=[metric_labels.get(c, c) for c in numeric_cols],
            cbar_kws={'label': 'Correlation'})
ax2.set_title('SLI Correlation Matrix', fontsize=14, fontweight='bold')

plt.suptitle('Network Metric Correlations', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_correlation_matrices.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig5_correlation_matrices.pdf", bbox_inches='tight')
print(f"  ✓ Saved to {FIG_DIR}/fig5_correlation_matrices.png")
plt.close()

# ----------------------------------------------------------------------------
# 1.6 MLU-Matched Comparison Visualization
# ----------------------------------------------------------------------------

print("Creating Figure 6: MLU-matched comparison...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Select top 3 metrics for visualization
mlu_metrics = ['avg_clustering', 'avg_path_length', 'density']

for idx, metric in enumerate(mlu_metrics):
    if metric in mlu_matched['metric'].values:
        row = mlu_matched[mlu_matched['metric'] == metric].iloc[0]

        # Create paired data visualization
        td_mean = row['td_mean']
        sli_mean = row['sli_mean']

        # Bar plot with error bars
        x = [0, 1]
        means = [td_mean, sli_mean]

        bars = axes[idx].bar(x, means,
                            color=['#3498db', '#e74c3c'],
                            alpha=0.7, edgecolor='black', linewidth=2)

        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(['TD\n(MLU-matched)', 'SLI'])
        axes[idx].set_ylabel('Mean Value')
        axes[idx].set_title(metric_labels.get(metric, metric),
                           fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)

        # Add significance annotation
        p_val = row['p_value']
        cohen_d = row['cohen_d']
        sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

        y_max = max(means) * 1.1
        axes[idx].plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
        axes[idx].text(0.5, y_max * 1.05,
                      f'{sig_marker}\np={p_val:.4f}\nd={cohen_d:.2f}',
                      ha='center', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('MLU-Matched Comparison: Network Differences Persist',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_mlu_matched.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig6_mlu_matched.pdf", bbox_inches='tight')
print(f"  ✓ Saved to {FIG_DIR}/fig6_mlu_matched.png")
plt.close()

print()
print("✓ Phase 1 visualizations complete!")
print(f"  All figures saved to: {FIG_DIR}")
print()

# ============================================================================
# PHASE 2: STATISTICAL TESTING FRAMEWORK
# ============================================================================

print("="*80)
print("PHASE 2: STATISTICAL TESTING FRAMEWORK")
print("="*80)
print()

# Container for all statistical results
stat_results = {}

# ----------------------------------------------------------------------------
# 2.1 Normality and Assumption Testing
# ----------------------------------------------------------------------------

print("2.1 Testing statistical assumptions...")

normality_results = []

for metric in key_metrics:
    metric_data = plot_df[plot_df['metric'] == metric_labels.get(metric, metric)]

    td_vals = metric_data[metric_data['group'] == 'TD']['value'].values
    sli_vals = metric_data[metric_data['group'] == 'SLI']['value'].values

    # Shapiro-Wilk test for normality
    if len(td_vals) > 3:
        stat_td, p_td = shapiro(td_vals)
    else:
        stat_td, p_td = np.nan, np.nan

    if len(sli_vals) > 3:
        stat_sli, p_sli = shapiro(sli_vals)
    else:
        stat_sli, p_sli = np.nan, np.nan

    # Levene's test for homogeneity of variance
    stat_levene, p_levene = levene(td_vals, sli_vals)

    normality_results.append({
        'metric': metric_labels.get(metric, metric),
        'shapiro_td_stat': stat_td,
        'shapiro_td_p': p_td,
        'shapiro_sli_stat': stat_sli,
        'shapiro_sli_p': p_sli,
        'td_normal': p_td > 0.05 if not np.isnan(p_td) else 'N/A',
        'sli_normal': p_sli > 0.05 if not np.isnan(p_sli) else 'N/A',
        'levene_stat': stat_levene,
        'levene_p': p_levene,
        'equal_variance': p_levene > 0.05
    })

normality_df = pd.DataFrame(normality_results)
normality_df.to_csv(STAT_DIR / "assumption_tests.csv", index=False)
print(f"  ✓ Normality tests saved to {STAT_DIR}/assumption_tests.csv")

# Print summary
print("\n  Normality Test Summary:")
for _, row in normality_df.iterrows():
    print(f"    {row['metric']: <25} TD: {row['td_normal']}, "
          f"SLI: {row['sli_normal']}, Equal Var: {row['equal_variance']}")

# ----------------------------------------------------------------------------
# 2.2 Primary Group Comparisons (Parametric and Non-Parametric)
# ----------------------------------------------------------------------------

print("\n2.2 Performing group comparisons...")

comparison_results = []

for metric in key_metrics:
    metric_data = plot_df[plot_df['metric'] == metric_labels.get(metric, metric)]

    td_vals = metric_data[metric_data['group'] == 'TD']['value'].values
    sli_vals = metric_data[metric_data['group'] == 'SLI']['value'].values

    # Parametric: Independent t-test
    t_stat, t_p = ttest_ind(td_vals, sli_vals)

    # Non-parametric: Mann-Whitney U test
    u_stat, u_p = mannwhitneyu(td_vals, sli_vals, alternative='two-sided')

    # Effect size: Cohen's d
    pooled_std = np.sqrt((np.std(td_vals, ddof=1)**2 + np.std(sli_vals, ddof=1)**2) / 2)
    cohen_d = (np.mean(td_vals) - np.mean(sli_vals)) / pooled_std

    # Descriptive statistics
    comparison_results.append({
        'metric': metric_labels.get(metric, metric),
        'td_mean': np.mean(td_vals),
        'td_std': np.std(td_vals, ddof=1),
        'td_median': np.median(td_vals),
        'td_n': len(td_vals),
        'sli_mean': np.mean(sli_vals),
        'sli_std': np.std(sli_vals, ddof=1),
        'sli_median': np.median(sli_vals),
        'sli_n': len(sli_vals),
        'mean_difference': np.mean(td_vals) - np.mean(sli_vals),
        't_statistic': t_stat,
        't_p_value': t_p,
        'u_statistic': u_stat,
        'u_p_value': u_p,
        'cohen_d': cohen_d,
        'effect_size_interpretation': (
            'Large' if abs(cohen_d) > 0.8 else
            'Medium' if abs(cohen_d) > 0.5 else
            'Small'
        )
    })

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv(STAT_DIR / "group_comparisons.csv", index=False)
print(f"  ✓ Group comparisons saved to {STAT_DIR}/group_comparisons.csv")

# Print key findings
print("\n  Key Statistical Findings:")
for _, row in comparison_df.iterrows():
    sig = "***" if row['t_p_value'] < 0.001 else "**" if row['t_p_value'] < 0.01 else "*" if row['t_p_value'] < 0.05 else "ns"
    print(f"    {row['metric']: <25} t={row['t_statistic']:6.2f}, p={row['t_p_value']:.4f} {sig}, d={row['cohen_d']:5.2f} ({row['effect_size_interpretation']})")

# ----------------------------------------------------------------------------
# 2.3 Age-Stratified Analysis
# ----------------------------------------------------------------------------

print("\n2.3 Performing age-stratified analysis...")

age_stratified_results = []

ages = sorted(individual_networks['age_years'].unique())

for age in ages:
    age_data = individual_networks[individual_networks['age_years'] == age]
    td_age = age_data[age_data['development_type'] == 'TD']
    sli_age = age_data[age_data['development_type'] == 'SLI']

    if len(td_age) < 3 or len(sli_age) < 3:
        continue

    for metric in key_metrics:
        if metric not in age_data.columns:
            continue

        td_vals = td_age[metric].dropna().values
        sli_vals = sli_age[metric].dropna().values

        if len(td_vals) < 3 or len(sli_vals) < 3:
            continue

        t_stat, p_val = ttest_ind(td_vals, sli_vals)
        pooled_std = np.sqrt((np.std(td_vals, ddof=1)**2 + np.std(sli_vals, ddof=1)**2) / 2)
        cohen_d = (np.mean(td_vals) - np.mean(sli_vals)) / pooled_std

        age_stratified_results.append({
            'age': age,
            'metric': metric_labels.get(metric, metric),
            'td_n': len(td_vals),
            'sli_n': len(sli_vals),
            'td_mean': np.mean(td_vals),
            'sli_mean': np.mean(sli_vals),
            'difference': np.mean(td_vals) - np.mean(sli_vals),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohen_d': cohen_d
        })

age_stratified_df = pd.DataFrame(age_stratified_results)
age_stratified_df.to_csv(STAT_DIR / "age_stratified_analysis.csv", index=False)
print(f"  ✓ Age-stratified analysis saved to {STAT_DIR}/age_stratified_analysis.csv")

print(f"\n  Analyzed {len(ages)} age groups")

# ----------------------------------------------------------------------------
# 2.4 Correlation Analysis
# ----------------------------------------------------------------------------

print("\n2.4 Computing correlation analyses...")

# Test if MLU correlates with network metrics
if metadata is not None and 'mlu' in metadata.columns:
    # Merge with individual networks
    merged_data = individual_networks.merge(
        metadata[['filename', 'mlu']],
        on='filename',
        how='left'
    )

    correlation_results = []

    for group in ['TD', 'SLI']:
        group_data = merged_data[merged_data['development_type'] == group]

        for metric in key_metrics:
            if metric in group_data.columns and 'mlu' in group_data.columns:
                valid_data = group_data[[metric, 'mlu']].dropna()

                if len(valid_data) > 3:
                    r, p = stats.pearsonr(valid_data[metric], valid_data['mlu'])

                    correlation_results.append({
                        'group': group,
                        'metric': metric_labels.get(metric, metric),
                        'pearson_r': r,
                        'p_value': p,
                        'n': len(valid_data)
                    })

    correlation_df = pd.DataFrame(correlation_results)
    correlation_df.to_csv(STAT_DIR / "mlu_correlations.csv", index=False)
    print(f"  ✓ MLU correlations saved to {STAT_DIR}/mlu_correlations.csv")

# ----------------------------------------------------------------------------
# 2.5 Effect Size Summary Across Ages
# ----------------------------------------------------------------------------

print("\n2.5 Summarizing effect sizes across development...")

effect_size_summary = age_stratified_df.pivot_table(
    index='metric',
    columns='age',
    values='cohen_d'
)

effect_size_summary.to_csv(STAT_DIR / "effect_sizes_by_age.csv")
print(f"  ✓ Effect size summary saved to {STAT_DIR}/effect_sizes_by_age.csv")

# ----------------------------------------------------------------------------
# 2.6 Generate Comprehensive Statistical Report
# ----------------------------------------------------------------------------

print("\n2.6 Generating comprehensive statistical report...")

report = f"""
COMPREHENSIVE STATISTICAL ANALYSIS REPORT
Psycholinguistic Network Analysis: TD vs SLI
{'='*80}

SAMPLE CHARACTERISTICS
----------------------
Total children: {len(individual_networks)}
  - TD children: {len(individual_networks[individual_networks['development_type'] == 'TD'])}
  - SLI children: {len(individual_networks[individual_networks['development_type'] == 'SLI'])}
Age range: {individual_networks['age_years'].min()}-{individual_networks['age_years'].max()} years

STATISTICAL ASSUMPTIONS
-----------------------
{normality_df.to_string(index=False)}

PRIMARY GROUP COMPARISONS
-------------------------
{comparison_df.to_string(index=False)}

INTERPRETATION OF EFFECT SIZES
------------------------------
Cohen's d interpretation:
  - Small: 0.2 - 0.5
  - Medium: 0.5 - 0.8
  - Large: > 0.8

KEY FINDINGS
------------
"""

# Add significant findings
sig_findings = comparison_df[comparison_df['t_p_value'] < 0.05].sort_values('cohen_d', key=abs, ascending=False)

report += "\nSignificant differences (p < 0.05):\n"
for idx, row in sig_findings.iterrows():
    direction = "higher" if row['mean_difference'] > 0 else "lower"
    report += f"  • {row['metric']}: TD {direction} than SLI (d={row['cohen_d']:.2f}, p={row['t_p_value']:.4f})\n"

report += f"\n\nMLU-MATCHED COMPARISON\n"
report += f"----------------------\n"
report += f"{mlu_matched.to_string(index=False)}\n"

report += f"\n\nAGE-STRATIFIED ANALYSIS\n"
report += f"-----------------------\n"
report += f"Number of age groups analyzed: {len(ages)}\n"
report += f"See 'age_stratified_analysis.csv' for detailed results\n"

report += f"\n\nFILES GENERATED\n"
report += f"---------------\n"
report += f"Figures (n=6):\n"
report += f"  - fig1_topology_comparison.png/pdf\n"
report += f"  - fig2_age_progression.png/pdf\n"
report += f"  - fig3_distributions.png/pdf\n"
report += f"  - fig4_effect_sizes.png/pdf\n"
report += f"  - fig5_correlation_matrices.png/pdf\n"
report += f"  - fig6_mlu_matched.png/pdf\n"
report += f"\nStatistical outputs:\n"
report += f"  - assumption_tests.csv\n"
report += f"  - group_comparisons.csv\n"
report += f"  - age_stratified_analysis.csv\n"
report += f"  - effect_sizes_by_age.csv\n"
if metadata is not None and 'mlu' in metadata.columns:
    report += f"  - mlu_correlations.csv\n"

report += f"\n{'='*80}\n"
report += f"Analysis complete: {pd.Timestamp.now()}\n"

# Save report
with open(STAT_DIR / "statistical_report.txt", 'w') as f:
    f.write(report)

print(f"  ✓ Statistical report saved to {STAT_DIR}/statistical_report.txt")

# Save summary as JSON for programmatic access
summary_dict = {
    'sample_size': {
        'total': len(individual_networks),
        'td': len(individual_networks[individual_networks['development_type'] == 'TD']),
        'sli': len(individual_networks[individual_networks['development_type'] == 'SLI'])
    },
    'age_range': {
        'min': int(individual_networks['age_years'].min()),
        'max': int(individual_networks['age_years'].max())
    },
    'significant_findings': sig_findings.to_dict('records'),
    'normality_tests': normality_df.to_dict('records'),
    'group_comparisons': comparison_df.to_dict('records')
}

with open(STAT_DIR / "analysis_summary.json", 'w') as f:
    json.dump(summary_dict, f, indent=2)

print(f"  ✓ Analysis summary saved to {STAT_DIR}/analysis_summary.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print()
print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print()
print(f"Figures generated: 6")
print(f"  Location: {FIG_DIR}")
print()
print(f"Statistical outputs: {len(list(STAT_DIR.glob('*.csv'))) + 2}")
print(f"  Location: {STAT_DIR}")
print()
print("Key Findings Summary:")
print("-" * 80)

for _, row in sig_findings.head(5).iterrows():
    direction = ">" if row['mean_difference'] > 0 else "<"
    print(f"  {row['metric']: <25} TD {direction} SLI: d={row['cohen_d']:5.2f}, p={row['t_p_value']:.4f}")

print()
print("Next steps for academic presentation:")
print("  1. Review figures in", FIG_DIR)
print("  2. Examine statistical report:", STAT_DIR / "statistical_report.txt")
print("  3. Consider additional analyses (e.g., machine learning classification)")
print()
print("="*80)
