"""
Create additional publication-quality figures for academic presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
age_stratified = pd.read_csv("analysis_outputs/statistics/age_stratified_analysis.csv")
effect_sizes_by_age = pd.read_csv("analysis_outputs/statistics/effect_sizes_by_age.csv")

FIG_DIR = Path("analysis_outputs/figures")

# ============================================================================
# Figure 7: Effect Size Heatmap by Age
# ============================================================================

print("Creating Figure 7: Effect size heatmap across ages...")

fig, ax = plt.subplots(figsize=(12, 6))

# Create heatmap
sns.heatmap(effect_sizes_by_age.set_index('metric'),
            annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1.5, vmax=1.5, square=False, ax=ax,
            cbar_kws={'label': "Cohen's d (TD - SLI)"},
            linewidths=0.5, linecolor='gray')

ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Network Metric', fontsize=12, fontweight='bold')
ax.set_title("Effect Sizes Across Developmental Ages\n(Positive = TD > SLI, Negative = TD < SLI)",
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig7_effect_size_heatmap.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig7_effect_size_heatmap.pdf", bbox_inches='tight')
print(f"  ✓ Saved to {FIG_DIR}/fig7_effect_size_heatmap.png")
plt.close()

# ============================================================================
# Figure 8: Significance Stars by Age
# ============================================================================

print("Creating Figure 8: Statistical significance across ages...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

metrics = age_stratified['metric'].unique()

for idx, metric in enumerate(metrics[:5]):
    metric_data = age_stratified[age_stratified['metric'] == metric]

    ages = metric_data['age'].values
    p_values = metric_data['p_value'].values
    effect_sizes = metric_data['cohen_d'].values

    # Create bar plot of effect sizes
    colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_values]
    bars = axes[idx].bar(ages, effect_sizes, color=colors, edgecolor='black', linewidth=1.5)

    # Add significance stars
    for i, (age, p, d) in enumerate(zip(ages, p_values, effect_sizes)):
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        y_pos = d + 0.1 if d > 0 else d - 0.1
        axes[idx].text(age, y_pos, sig, ha='center', va='center' if d > 0 else 'top',
                      fontsize=10, fontweight='bold')

    axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[idx].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[idx].axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    axes[idx].set_xlabel('Age (years)', fontsize=10)
    axes[idx].set_ylabel("Cohen's d", fontsize=10)
    axes[idx].set_title(metric, fontsize=11, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].set_xticks(ages)

# Remove extra subplot
axes[-1].axis('off')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', edgecolor='black', label='Significant (p < 0.05)'),
    Patch(facecolor='#95a5a6', edgecolor='black', label='Not Significant')
]
axes[-1].legend(handles=legend_elements, loc='center', fontsize=12, frameon=True)

plt.suptitle('Age-Stratified Effect Sizes with Significance Markers',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig8_significance_by_age.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig8_significance_by_age.pdf", bbox_inches='tight')
print(f"  ✓ Saved to {FIG_DIR}/fig8_significance_by_age.png")
plt.close()

# ============================================================================
# Figure 9: Sample Size Information
# ============================================================================

print("Creating Figure 9: Sample size visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Sample sizes by age
sample_sizes = age_stratified.groupby(['age', 'metric']).first()[['td_n', 'sli_n']].reset_index()
sample_sizes_unique = sample_sizes.groupby('age')[['td_n', 'sli_n']].first()

x = np.arange(len(sample_sizes_unique))
width = 0.35

bars1 = ax1.bar(x - width/2, sample_sizes_unique['td_n'], width,
               label='TD', color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, sample_sizes_unique['sli_n'], width,
               label='SLI', color='#e74c3c', edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Children', fontsize=12, fontweight='bold')
ax1.set_title('Sample Sizes by Age Group', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(sample_sizes_unique.index)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

# Plot 2: Total sample composition
total_td = sample_sizes_unique['td_n'].sum()
total_sli = sample_sizes_unique['sli_n'].sum()

colors = ['#3498db', '#e74c3c']
explode = (0.05, 0.05)

wedges, texts, autotexts = ax2.pie([total_td, total_sli],
                                     labels=['TD', 'SLI'],
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     explode=explode,
                                     shadow=True,
                                     startangle=90,
                                     textprops={'fontsize': 12, 'fontweight': 'bold'})

ax2.set_title(f'Overall Sample Composition\n(Total N = {total_td + total_sli})',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / "fig9_sample_sizes.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig9_sample_sizes.pdf", bbox_inches='tight')
print(f"  ✓ Saved to {FIG_DIR}/fig9_sample_sizes.png")
plt.close()

print("\n✓ All additional figures created successfully!")
print(f"  Total figures now: 9")
