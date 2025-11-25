"""
Analysis module for semantic network comparison.

Analyzes and compares network properties between TD and SLI groups.
"""

import logging
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """
    Analyze semantic networks and compare between groups.

    Performs statistical comparisons, calculates small-world metrics,
    and generates visualizations.
    """

    def __init__(self, config: Dict):
        """
        Initialize network analyzer.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.bootstrap_iterations = config['analysis']['bootstrap_iterations']
        self.significance_level = config['analysis']['significance_level']
        self.age_groups = config['analysis']['age_groups']

    def run(self, networks_dir: str, output_dir: str, force: bool = False) -> Dict:
        """
        Run network analysis and generate results.

        Args:
            networks_dir: Directory containing network files
            output_dir: Directory to save results
            force: Force re-analysis even if results exist

        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting network analysis")

        networks_path = Path(networks_dir)
        output_path = Path(output_dir)

        # Create output directories
        (output_path / 'tables').mkdir(parents=True, exist_ok=True)
        (output_path / 'figures').mkdir(parents=True, exist_ok=True)
        (output_path / 'reports').mkdir(parents=True, exist_ok=True)

        # Load all networks
        networks = self._load_networks(networks_path)
        logger.info(f"Loaded {len(networks)} networks")

        # Calculate small-world metrics
        logger.info("Calculating small-world metrics")
        networks = self._calculate_small_world_metrics(networks)

        # Perform group comparisons
        logger.info("Performing group comparisons")
        comparisons = self._compare_groups(networks)

        # Generate visualizations
        logger.info("Generating visualizations")
        self._generate_visualizations(networks, comparisons, output_path / 'figures')

        # Save results tables
        logger.info("Saving results tables")
        self._save_results_tables(networks, comparisons, output_path / 'tables')

        # Generate summary report
        logger.info("Generating summary report")
        report = self._generate_report(networks, comparisons)
        report_path = output_path / 'reports' / 'analysis_report.md'
        report_path.write_text(report)

        results = {
            'td_count': len([n for n in networks if n['development_type'] == 'TD']),
            'sli_count': len([n for n in networks if n['development_type'] == 'SLI']),
            'optimal_threshold': comparisons.get('optimal_threshold', 0),
            'max_sigma': comparisons.get('max_sigma', 0),
            'significant_differences': comparisons.get('significant_differences', [])
        }

        logger.info(f"Analysis complete. Results saved to {output_path}")
        return results

    def _load_networks(self, networks_dir: Path) -> List[Dict]:
        """Load all network files."""
        network_files = list(networks_dir.glob("*_network.pkl"))
        networks = []

        for net_file in tqdm(network_files, desc="Loading networks"):
            try:
                with open(net_file, 'rb') as f:
                    network_data = pickle.load(f)
                networks.append(network_data)
            except Exception as e:
                logger.error(f"Failed to load network {net_file}: {e}")
                continue

        return networks

    def _calculate_small_world_metrics(self, networks: List[Dict]) -> List[Dict]:
        """Calculate small-world coefficients for all networks."""
        for network_data in tqdm(networks, desc="Calculating small-world metrics"):
            G = network_data['network']

            if G.number_of_edges() > 0:
                # Calculate actual metrics
                C_actual = nx.average_clustering(G)

                if nx.is_connected(G):
                    L_actual = nx.average_shortest_path_length(G)
                else:
                    # Use largest connected component
                    largest_cc = max(nx.connected_components(G), key=len)
                    if len(largest_cc) > 1:
                        G_cc = G.subgraph(largest_cc)
                        L_actual = nx.average_shortest_path_length(G_cc)
                    else:
                        L_actual = 1.0

                # Generate random graph for comparison
                n = G.number_of_nodes()
                m = G.number_of_edges()
                G_random = nx.gnm_random_graph(n, m)

                C_random = nx.average_clustering(G_random) if m > 0 else 0.001
                if nx.is_connected(G_random):
                    L_random = nx.average_shortest_path_length(G_random)
                else:
                    L_random = L_actual

                # Calculate small-world coefficient
                gamma = C_actual / C_random if C_random > 0 else 0
                lambda_ = L_actual / L_random if L_random > 0 else 0
                sigma = gamma / lambda_ if lambda_ > 0 else 0

                network_data['small_world'] = {
                    'sigma': sigma,
                    'gamma': gamma,
                    'lambda': lambda_,
                    'C_actual': C_actual,
                    'L_actual': L_actual,
                    'C_random': C_random,
                    'L_random': L_random
                }
            else:
                network_data['small_world'] = {
                    'sigma': 0,
                    'gamma': 0,
                    'lambda': 0
                }

        return networks

    def _compare_groups(self, networks: List[Dict]) -> Dict:
        """Compare network metrics between TD and SLI groups."""
        # Separate networks by group
        td_networks = [n for n in networks if n['development_type'] == 'TD']
        sli_networks = [n for n in networks if n['development_type'] == 'SLI']

        # Extract metrics for comparison
        metrics_to_compare = ['sigma', 'C_actual', 'L_actual']
        comparisons = {}

        for metric in metrics_to_compare:
            td_values = [n['small_world'].get(metric, 0) for n in td_networks]
            sli_values = [n['small_world'].get(metric, 0) for n in sli_networks]

            # Remove zeros and NaNs
            td_values = [v for v in td_values if v > 0 and not np.isnan(v)]
            sli_values = [v for v in sli_values if v > 0 and not np.isnan(v)]

            if td_values and sli_values:
                # Perform statistical test
                t_stat, p_value = stats.ttest_ind(td_values, sli_values)

                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(td_values) - 1) * np.var(td_values) +
                                     (len(sli_values) - 1) * np.var(sli_values)) /
                                    (len(td_values) + len(sli_values) - 2))
                cohen_d = (np.mean(td_values) - np.mean(sli_values)) / pooled_std if pooled_std > 0 else 0

                comparisons[metric] = {
                    'td_mean': np.mean(td_values),
                    'td_std': np.std(td_values),
                    'sli_mean': np.mean(sli_values),
                    'sli_std': np.std(sli_values),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohen_d': cohen_d,
                    'significant': p_value < self.significance_level
                }

        # Find optimal threshold (if applicable)
        if td_networks:
            sigma_values = [n['small_world'].get('sigma', 0) for n in td_networks]
            valid_sigmas = [s for s in sigma_values if s > 0 and not np.isnan(s)]
            if valid_sigmas:
                comparisons['max_sigma'] = max(valid_sigmas)
                max_idx = sigma_values.index(comparisons['max_sigma'])
                comparisons['optimal_threshold'] = td_networks[max_idx].get('threshold', 0.7)

        # List significant differences
        comparisons['significant_differences'] = [
            metric for metric, results in comparisons.items()
            if isinstance(results, dict) and results.get('significant', False)
        ]

        return comparisons

    def _generate_visualizations(self, networks: List[Dict], comparisons: Dict, output_dir: Path):
        """Generate analysis visualizations."""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

        # 1. Small-world coefficient comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['sigma', 'C_actual', 'L_actual']
        titles = ['Small-World Coefficient (σ)', 'Clustering Coefficient', 'Average Path Length']

        for ax, metric, title in zip(axes, metrics, titles):
            td_values = [n['small_world'].get(metric, 0) for n in networks
                        if n['development_type'] == 'TD' and n['small_world'].get(metric, 0) > 0]
            sli_values = [n['small_world'].get(metric, 0) for n in networks
                         if n['development_type'] == 'SLI' and n['small_world'].get(metric, 0) > 0]

            data = pd.DataFrame({
                'TD': pd.Series(td_values),
                'SLI': pd.Series(sli_values)
            })

            data.boxplot(ax=ax)
            ax.set_title(title)
            ax.set_ylabel(metric)

            # Add significance marker if significant
            if metric in comparisons and comparisons[metric].get('significant', False):
                ax.text(1.5, ax.get_ylim()[1] * 0.95, '*', fontsize=20, ha='center')

        plt.tight_layout()
        plt.savefig(output_dir / 'group_comparison_boxplots.png', dpi=150)
        plt.close()

        # 2. Age progression plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for dev_type in ['TD', 'SLI']:
            age_means = []
            age_stds = []

            for age in self.age_groups:
                age_networks = [n for n in networks
                              if n['development_type'] == dev_type and
                              self._get_age_from_filename(n['filename']) == age]
                if age_networks:
                    sigmas = [n['small_world'].get('sigma', 0) for n in age_networks]
                    valid_sigmas = [s for s in sigmas if s > 0 and not np.isnan(s)]
                    if valid_sigmas:
                        age_means.append(np.mean(valid_sigmas))
                        age_stds.append(np.std(valid_sigmas))
                    else:
                        age_means.append(np.nan)
                        age_stds.append(np.nan)
                else:
                    age_means.append(np.nan)
                    age_stds.append(np.nan)

            # Plot with error bars
            valid_ages = [age for age, mean in zip(self.age_groups, age_means) if not np.isnan(mean)]
            valid_means = [mean for mean in age_means if not np.isnan(mean)]
            valid_stds = [std for std, mean in zip(age_stds, age_means) if not np.isnan(mean)]

            if valid_ages:
                ax.errorbar(valid_ages, valid_means, yerr=valid_stds,
                          marker='o', label=dev_type, capsize=5)

        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Small-World Coefficient (σ)')
        ax.set_title('Small-World Properties Across Age Groups')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'age_progression.png', dpi=150)
        plt.close()

        logger.info("Visualizations saved")

    def _save_results_tables(self, networks: List[Dict], comparisons: Dict, output_dir: Path):
        """Save analysis results to CSV tables."""
        # 1. Individual network metrics
        network_df = []
        for net in networks:
            row = {
                'filename': net['filename'],
                'development_type': net['development_type'],
                'n_nodes': net['metrics']['n_nodes'],
                'n_edges': net['metrics']['n_edges'],
                'density': net['metrics']['density'],
                'clustering': net['metrics'].get('clustering', np.nan),
                'avg_path_length': net['metrics'].get('avg_path_length', np.nan),
                'sigma': net['small_world'].get('sigma', np.nan),
                'gamma': net['small_world'].get('gamma', np.nan),
                'lambda': net['small_world'].get('lambda', np.nan)
            }
            network_df.append(row)

        network_df = pd.DataFrame(network_df)
        network_df.to_csv(output_dir / 'individual_network_metrics.csv', index=False)

        # 2. Group comparison statistics
        comparison_df = []
        for metric, results in comparisons.items():
            if isinstance(results, dict) and 'td_mean' in results:
                row = {
                    'metric': metric,
                    'td_mean': results['td_mean'],
                    'td_std': results['td_std'],
                    'sli_mean': results['sli_mean'],
                    'sli_std': results['sli_std'],
                    't_statistic': results['t_statistic'],
                    'p_value': results['p_value'],
                    'cohen_d': results['cohen_d'],
                    'significant': results['significant']
                }
                comparison_df.append(row)

        if comparison_df:
            comparison_df = pd.DataFrame(comparison_df)
            comparison_df.to_csv(output_dir / 'group_comparisons.csv', index=False)

        logger.info("Results tables saved")

    def _generate_report(self, networks: List[Dict], comparisons: Dict) -> str:
        """Generate markdown analysis report."""
        td_count = len([n for n in networks if n['development_type'] == 'TD'])
        sli_count = len([n for n in networks if n['development_type'] == 'SLI'])

        report = f"""# Semantic Network Analysis Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Dataset Overview

- **Total networks analyzed**: {len(networks)}
- **TD children**: {td_count}
- **SLI children**: {sli_count}

## Small-World Analysis Results

### Group Comparisons

| Metric | TD Mean (SD) | SLI Mean (SD) | t-statistic | p-value | Cohen's d | Significant |
|--------|--------------|---------------|-------------|---------|-----------|-------------|
"""

        for metric in ['sigma', 'C_actual', 'L_actual']:
            if metric in comparisons:
                r = comparisons[metric]
                report += f"| {metric} | {r['td_mean']:.3f} ({r['td_std']:.3f}) | "
                report += f"{r['sli_mean']:.3f} ({r['sli_std']:.3f}) | "
                report += f"{r['t_statistic']:.3f} | {r['p_value']:.4f} | "
                report += f"{r['cohen_d']:.3f} | {'Yes' if r['significant'] else 'No'} |\n"

        report += f"""

## Key Findings

1. **Small-world properties**: """

        if 'sigma' in comparisons:
            if comparisons['sigma']['td_mean'] > 1:
                report += "TD networks show small-world properties (σ > 1)"
            if comparisons['sigma']['sli_mean'] > 1:
                report += ", SLI networks also show small-world properties"
            if comparisons['sigma']['significant']:
                report += f". Significant difference between groups (p = {comparisons['sigma']['p_value']:.4f})"

        report += """

2. **Optimal threshold**: """

        if 'optimal_threshold' in comparisons:
            report += f"{comparisons['optimal_threshold']:.3f} (max σ = {comparisons.get('max_sigma', 0):.2f})"

        report += """

3. **Significant differences**: """

        if comparisons.get('significant_differences'):
            report += ", ".join(comparisons['significant_differences'])
        else:
            report += "No significant differences found"

        report += """

## Recommendations

Based on these results:
"""

        if comparisons.get('significant_differences'):
            report += "- Significant network differences suggest distinct semantic organization between groups\n"
            report += "- Consider MLU-matched comparisons to control for language complexity\n"
        else:
            report += "- No significant differences found; consider larger sample size or different network construction methods\n"

        if 'max_sigma' in comparisons and comparisons['max_sigma'] > 3:
            report += "- Strong small-world properties detected; networks show efficient organization\n"

        return report

    def _get_age_from_filename(self, filename: str) -> Optional[int]:
        """Extract age from filename (e.g., td_7y3m_f_xxx -> 7)."""
        try:
            parts = filename.split('_')
            age_part = parts[1]  # e.g., '7y3m'
            age = int(age_part.split('y')[0])
            return age
        except:
            return None