#!/usr/bin/env python3
"""
Precompute Network Data for Marimo Interactive Analysis
========================================================
This script runs on the cloud to precompute all necessary data for
interactive network analysis on a local machine with limited resources.

Output: Compressed data files optimized for fast loading and visualization
"""

import numpy as np
import pandas as pd
import pickle
import networkx as nx
from pathlib import Path
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.sparse import csr_matrix
from tqdm import tqdm
import json
import h5py
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NetworkPrecomputer:
    """Precompute network data at multiple thresholds for interactive analysis"""

    def __init__(self, embeddings_dir='embeddings/', output_dir='precomputed_data/'):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Define threshold range for precomputation
        self.thresholds = np.arange(0.30, 0.96, 0.01)  # 66 thresholds

    def load_embeddings_by_group(self):
        """Load embeddings organized by group (TD/SLI) and age"""
        embeddings_by_group = {
            'TD': defaultdict(list),
            'SLI': defaultdict(list),
            'ALL': []
        }

        metadata = pd.read_csv('metadata_with_mlu.csv')

        for _, row in tqdm(metadata.iterrows(), desc="Loading embeddings"):
            embedding_file = self.embeddings_dir / f"{row['filename'].replace('.txt', '.pkl')}"

            if embedding_file.exists():
                with open(embedding_file, 'rb') as f:
                    data = pickle.load(f)

                # Extract word embeddings and tokens
                word_embeddings = data['word_embeddings']
                word_tokens = data['word_tokens']

                # Store with metadata
                child_data = {
                    'id': row['filename'],
                    'embeddings': word_embeddings,
                    'words': word_tokens,
                    'dev_type': row['development_type'],
                    'age': row['age_years'],
                    'mlu': row['mlu']
                }

                embeddings_by_group[row['development_type']][row['age_years']].append(child_data)
                embeddings_by_group['ALL'].append(child_data)

        return embeddings_by_group

    def compute_group_similarity_matrix(self, group_data):
        """Compute average similarity matrix for a group of children"""
        # Collect all unique words and their embeddings
        word_embeddings_dict = defaultdict(list)

        for child in group_data:
            for word, embedding in zip(child['words'], child['embeddings']):
                word_embeddings_dict[word].append(embedding)

        # Average embeddings for each word across children
        unique_words = sorted(word_embeddings_dict.keys())
        avg_embeddings = []

        for word in unique_words:
            embeddings_list = word_embeddings_dict[word]
            avg_embedding = np.mean(embeddings_list, axis=0)
            avg_embeddings.append(avg_embedding)

        avg_embeddings = np.array(avg_embeddings)

        # Compute similarity matrix (using cosine similarity)
        # More efficient than pdist for our use case
        n_words = len(avg_embeddings)
        similarity_matrix = np.zeros((n_words, n_words))

        for i in range(n_words):
            for j in range(i, n_words):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = 1 - cosine(avg_embeddings[i], avg_embeddings[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        return unique_words, similarity_matrix

    def compute_network_metrics_at_threshold(self, similarity_matrix, threshold):
        """Compute network metrics for a given threshold"""
        # Create adjacency matrix
        adjacency = (similarity_matrix > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)  # Remove self-loops

        # Create network
        G = nx.from_numpy_array(adjacency)

        # Remove isolated nodes for cleaner metrics
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return None

        metrics = {
            'threshold': threshold,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_isolated': len(isolated),
            'density': nx.density(G),
            'avg_degree': np.mean([d for n, d in G.degree()]),
            'max_degree': max([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
            'clustering': nx.average_clustering(G),
            'transitivity': nx.transitivity(G),
        }

        # Connected components analysis
        components = list(nx.connected_components(G))
        metrics['n_components'] = len(components)
        metrics['largest_component_size'] = len(max(components, key=len)) if components else 0
        metrics['largest_component_frac'] = metrics['largest_component_size'] / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

        # Path length (only for largest connected component)
        if metrics['largest_component_size'] > 1:
            Gcc = G.subgraph(max(components, key=len))
            try:
                metrics['avg_path_length'] = nx.average_shortest_path_length(Gcc)
                metrics['diameter'] = nx.diameter(Gcc)
            except:
                metrics['avg_path_length'] = np.nan
                metrics['diameter'] = np.nan
        else:
            metrics['avg_path_length'] = np.nan
            metrics['diameter'] = np.nan

        # Small-world metrics (if connected enough)
        if metrics['largest_component_frac'] > 0.8 and metrics['n_edges'] > metrics['n_nodes']:
            metrics.update(self.calculate_small_world_coefficient(G))
        else:
            metrics['sigma'] = np.nan
            metrics['omega'] = np.nan

        return metrics

    def calculate_small_world_coefficient(self, G):
        """Calculate small-world metrics"""
        n = G.number_of_nodes()
        m = G.number_of_edges()

        # Actual metrics
        C_actual = nx.average_clustering(G)

        # Get largest connected component for path length
        Gcc = G.subgraph(max(nx.connected_components(G), key=len))
        L_actual = nx.average_shortest_path_length(Gcc)

        # Random graph equivalent
        p = 2 * m / (n * (n - 1)) if n > 1 else 0
        C_random = p
        L_random = np.log(n) / np.log(2 * m / n) if m > 0 and n > 1 else 1

        # Small-world metrics
        gamma = C_actual / C_random if C_random > 0 else 0
        lambda_ = L_actual / L_random if L_random > 0 else 0
        sigma = gamma / lambda_ if lambda_ > 0 else 0

        # Omega small-world metric (alternative)
        # -1 = lattice, 0 = small-world, 1 = random
        omega = (L_random / L_actual) - (C_actual / C_random) if C_random > 0 else 0

        return {
            'C_actual': C_actual,
            'L_actual': L_actual,
            'C_random': C_random,
            'L_random': L_random,
            'gamma': gamma,
            'lambda': lambda_,
            'sigma': sigma,
            'omega': omega
        }

    def precompute_all_networks(self):
        """Main function to precompute all network data"""
        print("Starting precomputation pipeline...")

        # Load all embeddings
        print("\n1. Loading embeddings...")
        embeddings_by_group = self.load_embeddings_by_group()

        # Process each group
        groups_to_process = [
            ('ALL', embeddings_by_group['ALL']),
            ('TD', [child for age_group in embeddings_by_group['TD'].values() for child in age_group]),
            ('SLI', [child for age_group in embeddings_by_group['SLI'].values() for child in age_group])
        ]

        # Add age-specific groups
        for age in range(5, 12):
            td_age = embeddings_by_group['TD'].get(age, [])
            sli_age = embeddings_by_group['SLI'].get(age, [])

            if td_age:
                groups_to_process.append((f'TD_{age}', td_age))
            if sli_age:
                groups_to_process.append((f'SLI_{age}', sli_age))

        all_results = {}

        for group_name, group_data in groups_to_process:
            if not group_data:
                continue

            print(f"\n2. Processing {group_name} ({len(group_data)} children)...")

            # Compute similarity matrix
            print(f"   Computing similarity matrix...")
            words, similarity_matrix = self.compute_group_similarity_matrix(group_data)

            # Store similarity matrix info (sparse format for efficiency)
            sim_flat = similarity_matrix[np.triu_indices(len(words), k=1)]
            similarity_stats = {
                'mean': np.mean(sim_flat),
                'std': np.std(sim_flat),
                'median': np.median(sim_flat),
                'percentiles': {
                    '5': np.percentile(sim_flat, 5),
                    '25': np.percentile(sim_flat, 25),
                    '50': np.percentile(sim_flat, 50),
                    '75': np.percentile(sim_flat, 75),
                    '95': np.percentile(sim_flat, 95),
                    '99': np.percentile(sim_flat, 99)
                },
                'n_words': len(words)
            }

            # Compute network metrics at each threshold
            print(f"   Computing network metrics at {len(self.thresholds)} thresholds...")
            metrics_list = []

            for threshold in tqdm(self.thresholds, desc=f"   {group_name}"):
                metrics = self.compute_network_metrics_at_threshold(similarity_matrix, threshold)
                if metrics:
                    metrics['group'] = group_name
                    metrics_list.append(metrics)

            # Store results
            group_results = {
                'group_name': group_name,
                'n_children': len(group_data),
                'words': words[:1000],  # Store top 1000 words for reference
                'similarity_stats': similarity_stats,
                'metrics_by_threshold': pd.DataFrame(metrics_list),
                'similarity_distribution': sim_flat[:10000] if len(sim_flat) > 10000 else sim_flat  # Sample for visualization
            }

            all_results[group_name] = group_results

            # Save individual group data
            group_output_file = self.output_dir / f'network_metrics_{group_name}.parquet'
            group_results['metrics_by_threshold'].to_parquet(group_output_file)
            print(f"   Saved to {group_output_file}")

        # Find optimal thresholds
        print("\n3. Finding optimal thresholds...")
        optimal_thresholds = {}

        for group_name, results in all_results.items():
            df = results['metrics_by_threshold']

            # Find max small-world coefficient
            valid_sigma = df[df['sigma'].notna()]
            if not valid_sigma.empty:
                optimal_idx = valid_sigma['sigma'].idxmax()
                optimal_thresholds[group_name] = {
                    'threshold': valid_sigma.loc[optimal_idx, 'threshold'],
                    'sigma': valid_sigma.loc[optimal_idx, 'sigma'],
                    'metrics': valid_sigma.loc[optimal_idx].to_dict()
                }

        # Save summary data
        summary = {
            'thresholds': self.thresholds.tolist(),
            'groups': list(all_results.keys()),
            'optimal_thresholds': optimal_thresholds,
            'similarity_stats_by_group': {
                group: results['similarity_stats']
                for group, results in all_results.items()
            }
        }

        with open(self.output_dir / 'network_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n4. Saving combined results...")

        # Create combined dataframe for easy loading
        all_metrics_df = pd.concat([
            results['metrics_by_threshold']
            for results in all_results.values()
        ], ignore_index=True)

        all_metrics_df.to_parquet(self.output_dir / 'all_network_metrics.parquet')

        # Save similarity distributions for visualization
        with h5py.File(self.output_dir / 'similarity_distributions.h5', 'w') as f:
            for group_name, results in all_results.items():
                f.create_dataset(f'{group_name}/distribution',
                               data=results['similarity_distribution'])
                f.create_dataset(f'{group_name}/words',
                               data=np.array(results['words'][:100], dtype='S'))

        print("\n✅ Precomputation complete!")
        print(f"Output directory: {self.output_dir}")
        print("\nGenerated files:")
        print("  - all_network_metrics.parquet (main data for marimo)")
        print("  - network_analysis_summary.json (metadata)")
        print("  - similarity_distributions.h5 (for histograms)")
        print(f"  - network_metrics_*.parquet ({len(all_results)} group files)")

        # Print size summary
        total_size = sum(f.stat().st_size for f in self.output_dir.glob('*'))
        print(f"\nTotal size: {total_size / 1024 / 1024:.1f} MB")

        return all_results

def main():
    """Run the precomputation pipeline"""
    precomputer = NetworkPrecomputer()
    results = precomputer.precompute_all_networks()

    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY OF OPTIMAL THRESHOLDS")
    print("="*50)

    summary_df = []
    for group_name, group_results in results.items():
        df = group_results['metrics_by_threshold']
        valid_sigma = df[df['sigma'].notna()]

        if not valid_sigma.empty:
            optimal_idx = valid_sigma['sigma'].idxmax()
            summary_df.append({
                'Group': group_name,
                'Optimal Threshold': valid_sigma.loc[optimal_idx, 'threshold'],
                'Max Sigma': valid_sigma.loc[optimal_idx, 'sigma'],
                'Clustering': valid_sigma.loc[optimal_idx, 'clustering'],
                'Path Length': valid_sigma.loc[optimal_idx, 'avg_path_length'],
                'N Nodes': valid_sigma.loc[optimal_idx, 'n_nodes'],
                'N Edges': valid_sigma.loc[optimal_idx, 'n_edges']
            })

    summary_df = pd.DataFrame(summary_df)
    print(summary_df.to_string())

    # Save summary table
    summary_df.to_csv('precomputed_data/optimal_thresholds_summary.csv', index=False)

if __name__ == "__main__":
    main()