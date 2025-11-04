#!/usr/bin/env python3
"""
Utility functions for loading and analyzing BERT embeddings.
Useful for downstream semantic network analysis.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


class EmbeddingLoader:
    """Load and manage BERT embeddings for analysis."""

    def __init__(self, embeddings_dir: str = "embeddings", metadata_path: str = "embedding_ready/metadata.csv"):
        """
        Initialize the embedding loader.

        Args:
            embeddings_dir: Directory containing embedding pickle files
            metadata_path: Path to the metadata CSV file
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.metadata = pd.read_csv(metadata_path)

        # Create lookup dict for metadata
        self.metadata_dict = {}
        for _, row in self.metadata.iterrows():
            self.metadata_dict[row['filename']] = row.to_dict()

    def load_child_embeddings(self, child_filename: str) -> Dict:
        """
        Load embeddings for a specific child.

        Args:
            child_filename: Name of the child's text file (e.g., 'td_7y6m_f_46076v.txt')

        Returns:
            Dictionary containing embeddings and metadata
        """
        # Load pickle file
        pkl_path = self.embeddings_dir / child_filename.replace('.txt', '_embeddings.pkl')

        if not pkl_path.exists():
            raise FileNotFoundError(f"Embeddings not found for {child_filename}")

        with open(pkl_path, 'rb') as f:
            embeddings = pickle.load(f)

        # Add metadata
        if child_filename in self.metadata_dict:
            embeddings['metadata'] = self.metadata_dict[child_filename]

        return embeddings

    def load_group_embeddings(self, development_type: str = None, age: int = None) -> Dict[str, Dict]:
        """
        Load embeddings for a group of children.

        Args:
            development_type: 'TD' or 'SLI' (optional)
            age: Age in years (optional)

        Returns:
            Dictionary mapping child filenames to their embeddings
        """
        # Filter metadata based on criteria
        filtered_metadata = self.metadata.copy()

        if development_type:
            filtered_metadata = filtered_metadata[
                filtered_metadata['development_type'] == development_type
            ]

        if age is not None:
            filtered_metadata = filtered_metadata[
                filtered_metadata['age_years'] == age
            ]

        # Load embeddings for filtered children
        group_embeddings = {}

        for _, row in filtered_metadata.iterrows():
            filename = row['filename']
            try:
                embeddings = self.load_child_embeddings(filename)
                group_embeddings[filename] = embeddings
            except FileNotFoundError:
                print(f"Warning: Embeddings not found for {filename}")

        return group_embeddings

    def get_vocabulary(self, embeddings_dict: Dict[str, Dict]) -> List[str]:
        """
        Get combined vocabulary from multiple children.

        Args:
            embeddings_dict: Dictionary of child embeddings

        Returns:
            List of unique words across all children
        """
        vocab = set()

        for child_emb in embeddings_dict.values():
            vocab.update(child_emb['word_tokens'])

        return sorted(list(vocab))

    def create_word_embedding_matrix(self, embeddings_dict: Dict[str, Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Create a matrix of word embeddings averaged across children.

        Args:
            embeddings_dict: Dictionary of child embeddings

        Returns:
            (embedding_matrix, word_list)
        """
        # Collect all word embeddings
        word_embeddings_all = {}

        for child_emb in embeddings_dict.values():
            for word, embedding in zip(child_emb['word_tokens'], child_emb['word_embeddings']):
                if word not in word_embeddings_all:
                    word_embeddings_all[word] = []
                word_embeddings_all[word].append(embedding)

        # Average embeddings for each word
        word_list = []
        embedding_list = []

        for word, emb_list in sorted(word_embeddings_all.items()):
            word_list.append(word)
            # Average across all occurrences
            avg_embedding = np.mean(emb_list, axis=0)
            embedding_list.append(avg_embedding)

        embedding_matrix = np.array(embedding_list)

        return embedding_matrix, word_list

    def compute_similarity_matrix(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix between word embeddings.

        Args:
            embedding_matrix: Matrix of word embeddings

        Returns:
            Similarity matrix
        """
        return cosine_similarity(embedding_matrix)

    def find_hub_words(
        self,
        similarity_matrix: np.ndarray,
        word_list: List[str],
        threshold: float = 0.7,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Identify hub words based on connectivity in the semantic network.

        Args:
            similarity_matrix: Word similarity matrix
            word_list: List of words corresponding to matrix rows/cols
            threshold: Similarity threshold for considering words connected
            top_n: Number of top hub words to return

        Returns:
            DataFrame with hub words and their connectivity scores
        """
        # Count connections above threshold for each word
        connections = (similarity_matrix > threshold).sum(axis=1) - 1  # Subtract self-connection

        # Create DataFrame
        hub_df = pd.DataFrame({
            'word': word_list,
            'connections': connections,
            'avg_similarity': similarity_matrix.mean(axis=1)
        })

        # Sort by connections
        hub_df = hub_df.sort_values('connections', ascending=False).head(top_n)

        return hub_df

    def calculate_clustering_coefficient(
        self,
        similarity_matrix: np.ndarray,
        threshold: float = 0.7
    ) -> float:
        """
        Calculate the average clustering coefficient of the semantic network.

        Args:
            similarity_matrix: Word similarity matrix
            threshold: Similarity threshold for considering words connected

        Returns:
            Average clustering coefficient
        """
        # Create adjacency matrix
        adjacency = (similarity_matrix > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)  # Remove self-connections

        n = len(adjacency)
        clustering_coeffs = []

        for i in range(n):
            # Find neighbors
            neighbors = np.where(adjacency[i] == 1)[0]
            k = len(neighbors)

            if k < 2:
                clustering_coeffs.append(0)
                continue

            # Count edges between neighbors
            neighbor_edges = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if adjacency[neighbors[j], neighbors[l]] == 1:
                        neighbor_edges += 1

            # Calculate clustering coefficient
            max_edges = k * (k - 1) / 2
            cc = neighbor_edges / max_edges if max_edges > 0 else 0
            clustering_coeffs.append(cc)

        return np.mean(clustering_coeffs)

    def compare_groups(
        self,
        td_embeddings: Dict[str, Dict],
        sli_embeddings: Dict[str, Dict]
    ) -> Dict:
        """
        Compare network statistics between TD and SLI groups.

        Args:
            td_embeddings: Embeddings for TD children
            sli_embeddings: Embeddings for SLI children

        Returns:
            Dictionary with comparison statistics
        """
        results = {}

        # Process TD group
        td_matrix, td_words = self.create_word_embedding_matrix(td_embeddings)
        td_similarity = self.compute_similarity_matrix(td_matrix)
        td_hubs = self.find_hub_words(td_similarity, td_words)
        td_clustering = self.calculate_clustering_coefficient(td_similarity)

        # Process SLI group
        sli_matrix, sli_words = self.create_word_embedding_matrix(sli_embeddings)
        sli_similarity = self.compute_similarity_matrix(sli_matrix)
        sli_hubs = self.find_hub_words(sli_similarity, sli_words)
        sli_clustering = self.calculate_clustering_coefficient(sli_similarity)

        results['TD'] = {
            'vocab_size': len(td_words),
            'clustering_coefficient': td_clustering,
            'top_hubs': td_hubs.head(10),
            'hub_connectivity_mean': td_hubs['connections'].mean(),
            'hub_connectivity_std': td_hubs['connections'].std()
        }

        results['SLI'] = {
            'vocab_size': len(sli_words),
            'clustering_coefficient': sli_clustering,
            'top_hubs': sli_hubs.head(10),
            'hub_connectivity_mean': sli_hubs['connections'].mean(),
            'hub_connectivity_std': sli_hubs['connections'].std()
        }

        # Calculate differences
        results['differences'] = {
            'vocab_size_diff': len(td_words) - len(sli_words),
            'clustering_diff': td_clustering - sli_clustering,
            'hub_connectivity_diff': td_hubs['connections'].mean() - sli_hubs['connections'].mean(),
            'shared_top_hubs': len(set(td_hubs.head(10)['word']) & set(sli_hubs.head(10)['word']))
        }

        return results


def visualize_semantic_network(
    similarity_matrix: np.ndarray,
    word_list: List[str],
    threshold: float = 0.7,
    max_words: int = 50,
    title: str = "Semantic Network"
):
    """
    Visualize a semantic network using networkx.

    Args:
        similarity_matrix: Word similarity matrix
        word_list: List of words
        threshold: Similarity threshold for edges
        max_words: Maximum number of words to display
        title: Title for the plot
    """
    try:
        import networkx as nx

        # Select top connected words if too many
        if len(word_list) > max_words:
            connections = (similarity_matrix > threshold).sum(axis=1)
            top_indices = np.argsort(connections)[-max_words:]
            similarity_matrix = similarity_matrix[top_indices][:, top_indices]
            word_list = [word_list[i] for i in top_indices]

        # Create graph
        G = nx.Graph()

        # Add nodes
        G.add_nodes_from(word_list)

        # Add edges
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                if similarity_matrix[i, j] > threshold:
                    G.add_edge(word_list[i], word_list[j], weight=similarity_matrix[i, j])

        # Plot
        plt.figure(figsize=(12, 8))

        # Calculate node sizes based on degree
        degrees = dict(G.degree())
        node_sizes = [degrees[node] * 100 for node in G.nodes()]

        # Use spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("networkx not installed. Install with: pip install networkx")


def example_analysis():
    """Example of how to use the embedding utilities."""
    print("Example Embedding Analysis")
    print("="*80)

    # Initialize loader
    loader = EmbeddingLoader()

    # Load TD children aged 7
    print("\nLoading TD children aged 7...")
    td_7 = loader.load_group_embeddings(development_type='TD', age=7)
    print(f"Loaded {len(td_7)} TD children")

    # Load SLI children aged 7
    print("\nLoading SLI children aged 7...")
    sli_7 = loader.load_group_embeddings(development_type='SLI', age=7)
    print(f"Loaded {len(sli_7)} SLI children")

    if td_7 and sli_7:
        # Compare groups
        print("\nComparing TD and SLI groups...")
        comparison = loader.compare_groups(td_7, sli_7)

        print("\nResults:")
        print(f"TD Clustering Coefficient: {comparison['TD']['clustering_coefficient']:.3f}")
        print(f"SLI Clustering Coefficient: {comparison['SLI']['clustering_coefficient']:.3f}")
        print(f"Difference: {comparison['differences']['clustering_diff']:.3f}")

        print(f"\nTD Mean Hub Connectivity: {comparison['TD']['hub_connectivity_mean']:.1f}")
        print(f"SLI Mean Hub Connectivity: {comparison['SLI']['hub_connectivity_mean']:.1f}")

        print("\nTop 5 TD Hub Words:")
        print(comparison['TD']['top_hubs'].head())

        print("\nTop 5 SLI Hub Words:")
        print(comparison['SLI']['top_hubs'].head())


if __name__ == "__main__":
    # Run example if embeddings exist
    if Path("embeddings").exists():
        example_analysis()
    else:
        print("Please run generate_bert_embeddings.py first to create embeddings.")