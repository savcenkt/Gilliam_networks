"""
Network construction module for semantic network analysis.

Constructs semantic networks from BERT embeddings using various methods.
"""

import logging
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NetworkConstructor:
    """
    Construct semantic networks from word embeddings.

    Supports multiple construction methods:
    - Threshold-based
    - k-Nearest Neighbors
    - Adaptive density
    - Minimum Spanning Tree + additional edges
    """

    def __init__(self, config: Dict):
        """
        Initialize network constructor.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.method = config['network']['construction_method']
        self.threshold = config['network']['similarity_threshold']
        self.knn_k = config['network'].get('knn_k', 10)
        self.adaptive_density = config['network'].get('adaptive_density', 0.05)

    def run(self, embeddings_dir: str, output_dir: str, force: bool = False) -> Dict:
        """
        Construct networks from embeddings.

        Args:
            embeddings_dir: Directory containing embedding files
            output_dir: Directory to save network data
            force: Force re-construction even if networks exist

        Returns:
            Dictionary with construction statistics
        """
        logger.info(f"Starting network construction using {self.method} method")

        embeddings_path = Path(embeddings_dir)
        output_path = Path(output_dir)

        # Check if networks already exist
        if not force and output_path.exists() and len(list(output_path.glob("*.pkl"))) > 0:
            logger.info("Networks already exist. Use --force to regenerate.")
            return {"status": "skipped", "message": "Networks already exist"}

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Load embeddings and construct networks
        results = self._construct_all_networks(embeddings_path, output_path)

        logger.info(f"Network construction complete: {results['total_networks']} networks created")
        return results

    def _construct_all_networks(self, embeddings_dir: Path, output_dir: Path) -> Dict:
        """Construct networks for all embedding files."""
        embedding_files = list(embeddings_dir.glob("*_embeddings.pkl"))
        logger.info(f"Found {len(embedding_files)} embedding files")

        networks = []
        stats = {
            'total_networks': 0,
            'td_networks': 0,
            'sli_networks': 0,
            'method': self.method,
            'avg_nodes': 0,
            'avg_edges': 0,
            'avg_density': 0
        }

        for emb_file in tqdm(embedding_files, desc="Constructing networks"):
            try:
                # Load embeddings
                with open(emb_file, 'rb') as f:
                    emb_data = pickle.load(f)

                # Extract child info from filename
                filename = emb_file.stem.replace('_embeddings', '')
                dev_type = 'TD' if filename.startswith('td') else 'SLI'

                # Construct network
                network = self._construct_network(
                    emb_data['word_embeddings'],
                    emb_data['word_tokens']
                )

                # Calculate network metrics
                metrics = self._calculate_network_metrics(network)

                # Save network and metrics
                network_data = {
                    'filename': filename,
                    'development_type': dev_type,
                    'network': network,
                    'metrics': metrics,
                    'method': self.method,
                    'threshold': self.threshold if self.method == 'threshold' else None
                }

                output_file = output_dir / f"{filename}_network.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(network_data, f)

                networks.append(network_data)
                stats['total_networks'] += 1
                if dev_type == 'TD':
                    stats['td_networks'] += 1
                else:
                    stats['sli_networks'] += 1

            except Exception as e:
                logger.error(f"Failed to construct network for {emb_file}: {e}")
                continue

        # Calculate average statistics
        if networks:
            stats['avg_nodes'] = np.mean([n['metrics']['n_nodes'] for n in networks])
            stats['avg_edges'] = np.mean([n['metrics']['n_edges'] for n in networks])
            stats['avg_density'] = np.mean([n['metrics']['density'] for n in networks])

        return stats

    def _construct_network(self, embeddings: np.ndarray, words: List[str]) -> nx.Graph:
        """Construct network based on selected method."""
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Apply construction method
        if self.method == 'threshold':
            adjacency = self._threshold_based(similarity_matrix, self.threshold)
        elif self.method == 'knn':
            adjacency = self._knn_based(similarity_matrix, self.knn_k)
        elif self.method == 'adaptive':
            adjacency = self._adaptive_threshold(similarity_matrix, self.adaptive_density)
        elif self.method == 'mst_plus':
            adjacency = self._mst_plus(similarity_matrix)
        else:
            raise ValueError(f"Unknown construction method: {self.method}")

        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(len(words)))

        # Add node labels
        for i, word in enumerate(words):
            G.nodes[i]['word'] = word

        # Add edges
        edges = np.where(adjacency > 0)
        for i, j in zip(edges[0], edges[1]):
            if i < j:  # Avoid duplicate edges
                G.add_edge(i, j, weight=similarity_matrix[i, j])

        return G

    def _threshold_based(self, sim_matrix: np.ndarray, threshold: float) -> np.ndarray:
        """Classic threshold-based construction."""
        adjacency = (sim_matrix > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)  # Remove self-loops
        return adjacency

    def _knn_based(self, sim_matrix: np.ndarray, k: int) -> np.ndarray:
        """k-nearest neighbors construction."""
        n = len(sim_matrix)
        adjacency = np.zeros_like(sim_matrix)

        for i in range(n):
            # Get k most similar words (excluding self)
            similarities = sim_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            top_k = np.argsort(similarities)[-k:]

            # Add edges
            adjacency[i, top_k] = 1
            adjacency[top_k, i] = 1  # Ensure symmetry

        return adjacency

    def _adaptive_threshold(self, sim_matrix: np.ndarray, target_density: float) -> np.ndarray:
        """Adaptive threshold to achieve target density."""
        n = len(sim_matrix)
        target_edges = int(target_density * n * (n - 1) / 2)

        # Find threshold that gives closest to target edges
        flat_sim = sim_matrix[np.triu_indices(n, k=1)]
        threshold = np.percentile(flat_sim, 100 * (1 - target_density))

        return self._threshold_based(sim_matrix, threshold)

    def _mst_plus(self, sim_matrix: np.ndarray) -> np.ndarray:
        """Minimum spanning tree plus additional high-similarity edges."""
        n = len(sim_matrix)

        # Create MST using dissimilarity
        dissimilarity = 1 - sim_matrix
        np.fill_diagonal(dissimilarity, 0)
        G_temp = nx.from_numpy_array(dissimilarity)
        mst = nx.minimum_spanning_tree(G_temp)

        # Convert MST to adjacency
        adjacency = nx.to_numpy_array(mst)

        # Add additional high-similarity edges
        additional_edges = self.config['network'].get('mst_additional_edges', 100)
        flat_indices = np.triu_indices(n, k=1)
        flat_sim = sim_matrix[flat_indices]
        sorted_indices = np.argsort(flat_sim)[::-1]

        added = 0
        for idx in sorted_indices:
            i, j = flat_indices[0][idx], flat_indices[1][idx]
            if adjacency[i, j] == 0:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                added += 1
                if added >= additional_edges:
                    break

        return adjacency

    def _calculate_network_metrics(self, G: nx.Graph) -> Dict:
        """Calculate basic network metrics."""
        metrics = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 1 else 0
        }

        # Add clustering coefficient if network has edges
        if G.number_of_edges() > 0:
            metrics['clustering'] = nx.average_clustering(G)
            metrics['transitivity'] = nx.transitivity(G)

            # Add path length for connected component
            if nx.is_connected(G):
                metrics['avg_path_length'] = nx.average_shortest_path_length(G)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                if len(largest_cc) > 1:
                    G_cc = G.subgraph(largest_cc)
                    metrics['avg_path_length'] = nx.average_shortest_path_length(G_cc)
                    metrics['largest_cc_size'] = len(largest_cc)

        return metrics