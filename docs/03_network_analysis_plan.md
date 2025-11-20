# Marimo Interactive Network Analysis Implementation Plan
## Dynamic Network Construction & Small-World Optimization

### Project Overview
Create an interactive marimo notebook that:
1. Explores cosine similarity distributions in BERT embeddings
2. Dynamically visualizes network properties across similarity thresholds
3. Calculates small-world coefficients
4. Optimizes threshold selection for maximum small-worldness
5. Compares TD vs SLI networks across different construction methods

---

## Part 1: Theoretical Foundation

### Small-World Networks
Small-world networks are characterized by:
- **High clustering coefficient** (C): Nodes cluster together in tight groups
- **Low average path length** (L): Short paths between any two nodes
- **Small-world coefficient (σ)**: σ = (C/C_random) / (L/L_random)
  - σ > 1 indicates small-world properties
  - Typical small-world networks have σ between 2-10

### Watts-Strogatz Small-World Measure
```
σ = γ / λ
where:
  γ = C_actual / C_random  (clustering ratio)
  λ = L_actual / L_random  (path length ratio)

C_random ≈ k/n (for sparse networks)
L_random ≈ ln(n)/ln(k)
where k = average degree, n = number of nodes
```

### Alternative: Small-World Propensity (SWP)
More robust measure that accounts for degree distribution:
```
SWP = 1 - |ΔC/ΔC_max| × |ΔL/ΔL_max|
where ΔC and ΔL are deviations from lattice and random networks
```

---

## Part 2: Marimo Notebook Structure

### Cell Architecture

```python
# Cell 1: Imports and Setup
import marimo as mo
import numpy as np
import pandas as pd
import pickle
import networkx as nx
from scipy.spatial.distance import cosine
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Cell 2: Data Loading Functions
def load_embeddings(embedding_dir='embeddings/'):
    """Load all embeddings and extract word vectors"""
    pass

def compute_similarity_matrix(embeddings, method='cosine'):
    """Compute pairwise similarity matrix"""
    pass

# Cell 3: Interactive Controls
similarity_slider = mo.ui.slider(
    start=0.3,
    stop=0.95,
    step=0.01,
    value=0.7,
    label="Cosine Similarity Threshold"
)

age_selector = mo.ui.dropdown(
    options=['All'] + list(range(5, 12)),
    value='All',
    label="Age Group"
)

group_selector = mo.ui.radio(
    options=['Both', 'TD', 'SLI'],
    value='Both',
    label="Development Type"
)

# Cell 4: Network Construction
@mo.cache
def construct_network(similarity_matrix, threshold):
    """Build network from similarity matrix at given threshold"""
    G = nx.Graph()
    # Add edges where similarity > threshold
    return G

# Cell 5: Small-World Calculations
def calculate_small_world_coefficient(G):
    """Calculate σ (sigma) small-world coefficient"""
    # Actual metrics
    C_actual = nx.average_clustering(G)
    L_actual = nx.average_shortest_path_length(G)

    # Random graph metrics (Erdős-Rényi)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    G_random = nx.gnm_random_graph(n, m)
    C_random = nx.average_clustering(G_random)
    L_random = nx.average_shortest_path_length(G_random)

    # Small-world coefficient
    gamma = C_actual / C_random if C_random > 0 else 0
    lambda_ = L_actual / L_random if L_random > 0 else 0
    sigma = gamma / lambda_ if lambda_ > 0 else 0

    return {
        'sigma': sigma,
        'C_actual': C_actual,
        'L_actual': L_actual,
        'C_random': C_random,
        'L_random': L_random,
        'gamma': gamma,
        'lambda': lambda_
    }

# Cell 6: Optimization Algorithm
def optimize_threshold_for_small_world(similarity_matrix,
                                       threshold_range=(0.3, 0.95),
                                       step=0.01):
    """Find threshold that maximizes small-world coefficient"""
    thresholds = np.arange(threshold_range[0], threshold_range[1], step)
    results = []

    for threshold in tqdm(thresholds):
        G = construct_network(similarity_matrix, threshold)
        if nx.is_connected(G):  # Only consider connected graphs
            sw_metrics = calculate_small_world_coefficient(G)
            results.append({
                'threshold': threshold,
                **sw_metrics,
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'density': nx.density(G)
            })

    return pd.DataFrame(results)

# Cell 7: Dynamic Visualization
mo.md(f"""
## Network Properties at Threshold: {similarity_slider.value:.2f}

Current network has:
- {network_stats['n_nodes']} nodes
- {network_stats['n_edges']} edges
- Clustering coefficient: {network_stats['clustering']:.3f}
- Average path length: {network_stats['path_length']:.3f}
- **Small-world σ: {network_stats['sigma']:.2f}**
""")

# Cell 8: Interactive Plots
def create_interactive_dashboard():
    """Create plotly dashboard with multiple synchronized plots"""

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Similarity Distribution',
                       'Network Metrics vs Threshold',
                       'Small-World Coefficient',
                       'Network Visualization',
                       'Degree Distribution',
                       'Component Size Distribution')
    )

    # Add traces for each plot
    return fig

# Cell 9: Comparative Analysis
comparison_table = mo.ui.table(
    data=comparison_df,
    selection='single',
    label="TD vs SLI Network Properties"
)
```

---

## Part 3: Implementation Roadmap

### Phase 1: Data Preparation (Week 1)

#### Task 1.1: Cosine Similarity Distribution Analysis
```python
def analyze_similarity_distribution(embeddings):
    """Analyze distribution of cosine similarities"""

    # Compute all pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            similarities.append(sim)

    # Statistical analysis
    stats_dict = {
        'mean': np.mean(similarities),
        'median': np.median(similarities),
        'std': np.std(similarities),
        'percentiles': np.percentile(similarities, [5, 25, 50, 75, 95]),
        'skewness': stats.skew(similarities),
        'kurtosis': stats.kurtosis(similarities)
    }

    return similarities, stats_dict
```

#### Task 1.2: Efficient Similarity Matrix Computation
```python
def compute_similarity_matrix_efficient(embeddings, batch_size=1000):
    """Memory-efficient similarity computation"""
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            similarity_matrix[i:end_i, j:end_j] = cosine_similarity(
                embeddings[i:end_i],
                embeddings[j:end_j]
            )

    return similarity_matrix
```

### Phase 2: Network Construction Methods (Week 1-2)

#### Task 2.1: Multiple Network Construction Approaches
```python
class NetworkConstructor:
    def __init__(self, similarity_matrix):
        self.sim_matrix = similarity_matrix

    def threshold_based(self, threshold):
        """Classic threshold-based construction"""
        return (self.sim_matrix > threshold).astype(int)

    def knn_based(self, k=10):
        """k-nearest neighbors construction"""
        adjacency = np.zeros_like(self.sim_matrix)
        for i in range(len(self.sim_matrix)):
            # Get k most similar words
            top_k = np.argsort(self.sim_matrix[i])[-k-1:-1]
            adjacency[i, top_k] = 1
            adjacency[top_k, i] = 1  # Ensure symmetry
        return adjacency

    def adaptive_threshold(self, target_density=0.05):
        """Adapt threshold to achieve target density"""
        n = len(self.sim_matrix)
        target_edges = int(target_density * n * (n-1) / 2)

        # Find threshold that gives closest to target edges
        flat_sim = self.sim_matrix[np.triu_indices(n, k=1)]
        threshold = np.percentile(flat_sim, 100 * (1 - target_density))
        return self.threshold_based(threshold)

    def minimum_spanning_tree_plus(self, additional_edges=100):
        """MST backbone plus high-similarity edges"""
        # Create MST first
        G = nx.from_numpy_array(1 - self.sim_matrix)
        mst = nx.minimum_spanning_tree(G)

        # Add top similarity edges not in MST
        adjacency = nx.to_numpy_array(mst)
        flat_sim = self.sim_matrix[np.triu_indices(n, k=1)]
        sorted_edges = np.argsort(flat_sim)[::-1]

        added = 0
        for edge_idx in sorted_edges:
            i, j = np.triu_indices(n, k=1)
            if adjacency[i[edge_idx], j[edge_idx]] == 0:
                adjacency[i[edge_idx], j[edge_idx]] = 1
                adjacency[j[edge_idx], i[edge_idx]] = 1
                added += 1
                if added >= additional_edges:
                    break

        return adjacency
```

#### Task 2.2: Network Metric Calculations
```python
def calculate_network_metrics(G):
    """Calculate comprehensive network metrics"""

    metrics = {}

    # Basic metrics
    metrics['n_nodes'] = G.number_of_nodes()
    metrics['n_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)

    # Clustering and path length
    metrics['clustering'] = nx.average_clustering(G)
    metrics['transitivity'] = nx.transitivity(G)

    if nx.is_connected(G):
        metrics['path_length'] = nx.average_shortest_path_length(G)
        metrics['diameter'] = nx.diameter(G)
    else:
        # Use largest connected component
        Gcc = G.subgraph(max(nx.connected_components(G), key=len))
        metrics['path_length'] = nx.average_shortest_path_length(Gcc)
        metrics['diameter'] = nx.diameter(Gcc)
        metrics['largest_cc_size'] = len(Gcc)

    # Degree statistics
    degrees = [d for n, d in G.degree()]
    metrics['avg_degree'] = np.mean(degrees)
    metrics['std_degree'] = np.std(degrees)
    metrics['max_degree'] = np.max(degrees)

    # Centrality measures
    metrics['betweenness'] = np.mean(list(nx.betweenness_centrality(G).values()))
    metrics['closeness'] = np.mean(list(nx.closeness_centrality(G).values()))

    # Small-world metrics
    sw_metrics = calculate_small_world_coefficient(G)
    metrics.update(sw_metrics)

    return metrics
```

### Phase 3: Small-World Optimization (Week 2)

#### Task 3.1: Enhanced Small-World Calculation
```python
def calculate_small_world_enhanced(G, n_random=10, n_lattice=10):
    """
    Enhanced small-world calculation with confidence intervals
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Actual metrics
    C_actual = nx.average_clustering(G)
    if nx.is_connected(G):
        L_actual = nx.average_shortest_path_length(G)
    else:
        Gcc = G.subgraph(max(nx.connected_components(G), key=len))
        L_actual = nx.average_shortest_path_length(Gcc)

    # Random graph ensemble
    C_random_list = []
    L_random_list = []
    for _ in range(n_random):
        G_random = nx.gnm_random_graph(n, m)
        C_random_list.append(nx.average_clustering(G_random))
        if nx.is_connected(G_random):
            L_random_list.append(nx.average_shortest_path_length(G_random))

    C_random = np.mean(C_random_list)
    L_random = np.mean(L_random_list) if L_random_list else L_actual

    # Lattice graph for comparison
    k = int(2 * m / n)  # Average degree
    if k >= 2 and k < n:
        G_lattice = nx.watts_strogatz_graph(n, k, 0)  # p=0 for perfect lattice
        C_lattice = nx.average_clustering(G_lattice)
        L_lattice = nx.average_shortest_path_length(G_lattice)
    else:
        C_lattice = C_actual
        L_lattice = L_actual

    # Calculate metrics
    gamma = C_actual / C_random if C_random > 0 else 0
    lambda_ = L_actual / L_random if L_random > 0 else 0
    sigma = gamma / lambda_ if lambda_ > 0 else 0

    # Humphries-Gurney small-worldness
    S_HG = (C_actual / C_lattice) * (L_lattice / L_actual) if L_actual > 0 else 0

    return {
        'sigma': sigma,
        'S_HG': S_HG,
        'C_actual': C_actual,
        'L_actual': L_actual,
        'C_random': C_random,
        'L_random': L_random,
        'C_lattice': C_lattice,
        'L_lattice': L_lattice,
        'gamma': gamma,
        'lambda': lambda_,
        'C_random_std': np.std(C_random_list),
        'L_random_std': np.std(L_random_list) if L_random_list else 0
    }
```

#### Task 3.2: Optimization Algorithm
```python
def optimize_small_world_threshold(similarity_matrix,
                                  method='grid_search',
                                  constraint='connected'):
    """
    Find optimal threshold for maximum small-worldness
    """

    if method == 'grid_search':
        results = []
        thresholds = np.arange(0.3, 0.95, 0.01)

        for threshold in tqdm(thresholds, desc="Optimizing threshold"):
            G = construct_network(similarity_matrix, threshold)

            # Apply constraints
            if constraint == 'connected' and not nx.is_connected(G):
                continue
            if constraint == 'giant_component':
                Gcc = G.subgraph(max(nx.connected_components(G), key=len))
                if len(Gcc) < 0.8 * len(G):
                    continue

            metrics = calculate_small_world_enhanced(G)
            metrics['threshold'] = threshold
            results.append(metrics)

        df = pd.DataFrame(results)
        optimal_idx = df['sigma'].idxmax()
        optimal_threshold = df.loc[optimal_idx, 'threshold']

    elif method == 'golden_section':
        # Golden section search for single peak
        from scipy.optimize import golden

        def negative_small_world(threshold):
            G = construct_network(similarity_matrix, threshold)
            if not nx.is_connected(G):
                return 1e6  # Penalty for disconnected
            metrics = calculate_small_world_enhanced(G)
            return -metrics['sigma']

        optimal_threshold = golden(negative_small_world,
                                  brack=(0.3, 0.7, 0.95))

        # Get metrics at optimal
        G_optimal = construct_network(similarity_matrix, optimal_threshold)
        metrics = calculate_small_world_enhanced(G_optimal)
        df = pd.DataFrame([metrics])
        df['threshold'] = optimal_threshold

    return optimal_threshold, df
```

### Phase 4: Interactive Marimo Components (Week 2-3)

#### Task 4.1: Real-time Network Visualization
```python
@mo.cache
def create_network_visualization(G, threshold, layout='spring'):
    """Create interactive network visualization with plotly"""

    # Calculate layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G)), iterations=50)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)

    # Create edge trace
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        ))

    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=10,
            color=[G.degree(n) for n in G.nodes()],
            colorbar=dict(
                thickness=15,
                title='Node Degree',
                xanchor='left',
                titleside='right'
            )
        ),
        text=[str(n) for n in G.nodes()],
        textposition="top center"
    )

    # Create figure
    fig = go.Figure(
        data=edge_trace + [node_trace],
        layout=go.Layout(
            title=f'Network at Threshold {threshold:.2f}',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )

    return fig
```

#### Task 4.2: Dynamic Metric Dashboard
```python
def create_metric_dashboard(threshold_results_df):
    """Create interactive dashboard showing metrics across thresholds"""

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Small-World Coefficient (σ)',
            'Clustering Coefficient',
            'Average Path Length',
            'Network Density',
            'Number of Components',
            'Largest Component Size'
        )
    )

    # Small-world coefficient
    fig.add_trace(
        go.Scatter(x=threshold_results_df['threshold'],
                   y=threshold_results_df['sigma'],
                   mode='lines+markers',
                   name='σ'),
        row=1, col=1
    )

    # Add reference line at σ=1
    fig.add_hline(y=1, line_dash="dash", line_color="red",
                  annotation_text="σ=1", row=1, col=1)

    # Continue for other metrics...

    fig.update_layout(height=900, showlegend=False)
    return fig
```

#### Task 4.3: Comparative Analysis Interface
```python
def create_comparison_interface():
    """Create interface for TD vs SLI comparison"""

    # Controls
    method_selector = mo.ui.dropdown(
        options=['threshold', 'knn', 'adaptive', 'mst_plus'],
        value='threshold',
        label="Network Construction Method"
    )

    age_range = mo.ui.range_slider(
        start=5, stop=11, step=1,
        value=[5, 11],
        label="Age Range"
    )

    # Reactive computation
    @mo.reactive
    def compute_comparison():
        td_data = load_group_data('TD', age_range.value)
        sli_data = load_group_data('SLI', age_range.value)

        # Compute networks
        td_network = construct_network(td_data, method_selector.value)
        sli_network = construct_network(sli_data, method_selector.value)

        # Calculate metrics
        td_metrics = calculate_network_metrics(td_network)
        sli_metrics = calculate_network_metrics(sli_network)

        # Statistical comparison
        comparison = statistical_comparison(td_metrics, sli_metrics)

        return comparison

    return method_selector, age_range, compute_comparison
```

### Phase 5: Analysis & Reporting (Week 3)

#### Task 5.1: Statistical Analysis Functions
```python
def analyze_threshold_sensitivity(similarity_matrix, group_label):
    """Analyze how network properties change with threshold"""

    thresholds = np.arange(0.3, 0.95, 0.01)
    metrics = []

    for threshold in thresholds:
        G = construct_network(similarity_matrix, threshold)
        if G.number_of_edges() > 0:
            m = calculate_network_metrics(G)
            m['threshold'] = threshold
            m['group'] = group_label
            metrics.append(m)

    df = pd.DataFrame(metrics)

    # Find critical thresholds
    critical_points = {
        'max_small_world': df.loc[df['sigma'].idxmax(), 'threshold'],
        'percolation': find_percolation_threshold(df),
        'fragmentation': find_fragmentation_threshold(df)
    }

    return df, critical_points

def find_percolation_threshold(df):
    """Find threshold where giant component emerges"""
    # Simplified - find where largest CC > 50% of nodes
    if 'largest_cc_size' in df.columns:
        mask = df['largest_cc_size'] > 0.5 * df['n_nodes'].max()
        if mask.any():
            return df.loc[mask.idxmin(), 'threshold']
    return None

def bootstrap_small_world_confidence(G, n_bootstrap=100):
    """Bootstrap confidence intervals for small-world metrics"""

    sigmas = []
    for _ in range(n_bootstrap):
        # Resample edges
        edges = list(G.edges())
        n_sample = int(0.9 * len(edges))
        sampled_edges = np.random.choice(len(edges), n_sample, replace=True)

        G_boot = nx.Graph()
        G_boot.add_nodes_from(G.nodes())
        G_boot.add_edges_from([edges[i] for i in sampled_edges])

        if nx.is_connected(G_boot):
            metrics = calculate_small_world_coefficient(G_boot)
            sigmas.append(metrics['sigma'])

    return {
        'sigma_mean': np.mean(sigmas),
        'sigma_std': np.std(sigmas),
        'sigma_ci_lower': np.percentile(sigmas, 2.5),
        'sigma_ci_upper': np.percentile(sigmas, 97.5)
    }
```

#### Task 5.2: Report Generation
```python
def generate_analysis_report(results):
    """Generate comprehensive analysis report"""

    report = mo.md(f"""
    # Network Construction Analysis Report

    ## 1. Cosine Similarity Distribution

    - Mean similarity: {results['sim_stats']['mean']:.3f}
    - Median similarity: {results['sim_stats']['median']:.3f}
    - Standard deviation: {results['sim_stats']['std']:.3f}
    - 95th percentile: {results['sim_stats']['p95']:.3f}

    ## 2. Optimal Thresholds

    | Criterion | Optimal Threshold | Small-World σ |
    |-----------|------------------|---------------|
    | Maximum σ | {results['optimal_sigma_threshold']:.3f} | {results['max_sigma']:.2f} |
    | Target density 5% | {results['density_threshold']:.3f} | {results['density_sigma']:.2f} |
    | Percolation point | {results['percolation_threshold']:.3f} | {results['percolation_sigma']:.2f} |

    ## 3. Group Comparison (TD vs SLI)

    At optimal threshold {results['optimal_sigma_threshold']:.3f}:

    | Metric | TD | SLI | p-value | Cohen's d |
    |--------|-------|--------|----------|--------|
    | Small-world σ | {results['td_sigma']:.2f} | {results['sli_sigma']:.2f} | {results['p_sigma']:.4f} | {results['d_sigma']:.2f} |
    | Clustering | {results['td_clustering']:.3f} | {results['sli_clustering']:.3f} | {results['p_clustering']:.4f} | {results['d_clustering']:.2f} |
    | Path length | {results['td_path_length']:.2f} | {results['sli_path_length']:.2f} | {results['p_path_length']:.4f} | {results['d_path_length']:.2f} |

    ## 4. Recommendations

    {generate_recommendations(results)}
    """)

    return report

def generate_recommendations(results):
    """Generate recommendations based on analysis"""
    recs = []

    if results['max_sigma'] > 3:
        recs.append("- Strong small-world properties detected (σ > 3)")

    if abs(results['td_sigma'] - results['sli_sigma']) > 0.5:
        recs.append("- Significant difference in small-world structure between groups")

    if results['optimal_sigma_threshold'] < 0.5:
        recs.append("- Low optimal threshold suggests sparse semantic relationships")

    return '\n'.join(recs)
```

---

## Part 4: Marimo-Specific Implementation

### Interactive Controls Setup
```python
# marimo_network_analysis.py

import marimo as mo
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Cell 1: Controls
threshold_slider = mo.ui.slider(
    start=0.30,
    stop=0.95,
    step=0.01,
    value=0.70,
    label="Cosine Similarity Threshold",
    show_value=True
)

construction_method = mo.ui.dropdown(
    options=[
        {'label': 'Threshold-based', 'value': 'threshold'},
        {'label': 'k-Nearest Neighbors', 'value': 'knn'},
        {'label': 'Adaptive Density', 'value': 'adaptive'},
        {'label': 'MST + Edges', 'value': 'mst_plus'}
    ],
    value='threshold',
    label="Network Construction Method"
)

group_comparison = mo.ui.checkbox(
    label="Compare TD vs SLI",
    value=True
)

auto_optimize = mo.ui.button(
    label="Find Optimal Threshold",
    kind="success"
)

# Cell 2: Reactive Network Construction
network_state = mo.state({
    'threshold': 0.7,
    'method': 'threshold',
    'network': None,
    'metrics': {}
})

@mo.memo
def construct_current_network():
    """Construct network based on current settings"""
    threshold = threshold_slider.value
    method = construction_method.value

    # Load similarity matrix (cached)
    sim_matrix = load_similarity_matrix()

    # Construct network
    if method == 'threshold':
        adjacency = (sim_matrix > threshold).astype(int)
    elif method == 'knn':
        k = int(10 * threshold / 0.7)  # Scale k with threshold
        adjacency = knn_construction(sim_matrix, k)
    # ... other methods

    G = nx.from_numpy_array(adjacency)

    # Calculate metrics
    metrics = calculate_network_metrics(G)
    metrics['small_world'] = calculate_small_world_enhanced(G)

    network_state.set({
        'threshold': threshold,
        'method': method,
        'network': G,
        'metrics': metrics
    })

    return G, metrics

# Cell 3: Dynamic Visualization
mo.md(f"""
### Current Network Properties

**Threshold:** {threshold_slider.value:.3f}
**Method:** {construction_method.value}

| Metric | Value |
|--------|--------|
| Nodes | {network_state.value['metrics'].get('n_nodes', 0)} |
| Edges | {network_state.value['metrics'].get('n_edges', 0)} |
| Density | {network_state.value['metrics'].get('density', 0):.4f} |
| **Small-World σ** | **{network_state.value['metrics'].get('small_world', {}).get('sigma', 0):.2f}** |
| Clustering | {network_state.value['metrics'].get('clustering', 0):.3f} |
| Path Length | {network_state.value['metrics'].get('path_length', 0):.2f} |
""")

# Cell 4: Interactive Plots
@mo.memo
def create_dashboard():
    """Create interactive dashboard"""

    # Get sweep data
    sweep_df = sweep_thresholds()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Small-World Coefficient',
                       'Network Visualization',
                       'Similarity Distribution',
                       'Component Analysis')
    )

    # Plot 1: Small-world across thresholds
    fig.add_trace(
        go.Scatter(
            x=sweep_df['threshold'],
            y=sweep_df['sigma'],
            mode='lines',
            name='σ'
        ),
        row=1, col=1
    )

    # Add current threshold marker
    current_metrics = network_state.value['metrics'].get('small_world', {})
    fig.add_trace(
        go.Scatter(
            x=[threshold_slider.value],
            y=[current_metrics.get('sigma', 0)],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Current'
        ),
        row=1, col=1
    )

    # Plot 2: Network visualization
    G = network_state.value['network']
    if G and len(G) > 0:
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G)))

        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=0.5, color='gray'),
                    showlegend=False,
                    hoverinfo='none'
                ),
                row=1, col=2
            )

        # Add nodes
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_colors = [G.degree(n) for n in G.nodes()]

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                marker=dict(
                    size=8,
                    color=node_colors,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f'Node {n}<br>Degree: {G.degree(n)}'
                      for n in G.nodes()],
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=2
        )

    # Plot 3: Similarity distribution with threshold
    sim_flat = get_similarity_distribution()
    fig.add_trace(
        go.Histogram(
            x=sim_flat,
            nbinsx=50,
            name='Distribution',
            opacity=0.7
        ),
        row=2, col=1
    )

    # Add threshold line
    fig.add_vline(
        x=threshold_slider.value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current: {threshold_slider.value:.2f}",
        row=2, col=1
    )

    # Plot 4: Connected components
    if G:
        cc_sizes = [len(c) for c in nx.connected_components(G)]
        fig.add_trace(
            go.Bar(
                x=list(range(len(cc_sizes))),
                y=sorted(cc_sizes, reverse=True),
                name='Component Sizes'
            ),
            row=2, col=2
        )

    fig.update_layout(
        height=800,
        showlegend=False,
        title_text=f"Network Analysis Dashboard"
    )

    return fig

dashboard = create_dashboard()
mo.ui.plotly(dashboard)

# Cell 5: Optimization Results
if auto_optimize.value:
    mo.md("## Optimization Running...")

    optimal_threshold, optimization_df = optimize_small_world_threshold(
        similarity_matrix,
        method='grid_search'
    )

    mo.md(f"""
    ## Optimization Complete!

    **Optimal Threshold:** {optimal_threshold:.3f}
    **Maximum Small-World σ:** {optimization_df['sigma'].max():.2f}

    This threshold provides the best balance between:
    - High clustering (local structure)
    - Low path length (global connectivity)

    Click "Apply Optimal" to use this threshold.
    """)

    apply_button = mo.ui.button("Apply Optimal", kind="primary")

    if apply_button.value:
        threshold_slider.set_value(optimal_threshold)

# Cell 6: Export Functionality
export_format = mo.ui.dropdown(
    options=['CSV', 'JSON', 'GraphML'],
    value='CSV',
    label="Export Format"
)

export_button = mo.ui.button("Export Results", kind="secondary")

if export_button.value:
    if export_format.value == 'CSV':
        results_df = pd.DataFrame([network_state.value['metrics']])
        results_df.to_csv('network_analysis_results.csv')
        mo.md("✓ Exported to network_analysis_results.csv")
    elif export_format.value == 'GraphML':
        nx.write_graphml(network_state.value['network'],
                        'network.graphml')
        mo.md("✓ Exported to network.graphml")
```

---

## Part 5: Testing & Validation Plan

### Unit Tests
```python
# test_network_analysis.py

def test_small_world_calculation():
    """Test small-world coefficient calculation"""
    # Create known small-world network
    G = nx.watts_strogatz_graph(100, 6, 0.3)
    sw = calculate_small_world_coefficient(G)

    assert sw['sigma'] > 1.5, "Small-world network should have σ > 1.5"
    assert sw['gamma'] > 1, "Clustering should be higher than random"

def test_threshold_optimization():
    """Test threshold optimization"""
    # Create synthetic similarity matrix
    n = 50
    embeddings = np.random.randn(n, 100)
    sim_matrix = cosine_similarity(embeddings)

    optimal, df = optimize_small_world_threshold(sim_matrix)

    assert 0.3 <= optimal <= 0.95, "Optimal should be in range"
    assert len(df) > 0, "Should return results dataframe"

def test_network_construction_methods():
    """Test different construction methods"""
    sim_matrix = np.random.rand(30, 30)
    sim_matrix = (sim_matrix + sim_matrix.T) / 2  # Symmetrize

    constructor = NetworkConstructor(sim_matrix)

    # Test each method
    methods = ['threshold', 'knn', 'adaptive', 'mst_plus']
    for method in methods:
        if method == 'threshold':
            adj = constructor.threshold_based(0.5)
        elif method == 'knn':
            adj = constructor.knn_based(5)
        # ... etc

        assert adj.shape == (30, 30), f"{method} should return correct shape"
        assert np.allclose(adj, adj.T), f"{method} should be symmetric"
```

### Performance Benchmarks
```python
def benchmark_construction_methods():
    """Benchmark different network construction methods"""
    import time

    sizes = [100, 500, 1000]
    methods = ['threshold', 'knn', 'adaptive']

    results = []
    for n in sizes:
        sim_matrix = np.random.rand(n, n)
        sim_matrix = (sim_matrix + sim_matrix.T) / 2

        constructor = NetworkConstructor(sim_matrix)

        for method in methods:
            start = time.time()

            if method == 'threshold':
                constructor.threshold_based(0.5)
            elif method == 'knn':
                constructor.knn_based(10)
            elif method == 'adaptive':
                constructor.adaptive_threshold(0.05)

            elapsed = time.time() - start
            results.append({
                'n': n,
                'method': method,
                'time': elapsed
            })

    return pd.DataFrame(results)
```

---

## Part 6: Deliverables & Timeline

### Week 1 Deliverables
1. ✅ Similarity distribution analysis functions
2. ✅ Basic network construction methods
3. ✅ Small-world coefficient calculation
4. ✅ Initial marimo notebook structure

### Week 2 Deliverables
1. ✅ Threshold optimization algorithm
2. ✅ Interactive marimo controls
3. ✅ Real-time visualization updates
4. ✅ TD vs SLI comparison framework

### Week 3 Deliverables
1. ✅ Complete dashboard implementation
2. ✅ Statistical analysis integration
3. ✅ Export functionality
4. ✅ Comprehensive testing
5. ✅ Documentation and report generation

### Final Package Structure
```
network_optimization/
├── marimo_network_analysis.py     # Main marimo notebook
├── network_construction.py        # Construction methods
├── small_world_metrics.py         # Metric calculations
├── optimization.py                 # Threshold optimization
├── visualization.py                # Plotting functions
├── statistical_analysis.py        # Group comparisons
├── tests/
│   ├── test_construction.py
│   ├── test_metrics.py
│   └── test_optimization.py
├── data/
│   ├── similarity_matrices/       # Precomputed matrices
│   └── results/                   # Analysis outputs
└── README.md                      # Documentation
```

---

## Conclusion

This comprehensive plan provides:
1. **Theoretical foundation** for small-world analysis
2. **Multiple network construction methods** beyond simple thresholding
3. **Optimization algorithms** to find ideal parameters
4. **Interactive marimo interface** for real-time exploration
5. **Statistical framework** for group comparisons
6. **Complete implementation** with testing and documentation

The marimo notebook will enable dynamic exploration of how cosine similarity thresholds affect network topology, with particular focus on maximizing small-world properties that are characteristic of efficient language networks.