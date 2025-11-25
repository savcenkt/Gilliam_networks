# BERT Embedding Generation for Gillam Corpus

This set of scripts generates BERT embeddings for individual child narratives from the Gillam corpus for semantic network analysis.

## Overview

The pipeline processes 668 individual child narrative files to generate word-level and sentence-level BERT embeddings. These embeddings will be used to construct semantic networks for comparing typical language development (TD) vs specific language impairment (SLI) children.

## Scripts

### 1. `generate_bert_embeddings.py`
Main script for generating embeddings from child narrative texts.

**Features:**
- Processes individual child files preserving identity
- Generates word-level and sentence-level embeddings
- Includes checkpointing for resuming interrupted processing
- Optimizes GPU/CPU usage with batch processing
- Comprehensive logging and progress tracking
- Handles subword tokenization properly

### 2. `test_embeddings.py`
Test script to validate the pipeline on a small subset before full processing.

### 3. `embedding_utils.py`
Utility functions for loading and analyzing generated embeddings.

**Features:**
- Load embeddings by child, age, or development type
- Calculate network statistics (clustering coefficient, hub words)
- Compare TD vs SLI groups
- Visualize semantic networks

## Installation

First, install required dependencies:

```bash
pip install torch transformers pandas numpy tqdm scikit-learn matplotlib seaborn
```

For network visualization (optional):
```bash
pip install networkx
```

## Usage

### Step 1: Test the Pipeline

First, test on a small subset to ensure everything works:

```bash
python test_embeddings.py
```

This will process 4 files (2 TD, 2 SLI) and verify the output.

### Step 2: Generate All Embeddings

Run the main generation script:

```bash
python generate_bert_embeddings.py
```

**With custom parameters:**

```bash
python generate_bert_embeddings.py \
    --input-dir embedding_ready \
    --output-dir embeddings \
    --model bert-base-uncased \
    --batch-size 16 \
    --device cuda
```

**Parameters:**
- `--input-dir`: Directory containing individual child text files (default: `embedding_ready`)
- `--output-dir`: Directory to save embeddings (default: `embeddings`)
- `--model`: HuggingFace model to use (default: `bert-base-uncased`)
- `--batch-size`: Batch size for processing (default: 8)
- `--device`: Device to use - cuda/cpu (auto-detects if not specified)
- `--checkpoint-dir`: Directory for checkpoints (default: `checkpoints`)

### Step 3: Analyze Embeddings

Use the utility functions to analyze generated embeddings:

```python
from embedding_utils import EmbeddingLoader

# Initialize loader
loader = EmbeddingLoader(embeddings_dir="embeddings")

# Load embeddings for a specific group
td_age7 = loader.load_group_embeddings(development_type='TD', age=7)
sli_age7 = loader.load_group_embeddings(development_type='SLI', age=7)

# Compare groups
comparison = loader.compare_groups(td_age7, sli_age7)
print(f"TD Clustering: {comparison['TD']['clustering_coefficient']:.3f}")
print(f"SLI Clustering: {comparison['SLI']['clustering_coefficient']:.3f}")
```

## Output Structure

```
embeddings/
├── {child_id}_embeddings.pkl          # Embeddings (pickle format)
├── {child_id}_embeddings_metadata.json # Metadata for each file
└── embedding_generation_report_*.json  # Summary report

checkpoints/
├── embedding_checkpoint.json          # Resume checkpoint
└── embedding_generation_*.log         # Detailed logs
```

## Embedding Format

Each pickle file contains:
- `word_embeddings`: numpy array of word-level embeddings (n_words × 768)
- `word_tokens`: list of words corresponding to embeddings
- `sentence_embedding`: single sentence-level embedding (768,)
- `full_sequence`: full BERT sequence output including special tokens
- `file_name`: original filename
- `text`: original text
- `timestamp`: generation timestamp
- `model`: model used

## Resource Estimates

- **Time**: ~1-2 seconds per file on GPU, ~3-5 seconds on CPU
- **Total time**: ~20-30 minutes on GPU, ~60-90 minutes on CPU
- **Memory**: ~2-4 GB GPU memory, ~4-6 GB system RAM
- **Storage**: ~500 KB per child (total ~350 MB for all embeddings)

## Monitoring Progress

The script provides multiple ways to monitor progress:

1. **Console output**: Real-time progress bar with statistics
2. **Log file**: Detailed logs in `checkpoints/embedding_generation_*.log`
3. **Checkpoint file**: `checkpoints/embedding_checkpoint.json` shows processed files

## Resuming Interrupted Processing

If processing is interrupted, simply run the script again:
```bash
python generate_bert_embeddings.py
```

The script will automatically resume from the last checkpoint.

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python generate_bert_embeddings.py --batch-size 4
```

### CUDA Not Available
The script will automatically fall back to CPU. To explicitly use CPU:
```bash
python generate_bert_embeddings.py --device cpu
```

### Failed Files
Check the log file for details about failed files. The final report will list all failed files.

## Embedding Analysis Utilities

The `embedding_utils.py` module provides comprehensive tools for analyzing the generated embeddings and constructing semantic networks. Here's a detailed guide to all available functions:

### 1. EmbeddingLoader Class

The main class for loading and managing embeddings:

```python
from embedding_utils import EmbeddingLoader

# Initialize the loader
loader = EmbeddingLoader(
    embeddings_dir="embeddings",           # Directory with .pkl files
    metadata_path="embedding_ready/metadata.csv"  # Child metadata
)
```

### 2. Loading Embeddings

#### Load Individual Child
```python
# Load embeddings for a specific child
child_emb = loader.load_child_embeddings("td_7y6m_f_46076v.txt")

# Access embedding data
word_embeddings = child_emb['word_embeddings']  # Shape: (n_words, 768)
word_tokens = child_emb['word_tokens']          # List of words
sentence_emb = child_emb['sentence_embedding']  # Shape: (768,)
metadata = child_emb['metadata']                # Child info (age, MLU, etc.)
```

#### Load Groups of Children
```python
# Load all TD children
td_all = loader.load_group_embeddings(development_type='TD')

# Load all SLI children aged 7
sli_age7 = loader.load_group_embeddings(development_type='SLI', age=7)

# Load all children aged 8 (both TD and SLI)
age8_all = loader.load_group_embeddings(age=8)

# The result is a dictionary: {filename: embeddings_dict}
print(f"Loaded {len(td_all)} TD children")
```

### 3. Vocabulary Analysis

#### Extract Combined Vocabulary
```python
# Get unique words across a group
vocab = loader.get_vocabulary(td_all)
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample words: {vocab[:10]}")
```

#### Create Word Embedding Matrix
```python
# Create averaged word embeddings across children
embedding_matrix, word_list = loader.create_word_embedding_matrix(td_all)

# embedding_matrix: Shape (n_unique_words, 768)
# word_list: List of words corresponding to matrix rows
print(f"Matrix shape: {embedding_matrix.shape}")
print(f"Unique words: {len(word_list)}")
```

### 4. Semantic Network Construction

#### Compute Similarity Matrix
```python
# Calculate cosine similarities between all word pairs
similarity_matrix = loader.compute_similarity_matrix(embedding_matrix)

# similarity_matrix: Shape (n_words, n_words), values in [0, 1]
print(f"Similarity matrix shape: {similarity_matrix.shape}")
print(f"Mean similarity: {similarity_matrix.mean():.3f}")
```

### 5. Network Analysis Metrics

#### Identify Hub Words
```python
# Find words with many semantic connections
hub_words = loader.find_hub_words(
    similarity_matrix,
    word_list,
    threshold=0.7,    # Similarity threshold for connection
    top_n=20          # Number of top hubs to return
)

print("\nTop 10 Hub Words:")
print(hub_words.head(10))
# Returns DataFrame with columns: word, connections, avg_similarity
```

#### Calculate Clustering Coefficient
```python
# Measure local clustering in the semantic network
clustering_coeff = loader.calculate_clustering_coefficient(
    similarity_matrix,
    threshold=0.7
)

print(f"Clustering coefficient: {clustering_coeff:.3f}")
# Higher values indicate more "small-world" structure
```

### 6. Group Comparisons

#### Compare TD vs SLI Networks
```python
# Comprehensive comparison of network properties
comparison = loader.compare_groups(td_all, sli_all)

# Access results
print(f"TD clustering: {comparison['TD']['clustering_coefficient']:.3f}")
print(f"SLI clustering: {comparison['SLI']['clustering_coefficient']:.3f}")
print(f"Difference: {comparison['differences']['clustering_diff']:.3f}")

print(f"\nTD hub connectivity: {comparison['TD']['hub_connectivity_mean']:.1f}")
print(f"SLI hub connectivity: {comparison['SLI']['hub_connectivity_mean']:.1f}")

print(f"\nShared top hubs: {comparison['differences']['shared_top_hubs']}")
```

### 7. Network Visualization

```python
from embedding_utils import visualize_semantic_network

# Visualize semantic network (requires networkx)
visualize_semantic_network(
    similarity_matrix,
    word_list,
    threshold=0.7,      # Edge threshold
    max_words=50,       # Limit for clarity
    title="TD Semantic Network - Age 7"
)
```

### 8. Advanced Analysis Examples

#### MLU-Matched Comparison
```python
# Load metadata
import pandas as pd
metadata = pd.read_csv("embedding_ready/metadata.csv")

# Find MLU-matched pairs
target_mlu = 7.5
mlu_tolerance = 0.5

# Get TD children near target MLU
td_matched = metadata[
    (metadata['development_type'] == 'TD') &
    (metadata['mlu'].between(target_mlu - mlu_tolerance, target_mlu + mlu_tolerance))
]

# Get SLI children near target MLU
sli_matched = metadata[
    (metadata['development_type'] == 'SLI') &
    (metadata['mlu'].between(target_mlu - mlu_tolerance, target_mlu + mlu_tolerance))
]

# Load their embeddings
td_matched_emb = {row['filename']: loader.load_child_embeddings(row['filename'])
                   for _, row in td_matched.iterrows()}
sli_matched_emb = {row['filename']: loader.load_child_embeddings(row['filename'])
                    for _, row in sli_matched.iterrows()}

# Compare MLU-matched groups
mlu_comparison = loader.compare_groups(td_matched_emb, sli_matched_emb)
```

#### Age Progression Analysis
```python
# Analyze how networks change with age
age_results = {}

for age in range(5, 12):
    td_age = loader.load_group_embeddings(development_type='TD', age=age)

    if len(td_age) > 0:
        matrix, words = loader.create_word_embedding_matrix(td_age)
        similarity = loader.compute_similarity_matrix(matrix)
        clustering = loader.calculate_clustering_coefficient(similarity)

        age_results[age] = {
            'n_children': len(td_age),
            'vocab_size': len(words),
            'clustering': clustering
        }

# Plot progression
import matplotlib.pyplot as plt

ages = list(age_results.keys())
clustering_values = [age_results[a]['clustering'] for a in ages]

plt.figure(figsize=(10, 6))
plt.plot(ages, clustering_values, marker='o')
plt.xlabel('Age (years)')
plt.ylabel('Clustering Coefficient')
plt.title('Semantic Network Clustering by Age (TD Children)')
plt.grid(True)
plt.show()
```

#### Word Frequency Weighting
```python
# Weight embeddings by word frequency
from collections import Counter

def get_weighted_embeddings(embeddings_dict):
    """Create frequency-weighted word embeddings."""
    word_counts = Counter()
    word_embeddings = {}

    # Count word frequencies across children
    for child_emb in embeddings_dict.values():
        for word in child_emb['word_tokens']:
            word_counts[word] += 1

    # Collect and weight embeddings
    for child_emb in embeddings_dict.values():
        for word, emb in zip(child_emb['word_tokens'], child_emb['word_embeddings']):
            if word not in word_embeddings:
                word_embeddings[word] = []
            # Weight by frequency
            weight = np.log(1 + word_counts[word])
            word_embeddings[word].append(emb * weight)

    # Average weighted embeddings
    word_list = []
    embedding_list = []

    for word, emb_list in sorted(word_embeddings.items()):
        word_list.append(word)
        avg_emb = np.mean(emb_list, axis=0)
        embedding_list.append(avg_emb)

    return np.array(embedding_list), word_list
```

#### Path Length Analysis
```python
import networkx as nx

def calculate_path_lengths(similarity_matrix, word_list, threshold=0.7):
    """Calculate average shortest path length in semantic network."""
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(len(word_list)))

    # Add edges above threshold
    for i in range(len(word_list)):
        for j in range(i+1, len(word_list)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j)

    # Calculate path lengths for connected component
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        # Use largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        avg_path_length = nx.average_shortest_path_length(subgraph)

    return avg_path_length

# Example usage
path_length = calculate_path_lengths(similarity_matrix, word_list)
print(f"Average path length: {path_length:.2f}")
```

#### Statistical Testing
```python
from scipy import stats

def compare_network_metrics(td_embeddings, sli_embeddings, n_bootstrap=1000):
    """Statistical comparison using bootstrap resampling."""
    td_clustering_samples = []
    sli_clustering_samples = []

    # Bootstrap sampling
    for _ in range(n_bootstrap):
        # Sample with replacement
        td_sample = np.random.choice(list(td_embeddings.keys()),
                                    size=len(td_embeddings),
                                    replace=True)
        sli_sample = np.random.choice(list(sli_embeddings.keys()),
                                     size=len(sli_embeddings),
                                     replace=True)

        # Create sampled dictionaries
        td_dict = {k: td_embeddings[k] for k in td_sample}
        sli_dict = {k: sli_embeddings[k] for k in sli_sample}

        # Calculate metrics
        td_matrix, td_words = loader.create_word_embedding_matrix(td_dict)
        td_sim = loader.compute_similarity_matrix(td_matrix)
        td_clust = loader.calculate_clustering_coefficient(td_sim)

        sli_matrix, sli_words = loader.create_word_embedding_matrix(sli_dict)
        sli_sim = loader.compute_similarity_matrix(sli_matrix)
        sli_clust = loader.calculate_clustering_coefficient(sli_sim)

        td_clustering_samples.append(td_clust)
        sli_clustering_samples.append(sli_clust)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(td_clustering_samples, sli_clustering_samples)

    return {
        'td_mean': np.mean(td_clustering_samples),
        'td_std': np.std(td_clustering_samples),
        'sli_mean': np.mean(sli_clustering_samples),
        'sli_std': np.std(sli_clustering_samples),
        't_statistic': t_stat,
        'p_value': p_value
    }
```

### 9. Custom Network Metrics

You can extend the utilities with your own metrics:

```python
def calculate_modularity(similarity_matrix, word_list, threshold=0.7):
    """Calculate network modularity using community detection."""
    import networkx as nx
    from networkx.algorithms import community

    # Create graph
    G = nx.Graph()
    for i in range(len(word_list)):
        for j in range(i+1, len(word_list)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(word_list[i], word_list[j],
                          weight=similarity_matrix[i, j])

    # Detect communities
    communities = community.greedy_modularity_communities(G)
    modularity = community.modularity(G, communities)

    return modularity, communities

def calculate_degree_distribution(similarity_matrix, threshold=0.7):
    """Calculate degree distribution for scale-free analysis."""
    degrees = np.sum(similarity_matrix > threshold, axis=0) - 1
    degree_counts = np.bincount(degrees)

    return degrees, degree_counts
```

## Next Steps for Your Research

With these utilities, you can:

1. **Test your small-world hypothesis**: Compare clustering coefficients and path lengths between TD and SLI groups
2. **Identify disrupted hubs**: Analyze which hub words differ between groups
3. **Control for MLU**: Use MLU-matched comparisons to isolate network effects
4. **Track development**: Analyze how network structure changes with age
5. **Statistical validation**: Use bootstrap resampling for robust comparisons
6. **Extend analysis**: Add custom metrics specific to your theoretical framework

The utilities are designed to be modular - you can combine them in various ways to test your specific hypotheses about semantic network disruption in SLI.

## Notes

- The script preserves child identity in filenames
- Embeddings are saved individually for maximum flexibility
- Checkpointing ensures no data loss from interruptions
- Word-level embeddings handle subword tokenization properly
- Both word and sentence embeddings are saved for different analyses