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

## Next Steps for Your Research

After generating embeddings:

1. **Construct semantic networks** using similarity matrices
2. **Calculate network metrics** (clustering coefficient, path length, hub prominence)
3. **Compare TD vs SLI** at different age levels
4. **Match by MLU** for controlled comparisons
5. **Test hypothesis** about small-world network disruption in SLI

The embedding utilities provide functions for these analyses. You can extend them for your specific network metrics and statistical tests.

## Notes

- The script preserves child identity in filenames
- Embeddings are saved individually for maximum flexibility
- Checkpointing ensures no data loss from interruptions
- Word-level embeddings handle subword tokenization properly
- Both word and sentence embeddings are saved for different analyses