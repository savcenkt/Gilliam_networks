# Gilliam Networks: Semantic Network Analysis of Child Language Development

A comprehensive pipeline for analyzing semantic networks in child language development using BERT embeddings to distinguish between typically developing (TD) and specific language impairment (SLI) children.

## 🎯 Project Overview

This project analyzes semantic networks constructed from child narratives to identify differences in language organization between typically developing children and those with specific language impairment. Using state-of-the-art BERT embeddings and network science, we explore whether semantic network properties can provide diagnostic value beyond traditional language metrics.

### Key Features
- 📊 Analyzes **668 child narratives** from the Gilliam corpus (497 TD, 171 SLI)
- 🤖 Generates BERT embeddings for semantic representation
- 🕸️ Constructs semantic networks using multiple methods
- 📈 Calculates small-world network properties
- 🔬 Performs statistical comparisons between groups
- 🎨 Interactive visualization with Marimo notebooks

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Gilliam_networks.git
cd Gilliam_networks

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run complete pipeline (all phases)
python scripts/run_pipeline.py --phase all

# Or run individual phases
python scripts/run_pipeline.py --phase prepare   # Data preparation
python scripts/run_pipeline.py --phase embed     # Generate embeddings
python scripts/run_pipeline.py --phase network   # Construct networks
python scripts/run_pipeline.py --phase analyze   # Analyze results

# With custom configuration
python scripts/run_pipeline.py --config my_config.yaml --verbose
```

### Using Make Commands

```bash
make setup        # Install dependencies
make test         # Run tests
make run-all      # Run complete pipeline
make run-embed    # Generate embeddings only
make run-analyze  # Run analysis only
make explore      # Launch interactive Marimo notebook
make clean        # Clean generated files
```

## 📁 Project Structure

```
Gilliam_networks/
├── README.md                 # This file
├── config.yaml              # Main configuration
├── requirements.txt         # Dependencies
├── Makefile                # Automation commands
├── scripts/                 # Entry point scripts
│   └── run_pipeline.py     # Main pipeline runner
├── src/                    # Source code
│   ├── pipeline/          # Core pipeline modules
│   ├── utils/            # Utility functions
│   └── interactive/      # Interactive tools
├── data/                  # Data directory
│   ├── raw/              # Original data
│   ├── processed/        # Generated embeddings/networks
│   └── results/          # Analysis outputs
├── docs/                 # Documentation
└── tests/               # Test suite
```

## 🔄 Pipeline Phases

### 1. Data Preparation
- Loads child narrative texts from the Gilliam corpus
- Validates metadata (age, gender, development type)
- Calculates Mean Length of Utterance (MLU)

### 2. Embedding Generation
- Generates BERT embeddings for each child's narrative
- Produces word-level and sentence-level representations
- Saves embeddings in efficient format

### 3. Network Construction
- Builds semantic networks from word embeddings
- Supports multiple construction methods:
  - Threshold-based
  - k-Nearest Neighbors
  - Adaptive density
  - Minimum Spanning Tree + edges

### 4. Analysis & Visualization
- Calculates small-world network properties
- Compares TD vs SLI groups statistically
- Generates visualizations and reports
- Exports results in multiple formats

## 📊 Key Results

- **Small-world properties**: Both TD and SLI networks exhibit small-world characteristics (σ > 1)
- **Group differences**: Significant differences in network organization between groups
- **Age effects**: Network properties evolve with age
- **Clinical relevance**: Network measures provide additional diagnostic value beyond traditional MLU

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
embeddings:
  model: "bert-base-uncased"    # BERT model to use
  batch_size: 8                 # Batch size for processing

network:
  similarity_threshold: 0.7      # Cosine similarity threshold
  construction_method: "threshold"  # Network construction method

analysis:
  bootstrap_iterations: 1000    # Bootstrap sampling iterations
  significance_level: 0.05      # Statistical significance level
```

## 📚 Documentation

Detailed documentation available in `docs/`:

- [Data Preparation Guide](docs/01_data_preparation.md)
- [Embedding Generation](docs/02_embedding_generation.md)
- [Network Analysis](docs/03_network_analysis_plan.md)
- [API Reference](docs/api_reference.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_pipeline.py -v
```

## 🎯 Interactive Exploration

Launch the Marimo interactive notebook for real-time exploration:

```bash
make explore
# Or directly:
marimo run src/interactive/marimo_explorer.py
```

Features:
- Dynamic threshold adjustment
- Real-time network visualization
- Group comparison interface
- Export capabilities

## 📈 Results & Outputs

Results are saved in `data/results/`:

- **Tables**: Network metrics, group comparisons (`data/results/tables/`)
- **Figures**: Visualizations and plots (`data/results/figures/`)
- **Reports**: Markdown analysis reports (`data/results/reports/`)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{gilliam_networks,
  title = {Gilliam Networks: Semantic Network Analysis of Child Language},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Gilliam_networks}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📮 Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub Issues: [Project Issues](https://github.com/yourusername/Gilliam_networks/issues)

## 🙏 Acknowledgments

- Gilliam corpus providers for the child narrative data
- HuggingFace for BERT models and transformers library
- NetworkX developers for network analysis tools

---

**Version**: 2.0.0 | **Last Updated**: November 2024