"""
Wrapper for the embedding generation module.

This wraps the existing generate_bert_embeddings.py functionality
into a class-based interface for the new pipeline structure.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate BERT embeddings for child narratives.

    This is a wrapper around the existing embedding generation code.
    """

    def __init__(self, config: Dict):
        """
        Initialize embedding generator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.model_name = config['embeddings']['model']
        self.batch_size = config['embeddings']['batch_size']
        self.device = config['embeddings']['device']
        self.checkpoint_dir = Path(config['paths']['checkpoints'])

        # Import the existing module
        sys.path.append(str(Path(__file__).parent))
        from embedding_generation import BERTEmbeddingGenerator

        self.generator = BERTEmbeddingGenerator

    def run(self, input_dir: str, output_dir: str, force: bool = False) -> Dict:
        """
        Run embedding generation.

        Args:
            input_dir: Directory containing text files
            output_dir: Directory to save embeddings
            force: Force re-generation even if embeddings exist

        Returns:
            Dictionary with generation statistics
        """
        logger.info("Starting embedding generation")

        # Convert paths
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Check if embeddings already exist and force is False
        if not force and output_path.exists() and len(list(output_path.glob("*.pkl"))) > 0:
            logger.info("Embeddings already exist. Use --force to regenerate.")
            return {"status": "skipped", "message": "Embeddings already exist"}

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize generator with configuration
            generator = self.generator(
                input_dir=input_path,
                output_dir=output_path,
                model_name=self.model_name,
                batch_size=self.batch_size,
                device=self.device,
                checkpoint_dir=self.checkpoint_dir
            )

            # Run generation
            results = generator.generate_all()

            logger.info(f"Embedding generation complete: {results['processed']} files processed")
            return results

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise