#!/usr/bin/env python3
"""
Generate BERT embeddings for individual child narratives from the Gillam corpus.
This script processes child language narratives to create word embeddings for
semantic network analysis comparing typical development (TD) and specific
language impairment (SLI) children.

Author: [Your Name]
Date: October 2024
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    logging as transformers_logging
)
from tqdm import tqdm
import pickle

# Set transformers logging to error only to reduce noise
transformers_logging.set_verbosity_error()

class ChildNarrativeDataset(Dataset):
    """Dataset class for loading child narrative texts."""

    def __init__(self, file_paths: List[Path], texts: List[str]):
        self.file_paths = file_paths
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'file_path': self.file_paths[idx],
            'text': self.texts[idx]
        }

class BERTEmbeddingGenerator:
    """
    Generate BERT embeddings for child language narratives.
    Optimized for resource efficiency and robustness.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 8,
        device: str = None,
        checkpoint_dir: str = "checkpoints",
        output_dir: str = "embeddings"
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for processing
            device: Device to use (cuda/cpu)
            checkpoint_dir: Directory for saving checkpoints
            output_dir: Directory for saving embeddings
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)

        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup logging
        self.setup_logging()

        # Load model and tokenizer
        self.logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"Model loaded successfully. Using device: {self.device}")

    def setup_logging(self):
        """Setup comprehensive logging system."""
        log_file = self.checkpoint_dir / f"embedding_generation_{datetime.now():%Y%m%d_%H%M%S}.log"

        # Create logger
        self.logger = logging.getLogger("BERTEmbeddings")
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("="*80)
        self.logger.info("BERT Embedding Generation for Gillam Corpus")
        self.logger.info("="*80)

    def load_checkpoint(self) -> Dict:
        """Load checkpoint if it exists."""
        checkpoint_file = self.checkpoint_dir / "embedding_checkpoint.json"

        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            self.logger.info(f"Checkpoint loaded: {len(checkpoint['processed_files'])} files already processed")
            return checkpoint

        return {"processed_files": [], "failed_files": []}

    def save_checkpoint(self, checkpoint: Dict):
        """Save checkpoint for resuming."""
        checkpoint_file = self.checkpoint_dir / "embedding_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def get_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a single text.
        Returns word-level and sentence-level embeddings.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            return_attention_mask=True,
            return_offsets_mapping=True
        )

        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

            # Get last hidden states (word-level embeddings)
            last_hidden_states = outputs.last_hidden_state.cpu().numpy()

            # Get pooled output (sentence-level embedding)
            if hasattr(outputs, 'pooler_output'):
                pooled = outputs.pooler_output.cpu().numpy()
            else:
                # Mean pooling if no pooler output
                masked_hidden = last_hidden_states[0] * attention_mask.cpu().numpy()[0][:, np.newaxis]
                pooled = masked_hidden.sum(axis=0) / attention_mask.cpu().sum().item()
                pooled = pooled.reshape(1, -1)

        # Get tokens for alignment
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Create word mapping (handle subwords)
        word_embeddings = []
        word_tokens = []
        current_word_emb = []
        current_word_tokens = []

        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            if token.startswith('##'):
                # Subword continuation
                current_word_emb.append(last_hidden_states[0, i])
                current_word_tokens.append(token)
            else:
                # New word - save previous if exists
                if current_word_emb:
                    word_embeddings.append(np.mean(current_word_emb, axis=0))
                    word_tokens.append(''.join(t.replace('##', '') for t in current_word_tokens))

                # Start new word
                current_word_emb = [last_hidden_states[0, i]]
                current_word_tokens = [token]

        # Save last word
        if current_word_emb:
            word_embeddings.append(np.mean(current_word_emb, axis=0))
            word_tokens.append(''.join(t.replace('##', '') for t in current_word_tokens))

        return {
            'word_embeddings': np.array(word_embeddings),
            'word_tokens': word_tokens,
            'sentence_embedding': pooled[0],
            'full_sequence': last_hidden_states[0]  # Include full sequence for flexibility
        }

    def process_file(self, file_path: Path) -> Tuple[bool, Optional[Dict]]:
        """
        Process a single file and generate embeddings.

        Returns:
            (success, embeddings_dict or None)
        """
        try:
            # Read text
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            if not text:
                self.logger.warning(f"Empty file: {file_path.name}")
                return False, None

            # Generate embeddings
            embeddings = self.get_embeddings(text)

            # Add metadata
            embeddings['file_name'] = file_path.name
            embeddings['text'] = text
            embeddings['timestamp'] = datetime.now().isoformat()
            embeddings['model'] = self.model_name

            return True, embeddings

        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {str(e)}")
            return False, None

    def process_batch(self, file_paths: List[Path]) -> List[Dict]:
        """Process a batch of files."""
        results = []

        for file_path in file_paths:
            success, embeddings = self.process_file(file_path)
            if success:
                results.append(embeddings)

        return results

    def save_embeddings(self, embeddings: Dict, file_name: str):
        """Save embeddings to file."""
        # Create output filename (preserve original name structure)
        output_name = file_name.replace('.txt', '_embeddings.pkl')
        output_path = self.output_dir / output_name

        # Save as pickle for efficiency
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)

        # Also save a smaller JSON with metadata only
        metadata = {
            'file_name': embeddings['file_name'],
            'timestamp': embeddings['timestamp'],
            'model': embeddings['model'],
            'num_words': len(embeddings['word_tokens']),
            'embedding_dims': embeddings['word_embeddings'].shape[1] if len(embeddings['word_embeddings']) > 0 else 0
        }

        metadata_path = self.output_dir / output_name.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def process_all_files(self, input_dir: str = "embedding_ready"):
        """
        Process all files in the input directory.
        """
        input_path = Path(input_dir)

        # Get all text files
        all_files = sorted(input_path.glob("*.txt"))

        # Filter out combined files
        individual_files = [f for f in all_files if not f.name.startswith("combined_")]

        self.logger.info(f"Found {len(individual_files)} individual child files to process")

        # Load checkpoint
        checkpoint = self.load_checkpoint()
        processed_files = set(checkpoint['processed_files'])
        failed_files = set(checkpoint['failed_files'])

        # Filter out already processed files
        files_to_process = [f for f in individual_files if f.name not in processed_files]

        if len(files_to_process) == 0:
            self.logger.info("All files have been processed!")
            return

        self.logger.info(f"Processing {len(files_to_process)} remaining files...")

        # Process statistics
        start_time = time.time()
        success_count = 0
        fail_count = len(failed_files)

        # Process files with progress bar
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            for i, file_path in enumerate(files_to_process):
                # Process file
                success, embeddings = self.process_file(file_path)

                if success:
                    # Save embeddings
                    self.save_embeddings(embeddings, file_path.name)
                    processed_files.add(file_path.name)
                    success_count += 1

                    # Log progress
                    if (i + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed
                        remaining = (len(files_to_process) - i - 1) / rate
                        self.logger.info(
                            f"Progress: {i+1}/{len(files_to_process)} files | "
                            f"Rate: {rate:.2f} files/sec | "
                            f"ETA: {remaining/60:.1f} minutes"
                        )
                else:
                    failed_files.add(file_path.name)
                    fail_count += 1

                # Update checkpoint every 20 files
                if (i + 1) % 20 == 0:
                    checkpoint = {
                        'processed_files': list(processed_files),
                        'failed_files': list(failed_files)
                    }
                    self.save_checkpoint(checkpoint)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'success': success_count,
                    'failed': fail_count
                })

                # Memory management - clear cache periodically
                if (i + 1) % 50 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()

        # Final checkpoint save
        checkpoint = {
            'processed_files': list(processed_files),
            'failed_files': list(failed_files)
        }
        self.save_checkpoint(checkpoint)

        # Final statistics
        total_time = time.time() - start_time
        self.logger.info("="*80)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info(f"Total files processed: {success_count}")
        self.logger.info(f"Failed files: {fail_count}")
        self.logger.info(f"Total time: {total_time/60:.2f} minutes")
        self.logger.info(f"Average time per file: {total_time/success_count:.2f} seconds")
        self.logger.info("="*80)

        # Generate summary report
        self.generate_summary_report(processed_files, failed_files)

    def generate_summary_report(self, processed_files: set, failed_files: set):
        """Generate a summary report of the embedding generation."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'device': self.device,
            'total_processed': len(processed_files),
            'total_failed': len(failed_files),
            'processed_files': sorted(list(processed_files)),
            'failed_files': sorted(list(failed_files))
        }

        # Count by development type
        td_count = sum(1 for f in processed_files if f.startswith('td_'))
        sli_count = sum(1 for f in processed_files if f.startswith('sli_'))

        report['td_processed'] = td_count
        report['sli_processed'] = sli_count

        # Save report
        report_path = self.output_dir / f"embedding_generation_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Summary report saved to: {report_path}")
        self.logger.info(f"TD files processed: {td_count}")
        self.logger.info(f"SLI files processed: {sli_count}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate BERT embeddings for Gillam corpus child narratives"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="embedding_ready",
        help="Directory containing individual child text files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="embeddings",
        help="Directory to save embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model to use for embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detects if not specified"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = BERTEmbeddingGenerator(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir
    )

    # Process all files
    generator.process_all_files(input_dir=args.input_dir)


if __name__ == "__main__":
    main()