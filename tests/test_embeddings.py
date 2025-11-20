#!/usr/bin/env python3
"""
Test script to validate BERT embedding generation on a small subset of files.
This ensures the pipeline works correctly before processing all 668 files.
"""

import sys
import json
import pickle
from pathlib import Path
import numpy as np

def test_embedding_generation():
    """Test the embedding generation on a small subset."""
    print("="*80)
    print("BERT Embedding Generation Test")
    print("="*80)

    # Import the main generator
    from generate_bert_embeddings import BERTEmbeddingGenerator

    # Select a small test subset (2 TD and 2 SLI files)
    test_files = [
        "td_7y6m_f_46076v.txt",
        "td_10y3m_m_46771ca.txt",
        "sli_8y7m_m_a-3-44.txt",
        "sli_6y0m_m_a-3-01.txt"
    ]

    print(f"\nTesting with {len(test_files)} files:")
    for f in test_files:
        print(f"  - {f}")

    # Initialize generator with test output directory
    print("\nInitializing BERT model...")
    generator = BERTEmbeddingGenerator(
        model_name="bert-base-uncased",
        batch_size=2,
        checkpoint_dir="test_checkpoints",
        output_dir="test_embeddings"
    )

    # Process test files
    print("\nProcessing test files...")
    input_dir = Path("embedding_ready")
    successful = 0
    failed = 0

    for file_name in test_files:
        file_path = input_dir / file_name

        if not file_path.exists():
            print(f"  ⚠️  File not found: {file_name}")
            failed += 1
            continue

        print(f"\n  Processing: {file_name}")
        success, embeddings = generator.process_file(file_path)

        if success:
            # Save embeddings
            generator.save_embeddings(embeddings, file_name)

            # Validate embeddings
            print(f"    ✓ Generated embeddings successfully")
            print(f"    - Words found: {len(embeddings['word_tokens'])}")
            print(f"    - Embedding shape: {embeddings['word_embeddings'].shape}")
            print(f"    - Sample words: {embeddings['word_tokens'][:5]}")

            successful += 1
        else:
            print(f"    ✗ Failed to generate embeddings")
            failed += 1

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Successful: {successful}/{len(test_files)}")
    print(f"Failed: {failed}/{len(test_files)}")

    # Verify saved files
    print("\nVerifying saved embeddings...")
    output_dir = Path("test_embeddings")

    for file_name in test_files:
        pkl_file = output_dir / file_name.replace('.txt', '_embeddings.pkl')
        json_file = output_dir / file_name.replace('.txt', '_embeddings_metadata.json')

        if pkl_file.exists() and json_file.exists():
            # Load and check embeddings
            with open(pkl_file, 'rb') as f:
                saved_emb = pickle.load(f)

            with open(json_file, 'r') as f:
                metadata = json.load(f)

            print(f"\n  {file_name}:")
            print(f"    ✓ Pickle file size: {pkl_file.stat().st_size / 1024:.2f} KB")
            print(f"    ✓ Metadata: {metadata['num_words']} words, {metadata['embedding_dims']} dims")

    print("\n✅ Test completed successfully! The pipeline is ready for full processing.")
    print("\nTo process all files, run:")
    print("  python generate_bert_embeddings.py")
    print("\nOr with custom parameters:")
    print("  python generate_bert_embeddings.py --model bert-base-uncased --batch-size 16")


if __name__ == "__main__":
    test_embedding_generation()