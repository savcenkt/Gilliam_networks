#!/usr/bin/env python3
"""Quick verification that embeddings can be loaded and used."""

import pickle
import numpy as np
from pathlib import Path

# Load a test embedding file
test_file = Path("test_embeddings/td_7y6m_f_46076v_embeddings.pkl")

with open(test_file, 'rb') as f:
    embeddings = pickle.load(f)

print("Embedding file loaded successfully!")
print(f"Keys in embedding dict: {list(embeddings.keys())}")
print(f"\nWord embeddings shape: {embeddings['word_embeddings'].shape}")
print(f"Number of words: {len(embeddings['word_tokens'])}")
print(f"Sentence embedding shape: {embeddings['sentence_embedding'].shape}")
print(f"First 10 words: {embeddings['word_tokens'][:10]}")

# Verify embeddings are valid numpy arrays
assert isinstance(embeddings['word_embeddings'], np.ndarray)
assert isinstance(embeddings['sentence_embedding'], np.ndarray)
assert embeddings['word_embeddings'].shape[1] == 768  # BERT base dimension
assert len(embeddings['word_tokens']) == embeddings['word_embeddings'].shape[0]

print("\n✅ All verification checks passed!")
print("The embeddings are ready for semantic network analysis.")