"""
Module for converting text tokens into dense vector embeddings.
This module ensures that embeddings are generated in isolation, preventing 
sequential context from leaking into the individual word representations.
"""

import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

class WordEmbedder:
    """
    A wrapper around SentenceTransformer to provide isolated word embeddings.
    
    This class is designed to treat every input as a standalone token, which is 
    critical for maintaining monosemanticity before the data enters the SAE.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', cache_folder: str = './model_cache') -> None:
        """
        Initializes the embedder with a specific pre-trained model and local cache.
        
        Args:
            model_name: The HuggingFace model ID to use for embedding.
            cache_folder: Local directory to store the downloaded model.
        """
        # Ensure cache directory exists
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
            
        self.model: SentenceTransformer = SentenceTransformer(model_name, cache_folder=cache_folder)
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()

    def embed(self, word: str) -> np.ndarray:
        """
        Encodes a single word into a dense vector.
        """
        embedding: np.ndarray = self.model.encode(word, convert_to_numpy=True)
        return embedding

    def embed_batch(self, words: list[str]) -> np.ndarray:
        """
        Encodes a list of words independently.
        """
        embeddings: np.ndarray = self.model.encode(words, convert_to_numpy=True)
        return embeddings

    def embed_contextual(self, text: str) -> np.ndarray:
        """
        Encodes a sentence and returns the contextual embedding for each token.
        
        This simulates the hidden states of a Transformer, which is what 
        would be stored in a KV cache.
        """
        # output_value='token_embeddings' returns hidden states. 
        # We must manually handle the conversion to CPU and NumPy if it's on GPU.
        embeddings = self.model.encode([text], output_value='token_embeddings')[0]
        
        if isinstance(embeddings, torch.Tensor):
            return embeddings.detach().cpu().numpy()
        return np.array(embeddings)

    def get_dimension(self) -> int:
        """
        Returns the size of the embedding vector produced by the model.
        """
        return self.embedding_dim

    def get_vocabulary_tokens(self, max_tokens: int = 1000) -> list[str]:
        """
        Extracts the most common tokens from the underlying model's tokenizer.
        """
        tokenizer = self.model.tokenizer
        vocab = tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        tokens = [token for token, idx in sorted_vocab[:max_tokens]]
        return tokens

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.
    """
    norm1: float = float(np.linalg.norm(v1))
    norm2: float = float(np.linalg.norm(v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))
