"""
Unit tests for the WordEmbedder module.
Ensures that embeddings are generated correctly, maintain semantic relationships,
and are produced in isolation.
"""

import pytest
import numpy as np
import logging
from word_embedder import WordEmbedder, cosine_similarity

# Configure logger for test output
logger = logging.getLogger(__name__)

# Silence external loggers to keep test output clean
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Default thresholds for semantic similarity tests
SAME_THRESHOLD = 0.99
SIMILAR_THRESHOLD = 0.65
DISSIMILAR_THRESHOLD = 0.45

@pytest.fixture(scope="module")
def embedder() -> WordEmbedder:
    """
    Fixture to provide a shared WordEmbedder instance for all tests in this module.
    """
    return WordEmbedder()

def test_embedding_dimension(embedder: WordEmbedder) -> None:
    """
    Verifies that the embedder produces vectors of the expected dimension.
    """
    dim: int = embedder.get_dimension()
    assert isinstance(dim, int)
    assert dim > 0
    
    emb: np.ndarray = embedder.embed("test")
    assert emb.shape == (dim,)

def test_identical_words(embedder: WordEmbedder) -> None:
    """
    Ensures that identical words have a cosine similarity above the SAME_THRESHOLD.
    """
    word: str = "apple"
    emb1: np.ndarray = embedder.embed(word)
    emb2: np.ndarray = embedder.embed(word)
    sim: float = cosine_similarity(emb1, emb2)
    
    logger.info(f"Identical words '{word}': similarity={sim:.4f}, threshold={SAME_THRESHOLD}")
    assert sim >= SAME_THRESHOLD, f"Similarity between identical words too low: {sim}"

def test_similar_words(embedder: WordEmbedder) -> None:
    """
    Verifies that semantically similar words have high cosine similarity.
    """
    pairs: list[tuple[str, str]] = [
        ("cat", "kitten"),
        ("dog", "puppy"),
        ("king", "monarch"),
        ("happy", "joyful"),
        ("ocean", "sea")
    ]
    for w1, w2 in pairs:
        sim: float = cosine_similarity(embedder.embed(w1), embedder.embed(w2))
        logger.info(f"Similar words '{w1}' vs '{w2}': similarity={sim:.4f}, threshold={SIMILAR_THRESHOLD}")
        assert sim > SIMILAR_THRESHOLD, f"Similarity between {w1} and {w2} too low: {sim}"

def test_dissimilar_words(embedder: WordEmbedder) -> None:
    """
    Verifies that unrelated words have low cosine similarity.
    """
    pairs: list[tuple[str, str]] = [
        ("apple", "bicycle"),
        ("cat", "computer"),
        ("blue", "democracy"),
        ("run", "philosophy")
    ]
    for w1, w2 in pairs:
        sim: float = cosine_similarity(embedder.embed(w1), embedder.embed(w2))
        logger.info(f"Dissimilar words '{w1}' vs '{w2}': similarity={sim:.4f}, threshold={DISSIMILAR_THRESHOLD}")
        assert sim < DISSIMILAR_THRESHOLD, f"Similarity between {w1} and {w2} too high: {sim}"

def test_numbers(embedder: WordEmbedder) -> None:
    """
    Ensures that numbers and their string representations are semantically linked.
    """
    test_cases: list[tuple[str, str]] = [("one", "1"), ("one", "two")]
    for w1, w2 in test_cases:
        sim: float = cosine_similarity(embedder.embed(w1), embedder.embed(w2))
        logger.info(f"Numbers '{w1}' vs '{w2}': similarity={sim:.4f}, threshold={DISSIMILAR_THRESHOLD}")
        assert sim > DISSIMILAR_THRESHOLD, f"Similarity between {w1} and {w2} too low: {sim}"

def test_batch_embedding(embedder: WordEmbedder) -> None:
    """
    Verifies that batch embedding produces the same results as individual embedding.
    This confirms the isolation property of the encoder.
    """
    dim: int = embedder.get_dimension()
    words: list[str] = ["apple", "banana", "cherry"]
    embs: np.ndarray = embedder.embed_batch(words)
    assert embs.shape == (len(words), dim)
    
    # Verify batch matches individual
    for i, word in enumerate(words):
        individual: np.ndarray = embedder.embed(word)
        assert np.allclose(embs[i], individual, atol=1e-5)
