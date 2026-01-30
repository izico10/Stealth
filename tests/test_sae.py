"""
Unit tests for the SparseAutoencoder and SDR modules.
Ensures that dense embeddings are correctly projected into sparse space and 
that semantic relationships are preserved.
"""

import pytest
import numpy as np
import logging
import os
from sae import SDR, SparseAutoencoder
from word_embedder import WordEmbedder, cosine_similarity

# Global parameters for quick testing and configuration
USE_TERNARY_PROJECTION = False
PROJECTION_WEIGHT_DENSITY = 0.1
SDR_DIMS = 10000
SPARSITY_K = int(0.005 * SDR_DIMS)

# Silence external loggers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Configure logger for test output
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def embedder() -> WordEmbedder:
    return WordEmbedder()

@pytest.fixture(scope="module")
def sae(embedder: WordEmbedder) -> SparseAutoencoder:
    """Initializes the SAE using global test parameters."""
    return SparseAutoencoder(
        input_dims=embedder.get_dimension(), 
        sdr_dims=SDR_DIMS, 
        sparsity_k=SPARSITY_K,
        use_ternary_projection=USE_TERNARY_PROJECTION,
        projection_weight_density=PROJECTION_WEIGHT_DENSITY
    )

def get_sdr_metrics(sdr1: SDR, sdr2: SDR) -> tuple[int, float, float]:
    """
    Calculates various similarity metrics between two SDRs.
    Returns: (Overlap Count, Overlap Percentage, Jaccard Similarity)
    """
    intersection = len(sdr1.active_indices.intersection(sdr2.active_indices))
    union = len(sdr1.active_indices.union(sdr2.active_indices))
    
    overlap_pct = intersection / len(sdr1) if len(sdr1) > 0 else 0.0
    jaccard = intersection / union if union > 0 else 0.0
    
    return intersection, overlap_pct, jaccard

def test_sdr_basic_properties() -> None:
    """Verifies SDR initialization and properties."""
    sdr = SDR(sdr_dims=100, active_indices={1, 5, 10})
    assert sdr.sdr_dims == 100
    assert len(sdr) == 3
    assert sdr.active_indices == {1, 5, 10}

def test_sdr_circular_shift() -> None:
    """Verifies that circular shifting wraps correctly."""
    sdr = SDR(sdr_dims=10, active_indices={1, 9})
    shifted = sdr.circular_shift(1)
    assert shifted.active_indices == {2, 0}
    assert sdr.circular_shift(10).active_indices == sdr.active_indices

def test_sae_encoding_sparsity(sae: SparseAutoencoder) -> None:
    """Ensures the SAE produces SDRs with the exact requested sparsity."""
    vec = np.random.rand(sae.input_dims).astype(np.float32)
    sdr = sae.encode(vec)
    assert len(sdr) == SPARSITY_K

def test_sae_semantic_preservation(embedder: WordEmbedder, sae: SparseAutoencoder) -> None:
    """
    Compares Cosine Similarity (Dense) vs SDR Metrics.
    Ensures that the relative semantic relationships are preserved after projection.
    """
    test_pairs = [
        ("cat", "kitten", "Similar"),
        ("dog", "puppy", "Similar"),
        ("apple", "bicycle", "Dissimilar"),
        ("king", "sandwich", "Dissimilar"),
        ("one", "1", "Similar"),
        ("happy", "joyful", "Similar")
    ]
    
    logger.info(f"Config: SDR_DIMS={SDR_DIMS}, SPARSITY_K={SPARSITY_K}, TERNARY={USE_TERNARY_PROJECTION}")
    header = f"{'Pair':<20} | {'Type':<10} | {'Dense Cos':<10} | {'Bits':<6} | {'Overlap %':<10} | {'Jaccard':<8}"
    logger.info(header)
    logger.info("-" * len(header))
    
    results = []
    for w1, w2, ptype in test_pairs:
        emb1 = embedder.embed(w1)
        emb2 = embedder.embed(w2)
        dense_sim = cosine_similarity(emb1, emb2)
        
        sdr1 = sae.encode(emb1)
        sdr2 = sae.encode(emb2)
        bits, overlap_pct, jaccard = get_sdr_metrics(sdr1, sdr2)
        
        row = f"{f'{w1}/{w2}':<20} | {ptype:<10} | {dense_sim:10.4f} | {bits:<6} | {overlap_pct:10.2%} | {jaccard:8.4f}"
        logger.info(row)
        results.append((ptype, dense_sim, overlap_pct))

    sim_overlaps = [r[2] for r in results if r[0] == "Similar"]
    dis_overlaps = [r[2] for r in results if r[0] == "Dissimilar"]
    
    avg_sim = sum(sim_overlaps) / len(sim_overlaps)
    avg_dis = sum(dis_overlaps) / len(dis_overlaps)
    
    logger.info("-" * len(header))
    logger.info(f"Average Similar Overlap %:    {avg_sim:.2%}")
    logger.info(f"Average Dissimilar Overlap %: {avg_dis:.2%}")
    
    assert avg_sim > avg_dis, "SDR space failed to preserve relative semantic distances"

def test_sae_batch_consistency(sae: SparseAutoencoder) -> None:
    """Ensures batch encoding matches individual encoding."""
    batch_vecs = np.random.rand(5, sae.input_dims).astype(np.float32)
    batch_sdrs = sae.encode_batch(batch_vecs)
    
    for i in range(5):
        individual_sdr = sae.encode(batch_vecs[i])
        assert batch_sdrs[i].active_indices == individual_sdr.active_indices
