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
from vocabulary import Vocabulary

# Global parameters for quick testing and configuration
USE_NN_PROJECTION = True
USE_TERNARY_WEIGHTS = True
PROJECTION_WEIGHT_DENSITY = 0.1
USE_CENTERING = True
SDR_DIMS = 1024
DENSITY_K = int((10/100) * SDR_DIMS)
N_EPOCHS = 50

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
def vocab() -> Vocabulary:
    return Vocabulary()

@pytest.fixture(scope="module")
def sae(embedder: WordEmbedder, vocab: Vocabulary) -> SparseAutoencoder:
    """
    Initializes the SAE using global test parameters.
    If USE_NN_PROJECTION is True, the SAE is trained on the vocabulary before being returned.
    """
    sae_instance = SparseAutoencoder(
        input_dims=embedder.get_dimension(), 
        sdr_dims=SDR_DIMS, 
        density_k=DENSITY_K,
        use_ternary_weights=USE_TERNARY_WEIGHTS,
        projection_weight_density=PROJECTION_WEIGHT_DENSITY,
        use_centering=USE_CENTERING
    )
    
    if USE_NN_PROJECTION:
        logger.info(f"USE_NN_PROJECTION is True. Training SAE for {N_EPOCHS} epochs...")
        training_data = embedder.embed_batch(vocab.get_words())
        sae_instance.train(training_data, epochs=N_EPOCHS, lr=0.01)
        logger.info("SAE training complete.")
        
    return sae_instance

def get_sdr_metrics(sdr1: SDR, sdr2: SDR, density_k: int, sdr_dims: int) -> tuple[int, float, float]:
    """
    Calculates various similarity metrics between two SDRs.
    Returns: (Overlap Count, Overlap %, Normalized Similarity)
    """
    intersection = len(sdr1.active_indices.intersection(sdr2.active_indices))
    
    # We use the length of sdr1 as the denominator for overlap percentage
    overlap_pct = intersection / len(sdr1) if len(sdr1) > 0 else 0.0
    
    # Chance overlap probability
    p_chance = density_k / sdr_dims
    # Normalized Similarity: (Observed - Chance) / (Max - Chance)
    norm_sim = (overlap_pct - p_chance) / (1.0 - p_chance)
    norm_sim = max(0.0, norm_sim)
    
    return intersection, overlap_pct, norm_sim

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
    assert len(sdr) == sae.density_k

def test_sae_semantic_preservation(embedder: WordEmbedder, sae: SparseAutoencoder) -> None:
    """
    Compares Cosine Similarity (Dense) vs SDR Metrics.
    Ensures that the relative semantic relationships are preserved after projection.
    """
    test_pairs = [
        ("apple", "apple", "Identity"),
        ("cat", "kitten", "Similar"),
        ("dog", "puppy", "Similar"),
        ("apple", "bicycle", "Dissimilar"),
        ("king", "sandwich", "Dissimilar"),
        ("one", "1", "Similar"),
        ("happy", "joyful", "Similar")
    ]
    
    logger.info(f"Config: SDR_DIMS={sae.sdr_dims}, DENSITY_K={sae.density_k}, TERNARY={sae.use_ternary_weights}, TRAINED={USE_NN_PROJECTION}")
    header = f"{'Pair':<20} | {'Type':<10} | {'Dense Cos':<10} | {'Bits':<5} | {'Overlap %':<10} | {'Norm Sim':<8}"
    logger.info(header)
    logger.info("-" * len(header))
    
    results = []
    for w1, w2, ptype in test_pairs:
        emb1 = embedder.embed(w1)
        emb2 = embedder.embed(w2)
        dense_sim = cosine_similarity(emb1, emb2)
        
        sdr1 = sae.encode(emb1)
        sdr2 = sae.encode(emb2)
        bits, overlap_pct, norm_sim = get_sdr_metrics(sdr1, sdr2, sae.density_k, sae.sdr_dims)
        
        row = f"{f'{w1}/{w2}':<20} | {ptype:<10} | {dense_sim:10.4f} | {bits:<5} | {overlap_pct:10.2%} | {norm_sim:8.4f}"
        logger.info(row)
        
        if ptype in ["Similar", "Dissimilar"]:
            results.append((ptype, dense_sim, norm_sim))

    sim_norms = [r[2] for r in results if r[0] == "Similar"]
    dis_norms = [r[2] for r in results if r[0] == "Dissimilar"]
    
    avg_sim = sum(sim_norms) / len(sim_norms)
    avg_dis = sum(dis_norms) / len(dis_norms)
    
    logger.info("-" * len(header))
    logger.info(f"Average Similar Norm Sim:    {avg_sim:.4f}")
    logger.info(f"Average Dissimilar Norm Sim: {avg_dis:.4f}")
    
    assert avg_sim > avg_dis

def test_sae_analogy_logic(embedder: WordEmbedder, sae: SparseAutoencoder, vocab: Vocabulary) -> None:
    """
    Tests the King - Man + Woman = Queen analogy in both spaces.
    """
    # 1. Prepare embeddings
    v_king = embedder.embed("king")
    v_man = embedder.embed("man")
    v_woman = embedder.embed("woman")
    v_queen = embedder.embed("queen")
    v_bike = embedder.embed("bike")
    all_words = vocab.get_words()
    word_embs = embedder.embed_batch(all_words)
    
    # 2. Dense Analogy
    v_target = v_king - v_man + v_woman
    nearest_dense, sim_dense = vocab.find_nearest(v_target, word_embs, exclude_words=["king", "man", "woman"])
    logger.info(f"Dense Analogy (King - Man + Woman): Result='{nearest_dense}', Similarity={sim_dense:.4f}")
    
    # 3. Dense Superposition (King + Woman)
    v_combined_dense = v_king + v_woman
    cos_combined_queen = cosine_similarity(v_combined_dense, v_queen)
    cos_combined_bike = cosine_similarity(v_combined_dense, v_bike)
    logger.info(f"Dense Superposition (King + Woman) vs Queen: Cosine={cos_combined_queen:.4f}")
    logger.info(f"Dense Superposition (King + Woman) vs Bike:  Cosine={cos_combined_bike:.4f}")
    
    # 4. SDR Superposition (King + Woman) - Logical OR
    sdr_king = sae.encode(v_king)
    sdr_woman = sae.encode(v_woman)
    sdr_queen = sae.encode(v_queen)
    sdr_bike = sae.encode(v_bike)
    
    combined_indices_or = sdr_king.active_indices.union(sdr_woman.active_indices)
    sdr_or = SDR(sae.sdr_dims, combined_indices_or)
    
    bits_or_q, overlap_or_q, _ = get_sdr_metrics(sdr_or, sdr_queen, sae.density_k, sae.sdr_dims)
    bits_or_b, overlap_or_b, _ = get_sdr_metrics(sdr_or, sdr_bike, sae.density_k, sae.sdr_dims)
    logger.info(f"SDR OR (King + Woman) vs Queen:     Bits={bits_or_q}, Overlap={overlap_or_q:.2%}, Sparsity={len(sdr_or)/sae.sdr_dims:.2%}")
    logger.info(f"SDR OR (King + Woman) vs Bike:      Bits={bits_or_b}, Overlap={overlap_or_b:.2%}")
    
    # 5. SDR Superposition (King + Woman) - Thresholded (Maintaining Sparsity K)
    # Note: We use the weights directly to simulate the SDM retrieval process
    x_king = v_king - sae.b_pre if sae.use_centering else v_king
    x_woman = v_woman - sae.b_pre if sae.use_centering else v_woman
    
    activations_king = np.dot(x_king, sae.encoder_weights) + sae.encoder_bias
    activations_woman = np.dot(x_woman, sae.encoder_weights) + sae.encoder_bias
    combined_activations = activations_king + activations_woman
    
    top_k_indices = np.argpartition(combined_activations, -sae.density_k)[-sae.density_k:]
    sdr_thresholded = SDR(sae.sdr_dims, top_k_indices)
    
    bits_th_q, overlap_th_q, _ = get_sdr_metrics(sdr_thresholded, sdr_queen, sae.density_k, sae.sdr_dims)
    bits_th_b, overlap_th_b, _ = get_sdr_metrics(sdr_thresholded, sdr_bike, sae.density_k, sae.sdr_dims)
    logger.info(f"SDR Top-K (King + Woman) vs Queen:  Bits={bits_th_q}, Overlap={overlap_th_q:.2%}, Sparsity={len(sdr_thresholded)/sae.sdr_dims:.2%}")
    logger.info(f"SDR Top-K (King + Woman) vs Bike:   Bits={bits_th_b}, Overlap={overlap_th_b:.2%}")
    
    assert overlap_th_q > overlap_th_b

def test_sae_batch_consistency(sae: SparseAutoencoder) -> None:
    """Ensures batch encoding matches individual encoding."""
    batch_vecs = np.random.rand(5, sae.input_dims).astype(np.float32)
    batch_sdrs = sae.encode_batch(batch_vecs)
    
    for i in range(5):
        individual_sdr = sae.encode(batch_vecs[i])
        assert batch_sdrs[i].active_indices == individual_sdr.active_indices
