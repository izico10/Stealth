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
DENSITY_K = int((6/100) * SDR_DIMS)
VOCAB_SIZE = 10000
N_EPOCHS = 120
BATCH_SIZE = 256
LR = .025

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
def vocab(embedder: WordEmbedder) -> Vocabulary:
    """Initializes the vocabulary with the top tokens from the embedder."""
    tokens = embedder.get_vocabulary_tokens(max_tokens=VOCAB_SIZE)
    return Vocabulary(custom_words=tokens)

# Global storage for split words to be used across tests
split_data = {}

@pytest.fixture(scope="module")
def sae(embedder: WordEmbedder, vocab: Vocabulary) -> SparseAutoencoder:
    """
    Initializes the SAE using global test parameters.
    Implements a formal Train/Val/Test split (70/15/15).
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
        essential_words = ["king", "queen", "man", "woman", "apple", "bike", "cat", "kitten", "dog", "puppy"]
        all_words = vocab.get_words()
        pool_words = [w for w in all_words if w not in essential_words]
        
        np.random.seed(42)
        np.random.shuffle(pool_words)
        
        n = len(pool_words)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_words = pool_words[:train_end]
        val_words = pool_words[train_end:val_end]
        test_words = pool_words[val_end:]
        
        split_data['train'] = train_words
        split_data['val'] = val_words
        split_data['test'] = test_words
        split_data['essential'] = essential_words
        
        logger.info(f"Config: SDR_DIMS={SDR_DIMS}, DENSITY_K={DENSITY_K}, VOCAB={VOCAB_SIZE}, EPOCHS={N_EPOCHS}, LR={LR}")
        logger.info(f"USE_NN_PROJECTION is True. Total Vocab: {len(all_words)}")
        logger.info(f"Training on:   {len(train_words)} words")
        logger.info(f"Validating on: {len(val_words)} words")
        logger.info(f"Testing on:    {len(test_words)} words (+ {len(essential_words)} essential)")
        
        train_data = embedder.embed_batch(train_words)
        val_data = embedder.embed_batch(val_words)
        
        sae_instance.train(train_data, val_data=val_data, epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE)
        logger.info("SAE training complete.")
        
    return sae_instance

def get_sdr_metrics(sdr1: SDR, sdr2: SDR, density_k: int, sdr_dims: int) -> tuple[int, float, float]:
    """Calculates similarity metrics."""
    intersection = len(sdr1.active_indices.intersection(sdr2.active_indices))
    overlap_pct = intersection / len(sdr1) if len(sdr1) > 0 else 0.0
    p_chance = density_k / sdr_dims
    norm_sim = (overlap_pct - p_chance) / (1.0 - p_chance)
    norm_sim = max(0.0, norm_sim)
    return intersection, overlap_pct, norm_sim

def run_analogy_test(name: str, combined_vector: np.ndarray, sae: SparseAutoencoder, embedder: WordEmbedder, vocab: Vocabulary, active_indices: np.ndarray) -> None:
    """Helper to decode and log analogy results."""
    result_sdr = SDR(sae.sdr_dims, active_indices)
    v_reconstructed = sae.decode(result_sdr)
    
    all_words = vocab.get_words()
    all_embs = embedder.embed_batch(all_words)
    
    top_10 = vocab.find_top_n(v_reconstructed, all_embs, n=10, exclude_words=["king", "man", "woman"])
    
    logger.info(f"--- Full Loop Result: {name} ---")
    logger.info(f"Active bits: {len(result_sdr)}")
    for i, (word, sim) in enumerate(top_10):
        logger.info(f"  {i+1}. '{word}' (Sim: {sim:.4f})")

def test_sae_full_loop_comparison(embedder: WordEmbedder, sae: SparseAutoencoder, vocab: Vocabulary) -> None:
    """
    Compares Top-K vs Fixed Threshold for bit-level analogy reconstruction.
    """
    v_king = sae.encode(embedder.embed("king")).to_dense()
    v_man = sae.encode(embedder.embed("man")).to_dense()
    v_woman = sae.encode(embedder.embed("woman")).to_dense()
    
    combined = v_king - v_man + v_woman
    
    logger.info("--- Analogy Bit Physics: King - Man + Woman ---")
    vals, counts = np.unique(combined, return_counts=True)
    logger.info(f"Value Distribution in Sum: {dict(zip(vals, counts))}")
    
    top_k_indices = np.argpartition(combined, -sae.density_k)[-sae.density_k:]
    run_analogy_test("Top-K (Global)", combined, sae, embedder, vocab, top_k_indices)
    
    threshold = 0.5
    thresh_indices = np.where(combined > threshold)[0]
    run_analogy_test(f"Threshold > {threshold} (Local)", combined, sae, embedder, vocab, thresh_indices)

def test_sae_large_scale_reconstruction(embedder: WordEmbedder, sae: SparseAutoencoder, vocab: Vocabulary) -> None:
    """Tests Top-1, Top-5, and Top-10 Accuracy."""
    test_words = split_data.get('test', [])
    if not test_words:
        pytest.skip("No test data available.")
    sample_size = min(500, len(test_words))
    test_sample = test_words[:sample_size]
    embs = embedder.embed_batch(test_sample)
    sdrs = sae.encode_batch(embs)
    all_vocab_words = vocab.get_words()
    all_vocab_embs = embedder.embed_batch(all_vocab_words)
    top1, top5, top10 = 0, 0, 0
    for i, word in enumerate(test_sample):
        v_rec = sae.decode(sdrs[i])
        top_n = [item[0] for item in vocab.find_top_n(v_rec, all_vocab_embs, n=10)]
        if word == top_n[0]: top1 += 1
        if word in top_n[:5]: top5 += 1
        if word in top_n[:10]: top10 += 1
    logger.info(f"Top-1: {top1/sample_size:.2%}, Top-5: {top5/sample_size:.2%}, Top-10: {top10/sample_size:.2%}")

def test_sae_neuron_utilization(sae: SparseAutoencoder, embedder: WordEmbedder, vocab: Vocabulary) -> None:
    all_words = vocab.get_words()
    sample_words = all_words[:2000]
    sdrs = sae.encode_batch(embedder.embed_batch(sample_words))
    active = {idx for sdr in sdrs for idx in sdr.active_indices}
    logger.info(f"Neuron Utilization: {len(active)}/{sae.sdr_dims} ({len(active)/sae.sdr_dims:.2%})")

def test_sae_analogy_logic(embedder: WordEmbedder, sae: SparseAutoencoder, vocab: Vocabulary) -> None:
    """Honest Analogy SNR Test using Subtraction and Thresholding."""
    v_king, v_man, v_woman, v_queen, v_bike = [embedder.embed(w) for w in ["king", "man", "woman", "queen", "bike"]]
    sdr_k, sdr_m, sdr_w, sdr_q, sdr_b = [sae.encode(v) for v in [v_king, v_man, v_woman, v_queen, v_bike]]
    
    # Analogy Math: King - Man + Woman
    combined = sdr_k.to_dense() - sdr_m.to_dense() + sdr_w.to_dense()
    
    # Threshold to get the resulting SDR
    res_indices = np.where(combined > 0.5)[0]
    res_sdr = SDR(sae.sdr_dims, res_indices)
    
    _, o_q, _ = get_sdr_metrics(res_sdr, sdr_q, sae.density_k, sae.sdr_dims)
    _, o_b, _ = get_sdr_metrics(res_sdr, sdr_b, sae.density_k, sae.sdr_dims)
    
    logger.info(f"Analogy SNR (Bit-Level): {o_q/o_b if o_b > 0 else o_q:.2f}x")
    logger.info(f"  Queen Overlap: {o_q:.2%}")
    logger.info(f"  Bike Overlap:  {o_b:.2%}")

def test_sae_semantic_preservation(embedder: WordEmbedder, sae: SparseAutoencoder) -> None:
    """Expanded semantic preservation test with more pairs."""
    test_pairs = [
        ("apple", "apple", "Identity"),
        ("cat", "kitten", "Similar"),
        ("dog", "puppy", "Similar"),
        ("happy", "joyful", "Similar"),
        ("car", "truck", "Similar"),
        ("one", "1", "Similar"),
        ("apple", "bicycle", "Dissimilar"),
        ("king", "sandwich", "Dissimilar"),
        ("blue", "democracy", "Dissimilar")
    ]
    for w1, w2, ptype in test_pairs:
        s1, s2 = sae.encode(embedder.embed(w1)), sae.encode(embedder.embed(w2))
        _, overlap, norm = get_sdr_metrics(s1, s2, sae.density_k, sae.sdr_dims)
        logger.info(f"{w1}/{w2} ({ptype}): Overlap={overlap:.2%}, Norm Sim={norm:.4f}")

def test_sdr_basic_properties() -> None:
    sdr = SDR(100, {1, 5, 10})
    assert len(sdr) == 3

def test_sdr_circular_shift() -> None:
    sdr = SDR(10, {1, 9})
    assert sdr.circular_shift(1).active_indices == {2, 0}

def test_sae_encoding_sparsity(sae: SparseAutoencoder) -> None:
    assert len(sae.encode(np.random.rand(sae.input_dims).astype(np.float32))) == sae.density_k
