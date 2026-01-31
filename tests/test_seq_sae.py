"""
Research Test: SAE as a Sequential KV Cache Compressor.
Trains an SAE on contextual embeddings from sentences and tests reconstruction.
"""

import pytest
import numpy as np
import logging
import os
from sae import SDR, SparseAutoencoder
from word_embedder import WordEmbedder, cosine_similarity
from vocabulary import Vocabulary

# Global parameters for the Sequential SAE Test
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

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def embedder() -> WordEmbedder:
    return WordEmbedder()

@pytest.fixture(scope="module")
def vocab(embedder: WordEmbedder) -> Vocabulary:
    tokens = embedder.get_vocabulary_tokens(max_tokens=VOCAB_SIZE)
    return Vocabulary(custom_words=tokens)

@pytest.fixture(scope="module")
def contextual_data(embedder: WordEmbedder) -> np.ndarray:
    """Generates a dataset of contextual embeddings from sample sentences."""
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the modern world.",
        "A king rules over his kingdom with wisdom and justice.",
        "The cat sat on the mat and watched the birds outside.",
        "Quantum computing relies on the principles of superposition.",
        "Climate change is a global challenge that requires urgent action.",
        "The history of the universe began with the big bang.",
        "Music is a universal language that connects all people.",
        "Space exploration opens new frontiers for humanity.",
        "Sustainable energy is key to a greener future."
    ]
    
    all_embs = []
    for sent in sentences:
        embs = embedder.embed_contextual(sent)
        all_embs.append(embs)
        
    return np.vstack(all_embs)

@pytest.fixture(scope="module")
def seq_sae(contextual_data: np.ndarray) -> SparseAutoencoder:
    """Trains an SAE specifically on contextual (sequential) data."""
    sae = SparseAutoencoder(
        input_dims=contextual_data.shape[1], 
        sdr_dims=SDR_DIMS, 
        density_k=DENSITY_K,
        use_ternary_weights=USE_TERNARY_WEIGHTS,
        projection_weight_density=PROJECTION_WEIGHT_DENSITY,
        use_centering=USE_CENTERING
    )
    
    logger.info(f"Config: SDR_DIMS={SDR_DIMS}, DENSITY_K={DENSITY_K}, EPOCHS={N_EPOCHS}, LR={LR}")
    logger.info(f"Training Sequential SAE on {len(contextual_data)} contextual tokens...")
    sae.train(contextual_data, epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE)
    return sae

def test_sequential_reconstruction(embedder: WordEmbedder, seq_sae: SparseAutoencoder, vocab: Vocabulary) -> None:
    """
    Tests if the SAE can reconstruct contextual embeddings from UNSEEN sentences.
    """
    unseen_sentences = [
        "The brown dog jumps over the fox.",
        "Wisdom and justice are important for a king.",
        "Action is required to address global challenges."
    ]
    
    logger.info(f"--- Sequential Reconstruction Test (Unseen Sentences) ---")
    
    total_sim = 0.0
    total_tokens = 0
    
    for sent in unseen_sentences:
        v_contextual = embedder.embed_contextual(sent)
        num_tokens = v_contextual.shape[0]
        
        logger.info(f"Sentence: '{sent}'")
        for i in range(num_tokens):
            v_orig = v_contextual[i]
            sdr = seq_sae.encode(v_orig)
            v_rec = seq_sae.decode(sdr)
            
            sim = cosine_similarity(v_orig, v_rec)
            total_sim += sim
            total_tokens += 1
            
            # Get token string for logging
            tokens = embedder.model.tokenizer.tokenize(sent)
            full_tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_str = full_tokens[i] if i < len(full_tokens) else "?"
            
            logger.info(f"  Token '{token_str}': Sim={sim:.4f}")

    avg_sim = total_sim / total_tokens
    logger.info(f"Average Reconstruction Similarity on Unseen Sequences: {avg_sim:.4f}")
    
    assert avg_sim > 0.60, "Sequential reconstruction similarity too low."
