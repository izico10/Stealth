"""
Research Test: SAE as a Sequential KV Cache Compressor.
Trains an SAE on contextual embeddings and tests if it preserves token identity.
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
SDR_DIMS = 1024 * 2
DENSITY_K = int((6/100) * SDR_DIMS)
DICTIONARY_SIZE = 10000 
MAX_TRAIN_TOKENS = 50000 
N_EPOCHS = 100
BATCH_SIZE = 256
LR = .025

# Silence external loggers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def embedder() -> WordEmbedder:
    return WordEmbedder()

@pytest.fixture(scope="module")
def vocab(embedder: WordEmbedder) -> Vocabulary:
    tokens = embedder.get_vocabulary_tokens(max_tokens=DICTIONARY_SIZE)
    return Vocabulary(custom_words=tokens)

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Ensures all vectors are unit-length for consistent SSE loss."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms

# Global storage for split sentences
split_sentences = {}

@pytest.fixture(scope="module")
def contextual_data(embedder: WordEmbedder) -> dict[str, np.ndarray]:
    """Generates a dataset of normalized contextual embeddings using wikitext."""
    try:
        from datasets import load_dataset
        logger.info("Loading wikitext dataset from Hugging Face...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        all_sentences = [line for line in dataset["text"] if len(line.strip()) > 40]
    except Exception as e:
        logger.warning(f"Could not load wikitext dataset ({e}). Falling back to hardcoded pool.")
        all_sentences = ["The quick brown fox jumps over the lazy dog."] * 1000

    np.random.seed(42)
    np.random.shuffle(all_sentences)
    
    n = len(all_sentences)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_sents = all_sentences[:train_end]
    val_sents = all_sentences[train_end:val_end]
    test_sents = all_sentences[val_end:val_end+20]
    
    split_sentences['test'] = test_sents
    
    def get_embs(sentences, max_tokens=None):
        all_embs = []
        token_count = 0
        for sent in sentences:
            try:
                embs = normalize_vectors(embedder.embed_contextual(sent))
                all_embs.append(embs)
                token_count += embs.shape[0]
                if max_tokens and token_count >= max_tokens:
                    break
            except:
                continue
        if not all_embs: return np.zeros((0, 384))
        return np.vstack(all_embs)[:max_tokens] if max_tokens else np.vstack(all_embs)
        
    logger.info(f"Processing up to {MAX_TRAIN_TOKENS} training tokens...")
    train_data = get_embs(train_sents, max_tokens=MAX_TRAIN_TOKENS)
    val_data = get_embs(val_sents, max_tokens=5000)
    
    return {"train": train_data, "val": val_data}

@pytest.fixture(scope="module")
def seq_sae(contextual_data: dict[str, np.ndarray]) -> SparseAutoencoder:
    """Trains an SAE specifically on contextual (sequential) data."""
    sae = SparseAutoencoder(
        input_dims=contextual_data["train"].shape[1], 
        sdr_dims=SDR_DIMS, 
        density_k=DENSITY_K,
        use_ternary_weights=USE_TERNARY_WEIGHTS,
        projection_weight_density=PROJECTION_WEIGHT_DENSITY,
        use_centering=USE_CENTERING
    )
    
    logger.info(f"Config: SDR_DIMS={SDR_DIMS}, K={DENSITY_K}, TRAIN_TOKENS={len(contextual_data['train'])}")
    sae.train(contextual_data["train"], val_data=contextual_data["val"], epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE)
    return sae

def test_kv_cache_fidelity(embedder: WordEmbedder, seq_sae: SparseAutoencoder, vocab: Vocabulary) -> None:
    """
    Tests if the SAE preserves the 'Identity' of tokens in a KV cache.
    """
    test_sents = split_sentences.get('test', [])
    if not test_sents:
        pytest.skip("No test data available.")
        
    logger.info(f"--- KV Cache Fidelity Test ({len(test_sents)} Unseen Sentences) ---")
    
    total_sim = 0.0
    total_tokens = 0
    identity_correct = 0
    
    for sent in test_sents:
        try:
            v_orig_seq = normalize_vectors(embedder.embed_contextual(sent))
        except:
            continue
            
        num_tokens = v_orig_seq.shape[0]
        
        # 1. Compress and Reconstruct the whole sentence
        v_rec_seq = []
        for i in range(num_tokens):
            sdr = seq_sae.encode(v_orig_seq[i])
            v_rec_seq.append(seq_sae.decode(sdr))
        v_rec_seq = np.array(v_rec_seq)
        
        # 2. Check Fidelity
        for i in range(num_tokens):
            v_orig = v_orig_seq[i]
            v_rec = v_rec_seq[i]
            
            sim = cosine_similarity(v_orig, v_rec)
            total_sim += sim
            total_tokens += 1
            
            # IDENTITY TEST: Is v_rec closer to v_orig than to any other v_j in the sentence?
            sentence_similarities = [cosine_similarity(v_rec, v_target) for v_target in v_orig_seq]
            if np.argmax(sentence_similarities) == i:
                identity_correct += 1
                
            if i < 3: # Log first 3 tokens per sentence
                tokens = embedder.model.tokenizer.tokenize(sent)
                full_tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_str = full_tokens[i] if i < len(full_tokens) else "?"
                logger.info(f"  Token '{token_str}': Sim={sim:.4f}, Identity Match={np.argmax(sentence_similarities) == i}")

    avg_sim = total_sim / total_tokens
    identity_acc = identity_correct / total_tokens
    
    logger.info(f"Average Reconstruction Similarity: {avg_sim:.4f}")
    logger.info(f"KV Cache Identity Accuracy:      {identity_acc:.2%}")
    
    assert identity_acc >= 0.90, f"KV Cache Identity Accuracy too low: {identity_acc:.2%}"
