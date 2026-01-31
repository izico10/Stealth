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
SDR_DIMS = 1024*10
DENSITY_K = int((6/100) * SDR_DIMS)
VOCAB_SIZE = 10000
N_EPOCHS = 300
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
    tokens = embedder.get_vocabulary_tokens(max_tokens=VOCAB_SIZE)
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
    """Generates a dataset of normalized contextual embeddings with a proper split."""
    all_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the modern world.",
        "A king rules over his kingdom with wisdom and justice.",
        "The cat sat on the mat and watched the birds outside.",
        "Quantum computing relies on the principles of superposition.",
        "Climate change is a global challenge that requires urgent action.",
        "The history of the universe began with the big bang.",
        "Music is a universal language that connects all people.",
        "Space exploration opens new frontiers for humanity.",
        "Sustainable energy is key to a greener future.",
        "The ocean is deep and full of mysterious creatures.",
        "Technology continues to evolve at an exponential rate.",
        "Reading books expands the mind and improves knowledge.",
        "Exercise and a healthy diet are essential for well-being.",
        "The mountains are covered in snow during the winter.",
        "Innovation drives progress in every field of science.",
        "Communication is the foundation of strong relationships.",
        "The stars twinkle brightly in the clear night sky.",
        "History repeats itself for those who do not learn from it.",
        "Art reflects the culture and values of a society.",
        "The wind blows softly through the trees in the park.",
        "Mathematics is the language of the physical universe.",
        "Patience is a virtue that leads to great success.",
        "The sun provides energy for all life on Earth.",
        "Dreams are the windows to our subconscious mind.",
        "A journey of a thousand miles begins with a single step.",
        "Knowledge is power but character is more important.",
        "The best way to predict the future is to create it.",
        "Success is not final, failure is not fatal.",
        "Imagination is more important than knowledge.",
        "Life is what happens when you are busy making other plans.",
        "The only thing we have to fear is fear itself.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The pen is mightier than the sword.",
        "Actions speak louder than words.",
        "Beauty is in the eye of the beholder.",
        "Better late than never.",
        "Cleanliness is next to godliness.",
        "Don't count your chickens before they hatch.",
        "Easy come, easy go.",
        "Every cloud has a silver lining.",
        "Fortune favors the bold.",
        "Honesty is the best policy.",
        "If it ain't broke, don't fix it.",
        "Laughter is the best medicine.",
        "Look before you leap.",
        "Practice makes perfect.",
        "The early bird catches the worm.",
        "There is no place like home.",
        "Two heads are better than one.",
        "When in Rome, do as the Romans do.",
        "Where there is a will, there is a way.",
        "You can't judge a book by its cover.",
        "A bird in the hand is worth two in the bush.",
        "Absence makes the heart grow fonder.",
        "All good things must come to an end.",
        "An apple a day keeps the doctor away.",
        "Barking dogs seldom bite.",
        "Beggars can't be choosers.",
        "Better safe than sorry.",
        "Blood is thicker than water.",
        "Don't put all your eggs in one basket.",
        "Every dog has its day.",
        "Great minds think alike.",
        "Haste makes waste.",
        "Keep your friends close and your enemies closer.",
        "Knowledge is power.",
        "Like father, like son.",
        "Love is blind.",
        "Necessity is the mother of invention.",
        "No man is an island.",
        "Out of sight, out of mind.",
        "Rome wasn't built in a day.",
        "The grass is always greener on the other side.",
        "The truth will set you free.",
        "Time heals all wounds.",
        "Variety is the spice of life.",
        "What goes around comes around.",
        "You reap what you sow.",
        "A chain is only as strong as its weakest link.",
        "A house divided against itself cannot stand.",
        "A penny saved is a penny earned.",
        "All's fair in love and war.",
        "Birds of a feather flock together.",
        "Don't bite the hand that feeds you.",
        "Don't cry over spilled milk.",
        "Familiarity breeds contempt.",
        "Good things come to those who wait.",
        "If you can't beat them, join them.",
        "Ignorance is bliss.",
        "It takes two to tango.",
        "Kill two birds with one stone.",
        "Let sleeping dogs lie.",
        "Money doesn't grow on trees.",
        "Necessity is the mother of invention.",
        "Old habits die hard.",
        "One man's trash is another man's treasure.",
        "Opposites attract.",
        "Slow and steady wins the race.",
        "The customer is always right.",
        "The show must go on.",
        "There's no such thing as a free lunch.",
        "Think before you speak.",
        "Too many cooks spoil the broth.",
        "Well begun is half done.",
        "What doesn't kill you makes you stronger.",
        "You can lead a horse to water, but you can't make it drink.",
        "The capital of France is Paris.",
        "Water boils at one hundred degrees Celsius.",
        "The Earth orbits the Sun once every year.",
        "Oxygen is necessary for human life to exist.",
        "The Great Wall of China is a famous landmark.",
        "Computers process information using binary code.",
        "The human heart pumps blood throughout the body.",
        "Photosynthesis is the process by which plants make food.",
        "Gravity is the force that pulls objects toward the center of the Earth.",
        "The Amazon rainforest is home to many species.",
        "The Pyramids of Giza were built in ancient Egypt.",
        "Light travels at a speed of approximately three hundred thousand kilometers per second.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Mount Everest is the highest mountain in the world.",
        "The Mona Lisa is a famous painting by Leonardo da Vinci.",
        "The Roman Empire was one of the most powerful in history.",
        "The Industrial Revolution changed the way goods were produced.",
        "The internet has revolutionized communication and information sharing.",
        "Democracy is a system of government by the whole population.",
        "The United Nations was founded to promote international cooperation.",
        "Global warming is caused by the increase of greenhouse gases.",
        "The human brain is the most complex organ in the body.",
        "DNA contains the genetic instructions for all living organisms.",
        "The solar system consists of the Sun and eight planets.",
        "The theory of relativity was developed by Albert Einstein.",
        "The Magna Carta is a famous document in English history.",
        "The Renaissance was a period of great cultural and artistic growth.",
        "The French Revolution led to the end of the monarchy in France.",
        "The American Civil War was fought between the North and the South.",
        "The Second World War was the largest conflict in human history.",
        "The Cold War was a period of tension between the US and the Soviet Union.",
        "The fall of the Berlin Wall marked the end of the Cold War.",
        "The digital age has transformed every aspect of modern life.",
        "Artificial intelligence has the potential to solve complex problems.",
        "Space exploration has led to many scientific discoveries.",
        "The discovery of penicillin revolutionized medicine.",
        "The invention of the printing press made books more accessible.",
        "The steam engine powered the Industrial Revolution.",
        "The telephone made long-distance communication possible.",
        "The light bulb changed the way people live and work.",
        "The automobile made travel faster and more convenient.",
        "The airplane made it possible to travel across the world in hours.",
        "The television became a major source of news and entertainment.",
        "The computer has become an essential tool in modern society.",
        "The smartphone has changed the way people communicate and access information.",
        "Social media has transformed the way people interact and share ideas.",
        "The blockchain technology has the potential to revolutionize finance.",
        "Renewable energy is essential for a sustainable future.",
        "Electric vehicles are becoming more popular as a way to reduce emissions.",
        "Genetic engineering has the potential to cure many diseases.",
        "Nanotechnology is being used to create new materials and devices.",
        "Robotics is being used to automate many tasks in industry and medicine.",
        "Virtual reality is being used for training and entertainment.",
        "Augmented reality is being used to enhance the real world with digital information.",
        "The Internet of Things connects everyday objects to the internet.",
        "Big data is being used to analyze and predict trends in many fields.",
        "Cloud computing allows people to store and access data over the internet.",
        "Cybersecurity is essential for protecting information in the digital age.",
        "The future of technology is full of possibilities and challenges.",
        "A stitch in time saves nine.",
        "All work and no play makes Jack a dull boy.",
        "Birds of a feather flock together.",
        "Curiosity killed the cat.",
        "Don't judge a book by its cover.",
        "Every cloud has a silver lining.",
        "Great minds think alike.",
        "Honesty is the best policy.",
        "It takes two to tango.",
        "Laughter is the best medicine.",
        "Necessity is the mother of invention.",
        "Practice makes perfect.",
        "The early bird catches the worm.",
        "There is no place like home.",
        "Two heads are better than one.",
        "When in Rome, do as the Romans do.",
        "Where there is a will, there is a way.",
        "You can't have your cake and eat it too.",
        "A penny saved is a penny earned.",
        "Actions speak louder than words.",
        "All's well that ends well.",
        "Better safe than sorry.",
        "Don't cry over spilled milk.",
        "Easy come, easy go.",
        "Fortune favors the bold.",
        "Good things come to those who wait.",
        "Haste makes waste.",
        "If it ain't broke, don't fix it.",
        "Keep your friends close and your enemies closer.",
        "Knowledge is power.",
        "Like father, like son.",
        "Love is blind.",
        "No man is an island.",
        "Out of sight, out of mind.",
        "Rome wasn't built in a day.",
        "The grass is always greener on the other side.",
        "The truth will set you free.",
        "Time heals all wounds.",
        "Variety is the spice of life.",
        "What goes around comes around.",
        "You reap what you sow.",
        "A chain is only as strong as its weakest link.",
        "A house divided against itself cannot stand.",
        "All's fair in love and war.",
        "Don't bite the hand that feeds you.",
        "Familiarity breeds contempt.",
        "If you can't beat them, join them.",
        "Ignorance is bliss.",
        "Kill two birds with one stone.",
        "Let sleeping dogs lie.",
        "Money doesn't grow on trees.",
        "Old habits die hard.",
        "One man's trash is another man's treasure.",
        "Opposites attract.",
        "Slow and steady wins the race.",
        "The customer is always right.",
        "The show must go on.",
        "There's no such thing as a free lunch.",
        "Think before you speak.",
        "Too many cooks spoil the broth.",
        "Well begun is half done.",
        "What doesn't kill you makes you stronger.",
        "You can lead a horse to water, but you can't make it drink.",
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The pen is mightier than the sword.",
        "Beauty is in the eye of the beholder.",
        "Better late than never.",
        "Cleanliness is next to godliness.",
        "Don't count your chickens before they hatch.",
        "Every dog has its day.",
        "The early bird catches the worm.",
        "There is no place like home.",
        "Two heads are better than one.",
        "When in Rome, do as the Romans do.",
        "Where there is a will, there is a way.",
        "You can't judge a book by its cover.",
        "A bird in the hand is worth two in the bush.",
        "Absence makes the heart grow fonder.",
        "All good things must come to an end.",
        "An apple a day keeps the doctor away.",
        "Barking dogs seldom bite.",
        "Beggars can't be choosers.",
        "Better safe than sorry.",
        "Blood is thicker than water.",
        "Don't put all your eggs in one basket.",
        "Every dog has its day.",
        "Great minds think alike.",
        "Haste makes waste.",
        "Keep your friends close and your enemies closer.",
        "Knowledge is power.",
        "Like father, like son.",
        "Love is blind.",
        "Necessity is the mother of invention.",
        "No man is an island.",
        "Out of sight, out of mind.",
        "Rome wasn't built in a day.",
        "The grass is always greener on the other side.",
        "The truth will set you free.",
        "Time heals all wounds.",
        "Variety is the spice of life.",
        "What goes around comes around.",
        "You reap what you sow.",
        "A chain is only as strong as its weakest link.",
        "A house divided against itself cannot stand.",
        "A penny saved is a penny earned.",
        "All's fair in love and war.",
        "Birds of a feather flock together.",
        "Don't bite the hand that feeds you.",
        "Don't cry over spilled milk.",
        "Familiarity breeds contempt.",
        "Good things come to those who wait.",
        "If you can't beat them, join them.",
        "Ignorance is bliss.",
        "It takes two to tango.",
        "Kill two birds with one stone.",
        "Let sleeping dogs lie.",
        "Money doesn't grow on trees.",
        "Necessity is the mother of invention.",
        "Old habits die hard.",
        "One man's trash is another man's treasure.",
        "Opposites attract.",
        "Slow and steady wins the race.",
        "The customer is always right.",
        "The show must go on.",
        "There's no such thing as a free lunch.",
        "Think before you speak.",
        "Too many cooks spoil the broth.",
        "Well begun is half done.",
        "What doesn't kill you makes you stronger.",
        "You can lead a horse to water, but you can't make it drink."
    ]
    
    # Proper 70/15/15 Split
    np.random.seed(42)
    np.random.shuffle(all_sentences)
    
    n = len(all_sentences)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_sents = all_sentences[:train_end]
    val_sents = all_sentences[train_end:val_end]
    test_sents = all_sentences[val_end:]
    
    split_sentences['train'] = train_sents
    split_sentences['val'] = val_sents
    split_sentences['test'] = test_sents
    
    def get_embs(sentences):
        all_embs = []
        for sent in sentences:
            embs = embedder.embed_contextual(sent)
            all_embs.append(normalize_vectors(embs))
        return np.vstack(all_embs)
        
    return {
        "train": get_embs(train_sents),
        "val": get_embs(val_sents)
    }

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
    
    logger.info(f"Config: SDR_DIMS={SDR_DIMS}, DENSITY_K={DENSITY_K}, EPOCHS={N_EPOCHS}, LR={LR}")
    logger.info(f"Training Sequential SAE on {len(contextual_data['train'])} contextual tokens...")
    
    sae.train(
        contextual_data["train"], 
        val_data=contextual_data["val"], 
        epochs=N_EPOCHS, 
        lr=LR, 
        batch_size=BATCH_SIZE
    )
    return sae

def test_sequential_reconstruction(embedder: WordEmbedder, seq_sae: SparseAutoencoder, vocab: Vocabulary) -> None:
    """
    Tests if the SAE can reconstruct contextual embeddings from UNSEEN sentences.
    """
    test_sents = split_sentences.get('test', [])
    if not test_sents:
        pytest.skip("No test data available.")
        
    logger.info(f"--- Sequential Reconstruction Test ({len(test_sents)} Unseen Sentences) ---")
    
    total_sim = 0.0
    total_tokens = 0
    
    all_vocab_words = vocab.get_words()
    all_vocab_embs = embedder.embed_batch(all_vocab_words)
    
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    
    for sent in test_sents:
        v_contextual = normalize_vectors(embedder.embed_contextual(sent))
        num_tokens = v_contextual.shape[0]
        
        logger.info(f"Sentence: '{sent}'")
        for i in range(num_tokens):
            v_orig = v_contextual[i]
            sdr = seq_sae.encode(v_orig)
            v_rec = seq_sae.decode(sdr)
            
            sim = cosine_similarity(v_orig, v_rec)
            total_sim += sim
            total_tokens += 1
            
            # Top-N Accuracy Check
            top_n = vocab.find_top_n(v_rec, all_vocab_embs, n=10)
            top_n_words = [item[0] for item in top_n]
            
            # Get token string for logging
            tokens = embedder.model.tokenizer.tokenize(sent)
            full_tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_str = full_tokens[i] if i < len(full_tokens) else "?"
            
            # Clean token for comparison
            clean_token = token_str.lower().replace("##", "")
            
            if clean_token == top_n_words[0].lower():
                top1_correct += 1
            if clean_token in [w.lower() for w in top_n_words[:5]]:
                top5_correct += 1
            if clean_token in [w.lower() for w in top_n_words[:10]]:
                top10_correct += 1
            
            logger.info(f"  Token '{token_str}': Sim={sim:.4f}, Nearest='{top_n_words[0]}'")

    avg_sim = total_sim / total_tokens
    top1_acc = top1_correct / total_tokens
    
    logger.info(f"Average Reconstruction Similarity: {avg_sim:.4f}")
    logger.info(f"Top-1 Accuracy:  {top1_acc:.2%}")
    logger.info(f"Top-5 Accuracy:  {top5_correct/total_tokens:.2%}")
    logger.info(f"Top-10 Accuracy: {top10_correct/total_tokens:.2%}")
    
    assert top1_acc >= 0.90, f"Sequential Top-1 accuracy too low: {top1_acc:.2%}"
