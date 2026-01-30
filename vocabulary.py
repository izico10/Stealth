"""
Module for managing the system's vocabulary.
Provides a mapping between string tokens and integer indices, acting as the 
final 'clean-up' dictionary for the retrieval process.
"""

import numpy as np

class Vocabulary:
    """
    A static dictionary of common words, numbers, and punctuation.
    
    In the SAE-SDM architecture, this serves as the ground truth for the 
    'Un-shifting' stage, allowing the system to map retrieved semantic bits 
    back to human-readable tokens.
    """
    
    def __init__(self) -> None:
        """
        Initializes the vocabulary with a predefined list of common tokens.
        """
        self.words: list[str] = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "cat", "dog", "man", "woman", "boy", "girl", "apple", "banana", "car", "bike",
            "run", "jump", "eat", "sleep", "happy", "sad", "big", "small", "red", "blue",
            "king", "queen", "monarch", "prince", "princess",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            ".", ",", "!", "?", ";", ":"
        ]
        self.word_to_idx: dict[str, int] = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word: dict[int, str] = {i: word for i, word in enumerate(self.words)}

    def get_words(self) -> list[str]:
        """Returns the full list of words in the vocabulary."""
        return self.words

    def __len__(self) -> int:
        """Returns the total number of tokens in the vocabulary."""
        return len(self.words)

    def get_word(self, idx: int) -> str:
        """Retrieves the word associated with a specific index."""
        return self.idx_to_word.get(idx, "<UNK>")

    def get_idx(self, word: str) -> int:
        """Retrieves the index associated with a specific word."""
        return self.word_to_idx.get(word, -1)

    def find_nearest(self, query_vector: np.ndarray, word_embeddings: np.ndarray, exclude_words: list[str] = None) -> tuple[str, float]:
        """
        Finds the word in the vocabulary whose embedding is closest to the query_vector.
        
        Args:
            query_vector: The vector to search for.
            word_embeddings: A 2D array of embeddings for every word in the vocab.
            exclude_words: Optional list of words to ignore (e.g., the input words).
            
        Returns:
            A tuple of (nearest_word, similarity_score).
        """
        # Compute cosine similarity
        dot_product = np.dot(word_embeddings, query_vector)
        norms = np.linalg.norm(word_embeddings, axis=1) * np.linalg.norm(query_vector)
        norms[norms == 0] = 1e-10
        similarities = dot_product / norms
        
        if exclude_words:
            for word in exclude_words:
                idx = self.get_idx(word)
                if idx != -1:
                    similarities[idx] = -1.0 # Force similarity to minimum
        
        nearest_idx = np.argmax(similarities)
        return self.words[nearest_idx], float(similarities[nearest_idx])
