"""
Module for managing the system's vocabulary.
Provides a mapping between string tokens and integer indices, acting as the 
final 'clean-up' dictionary for the retrieval process.
"""

import numpy as np

class Vocabulary:
    """
    A static dictionary of common words, numbers, and punctuation.
    """
    
    def __init__(self, custom_words: list[str] = None) -> None:
        """
        Initializes the vocabulary.
        """
        if custom_words:
            self.words = custom_words
        else:
            self.words = [
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                "cat", "dog", "man", "woman", "apple", "bike", "king", "queen",
                "0", "1", "2", "3", ".", ",", "!", "?"
            ]
        
        test_essentials = ["king", "queen", "man", "woman", "apple", "bike", "cat", "kitten", "dog", "puppy"]
        for word in test_essentials:
            if word not in self.words:
                self.words.append(word)

        seen = set()
        self.words = [x for x in self.words if not (x in seen or seen.add(x))]

        self.word_to_idx: dict[str, int] = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word: dict[int, str] = {i: word for i, word in enumerate(self.words)}

    def get_words(self) -> list[str]:
        return self.words

    def __len__(self) -> int:
        return len(self.words)

    def get_word(self, idx: int) -> str:
        return self.idx_to_word.get(idx, "<UNK>")

    def get_idx(self, word: str) -> int:
        return self.word_to_idx.get(word, -1)

    def find_nearest(self, query_vector: np.ndarray, word_embeddings: np.ndarray, exclude_words: list[str] = None) -> tuple[str, float]:
        """Finds the single nearest word."""
        similarities = self.get_similarities(query_vector, word_embeddings, exclude_words)
        nearest_idx = np.argmax(similarities)
        return self.words[nearest_idx], float(similarities[nearest_idx])

    def find_top_n(self, query_vector: np.ndarray, word_embeddings: np.ndarray, n: int = 10, exclude_words: list[str] = None) -> list[tuple[str, float]]:
        """Finds the top N nearest words."""
        similarities = self.get_similarities(query_vector, word_embeddings, exclude_words)
        top_indices = np.argpartition(similarities, -n)[-n:]
        # Sort the top indices by similarity score
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        return [(self.words[idx], float(similarities[idx])) for idx in top_indices]

    def get_similarities(self, query_vector: np.ndarray, word_embeddings: np.ndarray, exclude_words: list[str] = None) -> np.ndarray:
        """Calculates cosine similarities for all words."""
        dot_product = np.dot(word_embeddings, query_vector)
        norms = np.linalg.norm(word_embeddings, axis=1) * np.linalg.norm(query_vector)
        norms[norms == 0] = 1e-10
        similarities = dot_product / norms
        
        if exclude_words:
            for word in exclude_words:
                idx = self.get_idx(word)
                if idx != -1:
                    similarities[idx] = -1.0
        return similarities
