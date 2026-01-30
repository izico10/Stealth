"""
Module for managing the system's vocabulary.
Provides a mapping between string tokens and integer indices, acting as the 
final 'clean-up' dictionary for the retrieval process.
"""

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
        # A representative sample for the prototype.
        self.words: list[str] = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "cat", "dog", "man", "woman", "boy", "girl", "apple", "banana", "car", "bike",
            "run", "jump", "eat", "sleep", "happy", "sad", "big", "small", "red", "blue",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            ".", ",", "!", "?", ";", ":"
        ]
        self.word_to_idx: dict[str, int] = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word: dict[int, str] = {i: word for i, word in enumerate(self.words)}

    def get_words(self) -> list[str]:
        """
        Returns the full list of words in the vocabulary.
        """
        return self.words

    def __len__(self) -> int:
        """
        Returns the total number of tokens in the vocabulary.
        """
        return len(self.words)

    def get_word(self, idx: int) -> str:
        """
        Retrieves the word associated with a specific index.
        
        Args:
            idx: The integer index of the word.
            
        Returns:
            The string token, or '<UNK>' if the index is out of range.
        """
        return self.idx_to_word.get(idx, "<UNK>")

    def get_idx(self, word: str) -> int:
        """
        Retrieves the index associated with a specific word.
        
        Args:
            word: The string token to look up.
            
        Returns:
            The integer index, or -1 if the word is not in the vocabulary.
        """
        return self.word_to_idx.get(word, -1)
