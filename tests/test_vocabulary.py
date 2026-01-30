import pytest
from vocabulary import Vocabulary

def test_vocabulary_initialization():
    vocab = Vocabulary()
    assert len(vocab) > 0
    assert "the" in vocab.get_words()
    assert "1" in vocab.get_words()

def test_mapping():
    vocab = Vocabulary()
    word = "apple"
    idx = vocab.get_idx(word)
    assert idx != -1
    assert vocab.get_word(idx) == word

def test_unknown_word():
    vocab = Vocabulary()
    assert vocab.get_idx("nonexistentword") == -1
    assert vocab.get_word(9999) == "<UNK>"
