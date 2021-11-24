from typing import List, Callable


class BaseEncoder:
    def __init__(self, encoder_name: str = 'basic tokenizer'):
        self.encoder_name: str = encoder_name

    def encode(self, sentence: str) -> List:
        return sentence.strip().split()

    def encode_single_word(self, word: str) -> List:
        return [word]
