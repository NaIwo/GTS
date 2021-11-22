from typing import List


class BaseEncoder:
    def __init__(self, encoder_name: str = 'basic tokenizer'):
        self.encoder_name: str = encoder_name

    def encode(self, sentence: str) -> List:
        return [len(word) for word in sentence]

    def encode_single_word(self, word: str) -> List:
        return [len(word)]
