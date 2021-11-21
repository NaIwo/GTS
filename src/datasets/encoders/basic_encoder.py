from abc import abstractmethod
from typing import List


class BasicEncoder:
    def __init__(self, encoder_name: str = 'basic tokenizer'):
        self.encoder_name: str = encoder_name

    @abstractmethod
    def encode(self, sentence: str) -> List:
        return [idx for idx, word in enumerate(sentence)]
