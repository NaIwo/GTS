from typing import TypeVar, List
from functools import cached_property
import numpy as np

from src.config_reader import config

Sentence = TypeVar('Sentence')


class BaseCreator:
    def __init__(self, sentence: Sentence):
        self.sentence: Sentence = sentence

    @cached_property
    def token_range(self) -> List:
        token_range: List = list()
        token_start: int = self.get_offset_based_on_config()
        idx: int
        word: str
        for word in self.sentence.encoded_words_in_sentence:
            token_end = token_start + len(word)
            token_range.append([token_start, token_end - 1])
            token_start = token_end
        return token_range

    @staticmethod
    def get_offset_based_on_config() -> int:
        return 1 if config['encoder']['type'] == 'bert' else 0

    @staticmethod
    def fill_lower_diagonal_matrix(matrix: np.ndarray, value: int) -> np.ndarray:
        mask: np.ndarray = np.full(shape=(config['sentence']['max-length'], config['sentence']['max-length']),
                                   fill_value=1.)
        mask = np.triu(mask, k=0)
        matrix: np.ndarray = np.triu(matrix, k=0)
        return np.where(mask, matrix, value)
