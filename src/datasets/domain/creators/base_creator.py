from typing import TypeVar, List
from functools import cached_property
import numpy as np

from src.config_reader import config

Sentence = TypeVar('Sentence')


class BaseCreator:
    def __init__(self, sentence: Sentence):
        self.sentence: Sentence = sentence

    def __getattr__(self, name: str):
        try:
            return getattr(self.sentence, name)
        except AttributeError as e:
            raise e

    @staticmethod
    def fill_lower_diagonal_matrix(matrix: np.ndarray, value: int) -> np.ndarray:
        mask: np.ndarray = np.full(shape=(config['sentence']['max-length'], config['sentence']['max-length']),
                                   fill_value=1.)
        mask = np.triu(mask, k=0)
        matrix: np.ndarray = np.triu(matrix, k=0)
        return np.where(mask, matrix, value)
