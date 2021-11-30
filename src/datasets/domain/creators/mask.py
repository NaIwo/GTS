from typing import TypeVar
import numpy as np

from src.datasets.domain.enums import MaskID
from src.config_reader import config
from .base_creator import BaseCreator

Sentence = TypeVar('Sentence')


class MaskCreator(BaseCreator):

    def __init__(self, sentence: Sentence):
        super().__init__(sentence=sentence)

    def construct1d(self) -> np.ndarray:
        mask: np.ndarray = np.full(config['sentence']['max-length'], MaskID.RELEVANT.value)
        mask[self.encoded_sentence_length:] = MaskID.NOT_RELEVANT.value
        return mask

    def construct3d(self) -> np.ndarray:
        mask: np.ndarray = self.construct1d()
        mask = np.expand_dims(mask, axis=0)
        mask = np.repeat(mask, repeats=config['sentence']['max-length'], axis=0)
        mask_t: np.ndarray = np.transpose(mask, axes=[1, 0])
        mask *= mask_t
        mask = self.fill_lower_diagonal_matrix(matrix=mask, value=MaskID.NOT_RELEVANT.value)
        mask = np.expand_dims(mask, axis=-1)
        return np.repeat(mask, repeats=self.repeats_number, axis=-1)

    @property
    def repeats_number(self) -> int:
        return config['task']['class-number'][config['task']['type']]
