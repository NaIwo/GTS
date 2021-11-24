from typing import TypeVar
import numpy as np

from src.datasets.domain.enums import MaskID
from src.config_reader import config
from .base_creator import BaseCreator

Sentence = TypeVar('Sentence')


class MaskCreator(BaseCreator):

    def __init__(self, sentence: Sentence):
        super().__init__(sentence=sentence)

    def construct(self) -> np.ndarray:
        mask: np.ndarray = np.full(config['sentence']['max-length'], MaskID.RELEVANT.value)
        mask[self.sentence.encoded_sentence_length:] = MaskID.NOT_RELEVANT.value
        return mask
