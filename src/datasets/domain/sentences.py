from typing import Dict, List, TypeVar
import numpy as np
from functools import cached_property

from src.config_reader import config
from .enums import MaskID
from .triplets import Triplet
from ..encoders.encoder import encoder
from src.datasets.domain.creators import GtsMatrixCreator, TagsVectorCreator, MaskCreator

S = TypeVar('S', bound='Sentence')


class Sentence:
    def __init__(self, sentence_id: int, sentence: str, triplets: List[Triplet]):
        self.sentence_id: int = sentence_id
        self.sentence: str = sentence
        self._encoded_sentence: List = encoder.encode(sentence=self.sentence)
        self.encoded_words_in_sentence: List = encoder.encode_word_by_word(sentence=self.sentence)
        self.sentence_length: int = len(sentence.split())
        self.encoded_sentence_length: int = len(self._encoded_sentence)
        self.triplets: List[Triplet] = triplets

    @cached_property
    def token_range(self) -> List:
        token_range: List = list()
        token_start: int = self.get_offset_based_on_config()
        idx: int
        word: str
        for word in self.encoded_words_in_sentence:
            token_end = token_start + len(word)
            token_range.append([token_start, token_end - 1])
            token_start = token_end
        return token_range

    @staticmethod
    def get_offset_based_on_config() -> int:
        return 1 if config['encoder']['type'] == 'bert' else 0

    @property
    def mask(self) -> np.ndarray:
        return MaskCreator(self).construct1d()

    @property
    def mask3d(self) -> np.ndarray:
        return MaskCreator(self).construct3d()

    @property
    def target_tags_vector(self) -> np.ndarray:
        return TagsVectorCreator(self).construct_target_tags()

    @property
    def opinion_tags_vector(self) -> np.ndarray:
        return TagsVectorCreator(self).construct_opinion_tags()

    @cached_property
    def gts_matrix(self) -> np.ndarray:
        return GtsMatrixCreator(self).construct()

    @property
    def encoded_sentence(self) -> np.ndarray:
        if isinstance(self._encoded_sentence[0], str):
            encoded: np.ndarray = np.full(config['sentence']['max-length'], '', dtype=object)
            encoded[:len(self._encoded_sentence)] = np.array(self._encoded_sentence, dtype=object)
            return encoded
        else:
            encoded: np.ndarray = np.full(config['sentence']['max-length'], MaskID.NOT_RELEVANT.value)
            encoded[:len(self._encoded_sentence)] = self._encoded_sentence
            return encoded

    @classmethod
    def from_raw_data(cls, data: Dict) -> S:
        sentence_id: int = data['id']
        sentence: str = data['sentence']
        triplets: List[Triplet] = Triplet.from_list(data=data['triples'])
        return cls(sentence_id=sentence_id, sentence=sentence, triplets=triplets)

    def __str__(self) -> str:
        return str({
            'sentence_id': self.sentence_id,
            'sentence:': self.sentence,
            'triplets': self.triplets
        })
