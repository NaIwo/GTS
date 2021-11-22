from typing import Dict, List, TypeVar
import numpy as np
from functools import cached_property

from src.config_reader import config
from src.datasets.domain.enums import TagVectorID, MaskID
from .triplets import Triplet
from ..encoders.encoder import encoder
from ..matrix_creators.gtx_matrix_creator import gts_matrix_creator
from ..matrix_creators.tags_vector_creator import BasicTagCreator

T = TypeVar('T', bound='Sentence')


class Sentence:
    def __init__(self, sentence_id: int, sentence: str, triplets: List[Triplet]):
        self.sentence_id: int = sentence_id
        self.sentence: str = sentence
        self.encoded_sentence: List = encoder.encode(sentence=self.sentence)
        self.sentence_length: int = len(sentence.split())
        self.encoded_sentence_length: int = len(self.encoded_sentence)
        self.triplets: List[Triplet] = triplets

        self.mask: np.ndarray = self.construct_mask()
        self.gts_matrix: np.ndarray = gts_matrix_creator.construct_gts_matrix(triplets=self.triplets)

    def construct_mask(self) -> np.ndarray:
        mask: np.ndarray = np.full(config['sentence']['max-length'], MaskID.RELEVANT.value)
        mask[self.encoded_sentence_length:] = MaskID.NOT_RELEVANT.value
        return mask

    @cached_property
    def target_tags_vector(self) -> np.ndarray:
        tag_vector: np.ndarray = BasicTagCreator.get_raw_tag_vector()
        triplet: Triplet
        for triplet in self.triplets:
            target_tags_vector: np.ndarray = triplet.get_target_tags_vector(self.encoded_sentence_length)
            tag_vector = np.where(target_tags_vector != TagVectorID.OTHER.value, target_tags_vector, tag_vector)
        return tag_vector

    @cached_property
    def opinion_tags_vector(self) -> np.ndarray:
        tag_vector: np.ndarray = BasicTagCreator.get_raw_tag_vector()
        triplet: Triplet
        for triplet in self.triplets:
            opinion_tags_vector: np.ndarray = triplet.get_target_tags_vector(self.encoded_sentence_length)
            tag_vector = np.where(opinion_tags_vector != TagVectorID.OTHER.value, opinion_tags_vector, tag_vector)
        return tag_vector

    @classmethod
    def from_raw_data(cls, data: Dict) -> T:
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
