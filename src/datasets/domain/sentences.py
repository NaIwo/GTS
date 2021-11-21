from typing import Dict, List, TypeVar
import numpy as np

from .triplets import Triplet
from ..encoders.encoder import encoder
from ..matrix_creators.creator import creator

T = TypeVar('T', bound='Sentence')


class Sentence:
    def __init__(self, sentence_id: int, sentence: str, triplets: List[Triplet]):
        self.sentence_id: int = sentence_id
        self.sentence: str = sentence
        self.encoded_sentence: List = encoder.encode(sentence=self.sentence)
        self.sentence_length: int = len(sentence.split())
        self.encoded_sentence_length: int = len(self.encoded_sentence)
        self.triplets: List[Triplet] = triplets

        self.mask: np.ndarray = creator.construct_mask(sentence_length=self.sentence_length)
        self.gts_matrix: np.ndarray = creator.construct_gts_matrix(triplets=self.triplets)

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
