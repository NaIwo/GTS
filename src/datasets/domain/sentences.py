from typing import Dict, List, TypeVar
import numpy as np

from src.config_reader import config
from .bio_tags import BioTag
from .triplets import Triplet
from .enums import GTSMatrixID, MaskID

T = TypeVar('T', bound='Sentence')


class Sentence:
    def __init__(self, sentence_id: int, sentence: str, triplets: List[Triplet]):
        self.sentence_id: int = sentence_id
        self.sentence: str = sentence
        self.sentence_length: int = len(sentence.split())
        self.triplets: List[Triplet] = triplets

        self.mask: np.ndarray = self._construct_mask()
        self.gts_matrix: np.ndarray = self._construct_gts_matrix()

    def _construct_mask(self) -> np.ndarray:
        mask: np.ndarray = np.full(config['sentence']['max-length'], MaskID.RELEVANT.value)
        mask[self.sentence_length:] = MaskID.NOT_RELEVANT.value
        return mask

    def _construct_gts_matrix(self) -> np.ndarray:
        gts_matrix: np.ndarray = np.full(shape=(config['sentence']['max-length'], config['sentence']['max-length']),
                                         fill_value=GTSMatrixID.OTHER.value)

        triplet: Triplet
        for triplet in self.triplets:
            self._fill_matrix_based_on_span(gts_matrix, triplet.target_span, triplet.target_span,
                                            GTSMatrixID.TARGET.value)
            self._fill_matrix_based_on_span(gts_matrix, triplet.opinion_span, triplet.opinion_span,
                                            GTSMatrixID.OPINION.value)
            if triplet.target_span.start_idx < triplet.opinion_span.start_idx:
                self._fill_matrix_based_on_span(gts_matrix, triplet.target_span, triplet.opinion_span,
                                                GTSMatrixID[triplet.sentiment].value)
            else:
                self._fill_matrix_based_on_span(gts_matrix, triplet.opinion_span, triplet.target_span,
                                                GTSMatrixID[triplet.sentiment].value)

        return self._fill_lower_diagonal_matrix_with_not_relevant_values(gts_matrix=gts_matrix)

    @staticmethod
    def _fill_matrix_based_on_span(gts_matrix: np.ndarray, fist_span: BioTag, second_span: BioTag,
                                   value: int) -> None:
        gts_matrix[fist_span.start_idx:fist_span.end_idx + 1, second_span.start_idx:second_span.end_idx + 1] = value

    @staticmethod
    def _fill_lower_diagonal_matrix_with_not_relevant_values(gts_matrix: np.ndarray) -> np.ndarray:
        mask: np.ndarray = np.full(shape=(config['sentence']['max-length'], config['sentence']['max-length']),
                                   fill_value=1.)
        mask = np.triu(mask, k=0)
        gts_matrix: np.ndarray = np.triu(gts_matrix, k=0)
        return np.where(mask, gts_matrix, GTSMatrixID.NOT_RELEVANT.value)

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
