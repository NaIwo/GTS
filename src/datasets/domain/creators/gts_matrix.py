import numpy as np
from typing import List, TypeVar

from src.datasets.domain.bio_tags import BioTag
from src.datasets.domain.enums import GTSMatrixID, TagVectorID
from src.datasets.domain.triplets import Triplet
from src.config_reader import config
from .base_creator import BaseCreator


Sentence = TypeVar('Sentence')


class GtsMatrixCreator(BaseCreator):
    def __init__(self, sentence: Sentence):
        print('xDD')
        super().__init__(sentence=sentence)

    def construct(self) -> np.ndarray:
        print('xD')
        gts_matrix: np.ndarray = self._get_raw_gts_matrix()
        triplet: Triplet
        for triplet in self.sentence.triplets:
            self._fill_gts_with_target_span_dependencies(gts_matrix, triplet)
            self._fill_gts_based_on_span(gts_matrix, triplet.target_span, GTSMatrixID.TARGET.value)
            self._fill_gts_based_on_span(gts_matrix, triplet.opinion_span, GTSMatrixID.OPINION.value)

        return self.fill_lower_diagonal_matrix(matrix=gts_matrix, value=GTSMatrixID.NOT_RELEVANT.value)

    def _get_raw_gts_matrix(self) -> np.ndarray:
        gts_matrix: np.ndarray = np.full(shape=(config['sentence']['max-length'], config['sentence']['max-length']),
                                         fill_value=GTSMatrixID.OTHER.value)
        gts_matrix[:, self.sentence.encoded_sentence_length:] = GTSMatrixID.NOT_RELEVANT.value
        self._set_matrix_border_values(gts_matrix)

        return gts_matrix

    def _set_matrix_border_values(self, gts_matrix: np.ndarray) -> None:
        if config['encoder']['type'] == 'bert':
            gts_matrix[0, :] = TagVectorID.NOT_RELEVANT.value
            gts_matrix[:, self.sentence.encoded_sentence_length - 1] = GTSMatrixID.NOT_RELEVANT.value

    def _fill_gts_based_on_span(self, gts_matrix: np.ndarray, span: BioTag, value: int) -> None:
        start: List = self.token_range[span.start_idx][0]
        end: List = self.token_range[span.end_idx][1] + 1
        gts_matrix[start:end, start:end] = value
        idx: int
        for idx in range(span.start_idx, span.end_idx):
            tokens: List = self.token_range[idx]
            gts_matrix[tokens[0] + 1:tokens[1] + 1, :] = GTSMatrixID.NOT_RELEVANT.value
            gts_matrix[:, tokens[0] + 1:tokens[1] + 1] = GTSMatrixID.NOT_RELEVANT.value

    def _fill_gts_with_target_span_dependencies(self, gts_matrix: np.ndarray, triplet: Triplet) -> None:
        t_start: List = self.token_range[triplet.target_span.start_idx][0]
        t_end: List = self.token_range[triplet.target_span.end_idx][1] + 1
        o_start: List = self.token_range[triplet.opinion_span.start_idx][0]
        o_end: List = self.token_range[triplet.opinion_span.end_idx][1] + 1

        if triplet.target_span.start_idx < triplet.opinion_span.start_idx:
            gts_matrix[t_start:t_end, o_start:o_end] = GTSMatrixID[triplet.sentiment].value
        else:
            gts_matrix[o_start:o_end, t_start:t_end] = GTSMatrixID[triplet.sentiment].value
