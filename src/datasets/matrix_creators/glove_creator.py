from .basic_creator import BasicCreator
from src.config_reader import config
from ..domain.bio_tags import BioTag
from ..domain.triplets import Triplet
from src.datasets.domain.enums import GTSMatrixID, MaskID

import numpy as np
from typing import List


class GloveCreator(BasicCreator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def construct_mask(sentence_length: int) -> np.ndarray:
        mask: np.ndarray = np.full(config['sentence']['max-length'], MaskID.RELEVANT.value)
        mask[sentence_length:] = MaskID.NOT_RELEVANT.value
        return mask

    def construct_gts_matrix(self, triplets: List[Triplet]) -> np.ndarray:
        gts_matrix: np.ndarray = np.full(shape=(config['sentence']['max-length'], config['sentence']['max-length']),
                                         fill_value=GTSMatrixID.OTHER.value)

        triplet: Triplet
        for triplet in triplets:
            self.fill_matrix_based_on_span(gts_matrix, triplet.target_span, triplet.target_span,
                                           GTSMatrixID.TARGET.value)
            self.fill_matrix_based_on_span(gts_matrix, triplet.opinion_span, triplet.opinion_span,
                                           GTSMatrixID.OPINION.value)
            if triplet.target_span.start_idx < triplet.opinion_span.start_idx:
                self.fill_matrix_based_on_span(gts_matrix, triplet.target_span, triplet.opinion_span,
                                               GTSMatrixID[triplet.sentiment].value)
            else:
                self.fill_matrix_based_on_span(gts_matrix, triplet.opinion_span, triplet.target_span,
                                               GTSMatrixID[triplet.sentiment].value)

        return self.fill_lower_diagonal_matrix_with_not_relevant_values(gts_matrix=gts_matrix)

    @staticmethod
    def fill_matrix_based_on_span(gts_matrix: np.ndarray, fist_span: BioTag, second_span: BioTag,
                                  value: int) -> None:
        gts_matrix[fist_span.start_idx:fist_span.end_idx + 1, second_span.start_idx:second_span.end_idx + 1] = value

    @staticmethod
    def fill_lower_diagonal_matrix_with_not_relevant_values(gts_matrix: np.ndarray) -> np.ndarray:
        mask: np.ndarray = np.full(shape=(config['sentence']['max-length'], config['sentence']['max-length']),
                                   fill_value=1.)
        mask = np.triu(mask, k=0)
        gts_matrix: np.ndarray = np.triu(gts_matrix, k=0)
        return np.where(mask, gts_matrix, GTSMatrixID.NOT_RELEVANT.value)
