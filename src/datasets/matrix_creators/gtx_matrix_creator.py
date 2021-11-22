from ..domain.triplets import Triplet
from src.config_reader import config
from src.datasets.domain.enums import GTSMatrixID
from ..domain.bio_tags import BioTag

from abc import abstractmethod
import numpy as np
from typing import List


class BasicMatrixCreator:

    @abstractmethod
    def construct_gts_matrix(self, triplets: List[Triplet]) -> np.ndarray:
        pass

    @staticmethod
    def fill_lower_diagonal_matrix_with_not_relevant_values(gts_matrix: np.ndarray) -> np.ndarray:
        mask: np.ndarray = np.full(shape=(config['sentence']['max-length'], config['sentence']['max-length']),
                                   fill_value=1.)
        mask = np.triu(mask, k=0)
        gts_matrix: np.ndarray = np.triu(gts_matrix, k=0)
        return np.where(mask, gts_matrix, GTSMatrixID.NOT_RELEVANT.value)


class BertMatrixCreator(BasicMatrixCreator):
    def __init__(self):
        super().__init__()


class GloveMatrixCreator(BasicMatrixCreator):
    def __init__(self):
        super().__init__()

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


class GtsMatrixCreator:
    def __init__(self, creator: BasicMatrixCreator):
        self._creator: BasicMatrixCreator = creator

    def construct_gts_matrix(self, triplets: List[Triplet]) -> np.ndarray:
        return self._creator.construct_gts_matrix(triplets)


if config['encoder']['type'] == 'bert':
    gts_matrix_creator: GtsMatrixCreator = GtsMatrixCreator(BertMatrixCreator())
elif config['encoder']['type'] == 'glove':
    gts_matrix_creator: GtsMatrixCreator = GtsMatrixCreator(GloveMatrixCreator())
else:
    gts_matrix_creator: GtsMatrixCreator = GtsMatrixCreator(BasicMatrixCreator())
