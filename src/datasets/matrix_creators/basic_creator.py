from ..domain.triplets import Triplet
from abc import abstractmethod
import numpy as np
from typing import List


class BasicCreator:
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def construct_mask(sentence_length: int) -> np.ndarray:
        pass

    @abstractmethod
    def construct_gts_matrix(self, triplets: List[Triplet]) -> np.ndarray:
        pass
