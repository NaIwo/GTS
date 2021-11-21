from src.config_reader import config
from .basic_creator import BasicCreator
from .bert_creator import BertCreator
from .glove_creator import GloveCreator
from ..domain.triplets import Triplet

import numpy as np
from typing import List


class Creator:
    def __init__(self, creator: BasicCreator):
        self._creator: BasicCreator = creator

    def construct_mask(self, sentence_length: int) -> np.ndarray:
        return self._creator.construct_mask(sentence_length)

    def construct_gts_matrix(self, triplets: List[Triplet]) -> np.ndarray:
        return self._creator.construct_gts_matrix(triplets)


if config['encoder']['type'] == 'bert':
    creator: Creator = Creator(BertCreator())
elif config['encoder']['type'] == 'glove':
    creator: Creator = Creator(GloveCreator())
else:
    creator: Creator = Creator(BasicCreator())
