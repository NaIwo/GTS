import numpy as np
from typing import TypeVar

from src.datasets.domain.bio_tags import BioTag
from src.datasets.domain.enums import TagVectorID
from src.datasets.domain.triplets import Triplet
from src.config_reader import config
from .base_creator import BaseCreator

Sentence = TypeVar('Sentence')


class TagsVectorCreator(BaseCreator):
    def __init__(self, sentence: Sentence):
        super().__init__(sentence=sentence)

    def construct_target_tags(self) -> np.ndarray:
        return self._construct_tags_vector('target_span')

    def construct_opinion_tags(self) -> np.ndarray:
        return self._construct_tags_vector('opinion_span')

    def _construct_tags_vector(self, span_name: str) -> np.ndarray:
        tag_vector: np.ndarray = self._get_raw_tags_vector()
        triplet: Triplet
        for triplet in self.triplets:
            span: BioTag = getattr(triplet, span_name)
            idx: int
            for idx in range(span.start_idx, span.end_idx):
                tag_vector[self.token_range[idx][0]] = TagVectorID.INSIDE.value
                tag_vector[self.token_range[idx][0] + 1:self.token_range[idx][1] + 1] = TagVectorID.NOT_RELEVANT.value

            tag_vector[self.token_range[span.start_idx][0]] = TagVectorID.BEGIN.value

        return tag_vector

    def _get_raw_tags_vector(self) -> np.ndarray:
        tag_vector: np.ndarray = np.full(config['sentence']['max-length'], TagVectorID.OTHER.value)
        tag_vector[self.encoded_sentence_length:] = TagVectorID.NOT_RELEVANT.value
        self._set_vector_border_values(tag_vector)

        return tag_vector

    def _set_vector_border_values(self, tag_vector: np.ndarray) -> None:
        if config['encoder']['type'] == 'bert':
            tag_vector[0] = TagVectorID.NOT_RELEVANT.value
            tag_vector[self.encoded_sentence_length - 1] = TagVectorID.NOT_RELEVANT.value
