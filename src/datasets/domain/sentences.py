from typing import Dict, List, TypeVar
import numpy as np
from functools import cached_property

from src.config_reader import config
from src.datasets.domain.enums import TagVectorID, MaskID, GTSMatrixID
from .bio_tags import BioTag
from .triplets import Triplet
from ..encoders.encoder import encoder

T = TypeVar('T', bound='Sentence')


class Sentence:
    def __init__(self, sentence_id: int, sentence: str, triplets: List[Triplet]):
        self.sentence_id: int = sentence_id
        self.sentence: str = sentence
        self.encoded_sentence: List = encoder.encode(sentence=self.sentence)
        self.encoded_words_in_sentence: List = encoder.encode_word_by_word(sentence=self.sentence)
        self.sentence_length: int = len(sentence.split())
        self.encoded_sentence_length: int = len(self.encoded_sentence)
        self.triplets: List[Triplet] = triplets

        self.mask: np.ndarray = self._construct_mask()
        self.target_tags_vector: np.ndarray = self._construct_tags_vector('target_span')
        self.opinion_tags_vector: np.ndarray = self._construct_tags_vector('opinion_span')
        self.gts_matrix: np.ndarray = self._construct_gts_matrix()
        if self.sentence_id == '834':
            print(self.sentence_id)
            print(self.target_tags_vector)
            print(self.opinion_tags_vector)
            print(self.gts_matrix[:20, :20])

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

    def _construct_mask(self) -> np.ndarray:
        mask: np.ndarray = np.full(config['sentence']['max-length'], MaskID.RELEVANT.value)
        mask[self.encoded_sentence_length:] = MaskID.NOT_RELEVANT.value
        return mask

    def _construct_gts_matrix(self) -> np.ndarray:
        gts_matrix: np.ndarray = self._get_raw_gts_matrix()
        triplet: Triplet
        for triplet in self.triplets:
            self._fill_gts_with_target_span_dependencies(gts_matrix, triplet)
            self._fill_gts_based_on_span(gts_matrix, triplet.target_span, GTSMatrixID.TARGET.value)
            self._fill_gts_based_on_span(gts_matrix, triplet.opinion_span, GTSMatrixID.OPINION.value)

        return self.fill_lower_diagonal_matrix_with_not_relevant_values(gts_matrix)

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

    def _get_raw_gts_matrix(self) -> np.ndarray:
        gts_matrix: np.ndarray = np.full(shape=(config['sentence']['max-length'], config['sentence']['max-length']),
                                         fill_value=GTSMatrixID.OTHER.value)
        gts_matrix[:, self.encoded_sentence_length:] = GTSMatrixID.NOT_RELEVANT.value
        self._set_matrix_border_values(gts_matrix)

        return gts_matrix

    def _set_matrix_border_values(self, gts_matrix: np.ndarray) -> None:
        if config['encoder']['type'] == 'bert':
            gts_matrix[0, :] = TagVectorID.NOT_RELEVANT.value
            gts_matrix[:, self.encoded_sentence_length - 1] = GTSMatrixID.NOT_RELEVANT.value

    def _get_raw_tags_vector(self) -> np.ndarray:
        tag_vector: np.ndarray = np.full(config['sentence']['max-length'], TagVectorID.OTHER.value)
        tag_vector[self.encoded_sentence_length:] = TagVectorID.NOT_RELEVANT.value
        self._set_vector_border_values(tag_vector)

        return tag_vector

    def _set_vector_border_values(self, tag_vector: np.ndarray) -> None:
        if config['encoder']['type'] == 'bert':
            tag_vector[0] = TagVectorID.NOT_RELEVANT.value
            tag_vector[self.encoded_sentence_length - 1] = TagVectorID.NOT_RELEVANT.value

    @staticmethod
    def fill_lower_diagonal_matrix_with_not_relevant_values(gts_matrix: np.ndarray) -> np.ndarray:
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
