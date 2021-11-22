from src.config_reader import config
from ..domain.enums import TagVectorID
from ..domain.bio_tags import BioTag

from abc import abstractmethod
import numpy as np


class BasicTagCreator:
    @staticmethod
    def get_raw_tag_vector() -> np.ndarray:
        return np.full(config['sentence']['max-length'], TagVectorID.OTHER.value)

    @abstractmethod
    def construct_tags_vector(self, span: BioTag, sequence_length: int) -> np.ndarray:
        pass


class BertTagCreator(BasicTagCreator):

    def construct_tags_vector(self, span: BioTag, sequence_length: int) -> np.ndarray:
        pass


class GloveTagCreator(BasicTagCreator):

    def construct_tags_vector(self, span: BioTag, sequence_length: int) -> np.ndarray:
        tag_vector: np.ndarray = self.get_raw_tag_vector()

        tag_vector[sequence_length:] = TagVectorID.NOT_RELEVANT.value
        tag_vector[span.start_idx:span.end_idx] = TagVectorID.INSIDE.value
        tag_vector[span.start_idx] = TagVectorID.BEGIN.value

        return tag_vector


class TagVectorCreator:
    def __init__(self, creator: BasicTagCreator):
        self._creator: BasicTagCreator = creator

    def construct_tags_vector(self, span: BioTag, sequence_length: int) -> np.ndarray:
        return self._creator.construct_tags_vector(span, sequence_length)


if config['encoder']['type'] == 'bert':
    tag_vector_creator: TagVectorCreator = TagVectorCreator(BertTagCreator())
elif config['encoder']['type'] == 'glove':
    tag_vector_creator: TagVectorCreator = TagVectorCreator(GloveTagCreator())
else:
    tag_vector_creator: TagVectorCreator = TagVectorCreator(BasicTagCreator())
