from typing import Dict, List, TypeVar
import numpy as np

from src.config_reader import config
from .enums import TagVectorID
from .bio_tags import BioTag

T = TypeVar('T', bound='Triplet')


class Triplet:
    def __init__(self, uid: str, target_tags: str, opinion_tags: str, sentiment: str):
        self.uid: str = uid
        self.target_tags: str = target_tags
        self.opinion_tags: str = opinion_tags
        self.sentence_length: int = len(target_tags.split())
        self.sentiment: str = sentiment.upper()

        self.target_span: BioTag = BioTag.from_raw_tags(tags=self.target_tags)
        self.opinion_span: BioTag = BioTag.from_raw_tags(tags=self.opinion_tags)

        self.target_tags_vector: np.ndarray = self._construct_tags_vector(span=self.target_span)
        self.opinion_tags_vector: np.ndarray = self._construct_tags_vector(span=self.opinion_span)

    def _construct_tags_vector(self, span: BioTag) -> np.ndarray:
        tag_vector: np.ndarray = np.full(config['sentence']['max-length'], TagVectorID.OTHER.value)

        tag_vector[self.sentence_length:] = TagVectorID.NOT_RELEVANT.value
        tag_vector[span.start_idx:span.end_idx] = TagVectorID.INSIDE.value
        tag_vector[span.start_idx] = TagVectorID.BEGIN.value

        return tag_vector

    @classmethod
    def from_list(cls, data: List[Dict]) -> List[T]:
        triplets: List[Triplet] = list()
        for d in data:
            uid: str = d['uid']
            target_tags: str = d['target_tags']
            opinion_tags: str = d['opinion_tags']
            sentiment: str = d['sentiment']
            t_object = cls(uid=uid, target_tags=target_tags, opinion_tags=opinion_tags, sentiment=sentiment)
            triplets.append(t_object)

        return triplets

    def __str__(self) -> str:
        return str({
            'uid': self.uid,
            'target_tags': self.target_tags,
            'opinion_tags': self.opinion_tags,
            'sentiment': self.sentiment
        })

    def __repr__(self):
        return self.__str__()
