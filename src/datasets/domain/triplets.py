from typing import Dict, List, TypeVar
import numpy as np

from ..matrix_creators.tags_vector_creator import tag_vector_creator
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

    def get_target_tags_vector(self, sequence_length: int) -> np.ndarray:
        return tag_vector_creator.construct_tags_vector(span=self.target_span, sequence_length=sequence_length)

    def get_opinion_tags_vector(self, sequence_length: int) -> np.ndarray:
        return tag_vector_creator.construct_tags_vector(span=self.opinion_span, sequence_length=sequence_length)

    @classmethod
    def from_list(cls, data: List[Dict]) -> List[T]:
        triplets: List[cls] = list()
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
