from typing import Dict, List, TypeVar

from .triplets import Triplet

T = TypeVar('T', bound='Sentence')


class Sentence:
    def __init__(self, sentence_id: int, sentence: str, triplets: List[Triplet]):
        self.sentence_id: int = sentence_id
        self.sentence: str = sentence
        self.triplets: List[Triplet] = triplets

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
