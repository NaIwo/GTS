from typing import Dict, List, TypeVar, Optional

T = TypeVar('T', bound='Triplet')


class BioTag:
    def __init__(self,
                 start_idx: Optional[int] = None,
                 end_idx: Optional[int] = None,
                 span_words: str = ''):
        self.start_idx: Optional[int] = start_idx
        self.end_idx: Optional[int] = end_idx
        self.span_words: str = span_words

    def __str__(self) -> str:
        return str({
            'start idx': self.start_idx,
            'end idx': self.end_idx,
            'span words': self.span_words
        })

    def __repr__(self):
        return self.__str__()


class Triplet:
    def __init__(self, uid: str, target_tags: str, opinion_tags: str, sentiment: str):
        self.uid: str = uid
        self.target_tags: str = target_tags
        self.opinion_tags: str = opinion_tags
        self.sentiment: str = sentiment

        self.target_span: List[BioTag] = self.get_spans_from_bio_tagging(self.target_tags)
        self.opinion_tags: List[BioTag] = self.get_spans_from_bio_tagging(self.opinion_tags)

    @classmethod
    def from_list(cls, data: List[Dict]) -> List[T]:
        triplets: List[T] = list()
        for d in data:
            uid: str = d['uid']
            target_tags: str = d['target_tags']
            opinion_tags: str = d['opinion_tags']
            sentiment: str = d['sentiment']
            t_object = cls(uid=uid, target_tags=target_tags, opinion_tags=opinion_tags, sentiment=sentiment)
            triplets.append(t_object)

        return triplets

    @staticmethod
    def get_spans_from_bio_tagging(tags: str) -> List[BioTag]:
        spans: List = list()
        splitted_tags: List = tags.strip().split()

        span: Optional[BioTag] = None

        tag: str
        idx: int
        for idx, tag in enumerate(splitted_tags):
            word: str
            bio_tag: str
            word, bio_tag = tag.split('\\')
            if bio_tag == 'B':
                span = BioTag()
                span.start_idx = idx
                span.span_words += word
            elif (bio_tag == 'I') and (span is not None):
                span.span_words += f' {word}'
            if (bio_tag == 'O' or idx == len(splitted_tags) - 1) and (span is not None):
                span.end_idx = idx - (int(bio_tag == 'O'))  # if idx == len(splitted_tags) - 1
                spans.append(span)
                span = None

        return spans

    def __str__(self) -> str:
        return str({
            'uid': self.uid,
            'target_tags': self.target_tags,
            'opinion_tags': self.opinion_tags,
            'sentiment': self.sentiment
        })

    def __repr__(self):
        return self.__str__()
