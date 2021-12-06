from typing import List, TypeVar, Optional

B = TypeVar('B', bound='BioTag')


class BioTag:
    def __init__(self,
                 start_idx: Optional[int] = None,
                 end_idx: Optional[int] = None,
                 span_words: str = ''):
        self.start_idx: Optional[int] = start_idx
        self.end_idx: Optional[int] = end_idx
        self.span_words: str = span_words

    @classmethod
    def from_raw_tags(cls, tags: str) -> List[B]:
        splitted_tags: List = tags.strip().split()

        span: Optional[BioTag] = None
        spans: List[BioTag] = list()

        tag: str
        idx: int
        for idx, tag in enumerate(splitted_tags):
            word: str
            bio_tag: str
            word, bio_tag = tag.strip().split('\\')
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
            'start idx': self.start_idx,
            'end idx': self.end_idx,
            'span words': self.span_words
        })

    def __repr__(self):
        return self.__str__()
