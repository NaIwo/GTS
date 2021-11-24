from typing import TypeVar, List
from functools import cached_property

from src.config_reader import config

Sentence = TypeVar('Sentence')


class BaseCreator:
    def __init__(self, sentence: Sentence):
        self.sentence: Sentence = sentence

    @cached_property
    def token_range(self) -> List:
        token_range: List = list()
        token_start: int = self.get_offset_based_on_config()
        idx: int
        word: str
        for word in self.sentence.encoded_words_in_sentence:
            token_end = token_start + len(word)
            token_range.append([token_start, token_end - 1])
            token_start = token_end
        return token_range

    @staticmethod
    def get_offset_based_on_config() -> int:
        return 1 if config['encoder']['type'] == 'bert' else 0
