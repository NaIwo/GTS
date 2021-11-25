from .base_encoder import BaseEncoder
from src.config_reader import config
from typing import List, Dict
import json
import os
from pathlib import Path


class IndexEncoder(BaseEncoder):
    def __init__(self):
        super().__init__(encoder_name='index tokenizer')
        data_path: str = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, 'datasets',
                                      'embeddings_data', config['encoder']['indexer']['indexer-path'])
        self.encoder: Dict = json.load(open(data_path))

    def encode(self, sentence: str) -> List:
        encoded_words: List = list()
        word: str
        for word in sentence.strip().split():
            encoded_words += self.encode_single_word(word=word)
        return encoded_words

    def encode_single_word(self, word: str) -> List:
        encoded_words: List = list()
        encoded_words.append(self.encoder[word])
        return encoded_words
