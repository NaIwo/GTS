from src.datasets.encoders.bert_encoder import BertEncoder
from src.datasets.encoders.basic_encoder import BasicEncoder
from src.config_reader import config

from typing import List


class Encoder:
    def __init__(self, encoder: BasicEncoder):
        self._encoder: BasicEncoder = encoder

    def encode(self, sentence: str) -> List:
        return self._encoder.encode(sentence=sentence)


if config['encoder']['type'] == 'bert':
    encoder: Encoder = Encoder(BertEncoder())
else:
    encoder: Encoder = Encoder(BasicEncoder())
