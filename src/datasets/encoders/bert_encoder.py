from .basic_encoder import BasicEncoder
from src.config_reader import config
from typing import List

from transformers import BertTokenizer


class BertEncoder(BasicEncoder):
    def __init__(self):
        super().__init__(encoder_name='bert tokenizer')
        self.encoder: BertTokenizer = BertTokenizer.from_pretrained(config['encoder']['bert']['source'])

    def encode(self, sentence: str) -> List:
        return self.encoder.encode(sentence.split())

