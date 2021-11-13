import os
import json
import random
from typing import Dict, List

from .domain import Sentence


class DatasetReader:
    def __init__(self, dataset_name: str, batch_size: int = 32, seed: int = None):
        self.dataset_name: str = dataset_name
        datasets_dir_path: str = os.path.dirname(os.path.realpath(__file__))
        self.dataset_path: str = os.path.join(datasets_dir_path, 'datasets', self.dataset_name)

        self.batch_size: int = batch_size
        self.seed: int = seed

        self._train_dataset = self._construct_dataset('train')
        self.dev_dataset = self._construct_dataset('dev')
        self.test_dataset = self._construct_dataset('test')

    def _construct_dataset(self, ds_name: str = 'train') -> List[Sentence]:
        file_path: str = os.path.join(self.dataset_path, ds_name + '.json')
        with open(file_path, 'r') as file:
            sentences: list = json.load(file)

        all_sentences: List[Sentence] = list()

        raw_sentence: Dict
        for raw_sentence in sentences:
            sentence: Sentence = Sentence.from_raw_data(data=raw_sentence)
            all_sentences.append(sentence)

        return all_sentences

    @property
    def train_dataset(self) -> List[Sentence]:
        train_dataset: List[Sentence] = self._train_dataset
        random.seed(self.seed)
        random.shuffle(train_dataset)
        data: List[Sentence]
        for data in self.batch(train_dataset):
            yield data

    def batch(self, sentences: List[Sentence]) -> List[Sentence]:
        s_quantity: int = len(sentences)
        ndx: int
        for ndx in range(0, s_quantity, self.batch_size):
            yield sentences[ndx:min(ndx + self.batch_size, s_quantity)]
