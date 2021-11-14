import os
import json
import random
from typing import Dict, List, Optional, TypeVar

from .domain import Sentence

D = TypeVar('D', bound='Dataset')


class Dataset:
    def __init__(self, raw_dataset: List[Sentence]):
        self.raw_dataset: List[Sentence] = raw_dataset

    def batch(self, batch_size: int = 32, seed: Optional[int] = None) -> D:
        raw_dataset: List[Sentence] = self.raw_dataset
        random.seed(seed)
        random.shuffle(raw_dataset)
        data: List[Sentence]
        for data in self._get_batch(sentences=raw_dataset, batch_size=batch_size):
            yield Dataset(data)

    @staticmethod
    def _get_batch(sentences: List[Sentence], batch_size: int) -> List[Sentence]:
        s_quantity: int = len(sentences)
        ndx: int
        for ndx in range(0, s_quantity, batch_size):
            yield sentences[ndx:min(ndx + batch_size, s_quantity)]


class DatasetReader:
    def __init__(self, dataset_name: str):
        self.dataset_name: str = dataset_name
        datasets_dir_path: str = os.path.dirname(os.path.realpath(__file__))
        self.dataset_path: str = os.path.join(datasets_dir_path, 'datasets', self.dataset_name)

    def read(self, ds_name: str = 'train') -> Dataset:
        file_path: str = os.path.join(self.dataset_path, ds_name + '.json')
        with open(file_path, 'r') as file:
            sentences: list = json.load(file)

        all_sentences: List[Sentence] = list()

        raw_sentence: Dict
        for raw_sentence in sentences:
            sentence: Sentence = Sentence.from_raw_data(data=raw_sentence)
            all_sentences.append(sentence)

        return Dataset(all_sentences)
