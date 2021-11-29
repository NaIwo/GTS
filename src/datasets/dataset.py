import os
import json
import random
from typing import Dict, List, Optional, TypeVar
import numpy as np

from src.config_reader import config
from .domain import Sentence, IgnoreIndex

D = TypeVar('D', bound='Dataset')


class Dataset:
    def __init__(self, raw_dataset: List[Sentence]):
        self.sentences: List[Sentence] = raw_dataset
        self.ignore_index: int = IgnoreIndex.IGNORE_INDEX.value

        self.batch_size: int = config['dataset']['batch-size']
        self.shuffle_seed: Optional[int] = config['dataset']['shuffle-seed']

    def __iter__(self) -> D:
        sentences: List[Sentence] = self.sentences
        random.seed(self.shuffle_seed)
        random.shuffle(sentences)
        data: List[Sentence]
        for data in self._get_batch(sentences=sentences, batch_size=self.batch_size):
            yield Dataset(data)

    @staticmethod
    def _get_batch(sentences: List[Sentence], batch_size: int) -> List[Sentence]:
        s_quantity: int = len(sentences)
        ndx: int
        for ndx in range(0, s_quantity, batch_size):
            yield sentences[ndx:min(ndx + batch_size, s_quantity)]

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            sentence: Sentence
            stacked_data: List = [getattr(sentence, name) for sentence in self.sentences]
        except AttributeError as e:
            raise e
        return np.array(stacked_data)


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
