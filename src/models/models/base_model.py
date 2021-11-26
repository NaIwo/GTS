import tensorflow as tf
from typing import Optional, Union

from ..embeddings import embeddings_layer, GloveFasttext, Indexer, Bert
from src.datasets import Dataset


class BaseModel(tf.keras.Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.embeddings_layer: Union[GloveFasttext, Indexer, Bert] = embeddings_layer

    def call(self, inputs: Dataset, training: Optional[bool] = None, **kwargs) -> tf.Tensor:
        pass

    def get_config(self):
        pass
