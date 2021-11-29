from tensorflow import keras
import tensorflow as tf
from typing import Optional, Union
from abc import abstractmethod

from ..embeddings import embeddings_layer, GloveFasttext, Indexer, Bert
from src.datasets import Dataset
from ..layers.inference_layer import Inference


class BaseModel(keras.Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.embeddings_layer: Union[GloveFasttext, Indexer, Bert] = embeddings_layer
        self.inference_layer: Inference = Inference()
        self.dropout: tf.keras.layers.Layer = tf.keras.layers.Dropout(0.5)

    @abstractmethod
    def call(self, inputs: Dataset, training: Optional[bool] = None, **kwargs) -> tf.Tensor:
        pass

    def get_config(self):
        pass
