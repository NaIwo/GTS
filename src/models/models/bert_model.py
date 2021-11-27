import tensorflow as tf
from typing import Optional

from .base_model import BaseModel
from src.datasets import Dataset


class BertModel(BaseModel):
    def __init__(self):
        super(BertModel, self).__init__()

    def call(self, data: Dataset, training: Optional[bool] = None, **kwargs) -> tf.Tensor:
        embeddings: tf.Tensor = self.embeddings_layer(data.encoded_sentence, data.mask)

        out = self.inference_layer(embeddings, data.mask3d)
        return out

    def get_config(self):
        pass
