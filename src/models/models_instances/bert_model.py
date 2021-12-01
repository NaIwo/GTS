import tensorflow as tf
from typing import Optional

from .base_model import BaseModel
from src.datasets import Dataset


class BertModel(BaseModel):
    def __init__(self):
        super(BertModel, self).__init__()

    def call(self, inputs: Dataset, training: Optional[bool] = None, **kwargs) -> tf.Tensor:
        embeddings: tf.Tensor = self.embeddings_layer(inputs.encoded_sentence, inputs.mask)
        embeddings: tf.Tensor = self.dropout(embeddings, training=training)

        return self.inference_layer(embeddings, inputs.mask3d)

    def get_config(self):
        pass
