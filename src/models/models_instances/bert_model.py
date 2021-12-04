import tensorflow as tf
from typing import Optional

from .base_model import BaseModel


class BertModel(BaseModel):
    def __init__(self):
        super(BertModel, self).__init__()

    def call(self, encoded_sentence: tf.Tensor, mask: tf.Tensor, mask3d: tf.Tensor, training: Optional[bool] = None,
             **kwargs) -> tf.Tensor:
        embeddings: tf.Tensor = self.embeddings_layer(encoded_sentence, mask)
        embeddings: tf.Tensor = self.dropout(embeddings, training=training)

        return self.inference_layer(embeddings, mask3d)

    def get_config(self):
        pass
