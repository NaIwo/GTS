import tensorflow as tf
from typing import Optional

from .base_model import BaseModel
from ..utils import trim1d, trim2d
from ..layers.attention_layer import Attention


class BilstmModel(BaseModel):
    def __init__(self):
        super(BilstmModel, self).__init__()
        lstm: tf.keras.layers = tf.keras.layers.LSTM(50, return_sequences=True)
        self.bilstm: tf.keras.layers = tf.keras.layers.Bidirectional(lstm)
        self.relu: tf.keras.activations = tf.keras.activations.relu
        self.attention: tf.keras.layers.Layer = Attention()

    def call(self, encoded_sentence: tf.Tensor, mask: tf.Tensor, mask3d: tf.Tensor, sentence_length: tf.Tensor,
             training: Optional[bool] = None, **kwargs) -> tf.Tensor:

        embeddings: tf.Tensor = self.embeddings_layer(encoded_sentence, mask)
        embeddings = self.dropout(embeddings, training=training)

        trim_len: int = self.get_max_length(sentence_length=sentence_length)
        embeddings = trim1d(embeddings, trim_len)
        mask = trim1d(mask, trim_len)
        mask3d = trim2d(mask3d, trim_len)

        bilstm_out = self.bilstm(embeddings, mask=tf.cast(mask, dtype=bool), training=training)

        attention: tf.Tensor = self.attention(bilstm_out, mask3d)
        bilstm_out += attention

        return self.inference_layer(bilstm_out, mask3d)

    def get_config(self):
        pass
