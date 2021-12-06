import tensorflow as tf
from typing import Optional

from .base_model import BaseModel
from ..utils import trim1d, trim2d


class CnnModel(BaseModel):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.domain: tf.keras.layers.Layer = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same")
        self.general: tf.keras.layers.Layer = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same")
        self.conv1: tf.keras.layers.Layer = tf.keras.layers.Conv1D(256, kernel_size=5, padding="same")
        self.conv2: tf.keras.layers.Layer = tf.keras.layers.Conv1D(256, kernel_size=5, padding="same")
        self.conv3: tf.keras.layers.Layer = tf.keras.layers.Conv1D(256, kernel_size=5, padding="same")
        self.relu: tf.keras.layers.Layer = tf.keras.layers.ReLU()

    def call(self, encoded_sentence: tf.Tensor, mask: tf.Tensor, mask3d: tf.Tensor, sentence_length: tf.Tensor,
             training: Optional[bool] = None, **kwargs) -> tf.Tensor:

        embeddings: tf.Tensor = self.embeddings_layer(encoded_sentence, mask)
        embeddings = self.dropout(embeddings, training=training)
        domain_general_conv: tf.Tensor = tf.concat((self.general(embeddings), self.domain(embeddings)), axis=-1)
        conv_out: tf.Tensor = self.relu(domain_general_conv)
        conv_out = self.dropout(conv_out, training=training)

        conv: tf.keras.layers.Layer
        for conv in [self.conv1, self.conv2, self.conv3]:
            conv_out = conv(conv_out)
            conv_out = self.relu(conv_out)
            conv_out = self.dropout(conv_out, training=training)

        trim_len: tf.Tensor = tf.math.reduce_max(sentence_length)

        return self.inference_layer(trim1d(conv_out, trim_len), trim2d(mask3d, trim_len))

    def get_config(self):
        pass
