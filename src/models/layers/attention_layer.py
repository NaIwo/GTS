from src.utils import config

from tensorflow import keras
import tensorflow as tf


class Attention(keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        model_type: str = config['model']['type']
        attention_dim: int = config['model'][model_type]['attention-dimension']
        self.W1: keras.layers.Layer = tf.keras.layers.Dense(attention_dim, activation='linear')
        self.W2: keras.layers.Layer = tf.keras.layers.Dense(attention_dim, activation='linear')
        self.v: keras.layers.Layer = tf.keras.layers.Dense(1, activation='linear', use_bias=False)
        self.tanh: tf.keras.activations = tf.keras.activations.tanh
        self.softmax: keras.layers.activations = tf.keras.activations.softmax

    def call(self, matrix: tf.Tensor, mask: tf.Tensor, **kwargs) -> tf.Tensor:
        w1: tf.Tensor = self.linear_operation(layer=self.W1, matrix=matrix)
        w2: tf.Tensor = self.linear_operation(layer=self.W2, matrix=matrix)
        w2 = tf.transpose(w2, perm=[0, 2, 1, 3])

        tanh: tf.Tensor = self.tanh(w1 + w2)
        v: tf.Tensor = self.v(tanh)
        v = tf.squeeze(v)

        boolean_mask: tf.Tensor = tf.cast(mask[..., 0], dtype=bool)
        softmax_input: tf.Tensor = tf.where(boolean_mask, v, tf.float64.min)

        attention: tf.Tensor = self.softmax(softmax_input)
        attention = tf.where(tf.math.is_nan(attention), 0.0, attention)

        return attention @ matrix



    @staticmethod
    def linear_operation(layer: tf.keras.layers.Layer, matrix: tf.Tensor) -> tf.Tensor:
        w: tf.Tensor = layer(matrix)
        w = tf.expand_dims(w, axis=2)
        return tf.repeat(w, repeats=w.shape[1], axis=2)


